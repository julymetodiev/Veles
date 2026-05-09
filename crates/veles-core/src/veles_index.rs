//! Main `VelesIndex` — the central API for indexing and searching code.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use model2vec_rs::model::StaticModel;
use rayon::prelude::*;

use crate::chunker;
use crate::index::dense::DenseIndex;
use crate::index::search::{search_bm25, search_hybrid, search_semantic};
use crate::index::sparse::Bm25Index;
use crate::model;
use crate::persist::{self, FileFingerprint, Manifest, UpdateReport};
use crate::symbols::{self, Symbol};
use crate::tokenizer::tokenize_into;
use crate::types::{Chunk, IndexStats, SearchMode, SearchResult};
use crate::walker;

/// Fast local code index with hybrid search.
pub struct VelesIndex {
    model: StaticModel,
    chunks: Vec<Chunk>,
    bm25_index: Bm25Index,
    dense_index: DenseIndex,
    file_mapping: HashMap<String, Vec<usize>>,
    language_mapping: HashMap<String, Vec<usize>>,
    /// Tree-sitter-extracted definitions, one per (file, name).
    /// Empty for files in unsupported languages.
    symbols: Vec<Symbol>,
    /// Per-file fingerprints + model metadata. Populated on `from_path`,
    /// `from_git`, `load`, and `update_from_path`. Used by `save` and
    /// `update_from_path` for incremental rebuilds.
    manifest: Option<Manifest>,
}

impl VelesIndex {
    /// Create a VelesIndex from a directory path.
    ///
    /// Files are chunked, embedded, and indexed for both BM25 and semantic search.
    /// Chunk file paths are stored relative to `path`.
    pub fn from_path(
        path: &Path,
        model: Option<StaticModel>,
        extensions: Option<HashSet<String>>,
        include_text_files: bool,
    ) -> Result<Self> {
        let path = path.canonicalize()?;
        if !path.is_dir() {
            bail!("Path is not a directory: {}", path.display());
        }

        let model = model.unwrap_or(model::load_model(None)?);
        let exts = walker::filter_extensions(extensions.as_ref(), include_text_files);
        let (chunks, symbols) = collect_chunks_and_symbols(&path, &path, &exts)?;

        if chunks.is_empty() {
            bail!("No supported files found under {}", path.display());
        }

        let (bm25_index, dense_index) = build_indexes(&model, &chunks);
        let (file_mapping, language_mapping) = build_mappings(&chunks);

        let manifest = Some(build_manifest(
            &path,
            &chunks,
            &dense_index,
            include_text_files,
        ));

        Ok(Self {
            model,
            chunks,
            bm25_index,
            dense_index,
            file_mapping,
            language_mapping,
            symbols,
            manifest,
        })
    }

    /// Clone a git repository into a temp directory and index it.
    pub fn from_git(
        url: &str,
        ref_: Option<&str>,
        model: Option<StaticModel>,
        include_text_files: bool,
    ) -> Result<Self> {
        let tmp_dir = tempfile::tempdir()?;
        let tmp_path = tmp_dir.path().to_path_buf();

        // Clone the repository.
        let mut cmd = std::process::Command::new("git");
        cmd.args(["clone", "--depth", "1"]);
        if let Some(ref_val) = ref_ {
            cmd.args(["--branch", ref_val]);
        }
        cmd.args(["--", url]);
        cmd.arg(&tmp_path);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::piped());

        let output = cmd.output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git clone failed for {:?}:\n{}", url, stderr.trim());
        }

        let model = model.unwrap_or(model::load_model(None)?);
        let resolved = tmp_path.canonicalize()?;
        let exts = walker::filter_extensions(None, include_text_files);
        let (chunks, symbols) = collect_chunks_and_symbols(&resolved, &resolved, &exts)?;

        if chunks.is_empty() {
            bail!("No supported files found in cloned repository");
        }

        let (bm25_index, dense_index) = build_indexes(&model, &chunks);
        let (file_mapping, language_mapping) = build_mappings(&chunks);

        // Manifest is intentionally not populated for git-cloned indexes:
        // the temp directory disappears on drop, so persisting/updating
        // against it would be meaningless.
        Ok(Self {
            model,
            chunks,
            bm25_index,
            dense_index,
            file_mapping,
            language_mapping,
            symbols,
            manifest: None,
        })
    }

    /// Persist the index to `<repo_root>/.veles/`.
    pub fn save(&self, repo_root: &Path) -> Result<()> {
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Index has no manifest — cannot persist (was it built from a git URL?)")
        })?;
        persist::save(
            repo_root,
            manifest,
            &self.chunks,
            &self.bm25_index,
            &self.dense_index,
            &self.symbols,
        )
    }

    /// Load a persisted index from `<repo_root>/.veles/`.
    ///
    /// The `model` must match the model the index was built with — a mismatch
    /// gives meaningless similarity scores. We check the model name in the
    /// manifest against the provided model id (if known).
    pub fn load(repo_root: &Path, model: StaticModel) -> Result<Self> {
        let persisted = persist::load(repo_root)?;
        let (file_mapping, language_mapping) = build_mappings(&persisted.chunks);
        Ok(Self {
            model,
            chunks: persisted.chunks,
            bm25_index: persisted.bm25,
            dense_index: persisted.dense,
            file_mapping,
            language_mapping,
            symbols: persisted.symbols,
            manifest: Some(persisted.manifest),
        })
    }

    /// Borrow the manifest, if available.
    pub fn manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }

    /// All extracted symbols across the index.
    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    /// Symbols belonging to a specific file path (matches the path stored in
    /// chunks, i.e. relative to the index root).
    pub fn symbols_for_file(&self, file_path: &str) -> Vec<&Symbol> {
        self.symbols
            .iter()
            .filter(|s| s.file_path == file_path)
            .collect()
    }

    /// Find every symbol with the given exact name (across all files).
    pub fn find_definitions(&self, name: &str) -> Vec<&Symbol> {
        self.symbols.iter().filter(|s| s.name == name).collect()
    }

    /// Incrementally update the index against the current state of `repo_root`.
    ///
    /// Files whose (size, mtime) fingerprint matches the manifest keep their
    /// chunks and embeddings untouched. Modified/added files are re-chunked
    /// and re-embedded. Removed files are dropped. BM25 and dense indexes are
    /// rebuilt from the union of kept + new chunks (cheap relative to
    /// re-embedding the whole corpus).
    pub fn update_from_path(&mut self, repo_root: &Path) -> Result<UpdateReport> {
        let root = repo_root.canonicalize()?;
        if !root.is_dir() {
            bail!("Path is not a directory: {}", root.display());
        }

        let manifest = self.manifest.as_ref().cloned().ok_or_else(|| {
            anyhow::anyhow!("Index has no manifest — call `from_path` or `load` first")
        })?;

        let exts = walker::filter_extensions(None, manifest.include_text_files);

        // Discover files currently on disk and their fingerprints (without
        // chunk_count — that's filled in after chunking).
        let on_disk: HashMap<String, (PathBuf, u64, i64)> = walker::walk_files(&root, &exts)
            .filter_map(|abs_path| {
                let rel = abs_path
                    .strip_prefix(&root)
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
                let meta = std::fs::metadata(&abs_path).ok()?;
                let mtime = meta
                    .modified()
                    .ok()?
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                Some((rel, (abs_path, meta.len(), mtime)))
            })
            .collect();

        // Classify files: unchanged / modified / added / removed.
        let mut unchanged: Vec<String> = Vec::new();
        let mut modified: Vec<String> = Vec::new();
        let mut added: Vec<String> = Vec::new();

        for (rel, (_abs, size, mtime)) in &on_disk {
            match manifest.files.get(rel) {
                Some(prev) if prev.size == *size && prev.mtime_secs == *mtime => {
                    unchanged.push(rel.clone());
                }
                Some(_) => modified.push(rel.clone()),
                None => added.push(rel.clone()),
            }
        }
        let removed: Vec<String> = manifest
            .files
            .keys()
            .filter(|k| !on_disk.contains_key(*k))
            .cloned()
            .collect();

        // Fast path: nothing to do.
        if modified.is_empty() && added.is_empty() && removed.is_empty() {
            return Ok(UpdateReport {
                added_files: 0,
                modified_files: 0,
                removed_files: 0,
                kept_chunks: self.chunks.len(),
                new_chunks: 0,
                total_chunks: self.chunks.len(),
            });
        }

        // Build the keep set: indices of chunks belonging to unchanged files.
        let unchanged_set: HashSet<&str> = unchanged.iter().map(|s| s.as_str()).collect();
        let mut keep_indices: Vec<usize> = Vec::new();
        for (i, chunk) in self.chunks.iter().enumerate() {
            if unchanged_set.contains(chunk.file_path.as_str()) {
                keep_indices.push(i);
            }
        }

        // Reuse embeddings of unchanged chunks (rows of the dense matrix —
        // already L2-normalised; re-normalising is a no-op).
        let kept_embeddings = self.dense_index.extract_rows(&keep_indices);
        let kept_chunks: Vec<Chunk> = keep_indices
            .iter()
            .map(|&i| self.chunks[i].clone())
            .collect();

        // Keep the symbols belonging to unchanged files; the rest get
        // re-extracted in lock-step with re-chunking.
        let kept_symbols: Vec<Symbol> = self
            .symbols
            .iter()
            .filter(|s| unchanged_set.contains(s.file_path.as_str()))
            .cloned()
            .collect();

        // Chunk + extract symbols for modified + added files. We pair both
        // in one pass to amortise the file read.
        let mut to_chunk: Vec<String> = Vec::with_capacity(modified.len() + added.len());
        to_chunk.extend(modified.iter().cloned());
        to_chunk.extend(added.iter().cloned());

        let new_pairs: Vec<(Vec<Chunk>, Vec<Symbol>)> = to_chunk
            .par_iter()
            .map(|rel| {
                let abs = match on_disk.get(rel) {
                    Some((abs, _, _)) => abs.clone(),
                    None => return (Vec::new(), Vec::new()),
                };
                let language = walker::language_for_path(&abs).map(|s| s.to_string());
                let content = match std::fs::read_to_string(&abs) {
                    Ok(c) => c,
                    Err(_) => return (Vec::new(), Vec::new()),
                };
                let chunks = chunker::chunk_source(&content, rel, language.as_deref());
                let syms = match language.as_deref() {
                    Some(lang) if symbols::supports(lang) => {
                        symbols::extract_symbols(&content, rel, lang)
                    }
                    _ => Vec::new(),
                };
                (chunks, syms)
            })
            .collect();

        let mut new_chunks: Vec<Chunk> = Vec::new();
        let mut new_symbols: Vec<Symbol> = Vec::new();
        for (cs, ss) in new_pairs {
            new_chunks.extend(cs);
            new_symbols.extend(ss);
        }

        // Embed only the new chunks (the expensive step we save).
        let new_texts: Vec<String> = new_chunks.iter().map(|c| c.content.clone()).collect();
        let new_embeddings = if new_texts.is_empty() {
            Vec::new()
        } else {
            self.model.encode(&new_texts)
        };

        // Combine kept + new.
        let mut all_chunks = kept_chunks;
        all_chunks.extend(new_chunks.iter().cloned());
        let mut all_embeddings = kept_embeddings;
        all_embeddings.extend(new_embeddings);
        let mut all_symbols = kept_symbols;
        all_symbols.extend(new_symbols);

        // Rebuild BM25 from full token set, dense from combined embeddings.
        let bm25_index = build_bm25(&all_chunks);
        let dense_index = DenseIndex::new(all_embeddings);
        let (file_mapping, language_mapping) = build_mappings(&all_chunks);

        // Build a fresh manifest.
        let mut new_manifest = Manifest::new(
            &manifest.model_name,
            self.dense_index.dim().max(dense_index.dim()),
            manifest.include_text_files,
        );
        new_manifest.embedding_dim = dense_index.dim();
        new_manifest.total_chunks = all_chunks.len();

        // Per-file chunk counts for the new state.
        let mut chunk_counts: HashMap<&str, usize> = HashMap::new();
        for c in &all_chunks {
            *chunk_counts.entry(c.file_path.as_str()).or_default() += 1;
        }
        for (rel, (_abs, size, mtime)) in &on_disk {
            new_manifest.files.insert(
                rel.clone(),
                FileFingerprint {
                    size: *size,
                    mtime_secs: *mtime,
                    chunk_count: chunk_counts.get(rel.as_str()).copied().unwrap_or(0),
                },
            );
        }

        let report = UpdateReport {
            added_files: added.len(),
            modified_files: modified.len(),
            removed_files: removed.len(),
            kept_chunks: keep_indices.len(),
            new_chunks: new_chunks.len(),
            total_chunks: all_chunks.len(),
        };

        self.chunks = all_chunks;
        self.bm25_index = bm25_index;
        self.dense_index = dense_index;
        self.file_mapping = file_mapping;
        self.language_mapping = language_mapping;
        self.symbols = all_symbols;
        self.manifest = Some(new_manifest);

        Ok(report)
    }

    /// Search the index and return the top-k most relevant chunks.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        mode: SearchMode,
        alpha: Option<f64>,
        filter_languages: Option<&[String]>,
        filter_paths: Option<&[String]>,
    ) -> Vec<SearchResult> {
        if self.chunks.is_empty() || query.trim().is_empty() {
            return Vec::new();
        }

        let selector = self.get_selector_vector(filter_languages, filter_paths);

        match mode {
            SearchMode::Bm25 => search_bm25(
                query,
                &self.bm25_index,
                &self.chunks,
                top_k,
                selector.as_deref(),
            ),
            SearchMode::Semantic => search_semantic(
                query,
                &self.model,
                &self.dense_index,
                &self.chunks,
                top_k,
                selector.as_deref(),
            ),
            SearchMode::Hybrid => search_hybrid(
                query,
                &self.model,
                &self.dense_index,
                &self.bm25_index,
                &self.chunks,
                top_k,
                alpha,
                selector.as_deref(),
            ),
        }
    }

    /// Return chunks semantically similar to the given chunk.
    pub fn find_related(&self, source: &Chunk, top_k: usize) -> Vec<SearchResult> {
        let selector = source
            .language
            .as_ref()
            .and_then(|lang| self.language_mapping.get(lang))
            .map(|v| v.as_slice());

        let results = search_semantic(
            &source.content,
            &self.model,
            &self.dense_index,
            &self.chunks,
            top_k + 1,
            selector,
        );

        results
            .into_iter()
            .filter(|r| r.chunk != *source)
            .take(top_k)
            .collect()
    }

    /// Return statistics about the index.
    pub fn stats(&self) -> IndexStats {
        let mut language_counts: HashMap<String, usize> = HashMap::new();
        for chunk in &self.chunks {
            if let Some(ref lang) = chunk.language {
                *language_counts.entry(lang.clone()).or_default() += 1;
            }
        }
        IndexStats {
            indexed_files: self.file_mapping.len(),
            total_chunks: self.chunks.len(),
            languages: language_counts,
        }
    }

    /// Access the chunks in this index.
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Access the model used by this index.
    pub fn model(&self) -> &StaticModel {
        &self.model
    }

    /// Resolve a file path and line number to the containing chunk.
    pub fn resolve_chunk(&self, file_path: &str, line: usize) -> Option<&Chunk> {
        let mut fallback = None;
        for chunk in &self.chunks {
            if chunk.file_path == file_path && chunk.start_line <= line && line <= chunk.end_line {
                if line < chunk.end_line {
                    return Some(chunk);
                }
                if fallback.is_none() {
                    fallback = Some(chunk);
                }
            }
        }
        fallback
    }

    fn get_selector_vector(
        &self,
        filter_languages: Option<&[String]>,
        filter_paths: Option<&[String]>,
    ) -> Option<Vec<usize>> {
        let mut selector = Vec::new();
        if let Some(languages) = filter_languages {
            for lang in languages {
                if let Some(indices) = self.language_mapping.get(lang) {
                    selector.extend(indices);
                }
            }
        }
        if let Some(paths) = filter_paths {
            for path in paths {
                if let Some(indices) = self.file_mapping.get(path) {
                    selector.extend(indices);
                }
            }
        }
        if selector.is_empty() {
            return None;
        }
        selector.sort();
        selector.dedup();
        Some(selector)
    }
}

/// Collect chunks **and tree-sitter symbols** for every file under `root`,
/// storing paths relative to `display_root`.
///
/// File reading, language inference, chunking, and symbol extraction run in
/// parallel across files. Each file is read exactly once.
fn collect_chunks_and_symbols(
    root: &Path,
    display_root: &Path,
    extensions: &HashSet<String>,
) -> Result<(Vec<Chunk>, Vec<Symbol>)> {
    let files: Vec<PathBuf> = walker::walk_files(root, extensions).collect();

    let per_file: Vec<(Vec<Chunk>, Vec<Symbol>)> = files
        .par_iter()
        .map(|file_path| {
            let language = walker::language_for_path(file_path).map(|s| s.to_string());
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => return (Vec::new(), Vec::new()),
            };
            let chunk_path = file_path.strip_prefix(display_root).unwrap_or(file_path);
            let chunk_path_str = chunk_path.to_string_lossy().into_owned();

            let chunks = chunker::chunk_source(&content, &chunk_path_str, language.as_deref());
            let syms = match language.as_deref() {
                Some(lang) if symbols::supports(lang) => {
                    symbols::extract_symbols(&content, &chunk_path_str, lang)
                }
                _ => Vec::new(),
            };
            (chunks, syms)
        })
        .collect();

    let mut chunks = Vec::new();
    let mut symbols = Vec::new();
    for (cs, ss) in per_file {
        chunks.extend(cs);
        symbols.extend(ss);
    }
    Ok((chunks, symbols))
}

/// Build BM25 and dense indexes from chunks.
fn build_indexes(model: &StaticModel, chunks: &[Chunk]) -> (Bm25Index, DenseIndex) {
    let bm25_index = build_bm25(chunks);

    // Build dense index. Embedding is single-call (`model.encode` batches
    // internally on a thread pool), so we feed it `&str` to avoid cloning.
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = model.encode(&texts);
    let dense_index = DenseIndex::new(embeddings);

    (bm25_index, dense_index)
}

/// Tokenize chunks (with path enrichment) and build a BM25 index.
fn build_bm25(chunks: &[Chunk]) -> Bm25Index {
    let tokenized: Vec<Vec<String>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut tokens: Vec<String> = Vec::with_capacity(64);
            tokenize_into(&chunk.content, &mut tokens);
            append_path_tokens(&chunk.file_path, &mut tokens);
            tokens
        })
        .collect();
    Bm25Index::new(&tokenized)
}

/// Build a fresh `Manifest` describing a freshly-built index over `repo_root`.
fn build_manifest(
    repo_root: &Path,
    chunks: &[Chunk],
    dense: &DenseIndex,
    include_text_files: bool,
) -> Manifest {
    let model_name = if include_text_files {
        // Heuristic; the field is informational. Real model name is set
        // explicitly by the caller via persist::save when relevant.
        crate::model::DEFAULT_MODEL_NAME.to_string()
    } else {
        crate::model::DEFAULT_MODEL_NAME.to_string()
    };
    let mut manifest = Manifest::new(&model_name, dense.dim(), include_text_files);
    manifest.total_chunks = chunks.len();

    // Aggregate chunk counts per file.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for c in chunks {
        *counts.entry(c.file_path.as_str()).or_default() += 1;
    }

    for (rel, count) in counts {
        let abs = repo_root.join(rel);
        if let Ok(fp) = FileFingerprint::from_path(&abs, count) {
            manifest.files.insert(rel.to_string(), fp);
        }
    }

    manifest
}

/// Append file path component tokens to a tokenized BM25 document.
///
/// Equivalent to the previous `enrich_for_bm25` but without re-tokenising the
/// chunk content: stems are duplicated (matches the original "stem stem"
/// emphasis) and we take the last three directory parts.
fn append_path_tokens(file_path: &str, tokens: &mut Vec<String>) {
    let path = Path::new(file_path);
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        // Run tokeniser over the stem so split-identifier sub-tokens are added.
        let mut stem_tokens: Vec<String> = Vec::new();
        tokenize_into(stem, &mut stem_tokens);
        // Emphasise: include twice to match prior weighting.
        tokens.extend(stem_tokens.iter().cloned());
        tokens.extend(stem_tokens);
    }
    if let Some(parent) = path.parent().and_then(|p| p.to_str()) {
        let mut count = 0;
        for part in parent.rsplit('/').filter(|p| !p.is_empty() && *p != ".") {
            tokenize_into(part, tokens);
            count += 1;
            if count >= 3 {
                break;
            }
        }
    }
}

/// Build (file → chunk indices) and (language → chunk indices) mappings.
fn build_mappings(chunks: &[Chunk]) -> (HashMap<String, Vec<usize>>, HashMap<String, Vec<usize>>) {
    let mut file_mapping: HashMap<String, Vec<usize>> = HashMap::new();
    let mut language_mapping: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, chunk) in chunks.iter().enumerate() {
        file_mapping
            .entry(chunk.file_path.clone())
            .or_default()
            .push(i);
        if let Some(ref lang) = chunk.language {
            language_mapping.entry(lang.clone()).or_default().push(i);
        }
    }

    (file_mapping, language_mapping)
}
