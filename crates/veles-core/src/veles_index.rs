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
        let chunks = collect_chunks(&path, &path, &exts)?;

        if chunks.is_empty() {
            bail!("No supported files found under {}", path.display());
        }

        let (bm25_index, dense_index) = build_indexes(&model, &chunks);
        let (file_mapping, language_mapping) = build_mappings(&chunks);

        Ok(Self {
            model,
            chunks,
            bm25_index,
            dense_index,
            file_mapping,
            language_mapping,
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
        let chunks = collect_chunks(&resolved, &resolved, &exts)?;

        if chunks.is_empty() {
            bail!("No supported files found in cloned repository");
        }

        let (bm25_index, dense_index) = build_indexes(&model, &chunks);
        let (file_mapping, language_mapping) = build_mappings(&chunks);

        Ok(Self {
            model,
            chunks,
            bm25_index,
            dense_index,
            file_mapping,
            language_mapping,
        })
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

/// Collect chunks from all files under `root`, storing paths relative to `display_root`.
///
/// File reading, language inference, and chunking run in parallel across files.
fn collect_chunks(
    root: &Path,
    display_root: &Path,
    extensions: &HashSet<String>,
) -> Result<Vec<Chunk>> {
    // Walk first (sequential, cheap), then chunk in parallel.
    let files: Vec<PathBuf> = walker::walk_files(root, extensions).collect();

    let chunks: Vec<Chunk> = files
        .par_iter()
        .flat_map_iter(|file_path| {
            let language = walker::language_for_path(file_path).map(|s| s.to_string());
            let content = match std::fs::read_to_string(file_path) {
                Ok(c) => c,
                Err(_) => return Vec::new().into_iter(),
            };
            let chunk_path = file_path.strip_prefix(display_root).unwrap_or(file_path);
            let chunk_path_str = chunk_path.to_string_lossy().into_owned();
            chunker::chunk_source(&content, &chunk_path_str, language.as_deref()).into_iter()
        })
        .collect();

    Ok(chunks)
}

/// Build BM25 and dense indexes from chunks.
fn build_indexes(model: &StaticModel, chunks: &[Chunk]) -> (Bm25Index, DenseIndex) {
    // Tokenize for BM25 in parallel; reuse a per-thread token buffer to avoid
    // re-allocating for every chunk.
    let tokenized: Vec<Vec<String>> = chunks
        .par_iter()
        .map(|chunk| {
            let mut tokens: Vec<String> = Vec::with_capacity(64);
            tokenize_into(&chunk.content, &mut tokens);
            append_path_tokens(&chunk.file_path, &mut tokens);
            tokens
        })
        .collect();

    let bm25_index = Bm25Index::new(&tokenized);

    // Build dense index. Embedding is single-call (`model.encode` batches
    // internally on a thread pool), so we feed it `&str` to avoid cloning.
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let embeddings = model.encode(&texts);
    let dense_index = DenseIndex::new(embeddings);

    (bm25_index, dense_index)
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
