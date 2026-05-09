//! Persistent on-disk index format.
//!
//! Layout under `<repo>/.veles/`:
//!
//! ```text
//! .veles/
//!   manifest.json   - format version, model, per-file fingerprints
//!   chunks.bin      - bincode Vec<Chunk>
//!   bm25.bin        - bincode Bm25Index
//!   dense.bin       - bincode DenseIndex
//! ```
//!
//! The manifest records a (size, mtime, chunk_count) fingerprint per file so
//! `update` can detect added / removed / modified files without re-reading
//! everything.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::index::dense::DenseIndex;
use crate::index::sparse::Bm25Index;
use crate::symbols::Symbol;
use crate::types::Chunk;

/// Directory name used under the indexed repo to store the on-disk index.
pub const INDEX_DIR_NAME: &str = ".veles";

/// Bumped whenever the on-disk format changes incompatibly. Bumped to 2
/// when symbols.bin was added — older indexes lack tree-sitter symbols.
pub const FORMAT_VERSION: u32 = 2;

const MANIFEST_FILE: &str = "manifest.json";
const CHUNKS_FILE: &str = "chunks.bin";
const BM25_FILE: &str = "bm25.bin";
const DENSE_FILE: &str = "dense.bin";
const SYMBOLS_FILE: &str = "symbols.bin";

/// Cheap fingerprint for change detection.
///
/// `(size, mtime)` is fast to compute and covers almost all real edits;
/// `content_hash` (BLAKE3 of the file bytes) is the fallback used by
/// incremental update when mtime drifts but the bytes haven't changed
/// (`touch`, `git checkout` of an identical version, no-op formatter
/// runs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFingerprint {
    /// File size in bytes.
    pub size: u64,
    /// Modification time as Unix epoch seconds.
    pub mtime_secs: i64,
    /// Number of chunks this file produced.
    pub chunk_count: usize,
    /// BLAKE3 hex digest of the file bytes. `None` for fingerprints
    /// loaded from a pre-content-hash manifest; new fingerprints
    /// always populate it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

impl FileFingerprint {
    /// Compute the fingerprint for a path on disk. `chunk_count` is provided
    /// by the caller after chunking.
    pub fn from_path(path: &Path, chunk_count: usize) -> Result<Self> {
        let meta = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
        let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
        let mtime_secs = mtime
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let content_hash = Some(content_hash(path)?);
        Ok(Self {
            size: meta.len(),
            mtime_secs,
            chunk_count,
            content_hash,
        })
    }
}

/// BLAKE3 hex digest of `path`'s bytes. Used both at index-build time
/// (to populate the manifest) and at update time (to verify whether a
/// touched file's content actually changed).
pub fn content_hash(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Small JSON sidecar describing a persisted index.
///
/// Human-readable on purpose so users can `cat .veles/manifest.json` to
/// debug staleness or model mismatches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Version of `veles` that wrote this index (from `CARGO_PKG_VERSION`).
    pub veles_version: String,
    /// On-disk format version. Bumped on incompatible layout changes; a
    /// mismatch on `load` forces a `veles index --force`.
    pub format_version: u32,
    /// Embedding model used at build time (e.g. `"minishlab/potion-code-16M"`).
    /// Loading with a different model is rejected.
    pub model_name: String,
    /// Dimensionality of the dense vectors.
    pub embedding_dim: usize,
    /// Whether text/document files (markdown, yaml, ...) were indexed
    /// alongside source code.
    pub include_text_files: bool,
    /// Unix epoch seconds when the index was last written.
    pub indexed_at: i64,
    /// Per-file fingerprints used by incremental update.
    pub files: BTreeMap<String, FileFingerprint>,
    /// Total chunks across all files.
    pub total_chunks: usize,
}

impl Manifest {
    pub fn new(model_name: &str, embedding_dim: usize, include_text_files: bool) -> Self {
        Self {
            veles_version: env!("CARGO_PKG_VERSION").to_string(),
            format_version: FORMAT_VERSION,
            model_name: model_name.to_string(),
            embedding_dim,
            include_text_files,
            indexed_at: now_secs(),
            files: BTreeMap::new(),
            total_chunks: 0,
        }
    }

    pub fn touch(&mut self) {
        self.indexed_at = now_secs();
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Path of the `.veles/` directory under a given repo root.
pub fn index_dir_for(repo_root: &Path) -> PathBuf {
    repo_root.join(INDEX_DIR_NAME)
}

/// Returns true if a saved index appears to exist at the given path.
pub fn index_exists(repo_root: &Path) -> bool {
    let dir = index_dir_for(repo_root);
    dir.join(MANIFEST_FILE).is_file()
        && dir.join(CHUNKS_FILE).is_file()
        && dir.join(BM25_FILE).is_file()
        && dir.join(DENSE_FILE).is_file()
}

/// Components of a loaded index — the model is provided separately at load
/// time so the heavy weights aren't serialised.
pub struct PersistedIndex {
    pub manifest: Manifest,
    pub chunks: Vec<Chunk>,
    pub bm25: Bm25Index,
    pub dense: DenseIndex,
    pub symbols: Vec<Symbol>,
}

/// Write all index artefacts to `<repo_root>/.veles/`.
pub fn save(
    repo_root: &Path,
    manifest: &Manifest,
    chunks: &[Chunk],
    bm25: &Bm25Index,
    dense: &DenseIndex,
    symbols: &[Symbol],
) -> Result<()> {
    let dir = index_dir_for(repo_root);
    fs::create_dir_all(&dir).with_context(|| format!("create index dir {}", dir.display()))?;

    write_json(&dir.join(MANIFEST_FILE), manifest)?;
    write_bincode(&dir.join(CHUNKS_FILE), &chunks.to_vec())?;
    write_bincode(&dir.join(BM25_FILE), bm25)?;
    write_bincode(&dir.join(DENSE_FILE), dense)?;
    write_bincode(&dir.join(SYMBOLS_FILE), &symbols.to_vec())?;
    Ok(())
}

/// Load all index artefacts from `<repo_root>/.veles/`.
pub fn load(repo_root: &Path) -> Result<PersistedIndex> {
    let dir = index_dir_for(repo_root);
    if !dir.is_dir() {
        bail!("No index found at {}", dir.display());
    }

    let manifest: Manifest = read_json(&dir.join(MANIFEST_FILE))?;
    if manifest.format_version != FORMAT_VERSION {
        bail!(
            "Index format version {} is incompatible (expected {}). Run `veles index --force` to rebuild.",
            manifest.format_version,
            FORMAT_VERSION
        );
    }
    let chunks: Vec<Chunk> = read_bincode(&dir.join(CHUNKS_FILE))?;
    let bm25: Bm25Index = read_bincode(&dir.join(BM25_FILE))?;
    let dense: DenseIndex = read_bincode(&dir.join(DENSE_FILE))?;
    // Symbols file may be missing on a partially-written index; treat as empty.
    let symbols: Vec<Symbol> = if dir.join(SYMBOLS_FILE).is_file() {
        read_bincode(&dir.join(SYMBOLS_FILE))?
    } else {
        Vec::new()
    };

    Ok(PersistedIndex {
        manifest,
        chunks,
        bm25,
        dense,
        symbols,
    })
}

/// Read just the manifest (cheap — used by `status` and to check compatibility).
pub fn load_manifest(repo_root: &Path) -> Result<Manifest> {
    let dir = index_dir_for(repo_root);
    read_json(&dir.join(MANIFEST_FILE))
}

/// Remove the on-disk index directory if it exists.
pub fn clean(repo_root: &Path) -> Result<bool> {
    let dir = index_dir_for(repo_root);
    if dir.is_dir() {
        fs::remove_dir_all(&dir).with_context(|| format!("remove {}", dir.display()))?;
        return Ok(true);
    }
    Ok(false)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(f, value).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let value = serde_json::from_reader(std::io::BufReader::new(f))
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(value)
}

fn write_bincode<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    bincode::serialize_into(&mut w, value).with_context(|| format!("encode {}", path.display()))?;
    Ok(())
}

fn read_bincode<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let f = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = std::io::BufReader::new(f);
    let value =
        bincode::deserialize_from(r).with_context(|| format!("decode {}", path.display()))?;
    Ok(value)
}

/// Outcome of an incremental update — returned by
/// [`crate::VelesIndex::update_from_path`].
#[derive(Debug, Default, Clone)]
pub struct UpdateReport {
    /// Files seen on disk that weren't in the previous manifest.
    pub added_files: usize,
    /// Files whose `(size, mtime)` fingerprint changed.
    pub modified_files: usize,
    /// Files in the previous manifest no longer present on disk.
    pub removed_files: usize,
    /// Chunks reused from the previous index without re-embedding.
    pub kept_chunks: usize,
    /// Chunks freshly embedded for added/modified files.
    pub new_chunks: usize,
    /// Total chunks in the updated index (`kept + new`).
    pub total_chunks: usize,
}

impl UpdateReport {
    /// True when no files were added, modified, or removed.
    pub fn is_noop(&self) -> bool {
        self.added_files == 0 && self.modified_files == 0 && self.removed_files == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip_via_json() {
        let mut m = Manifest::new("test-model", 64, false);
        m.files.insert(
            "src/lib.rs".to_string(),
            FileFingerprint {
                size: 100,
                mtime_secs: 1_000_000,
                chunk_count: 2,
                content_hash: Some("deadbeef".to_string()),
            },
        );
        m.total_chunks = 2;

        let s = serde_json::to_string(&m).unwrap();
        let m2: Manifest = serde_json::from_str(&s).unwrap();
        assert_eq!(m2.model_name, "test-model");
        assert_eq!(m2.embedding_dim, 64);
        assert_eq!(m2.files.len(), 1);
        assert_eq!(m2.files["src/lib.rs"].size, 100);
        assert_eq!(
            m2.files["src/lib.rs"].content_hash.as_deref(),
            Some("deadbeef")
        );
    }

    #[test]
    fn legacy_manifest_without_content_hash_loads() {
        // Pre-content-hash manifests omit the field entirely. Serde
        // must default it to None, not bail.
        let json = r#"{
            "veles_version": "0.2.3",
            "format_version": 2,
            "model_name": "test-model",
            "embedding_dim": 64,
            "include_text_files": false,
            "indexed_at": 0,
            "files": {
                "src/lib.rs": {
                    "size": 100,
                    "mtime_secs": 1000000,
                    "chunk_count": 2
                }
            },
            "total_chunks": 2
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.files["src/lib.rs"].size, 100);
        assert!(m.files["src/lib.rs"].content_hash.is_none());
    }

    #[test]
    fn content_hash_is_deterministic_and_discriminates() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.txt");

        std::fs::write(&p, b"hello").unwrap();
        let h1 = content_hash(&p).unwrap();
        let h2 = content_hash(&p).unwrap();
        assert_eq!(h1, h2, "same bytes must hash the same");

        std::fs::write(&p, b"hello world").unwrap();
        let h3 = content_hash(&p).unwrap();
        assert_ne!(h1, h3, "different bytes must hash differently");
    }
}
