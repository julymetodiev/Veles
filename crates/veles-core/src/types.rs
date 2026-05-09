//! Core types shared across the search surface.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How a query is scored against the index.
///
/// Used by [`crate::VelesIndex::search`] to pick a backend:
///
/// * [`SearchMode::Hybrid`] — BM25 + dense embeddings, fused with
///   Reciprocal Rank Fusion. The default for most queries.
/// * [`SearchMode::Semantic`] — dense (model2vec) embeddings only. Best
///   for fuzzy, concept-level queries.
/// * [`SearchMode::Bm25`] — sparse BM25 only. Best for exact identifier
///   or token lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// BM25 + semantic, blended with RRF and post-filtered.
    Hybrid,
    /// Dense embeddings only.
    Semantic,
    /// BM25 only.
    Bm25,
}

impl SearchMode {
    /// Stable lowercase name used by the CLI, gRPC, and MCP surfaces.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Semantic => "semantic",
            Self::Bm25 => "bm25",
        }
    }
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hybrid" => Ok(Self::Hybrid),
            "semantic" => Ok(Self::Semantic),
            "bm25" => Ok(Self::Bm25),
            other => Err(format!("Unknown search mode: {other:?}")),
        }
    }
}

/// A single indexable unit of code — a contiguous slice of one source
/// file, materialised once at index time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Chunk {
    /// The raw text of this chunk, joined back from the original lines.
    pub content: String,
    /// File path **relative to the index root**, using `/` separators
    /// even on Windows (the value is reused as a cache key).
    pub file_path: String,
    /// 1-indexed start line.
    pub start_line: usize,
    /// 1-indexed inclusive end line.
    pub end_line: usize,
    /// Detected language (e.g. `rust`, `python`), or `None` for files
    /// without a recognised extension.
    pub language: Option<String>,
}

impl Chunk {
    /// Format the location as `path:start-end` (e.g. `src/main.rs:10-25`).
    pub fn location(&self) -> String {
        format!("{}:{}-{}", self.file_path, self.start_line, self.end_line)
    }
}

/// One ranked hit returned from a search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching chunk (cloned out of the index).
    pub chunk: Chunk,
    /// Final ranking score. The scale depends on `source`:
    /// BM25 is unbounded positive, semantic is cosine similarity in
    /// `[-1, 1]`, and hybrid is a blended RRF score (small positive
    /// numbers around `1 / RRF_K`).
    pub score: f64,
    /// Which mode produced this result.
    pub source: SearchMode,
}

/// Summary statistics about an index — returned by
/// [`crate::VelesIndex::stats`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    /// Distinct files contributing at least one chunk.
    pub indexed_files: usize,
    /// Total chunks across all files.
    pub total_chunks: usize,
    /// Per-language chunk counts (e.g. `"rust" -> 1234`).
    pub languages: HashMap<String, usize>,
}
