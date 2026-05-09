//! **veles-core** — Fast code search library for agents.
//!
//! Provides indexing, chunking, BM25, dense vector search, and hybrid ranking.
//! No Python dependencies — pure Rust using [model2vec-rs](https://github.com/MinishLab/model2vec-rs)
//! for static embeddings.

pub mod chunker;
pub mod index;
pub mod model;
pub mod persist;
pub mod ranking;
pub mod veles_index;
pub mod tokenizer;
pub mod types;
pub mod walker;

// Re-export the main types.
pub use veles_index::VelesIndex;
pub use types::{Chunk, IndexStats, SearchMode, SearchResult};
