//! Search functions — semantic, BM25, and hybrid with Reciprocal Rank Fusion.
//!
//! All ranking is index-keyed (`Vec<f64>` of length `chunks.len()`); chunks are
//! cloned exactly once at the very end when materialising `SearchResult`s.

use crate::index::dense::DenseIndex;
use crate::index::sparse::Bm25Index;
use crate::ranking::{apply_query_boost, boost_multi_chunk_files, rerank_topk, resolve_alpha};
use crate::tokenizer::tokenize;
use crate::types::{Chunk, SearchMode, SearchResult};

const RRF_K: f64 = 60.0;

/// Convert an ordered `(idx, raw_score)` ranking into RRF scores.
///
/// `out[idx] = 1 / (RRF_K + rank + 1)` for each ranked entry; remaining slots stay 0.0.
fn fill_rrf(out: &mut [f64], ranked: &[(usize, f64)]) {
    for (rank, (idx, _)) in ranked.iter().enumerate() {
        if *idx < out.len() {
            out[*idx] = 1.0 / (RRF_K + (rank + 1) as f64);
        }
    }
}

/// Run semantic (dense) search for a query.
pub fn search_semantic(
    query: &str,
    model: &model2vec_rs::model::StaticModel,
    dense_index: &DenseIndex,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    let query_embedding = model.encode(&[query.to_string()]);
    let query_vec = &query_embedding[0];
    let (indices, similarities) = dense_index.query(query_vec, top_k, selector);

    indices
        .into_iter()
        .zip(similarities)
        .map(|(index, similarity)| SearchResult {
            chunk: chunks[index].clone(),
            score: similarity as f64,
            source: SearchMode::Semantic,
        })
        .collect()
}

/// Run BM25 (sparse) search for a query.
pub fn search_bm25(
    query: &str,
    bm25_index: &Bm25Index,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    let tokens = tokenize(query);
    if tokens.is_empty() {
        return Vec::new();
    }
    let results = bm25_index.top_k(&tokens, top_k, selector);
    results
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: chunks[idx].clone(),
            score,
            source: SearchMode::Bm25,
        })
        .collect()
}

/// Hybrid search: alpha-weighted combination of semantic and BM25 scores after RRF.
pub fn search_hybrid(
    query: &str,
    model: &model2vec_rs::model::StaticModel,
    dense_index: &DenseIndex,
    bm25_index: &Bm25Index,
    chunks: &[Chunk],
    top_k: usize,
    alpha: Option<f64>,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    if chunks.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let alpha_weight = resolve_alpha(query, alpha);
    let candidate_count = top_k * 5;
    let n = chunks.len();

    // Semantic candidates → indexed RRF scores.
    let query_emb = model.encode(&[query.to_string()]);
    let (sem_idx, sem_sim) = dense_index.query(&query_emb[0], candidate_count, selector);
    let sem_topk: Vec<(usize, f64)> = sem_idx
        .into_iter()
        .zip(sem_sim.into_iter())
        .map(|(i, s)| (i, s as f64))
        .collect();

    // BM25 candidates → indexed RRF scores.
    let tokens = tokenize(query);
    let bm25_topk = if tokens.is_empty() {
        Vec::new()
    } else {
        bm25_index.top_k(&tokens, candidate_count, selector)
    };

    let mut sem_rrf = vec![0.0f64; n];
    fill_rrf(&mut sem_rrf, &sem_topk);
    let mut bm25_rrf = vec![0.0f64; n];
    fill_rrf(&mut bm25_rrf, &bm25_topk);

    // Combine.
    let mut combined: Vec<f64> = vec![0.0f64; n];
    for i in 0..n {
        let s = sem_rrf[i];
        let b = bm25_rrf[i];
        if s > 0.0 || b > 0.0 {
            combined[i] = alpha_weight * s + (1.0 - alpha_weight) * b;
        }
    }

    // Boost multi-chunk files, then apply query-type boosts.
    boost_multi_chunk_files(&mut combined, chunks);
    apply_query_boost(&mut combined, query, chunks);

    // Rerank top-k with path penalties + file saturation.
    let ranked = rerank_topk(&combined, chunks, top_k, alpha_weight < 1.0);

    ranked
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: chunks[idx].clone(),
            score,
            source: SearchMode::Hybrid,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_rrf() {
        let mut out = vec![0.0f64; 4];
        let ranked = vec![(2, 10.0), (0, 5.0)];
        fill_rrf(&mut out, &ranked);
        assert!(out[2] > out[0]); // higher raw score → lower rank → higher RRF
        assert_eq!(out[1], 0.0);
        assert_eq!(out[3], 0.0);
    }
}
