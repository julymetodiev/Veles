//! Search functions — semantic, BM25, and hybrid with Reciprocal Rank Fusion.
//!
//! All ranking is index-keyed (`Vec<f64>` of length `chunks.len()`); chunks are
//! cloned exactly once at the very end when materialising `SearchResult`s.

use crate::index::dense::DenseIndex;
use crate::index::sparse::Bm25Index;
use crate::ranking::{
    apply_query_boost, boost_multi_chunk_files, is_symbol_query, rerank_topk, resolve_alpha,
};
use crate::tokenizer::tokenize;
use crate::types::{Chunk, SearchMode, SearchResult};

const RRF_K: f64 = 60.0;

/// How many candidates the pure (non-hybrid) modes pull before reranking.
/// The rerank pipeline (path penalties + per-file saturation decay) needs
/// headroom to actually shuffle results — 5× matches what hybrid uses
/// internally for each side of the fusion.
const PURE_MODE_CANDIDATE_OVERSHOOT: usize = 5;

/// Common short-circuit guard for all three search modes.
///
/// An empty index, a `top_k` of zero, or a query that's only whitespace
/// would all produce meaningless results — we'd waste an embedding pass
/// and (for semantic / hybrid) return arbitrary chunks ranked against
/// the embedding of `""`. Returning `Vec::new()` early keeps the public
/// behaviour consistent with `VelesIndex::search`'s own outer guard.
#[inline]
fn should_skip_search(query: &str, top_k: usize, chunks: &[Chunk]) -> bool {
    chunks.is_empty() || top_k == 0 || query.trim().is_empty()
}

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
///
/// Over-fetches a candidate pool, then runs the same boost + rerank pipeline
/// as hybrid search so path penalties (test files, `__init__.py`, compat
/// directories) and definition boosts apply uniformly across modes. Without
/// this, pure semantic mode tends to surface short generic test chunks for
/// short identifier queries because static embeddings have weak signal at
/// that granularity.
pub fn search_semantic(
    query: &str,
    model: &model2vec_rs::model::StaticModel,
    dense_index: &DenseIndex,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    let _span = tracing::trace_span!("search.semantic", top_k, n_chunks = chunks.len()).entered();
    if should_skip_search(query, top_k, chunks) {
        return Vec::new();
    }
    let query_embedding = {
        let _s = tracing::trace_span!("search.encode_query").entered();
        model.encode(&[query.to_string()])
    };
    let query_vec = &query_embedding[0];

    let candidate_count = top_k.saturating_mul(PURE_MODE_CANDIDATE_OVERSHOOT);
    let (indices, similarities) = {
        let _s = tracing::trace_span!("search.dense_query", candidate_count).entered();
        dense_index.query(query_vec, candidate_count, selector)
    };

    let mut scores = vec![0.0f64; chunks.len()];
    for (idx, sim) in indices.iter().zip(similarities.iter()) {
        if *idx < scores.len() {
            scores[*idx] = *sim as f64;
        }
    }

    finalize_pure_mode(scores, query, chunks, top_k, SearchMode::Semantic)
}

/// Run BM25 (sparse) search for a query.
///
/// Bare-identifier queries match against the whole token only — splitting
/// `handle_refs` into `[handle_refs, handle, refs]` lets chunks that
/// reference many `handle_*` functions outrank the chunk that actually
/// defines `handle_refs`. The index still stores sub-tokens, so other
/// queries that just say `handle` continue to match.
///
/// After scoring, the same boost + rerank pipeline as hybrid runs so the
/// definition site is lifted and test/compat paths are demoted.
pub fn search_bm25(
    query: &str,
    bm25_index: &Bm25Index,
    chunks: &[Chunk],
    top_k: usize,
    selector: Option<&[usize]>,
) -> Vec<SearchResult> {
    let _span = tracing::trace_span!("search.bm25", top_k, n_chunks = chunks.len()).entered();
    if should_skip_search(query, top_k, chunks) {
        return Vec::new();
    }
    let tokens = bm25_query_tokens(query);
    if tokens.is_empty() {
        // Non-whitespace query that still yields no tokens (e.g. all
        // punctuation) — BM25 has nothing to score against, so bail out.
        return Vec::new();
    }

    let candidate_count = top_k.saturating_mul(PURE_MODE_CANDIDATE_OVERSHOOT);
    let raw = {
        let _s = tracing::trace_span!("search.bm25_topk", n_tokens = tokens.len()).entered();
        bm25_index.top_k(&tokens, candidate_count, selector)
    };

    let mut scores = vec![0.0f64; chunks.len()];
    for (idx, score) in &raw {
        if *idx < scores.len() {
            scores[*idx] = *score;
        }
    }

    finalize_pure_mode(scores, query, chunks, top_k, SearchMode::Bm25)
}

/// Tokenize a query for BM25 lookup.
///
/// Bare-identifier queries skip sub-token splitting so `handle_refs` only
/// matches `handle_refs`, not every chunk that mentions `handle`. Natural-
/// language queries fall through to the standard splitter.
fn bm25_query_tokens(query: &str) -> Vec<String> {
    if is_symbol_query(query) {
        let trimmed = query.trim().to_lowercase();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed]
        }
    } else {
        tokenize(query)
    }
}

/// Apply boost + rerank to a raw-scored candidate pool and materialise the
/// top-k as `SearchResult`s. Shared between `search_bm25` and `search_semantic`.
fn finalize_pure_mode(
    mut scores: Vec<f64>,
    query: &str,
    chunks: &[Chunk],
    top_k: usize,
    source: SearchMode,
) -> Vec<SearchResult> {
    {
        let _s = tracing::trace_span!("search.boost_multi_chunk").entered();
        boost_multi_chunk_files(&mut scores, chunks);
    }
    {
        let _s = tracing::trace_span!("search.apply_query_boost").entered();
        apply_query_boost(&mut scores, query, chunks);
    }
    let ranked = {
        let _s = tracing::trace_span!("search.rerank_topk").entered();
        rerank_topk(&scores, chunks, top_k, true)
    };
    ranked
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk: chunks[idx].clone(),
            score,
            source,
        })
        .collect()
}

/// Hybrid search: alpha-weighted combination of semantic and BM25 scores after RRF.
#[allow(clippy::too_many_arguments)]
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
    let _span = tracing::trace_span!("search.hybrid", top_k, n_chunks = chunks.len()).entered();
    if should_skip_search(query, top_k, chunks) {
        return Vec::new();
    }
    let alpha_weight = resolve_alpha(query, alpha);
    let candidate_count = top_k * 5;
    let n = chunks.len();

    // Semantic candidates → indexed RRF scores.
    let query_emb = {
        let _s = tracing::trace_span!("search.encode_query").entered();
        model.encode(&[query.to_string()])
    };
    let (sem_idx, sem_sim) = {
        let _s = tracing::trace_span!("search.dense_query", candidate_count).entered();
        dense_index.query(&query_emb[0], candidate_count, selector)
    };
    let sem_topk: Vec<(usize, f64)> = sem_idx
        .into_iter()
        .zip(sem_sim)
        .map(|(i, s)| (i, s as f64))
        .collect();

    // BM25 candidates → indexed RRF scores.
    let tokens = tokenize(query);
    let bm25_topk = if tokens.is_empty() {
        Vec::new()
    } else {
        let _s = tracing::trace_span!("search.bm25_topk", n_tokens = tokens.len()).entered();
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
    {
        let _s = tracing::trace_span!("search.boost_multi_chunk").entered();
        boost_multi_chunk_files(&mut combined, chunks);
    }
    {
        let _s = tracing::trace_span!("search.apply_query_boost").entered();
        apply_query_boost(&mut combined, query, chunks);
    }

    // Rerank top-k with path penalties + file saturation.
    let ranked = {
        let _s = tracing::trace_span!("search.rerank_topk").entered();
        rerank_topk(&combined, chunks, top_k, alpha_weight < 1.0)
    };

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

    fn dummy_chunk() -> Chunk {
        Chunk {
            content: "fn foo() {}".to_string(),
            file_path: "test.rs".to_string(),
            start_line: 1,
            end_line: 1,
            language: Some("rust".to_string()),
        }
    }

    #[test]
    fn skip_when_chunks_empty() {
        let none: Vec<Chunk> = Vec::new();
        assert!(should_skip_search("anything", 5, &none));
    }

    #[test]
    fn skip_when_top_k_zero() {
        let chunks = vec![dummy_chunk()];
        assert!(should_skip_search("anything", 0, &chunks));
    }

    #[test]
    fn skip_when_query_blank() {
        let chunks = vec![dummy_chunk()];
        assert!(should_skip_search("", 5, &chunks));
        assert!(should_skip_search("   ", 5, &chunks));
        assert!(should_skip_search("\t\n", 5, &chunks));
    }

    #[test]
    fn proceed_with_real_inputs() {
        let chunks = vec![dummy_chunk()];
        assert!(!should_skip_search("hello", 5, &chunks));
    }
}
