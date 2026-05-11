//! Dense vector index — brute-force cosine similarity search.
//!
//! Layout: a single flat `Vec<f32>` matrix (N×D), so each row is contiguous in
//! memory and the inner product loop is auto-vectorisable. All stored
//! embeddings are L2-normalised at construction time, which collapses cosine
//! similarity to a plain dot product at query time.
//!
//! Scoring across candidates is parallelised with rayon, and the top-k is
//! computed via a bounded min-heap (O(N log k)) instead of a full sort.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::index::topk::top_k_from_iter_f32;

/// Below this candidate count, parallelism overhead exceeds gains.
const PARALLEL_THRESHOLD: usize = 1024;

/// A dense vector index supporting top-k nearest-neighbor search via cosine similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseIndex {
    /// Flat row-major matrix: N rows of `dim` f32 values, all L2-normalised.
    matrix: Vec<f32>,
    /// Number of vectors.
    n: usize,
    /// Embedding dimensionality.
    dim: usize,
}

impl DenseIndex {
    /// Build a dense index from a matrix of embeddings.
    ///
    /// Each inner `Vec<f32>` is one embedding vector. Vectors are L2-normalised
    /// at insertion so cosine similarity reduces to dot product at query time.
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        let n = embeddings.len();
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);

        let mut matrix = Vec::with_capacity(n * dim);
        for v in &embeddings {
            // Pad/truncate defensively if a vector has unexpected length.
            let mut buf = vec![0.0f32; dim];
            let copy = v.len().min(dim);
            buf[..copy].copy_from_slice(&v[..copy]);
            normalise_in_place(&mut buf);
            matrix.extend_from_slice(&buf);
        }

        Self { matrix, n, dim }
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Extract the (already L2-normalised) row vectors for the given chunk
    /// indices, in the order provided. Used by incremental update to reuse
    /// embeddings of unchanged chunks without re-running the model.
    pub fn extract_rows(&self, indices: &[usize]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            if i < self.n {
                out.push(self.row(i).to_vec());
            }
        }
        out
    }

    /// In-place compaction + extension for incremental update (§2.3).
    ///
    /// `keep_indices` must be sorted ascending and contain valid row
    /// indices into the current matrix; rows are shifted to the front
    /// in that order. The matrix is then truncated to `keep_indices.len()`
    /// rows, the new embeddings are appended and L2-normalised in place.
    ///
    /// The previous flow allocated a `Vec<Vec<f32>>` of kept rows and
    /// then re-allocated the flat matrix from scratch via `DenseIndex::new`
    /// — two full-corpus copies per update. The compact path does one
    /// pass of in-bounds row moves and a single resize. For a single-file
    /// edit on a large corpus, most rows don't move (kept rows already
    /// at their new positions), so the `memmove` cost is dominated by
    /// the small "shift past the gap" work.
    pub fn compact_and_extend(&mut self, keep_indices: &[usize], new_embeddings: Vec<Vec<f32>>) {
        debug_assert!(
            keep_indices.windows(2).all(|w| w[0] < w[1]),
            "keep_indices must be sorted ascending and unique"
        );
        let dim = self.dim;
        let kept = keep_indices.len();

        // Shift kept rows to the front in order. Safe because
        // keep_indices is ascending: for every iteration the source
        // index (`old`) is ≥ the destination (`new`), and no later
        // iteration reads from a destination we've already written
        // (later iterations have new' > new and old' ≥ new' > new, so
        // they read from `old'` which is past `new`).
        for (new_pos, &old_pos) in keep_indices.iter().enumerate() {
            if new_pos == old_pos {
                continue;
            }
            let src_start = old_pos * dim;
            let dst_start = new_pos * dim;
            // copy_within handles non-overlapping or backward-overlap
            // safely; old_pos > new_pos guarantees non-overlap in our
            // direction.
            self.matrix.copy_within(src_start..src_start + dim, dst_start);
        }

        // Truncate to kept rows; append new ones; normalise just the appended slice.
        self.matrix.truncate(kept * dim);
        let total_new = new_embeddings.len();
        self.matrix.reserve(total_new * dim);
        for emb in &new_embeddings {
            let copy = emb.len().min(dim);
            // Append row, padding with zeros if shorter than dim.
            self.matrix.extend_from_slice(&emb[..copy]);
            self.matrix.extend(std::iter::repeat(0.0).take(dim - copy));
        }
        self.n = kept + total_new;

        // Kept rows were already L2-normalised; only the freshly
        // appended rows need normalisation.
        for i in kept..self.n {
            let row = &mut self.matrix[i * dim..(i + 1) * dim];
            normalise_in_place(row);
        }
    }

    /// Borrow row `i` as a slice.
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        let start = i * self.dim;
        &self.matrix[start..start + self.dim]
    }

    /// Query for the top-k nearest neighbors of a single vector.
    ///
    /// Returns `(indices, scores)` where scores are cosine similarity (higher = better).
    /// If `selector` is provided, only vectors at those indices are considered.
    pub fn query(
        &self,
        query: &[f32],
        k: usize,
        selector: Option<&[usize]>,
    ) -> (Vec<usize>, Vec<f32>) {
        let _span = tracing::trace_span!("dense.query", n = self.n, k, dim = self.dim).entered();
        if self.n == 0 || k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Normalise the query so we score by plain dot product.
        let mut q = vec![0.0f32; self.dim];
        let copy = query.len().min(self.dim);
        q[..copy].copy_from_slice(&query[..copy]);
        normalise_in_place(&mut q);

        let candidates: &[usize] = match selector {
            Some(sel) => sel,
            None => &[],
        };
        let n_candidates = if selector.is_some() {
            candidates.len()
        } else {
            self.n
        };
        if n_candidates == 0 {
            return (Vec::new(), Vec::new());
        }

        // Score: parallel for large pools, serial for small.
        let scored: Vec<(usize, f32)> = if n_candidates >= PARALLEL_THRESHOLD {
            if let Some(sel) = selector {
                sel.par_iter()
                    .filter_map(|&idx| {
                        if idx < self.n {
                            Some((idx, dot(self.row(idx), &q)))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                (0..self.n)
                    .into_par_iter()
                    .map(|idx| (idx, dot(self.row(idx), &q)))
                    .collect()
            }
        } else if let Some(sel) = selector {
            sel.iter()
                .filter_map(|&idx| {
                    if idx < self.n {
                        Some((idx, dot(self.row(idx), &q)))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            (0..self.n)
                .map(|idx| (idx, dot(self.row(idx), &q)))
                .collect()
        };

        let topk = top_k_from_iter_f32(scored, k);
        let mut indices = Vec::with_capacity(topk.len());
        let mut scores = Vec::with_capacity(topk.len());
        for (i, s) in topk {
            indices.push(i);
            scores.push(s);
        }
        (indices, scores)
    }

    /// Batched query: query multiple vectors at once.
    ///
    /// Returns a list of `(indices, scores)` tuples, one per query.
    pub fn query_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        selector: Option<&[usize]>,
    ) -> Vec<(Vec<usize>, Vec<f32>)> {
        // Run queries in parallel — each query is independent.
        queries
            .par_iter()
            .map(|q| self.query(q, k, selector))
            .collect()
    }
}

/// L2-normalise a vector in place. Vectors with zero norm are left as zeros.
#[inline]
fn normalise_in_place(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    if sum_sq > 0.0 {
        let inv = sum_sq.sqrt().recip();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Dot product of two equal-length f32 slices. Auto-vectorises on x86-64/aarch64.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Manual chunking helps LLVM emit fma/avx code on x86-64; on aarch64 it
    // becomes neon. We don't need explicit intrinsics for this scale.
    let mut acc = 0.0f32;
    let mut i = 0;
    let chunks = a.len() / 8;
    while i < chunks * 8 {
        // Unroll by 8 — gives the auto-vectoriser an obvious window.
        acc += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3]
            + a[i + 4] * b[i + 4]
            + a[i + 5] * b[i + 5]
            + a[i + 6] * b[i + 6]
            + a[i + 7] * b[i + 7];
        i += 8;
    }
    while i < a.len() {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = DenseIndex::new(vec![]);
        let (indices, _) = index.query(&[1.0, 0.0, 0.0], 5, None);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_cosine_search() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0], // aligned with query
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![0.9, 0.1, 0.0], // close to query
        ];
        let index = DenseIndex::new(embeddings);
        let (indices, scores) = index.query(&[1.0, 0.0, 0.0], 2, None);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0);
        assert!((scores[0] - 1.0).abs() < 1e-4);
        assert_eq!(indices[1], 2);
    }

    #[test]
    fn test_with_selector() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let index = DenseIndex::new(embeddings);
        let (indices, _) = index.query(&[0.0, 1.0], 2, Some(&[1, 2]));
        assert_eq!(indices[0], 1);
    }

    #[test]
    fn compact_and_extend_preserves_kept_rows() {
        // Start with 4 rows; keep [0, 2] (drop 1 and 3); append 1 new row.
        let mut index = DenseIndex::new(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.7, 0.7],
            vec![0.5, 0.5],
        ]);
        let kept_row0_before = index.row(0).to_vec();
        let kept_row2_before = index.row(2).to_vec();

        index.compact_and_extend(&[0, 2], vec![vec![1.0, 1.0]]);
        assert_eq!(index.len(), 3);
        // Row 0 unchanged (was already at index 0).
        assert_eq!(index.row(0), kept_row0_before.as_slice());
        // Former row 2 now at index 1.
        assert_eq!(index.row(1), kept_row2_before.as_slice());
        // Appended row should be L2-normalised: [1,1] → [1/√2, 1/√2].
        let row2 = index.row(2);
        let norm_sq: f32 = row2.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5, "appended row not normalised");
    }

    #[test]
    fn compact_and_extend_full_drop() {
        // Edge case: drop everything, then append.
        let mut index = DenseIndex::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        index.compact_and_extend(&[], vec![vec![1.0, 0.0]]);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn compact_and_extend_no_new() {
        // Compaction with no appended rows — just drop a row in the middle.
        let mut index = DenseIndex::new(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.7, 0.7],
        ]);
        let kept2 = index.row(2).to_vec();
        index.compact_and_extend(&[0, 2], vec![]);
        assert_eq!(index.len(), 2);
        assert_eq!(index.row(1), kept2.as_slice());
    }
}
