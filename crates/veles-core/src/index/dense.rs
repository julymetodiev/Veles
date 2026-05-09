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

use crate::index::topk::top_k_from_iter_f32;

/// Below this candidate count, parallelism overhead exceeds gains.
const PARALLEL_THRESHOLD: usize = 1024;

/// A dense vector index supporting top-k nearest-neighbor search via cosine similarity.
#[derive(Debug, Clone)]
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
            (0..self.n).map(|idx| (idx, dot(self.row(idx), &q))).collect()
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
}
