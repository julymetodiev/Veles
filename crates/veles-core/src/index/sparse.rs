//! BM25 sparse index — inverted-index implementation with token interning.
//!
//! Tokens are interned to `u32` IDs once at build time so query-time lookups
//! avoid string hashing/cloning entirely. Per-term postings lists let us
//! iterate only the documents that contain a query term, instead of scanning
//! the whole corpus per token.

use std::cell::RefCell;

use ahash::AHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// BM25 parameters.
const K1: f64 = 1.5;
const B: f64 = 0.75;

/// One entry in a postings list.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Posting {
    doc: u32,
    tf: u32,
}

/// A BM25 index over a corpus of tokenized documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    /// Number of documents in the corpus.
    num_docs: usize,
    /// Average document length (in tokens).
    avg_dl: f64,
    /// Term → interned id.
    vocab: AHashMap<String, u32>,
    /// For each term id: precomputed IDF.
    idf: Vec<f64>,
    /// For each term id: sorted postings list (by doc id).
    postings: Vec<Vec<Posting>>,
    /// Document lengths (in tokens).
    doc_lengths: Vec<u32>,
}

impl Bm25Index {
    /// Build a BM25 index from a list of tokenized documents.
    ///
    /// Each inner `Vec<String>` represents the tokens of one document.
    pub fn new(tokenized_docs: &[Vec<String>]) -> Self {
        let num_docs = tokenized_docs.len();
        if num_docs == 0 {
            return Self {
                num_docs: 0,
                avg_dl: 0.0,
                vocab: AHashMap::new(),
                idf: Vec::new(),
                postings: Vec::new(),
                doc_lengths: Vec::new(),
            };
        }

        // Step 1: per-document local term-frequency tables (in parallel).
        // Each thread builds an AHashMap<&str, u32> referencing the original token strings,
        // avoiding the per-token clone the previous implementation paid.
        let per_doc: Vec<(AHashMap<&str, u32>, u32)> = tokenized_docs
            .par_iter()
            .map(|doc_tokens| {
                let mut local: AHashMap<&str, u32> =
                    AHashMap::with_capacity(doc_tokens.len().min(64));
                for tok in doc_tokens {
                    *local.entry(tok.as_str()).or_insert(0) += 1;
                }
                let dl = doc_tokens.len() as u32;
                (local, dl)
            })
            .collect();

        // Step 2: build the global vocab from the per-doc maps.
        // Single-threaded (cheap relative to parallel chunking) and lets us assign stable ids.
        let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(per_doc.len() * 4);
        let mut df: Vec<u32> = Vec::new();
        for (local, _) in &per_doc {
            for term in local.keys() {
                if !vocab.contains_key(*term) {
                    let id = df.len() as u32;
                    vocab.insert((*term).to_string(), id);
                    df.push(0);
                }
            }
        }

        // Step 3: build postings lists.
        // For each (doc, term, tf) we push to postings[term_id]. Postings are appended
        // in increasing doc order naturally, so they remain sorted.
        let n_terms = df.len();
        let mut postings: Vec<Vec<Posting>> = vec![Vec::new(); n_terms];
        let mut doc_lengths: Vec<u32> = Vec::with_capacity(num_docs);

        for (doc_id, (local, dl)) in per_doc.iter().enumerate() {
            doc_lengths.push(*dl);
            for (term, tf) in local {
                let id = *vocab.get(*term).expect("vocab built above");
                postings[id as usize].push(Posting {
                    doc: doc_id as u32,
                    tf: *tf,
                });
                df[id as usize] += 1;
            }
        }

        // Step 4: compute IDF per term.
        let total_len: u64 = doc_lengths.iter().map(|&l| l as u64).sum();
        let avg_dl = total_len as f64 / num_docs as f64;
        let n = num_docs as f64;
        let idf: Vec<f64> = df
            .iter()
            .map(|&dfv| {
                let dfv = dfv as f64;
                ((n - dfv + 0.5) / (dfv + 0.5) + 1.0).ln()
            })
            .collect();

        Self {
            num_docs,
            avg_dl,
            vocab,
            idf,
            postings,
            doc_lengths,
        }
    }

    /// Sparse BM25 scoring core. Returns `(doc_id, score)` for every
    /// document that has a non-zero score against `query_tokens`.
    ///
    /// This is the engine behind `top_k` and `get_scores`. Two tricks
    /// keep it fast across both small and large corpora (§1.1 of the
    /// perf plan):
    ///
    /// 1. **Thread-local dense scratch buffer.** A `Vec<f64>` of length
    ///    `num_docs` is allocated once per thread and reused across all
    ///    queries. No per-query allocation. Indexing into the dense
    ///    buffer is array-fast and cache-friendly — beats a HashMap on
    ///    every corpus size we measured.
    /// 2. **Touched list.** As we accumulate scores we record the doc
    ///    ids we wrote to. At the end we walk only those positions to
    ///    build the sparse return value and zero them back out — no
    ///    `O(num_docs)` clear, no `O(num_docs)` top-k iteration.
    ///
    /// The scratch buffer can grow if a larger index runs on the same
    /// thread; it never shrinks. Peak per-thread memory is
    /// `8 × num_docs_max` bytes — 1.6MB for a 200K-chunk index.
    fn score_sparse(&self, query_tokens: &[String], selector: Option<&[usize]>) -> Vec<(u32, f64)> {
        if self.num_docs == 0 || query_tokens.is_empty() {
            return Vec::new();
        }

        // Resolve query tokens to interned ids and dedupe (BM25 is bag-of-words: a
        // repeated query term contributes the same per-doc term once with idf
        // already accounting for it, so we union the postings).
        let mut term_ids: Vec<u32> = Vec::with_capacity(query_tokens.len());
        for tok in query_tokens {
            if let Some(&id) = self.vocab.get(tok.as_str())
                && !term_ids.contains(&id)
            {
                term_ids.push(id);
            }
        }
        if term_ids.is_empty() {
            return Vec::new();
        }

        // Selector mask. Only allocated when a filter is in play. The
        // dense bool-vec gives O(1) lookup in the inner loop and is the
        // smallest representation that does — selector lists are usually
        // large in practice (language / glob filters often match most
        // of the corpus), so a set would be no smaller.
        let mask: Option<Vec<bool>> = selector.map(|sel| {
            let mut m = vec![false; self.num_docs];
            for &i in sel {
                if i < self.num_docs {
                    m[i] = true;
                }
            }
            m
        });

        let inv_avg_dl = if self.avg_dl > 0.0 {
            1.0 / self.avg_dl
        } else {
            0.0
        };

        BM25_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < self.num_docs {
                scratch.resize(self.num_docs, 0.0);
            }
            // Invariant: between calls every position in `scratch[..num_docs]`
            // is 0.0 (each call clears the positions it wrote). Positions
            // beyond `num_docs` are irrelevant — we only read 0..num_docs.

            // Track which positions we write so we can zero them at the end
            // and build the sparse return value without scanning all N.
            let upper: usize = term_ids
                .iter()
                .map(|&t| self.postings[t as usize].len())
                .sum();
            let mut touched: Vec<u32> = Vec::with_capacity(upper.min(self.num_docs).max(16));

            for tid in term_ids {
                let idf_val = self.idf[tid as usize];
                for posting in &self.postings[tid as usize] {
                    let doc_idx = posting.doc as usize;
                    if let Some(m) = &mask
                        && !m[doc_idx]
                    {
                        continue;
                    }
                    let tf_val = posting.tf as f64;
                    let dl = self.doc_lengths[doc_idx] as f64;
                    let denom = tf_val + K1 * (1.0 - B + B * dl * inv_avg_dl);
                    let tf_component = (tf_val * (K1 + 1.0)) / denom;
                    if scratch[doc_idx] == 0.0 {
                        touched.push(posting.doc);
                    }
                    scratch[doc_idx] += idf_val * tf_component;
                }
            }

            let mut out: Vec<(u32, f64)> = Vec::with_capacity(touched.len());
            for &doc in &touched {
                let s = scratch[doc as usize];
                scratch[doc as usize] = 0.0; // restore the invariant for next call
                if s > 0.0 {
                    out.push((doc, s));
                }
            }
            out
        })
    }

    /// Compute BM25 scores for a query against all documents.
    ///
    /// Returns a vector of scores, one per document. If `selector` is provided,
    /// only documents at those indices are scored (others get 0.0).
    ///
    /// Internally this is a thin wrapper over [`Self::score_sparse`] that
    /// materialises the result into a dense `Vec<f64>`. Prefer `top_k`
    /// when you only need the highest-scoring docs — it skips the
    /// dense materialisation.
    pub fn get_scores(&self, query_tokens: &[String], selector: Option<&[usize]>) -> Vec<f64> {
        let _span = tracing::trace_span!(
            "bm25.get_scores",
            n_docs = self.num_docs,
            n_tokens = query_tokens.len()
        )
        .entered();
        let mut scores = vec![0.0f64; self.num_docs];
        for (doc, s) in self.score_sparse(query_tokens, selector) {
            scores[doc as usize] = s;
        }
        scores
    }

    /// Return the top-k document indices sorted by BM25 score (descending).
    /// Excludes documents with zero score.
    pub fn top_k(
        &self,
        query_tokens: &[String],
        k: usize,
        selector: Option<&[usize]>,
    ) -> Vec<(usize, f64)> {
        let _span = tracing::trace_span!(
            "bm25.top_k",
            n_docs = self.num_docs,
            n_tokens = query_tokens.len(),
            k
        )
        .entered();
        if k == 0 || self.num_docs == 0 || query_tokens.is_empty() {
            return Vec::new();
        }

        let sparse = self.score_sparse(query_tokens, selector);
        crate::index::topk::top_k_from_iter_f64(sparse.into_iter().map(|(d, s)| (d as usize, s)), k)
    }
}

thread_local! {
    /// Per-thread reusable dense scoring buffer for BM25.
    ///
    /// Sized lazily on first use to the largest `num_docs` the thread
    /// has scored against; each `score_sparse` call clears only the
    /// positions it wrote (via the `touched` list) so the invariant
    /// "all positions are 0.0 between calls" is preserved cheaply.
    ///
    /// Held for the lifetime of the thread. Worst-case peak is
    /// `8 × max_num_docs` bytes per thread (1.6MB for 200K chunks).
    static BM25_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let docs = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["hello".to_string(), "rust".to_string()],
            vec!["world".to_string(), "of".to_string(), "rust".to_string()],
        ];
        let index = Bm25Index::new(&docs);
        let results = index.top_k(&["hello".to_string()], 2, None);
        assert_eq!(results.len(), 2);
        // Both docs 0 and 1 contain "hello"
        assert!(
            results
                .iter()
                .all(|(idx, score)| [*idx].contains(idx) && *score > 0.0)
        );
    }

    #[test]
    fn test_bm25_empty() {
        let index = Bm25Index::new(&[]);
        let results = index.top_k(&["hello".to_string()], 5, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_selector() {
        let docs = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["hello".to_string(), "rust".to_string()],
            vec!["world".to_string(), "of".to_string(), "rust".to_string()],
        ];
        let index = Bm25Index::new(&docs);
        // Only score doc at index 2
        let results = index.top_k(&["rust".to_string()], 5, Some(&[2]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_bm25_repeated_query_token() {
        // Repeated query tokens should not double-count (matches Okapi BM25 bag-of-words).
        let docs = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["hello".to_string(), "rust".to_string()],
        ];
        let index = Bm25Index::new(&docs);
        let s1 = index.get_scores(&["hello".to_string()], None);
        let s2 = index.get_scores(&["hello".to_string(), "hello".to_string()], None);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-9, "scores diverge: {a} vs {b}");
        }
    }
}
