//! BM25 sparse index — inverted-index implementation with token interning.
//!
//! Tokens are interned to `u32` IDs once at build time so query-time lookups
//! avoid string hashing/cloning entirely. Per-term postings lists let us
//! iterate only the documents that contain a query term, instead of scanning
//! the whole corpus per token.

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

    /// Compute BM25 scores for a query against all documents.
    ///
    /// Returns a vector of scores, one per document. If `selector` is provided,
    /// only documents at those indices are scored (others get 0.0).
    pub fn get_scores(&self, query_tokens: &[String], selector: Option<&[usize]>) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.num_docs];
        if self.num_docs == 0 || query_tokens.is_empty() {
            return scores;
        }

        // Build a selector bitmask once if provided.
        let mask: Option<Vec<bool>> = selector.map(|sel| {
            let mut m = vec![false; self.num_docs];
            for &i in sel {
                if i < self.num_docs {
                    m[i] = true;
                }
            }
            m
        });

        // Resolve query tokens to interned ids and dedupe (BM25 is bag-of-words: a
        // repeated query term contributes the same per-doc term once with idf
        // already accounting for it, so we union the postings).
        let mut term_ids: Vec<u32> = Vec::with_capacity(query_tokens.len());
        for tok in query_tokens {
            if let Some(&id) = self.vocab.get(tok.as_str()) {
                if !term_ids.contains(&id) {
                    term_ids.push(id);
                }
            }
        }
        if term_ids.is_empty() {
            return scores;
        }

        let inv_avg_dl = if self.avg_dl > 0.0 {
            1.0 / self.avg_dl
        } else {
            0.0
        };

        for tid in term_ids {
            let idf_val = self.idf[tid as usize];
            for posting in &self.postings[tid as usize] {
                let doc_idx = posting.doc as usize;
                if let Some(m) = &mask {
                    if !m[doc_idx] {
                        continue;
                    }
                }
                let tf_val = posting.tf as f64;
                let dl = self.doc_lengths[doc_idx] as f64;
                let denom = tf_val + K1 * (1.0 - B + B * dl * inv_avg_dl);
                let tf_component = (tf_val * (K1 + 1.0)) / denom;
                scores[doc_idx] += idf_val * tf_component;
            }
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
        if k == 0 || self.num_docs == 0 || query_tokens.is_empty() {
            return Vec::new();
        }

        let scores = self.get_scores(query_tokens, selector);
        crate::index::topk::top_k_indexed(&scores, k)
    }
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
