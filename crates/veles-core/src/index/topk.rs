//! Top-k selection utilities — partial sort via a bounded min-heap.
//!
//! For top-k from N scores, this is O(N log k) instead of O(N log N) for
//! full sort. Materially faster when N >> k.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A score paired with its index, ordered so that the smaller score is "greater"
/// (so the std max-heap behaves as a min-heap on the score).
#[derive(Debug, Clone, Copy)]
struct MinEntry {
    idx: u32,
    score: f64,
}

impl PartialEq for MinEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.idx == other.idx
    }
}
impl Eq for MinEntry {}

impl PartialOrd for MinEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller score wins (for min-heap behaviour atop std max-heap).
        // NaNs are treated as smaller than anything (they get evicted first).
        match other.score.partial_cmp(&self.score) {
            Some(o) => o.then_with(|| other.idx.cmp(&self.idx)),
            None => {
                // NaN handling: classify NaN as smallest.
                let self_nan = self.score.is_nan();
                let other_nan = other.score.is_nan();
                match (self_nan, other_nan) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            }
        }
    }
}

/// Select the top-k highest-scoring entries from a `scores` vector indexed by position.
///
/// Excludes entries with score `<= 0.0`. Returns `(index, score)` pairs sorted
/// descending by score.
pub fn top_k_indexed(scores: &[f64], k: usize) -> Vec<(usize, f64)> {
    if k == 0 || scores.is_empty() {
        return Vec::new();
    }

    let mut heap: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);

    for (i, &s) in scores.iter().enumerate() {
        if s <= 0.0 {
            continue;
        }
        if heap.len() < k {
            heap.push(MinEntry {
                idx: i as u32,
                score: s,
            });
        } else if let Some(top) = heap.peek()
            && s > top.score
        {
            heap.pop();
            heap.push(MinEntry {
                idx: i as u32,
                score: s,
            });
        }
    }

    let mut out: Vec<(usize, f64)> = heap
        .into_iter()
        .map(|e| (e.idx as usize, e.score))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out
}

/// Select the top-k highest-scoring `(usize, f32)` pairs from an iterator.
///
/// Used by the dense index where scores are f32.
pub fn top_k_from_iter_f32<I>(iter: I, k: usize) -> Vec<(usize, f32)>
where
    I: IntoIterator<Item = (usize, f32)>,
{
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<MinEntryF32> = BinaryHeap::with_capacity(k + 1);
    for (i, s) in iter {
        if heap.len() < k {
            heap.push(MinEntryF32 {
                idx: i as u32,
                score: s,
            });
        } else if let Some(top) = heap.peek()
            && s > top.score
        {
            heap.pop();
            heap.push(MinEntryF32 {
                idx: i as u32,
                score: s,
            });
        }
    }
    let mut out: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|e| (e.idx as usize, e.score))
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    out
}

#[derive(Debug, Clone, Copy)]
struct MinEntryF32 {
    idx: u32,
    score: f32,
}

impl PartialEq for MinEntryF32 {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.idx == other.idx
    }
}
impl Eq for MinEntryF32 {}
impl PartialOrd for MinEntryF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinEntryF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.score.partial_cmp(&self.score) {
            Some(o) => o.then_with(|| other.idx.cmp(&self.idx)),
            None => {
                let self_nan = self.score.is_nan();
                let other_nan = other.score.is_nan();
                match (self_nan, other_nan) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_basic() {
        let scores = vec![0.1, 0.9, 0.0, 0.5, 0.7];
        let r = top_k_indexed(&scores, 3);
        assert_eq!(r, vec![(1, 0.9), (4, 0.7), (3, 0.5)]);
    }

    #[test]
    fn top_k_excludes_zero() {
        let scores = vec![0.0, 0.0, 0.0];
        let r = top_k_indexed(&scores, 5);
        assert!(r.is_empty());
    }

    #[test]
    fn top_k_k_larger_than_n() {
        let scores = vec![0.5, 0.2];
        let r = top_k_indexed(&scores, 10);
        assert_eq!(r, vec![(0, 0.5), (1, 0.2)]);
    }
}
