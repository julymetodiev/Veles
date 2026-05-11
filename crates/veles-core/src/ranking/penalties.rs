//! Path-based penalties — test file penalties, compat/legacy penalties, file saturation.

use ahash::AHashMap;
use regex::RegexSet;
use std::sync::LazyLock;

use crate::index::topk::top_k_indexed;
use crate::types::Chunk;

/// Indices into [`PENALTY_SET`] — the order is load-bearing because
/// each index maps to a specific penalty multiplier in
/// [`file_path_penalty`]. Adding a pattern means appending here AND in
/// the `RegexSet` body, in the same order.
mod p {
    pub const TEST_FILE: usize = 0;
    pub const TEST_DIR: usize = 1;
    pub const COMPAT_DIR: usize = 2;
    pub const EXAMPLES_DIR: usize = 3;
    pub const TYPE_DEFS: usize = 4;
}

/// One `RegexSet` over all path-penalty patterns (§5.1 of the perf
/// plan). A single DFA scan returns the set of pattern indices that
/// matched the file path; the previous code ran 5 independent
/// `is_match` passes per file. `RegexSet` shares prefixes and the
/// inner state machine, so even when no patterns match it walks the
/// path once.
static PENALTY_SET: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        // index 0 — TEST_FILE
        r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py\
            |[^/]*_test\.go\
            |[^/]*Tests?\.java\
            |[^/]*Test\.php\
            |[^/]*_spec\.rb|[^/]*_test\.rb\
            |[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?\
            |[^/]*Tests?\.kt|[^/]*Spec\.kt\
            |[^/]*Tests?\.swift|[^/]*Spec\.swift\
            |[^/]*Tests?\.cs\
            |test_[^/]*\.cpp|[^/]*_test\.cpp|test_[^/]*\.c|[^/]*_test\.c\
            |[^/]*Spec\.scala|[^/]*Suite\.scala|[^/]*Test\.scala\
            |[^/]*_test\.dart|test_[^/]*\.dart\
            |[^/]*_spec\.lua|[^/]*_test\.lua|test_[^/]*\.lua\
            |test_helpers?[^/]*\.\w+)$",
        // index 1 — TEST_DIR
        r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)",
        // index 2 — COMPAT_DIR
        r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)",
        // index 3 — EXAMPLES_DIR
        r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)",
        // index 4 — TYPE_DEFS
        r"\.d\.ts$",
    ])
    .expect("penalty RegexSet must compile")
});

const STRONG_PENALTY: f64 = 0.3;
const MODERATE_PENALTY: f64 = 0.5;
const MILD_PENALTY: f64 = 0.7;

const REEXPORT_FILENAMES: &[&str] = &["__init__.py", "package-info.java"];

const FILE_SATURATION_THRESHOLD: usize = 1;
const FILE_SATURATION_DECAY: f64 = 0.5;

/// Select top-k results with optional file-path penalties and file-saturation decay.
///
/// `scores[i]` is the raw score for `chunks[i]`. Chunks with `score <= 0.0`
/// are treated as out of the candidate pool and skipped.
pub fn rerank_topk(
    scores: &[f64],
    chunks: &[Chunk],
    top_k: usize,
    penalise_paths: bool,
) -> Vec<(usize, f64)> {
    if scores.is_empty() || top_k == 0 {
        return Vec::new();
    }

    // Apply file-path penalties into a fresh penalised score vector.
    // Penalties are cached per file_path so each file is scanned by regex only once.
    let mut penalty_cache: AHashMap<&str, f64> = AHashMap::new();
    let mut penalised: Vec<f64> = Vec::with_capacity(scores.len());

    for (i, &score) in scores.iter().enumerate() {
        if score <= 0.0 || score.is_nan() {
            penalised.push(0.0);
            continue;
        }
        let pen = if penalise_paths {
            let fp = chunks[i].file_path.as_str();
            *penalty_cache
                .entry(fp)
                .or_insert_with(|| file_path_penalty(fp))
        } else {
            1.0
        };
        penalised.push(score * pen);
    }

    // Over-fetch a multiple of top_k so the file-saturation decay below has
    // room to demote chunks from the same file. We pick max(top_k * 4, 32).
    let pool_size = (top_k.saturating_mul(4)).max(32);
    let pool = top_k_indexed(&penalised, pool_size);
    if pool.is_empty() {
        return Vec::new();
    }

    // File saturation: as more chunks from the same file are selected, decay
    // their effective score so other files have a chance.
    let mut file_selected: AHashMap<&str, usize> = AHashMap::new();
    let mut selected: Vec<(usize, f64)> = Vec::with_capacity(top_k);
    let mut min_selected = f64::INFINITY;

    for (idx, pen_score) in pool {
        if selected.len() >= top_k && pen_score <= min_selected {
            break;
        }

        let fp = chunks[idx].file_path.as_str();
        let already_selected = *file_selected.get(fp).unwrap_or(&0);
        let eff_score = if already_selected >= FILE_SATURATION_THRESHOLD {
            let excess = (already_selected - FILE_SATURATION_THRESHOLD + 1) as i32;
            pen_score * FILE_SATURATION_DECAY.powi(excess)
        } else {
            pen_score
        };

        selected.push((idx, eff_score));
        *file_selected.entry(fp).or_default() += 1;

        if selected.len() >= top_k {
            min_selected = selected
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::INFINITY, f64::min);
        }
    }

    selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    selected.truncate(top_k);
    selected
}

/// Return a combined multiplicative penalty for all applicable path patterns.
///
/// Single `RegexSet::matches` call instead of five independent
/// `is_match` passes (§5.1). The set's compiled DFA shares prefix work
/// across all five patterns and returns the set of matched indices in
/// one path scan.
fn file_path_penalty(file_path: &str) -> f64 {
    // Avoid a heap allocation on the common case (no backslashes).
    let normalised: std::borrow::Cow<'_, str> = if file_path.contains('\\') {
        std::borrow::Cow::Owned(file_path.replace('\\', "/"))
    } else {
        std::borrow::Cow::Borrowed(file_path)
    };
    let s: &str = &normalised;
    let mut penalty = 1.0;

    let hits = PENALTY_SET.matches(s);
    if hits.matched(p::TEST_FILE) || hits.matched(p::TEST_DIR) {
        penalty *= STRONG_PENALTY;
    }
    if hits.matched(p::COMPAT_DIR) {
        penalty *= STRONG_PENALTY;
    }
    if hits.matched(p::EXAMPLES_DIR) {
        penalty *= STRONG_PENALTY;
    }
    if hits.matched(p::TYPE_DEFS) {
        penalty *= MILD_PENALTY;
    }

    let filename = std::path::Path::new(file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if REEXPORT_FILENAMES.contains(&filename) {
        penalty *= MODERATE_PENALTY;
    }

    penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_test_file() {
        let p = file_path_penalty("tests/test_foo.py");
        assert!(p < 1.0);
    }

    #[test]
    fn test_penalty_normal_file() {
        let p = file_path_penalty("src/main.rs");
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_penalty_init_py() {
        let p = file_path_penalty("src/__init__.py");
        assert!(p < 1.0);
    }

    #[test]
    fn test_penalty_d_ts() {
        let p = file_path_penalty("src/types.d.ts");
        assert!(p < 1.0);
        assert!(p > 0.3); // mild, not strong
    }
}
