//! Query-type boosting — definition boosts, multi-chunk file boosts, stem matches.
//!
//! All scoring is index-keyed: a slice `scores[idx]` holds the running score
//! for `chunks[idx]`. A score of `0.0` means "not in the candidate pool" — the
//! invariant is preserved because RRF scores are strictly positive for ranked
//! entries.

use ahash::AHashMap;
use ahash::AHashSet;
use dashmap::DashMap;
use rayon::prelude::*;
use regex::Regex;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use crate::tokenizer::split_identifier;
use crate::types::Chunk;

// ── Symbol query detection ────────────────────────────────────────────────

// `(?x)` enables verbose mode so unescaped whitespace in the pattern is
// ignored — without it, the literal newlines and indentation become part
// of the regex and the symbol-detection branch silently never matches
// any query.
static SYMBOL_QUERY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        ^(?:
            [A-Za-z_][A-Za-z0-9_]* (?: (?:::|\\|->|\.) [A-Za-z_][A-Za-z0-9_]* )+   # qualified: foo::bar, Foo.bar
          | _[A-Za-z0-9_]*                                                          # leading underscore
          | [A-Za-z][A-Za-z0-9]* [A-Z_] [A-Za-z0-9_]*                               # camelCase / SCREAMING_SNAKE
          | [A-Z][A-Za-z0-9]*                                                       # leading uppercase: Foo, Manifest
        )$",
    )
    .unwrap()
});

/// Embedded CamelCase identifiers in natural-language queries.
static EMBEDDED_SYMBOL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        \b(?:
            [A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*    # PascalCase: Foo, FooBar
          | [a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+         # camelCase: fooBar
        )\b",
    )
    .unwrap()
});

/// Token regex for natural-language stem boosts. Unicode-aware (matches
/// Cyrillic / CJK / Greek / Arabic etc. in addition to ASCII).
static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}_][\p{L}\p{N}_]*").unwrap());

/// Minimum stem length for prefix-based non-candidate scan.
const EMBEDDED_STEM_MIN_LEN: usize = 4;
/// Half-strength for embedded symbols.
const EMBEDDED_SYMBOL_BOOST_SCALE: f64 = 0.5;

/// Minimum chunk count before def-boost loops fan out via rayon.
///
/// Below this, the per-chunk work (substring filter — usually a fast
/// reject — plus an occasional regex match) is dominated by rayon's
/// spawn / join overhead (~10–50 µs end-to-end). Above it, the regex
/// work per matching chunk dwarfs the overhead and parallelism wins
/// linearly. Mirrors `index::dense::PARALLEL_THRESHOLD` (1024).
const PARALLEL_BOOST_THRESHOLD: usize = 1024;

/// Definition keywords across common languages.
const DEFINITION_KEYWORDS: &[&str] = &[
    "class",
    "module",
    "defmodule",
    "def",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "func",
    "function",
    "object",
    "abstract class",
    "data class",
    "fn",
    "fun",
    "package",
    "namespace",
    "protocol",
    "record",
    "typedef",
    // Top-level bindings — covers Rust/C/C++/Go/JS/TS constants and globals.
    // Without these, queries for SCREAMING_SNAKE_CASE constants miss the
    // definition site entirely.
    "const",
    "static",
];

/// SQL DDL keywords.
const SQL_DEFINITION_KEYWORDS: &[&str] = &[
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE PROCEDURE",
    "CREATE FUNCTION",
];

/// Additive boost multiplier for chunks that define a queried symbol.
const DEFINITION_BOOST_MULTIPLIER: f64 = 3.0;
/// Additive boost multiplier for NL queries when file stems match.
const STEM_BOOST_MULTIPLIER: f64 = 1.0;
/// Fraction of max_score added for file coherence.
const FILE_COHERENCE_BOOST_FRAC: f64 = 0.2;

/// Common English stopwords.
static STOPWORDS: LazyLock<AHashSet<&'static str>> = LazyLock::new(|| {
    "a an and are as at be by do does for from has have how if in is it not of on or the to was \
     what when where which who why with"
        .split_whitespace()
        .collect()
});

// ── Public API ────────────────────────────────────────────────────────────

/// Return true if the query looks like a bare symbol or namespace-qualified identifier.
pub fn is_symbol_query(query: &str) -> bool {
    SYMBOL_QUERY_RE.is_match(query.trim())
}

/// Resolve the blending weight for semantic scores, auto-detecting from query type.
pub fn resolve_alpha(query: &str, alpha: Option<f64>) -> f64 {
    match alpha {
        Some(a) => a,
        None => {
            if is_symbol_query(query) {
                0.3 // lean BM25 for exact keyword matching
            } else {
                0.5 // balanced semantic + BM25
            }
        }
    }
}

/// Apply query-type boosts to candidate scores in place.
///
/// `scores[i]` is the running score for `chunks[i]`; entries with score `0.0`
/// are treated as non-candidates (boosts may still add them).
pub fn apply_query_boost(scores: &mut [f64], query: &str, chunks: &[Chunk]) {
    if scores.is_empty() {
        return;
    }
    let max_score = current_max(scores);
    if max_score <= 0.0 || max_score.is_nan() {
        return;
    }

    if is_symbol_query(query) {
        boost_symbol_definitions(scores, query, max_score, chunks);
    } else {
        boost_stem_matches(scores, query, max_score, chunks);
        boost_embedded_symbols(scores, query, max_score, chunks);
    }
}

/// Promote files with multiple high-scoring chunks by boosting their top chunk in place.
pub fn boost_multi_chunk_files(scores: &mut [f64], chunks: &[Chunk]) {
    let max_score = current_max(scores);
    if max_score <= 0.0 || max_score.is_nan() {
        return;
    }

    // Aggregate per-file totals and best chunk index in a single pass over candidates.
    let mut file_sum: AHashMap<&str, f64> = AHashMap::new();
    let mut best_idx: AHashMap<&str, (usize, f64)> = AHashMap::new();

    for (i, &score) in scores.iter().enumerate() {
        if score <= 0.0 || score.is_nan() {
            continue;
        }
        let fp = chunks[i].file_path.as_str();
        *file_sum.entry(fp).or_insert(0.0) += score;
        let entry = best_idx.entry(fp).or_insert((i, f64::NEG_INFINITY));
        if score > entry.1 {
            *entry = (i, score);
        }
    }

    let max_file_sum = file_sum.values().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_file_sum <= 0.0 || max_file_sum.is_nan() {
        return;
    }
    let boost_unit = max_score * FILE_COHERENCE_BOOST_FRAC;

    for (fp, (idx, _)) in &best_idx {
        let sum = *file_sum.get(*fp).unwrap_or(&0.0);
        scores[*idx] += boost_unit * sum / max_file_sum;
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn current_max(scores: &[f64]) -> f64 {
    scores.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn extract_symbol_name(query: &str) -> String {
    let query = query.trim();
    for separator in ["::", "\\", "->", "."] {
        if query.contains(separator) {
            return query.rsplit(separator).next().unwrap_or(query).to_string();
        }
    }
    query.to_string()
}

/// A compiled definition matcher for a single symbol name.
///
/// Two regexes — `general` covers the curly-language `fn`/`struct`/`def`/...
/// keywords, `sql` covers `CREATE TABLE`/`CREATE FUNCTION`/...  The pair
/// is wrapped in an `Arc` and cached so we pay the regex compile cost
/// (≈ ms per pattern) at most once per symbol name across the process
/// lifetime — see [`DEF_PATTERN_CACHE`].
#[derive(Debug)]
pub(crate) struct DefPattern {
    general: Regex,
    sql: Regex,
}

impl DefPattern {
    fn matches(&self, content: &str) -> bool {
        self.general.is_match(content) || self.sql.is_match(content)
    }
}

/// Pre-computed `|`-joined body of escaped general definition keywords.
/// Identical for every symbol query, so we LazyLock it instead of
/// rebuilding the join + the per-keyword `regex::escape` on every call.
static DEF_KW_BODY: LazyLock<String> = LazyLock::new(|| {
    DEFINITION_KEYWORDS
        .iter()
        .map(|k| regex::escape(k))
        .collect::<Vec<_>>()
        .join("|")
});

/// Same idea for SQL DDL keywords.
static DEF_SQL_BODY: LazyLock<String> = LazyLock::new(|| {
    SQL_DEFINITION_KEYWORDS
        .iter()
        .map(|k| regex::escape(k))
        .collect::<Vec<_>>()
        .join("|")
});

/// Process-wide cache of compiled `DefPattern`s keyed by symbol name.
///
/// Lockfree-ish: `DashMap` shards the map internally, so concurrent
/// reads on different shards never contend. Reads in the common case
/// (cache hit) take a per-shard read lock that's effectively atomic.
/// Writes (first compile of a symbol) lock only their own shard.
///
/// Unbounded: in practice users search for tens of distinct symbols
/// per session, and each `DefPattern` is a few KB of regex DFA —
/// bounded growth even on long-running MCP / gRPC servers. Switch to
/// LRU if a session ever sustains thousands of unique symbol queries
/// (it won't).
static DEF_PATTERN_CACHE: LazyLock<DashMap<String, Arc<DefPattern>>> =
    LazyLock::new(|| DashMap::with_capacity(64));

/// Get (or build, on first call) the `DefPattern` for `symbol_name`.
///
/// Returns `Arc<DefPattern>` so callers can fan out across many chunks
/// (including via rayon — see §1.4) without holding any cache reference
/// and without cloning the underlying regexes (an `Arc::clone` is one
/// atomic op).
pub(crate) fn definition_pattern(symbol_name: &str) -> Arc<DefPattern> {
    // Fast path: shard read, hit.
    if let Some(p) = DEF_PATTERN_CACHE.get(symbol_name) {
        return Arc::clone(p.value());
    }
    // Slow path: compile outside any lock so peer readers stay unblocked,
    // then insert via DashMap's CAS-style entry API. If a concurrent
    // thread won the race, `or_insert_with` will pick up the existing
    // value rather than overwriting.
    let built = Arc::new(build_def_pattern(symbol_name));
    let entry = DEF_PATTERN_CACHE
        .entry(symbol_name.to_string())
        .or_insert(built);
    Arc::clone(entry.value())
}

fn build_def_pattern(symbol_name: &str) -> DefPattern {
    let escaped = regex::escape(symbol_name);
    let ns_prefix = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*";
    let suffix = format!(r")\s+{ns_prefix}{escaped}(?:\s|[<({{\[:;]|$)");

    // The Rust `regex` crate does not support lookbehind, so we use `\b`
    // to require a word boundary before the keyword. All DEFINITION_KEYWORDS
    // start with a letter, so `\b` correctly distinguishes `class` from
    // `subclass` (no boundary inside the latter).
    let general =
        Regex::new(&format!(r"\b(?:{kw}{suffix}", kw = *DEF_KW_BODY)).expect("def general regex");
    let sql =
        Regex::new(&format!(r"(?i)\b(?:{kw}{suffix}", kw = *DEF_SQL_BODY)).expect("def sql regex");
    DefPattern { general, sql }
}

/// A bundle of definition patterns for a set of symbol names.
struct DefinitionMatchers {
    patterns: Vec<Arc<DefPattern>>,
    names: Vec<String>,
}

impl DefinitionMatchers {
    fn for_names<I: IntoIterator<Item = String>>(names: I) -> Self {
        let names: Vec<String> = names.into_iter().collect();
        let patterns = names.iter().map(|n| definition_pattern(n)).collect();
        Self { patterns, names }
    }

    fn defines_any(&self, content: &str) -> bool {
        self.patterns.iter().any(|p| p.matches(content))
    }
}

fn stem_matches(stem: &str, name: &str) -> bool {
    let stem_norm = stem.replace('_', "");
    stem == name
        || stem_norm == name
        || stem.trim_end_matches('s') == name
        || stem_norm.trim_end_matches('s') == name
}

fn definition_tier(chunk: &Chunk, matchers: &DefinitionMatchers, boost_unit: f64) -> f64 {
    if !matchers.defines_any(&chunk.content) {
        return 0.0;
    }
    let stem = file_stem_lower(&chunk.file_path);
    if matchers
        .names
        .iter()
        .any(|n| stem_matches(&stem, &n.to_lowercase()))
    {
        boost_unit * 1.5
    } else {
        boost_unit
    }
}

fn file_stem_lower(file_path: &str) -> String {
    Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn boost_symbol_definitions(scores: &mut [f64], query: &str, max_score: f64, chunks: &[Chunk]) {
    let symbol_name = extract_symbol_name(query);
    let trimmed = query.trim().to_string();

    let mut names = vec![symbol_name.clone()];
    if symbol_name != trimmed {
        names.push(trimmed);
    }
    let matchers = DefinitionMatchers::for_names(names.clone());

    let boost_unit = max_score * DEFINITION_BOOST_MULTIPLIER;

    // Scan every chunk for a definition match — not just candidates and not
    // just file-stem-matching chunks. For SCREAMING_SNAKE_CASE constants
    // (and any short identifier whose name doesn't match its file stem),
    // BM25 buries the definition site under reference-heavy chunks; the
    // unconditional scan is what makes the definition boost actually fire.
    //
    // Parallelised via rayon (§1.4) above PARALLEL_BOOST_THRESHOLD. Below
    // it, sparse-match queries (rare identifiers where the substring
    // filter rejects almost every chunk) pay rayon spawn overhead for
    // microseconds of actual work — net regression. Above it, the regex
    // work dominates and parallel wins linearly with N.
    if chunks.len() >= PARALLEL_BOOST_THRESHOLD {
        let updates: Vec<(usize, f64)> = chunks
            .par_iter()
            .enumerate()
            .filter_map(|(i, chunk)| {
                if !names.iter().any(|n| chunk.content.contains(n)) {
                    return None;
                }
                let tier = definition_tier(chunk, &matchers, boost_unit);
                if tier > 0.0 { Some((i, tier)) } else { None }
            })
            .collect();
        for (i, tier) in updates {
            scores[i] += tier;
        }
    } else {
        for (i, chunk) in chunks.iter().enumerate() {
            if !names.iter().any(|n| chunk.content.contains(n)) {
                continue;
            }
            let tier = definition_tier(chunk, &matchers, boost_unit);
            if tier > 0.0 {
                scores[i] += tier;
            }
        }
    }
}

fn boost_embedded_symbols(scores: &mut [f64], query: &str, max_score: f64, chunks: &[Chunk]) {
    let names: Vec<String> = EMBEDDED_SYMBOL_RE
        .find_iter(query)
        .map(|m| m.as_str().to_string())
        .collect();
    if names.is_empty() {
        return;
    }

    let matchers = DefinitionMatchers::for_names(names.clone());
    let boost_unit = max_score * DEFINITION_BOOST_MULTIPLIER * EMBEDDED_SYMBOL_BOOST_SCALE;
    let symbols_lower: Vec<String> = names.iter().map(|s| s.to_lowercase()).collect();

    // Threshold-gated parallel scan (§1.4) — see `boost_symbol_definitions`
    // for the rationale. The parallel branch reads `scores[i]` via an
    // immutable reborrow and collects writes as `(idx, tier)`; the
    // serial branch reads + writes scores directly.
    if chunks.len() >= PARALLEL_BOOST_THRESHOLD {
        let scores_view: &[f64] = scores;
        let updates: Vec<(usize, f64)> = chunks
            .par_iter()
            .enumerate()
            .filter_map(|(i, chunk)| {
                let in_pool = scores_view[i] > 0.0;
                let tier = if in_pool {
                    definition_tier(chunk, &matchers, boost_unit)
                } else {
                    if !embedded_stem_matches(&chunk.file_path, &symbols_lower) {
                        return None;
                    }
                    definition_tier(chunk, &matchers, boost_unit)
                };
                if tier > 0.0 { Some((i, tier)) } else { None }
            })
            .collect();
        for (i, tier) in updates {
            scores[i] += tier;
        }
    } else {
        for (i, chunk) in chunks.iter().enumerate() {
            let in_pool = scores[i] > 0.0;
            let tier = if in_pool {
                definition_tier(chunk, &matchers, boost_unit)
            } else {
                if !embedded_stem_matches(&chunk.file_path, &symbols_lower) {
                    continue;
                }
                definition_tier(chunk, &matchers, boost_unit)
            };
            if tier > 0.0 {
                scores[i] += tier;
            }
        }
    }
}

/// File-stem gate for `boost_embedded_symbols` non-candidate scan.
/// Pulled out so the parallel and serial branches stay in sync.
fn embedded_stem_matches(file_path: &str, symbols_lower: &[String]) -> bool {
    let stem = file_stem_lower(file_path);
    let stem_norm = stem.replace('_', "");
    symbols_lower.iter().any(|sym_lower| {
        stem == *sym_lower
            || stem_norm == *sym_lower
            || (stem.len() >= EMBEDDED_STEM_MIN_LEN && sym_lower.starts_with(&stem))
            || (stem_norm.len() >= EMBEDDED_STEM_MIN_LEN && sym_lower.starts_with(&stem_norm))
    })
}

fn boost_stem_matches(scores: &mut [f64], query: &str, max_score: f64, chunks: &[Chunk]) {
    let keywords: AHashSet<String> = TOKEN_RE
        .find_iter(query)
        .map(|m| m.as_str().to_lowercase())
        // Use char count (not byte length) so a 2-letter Cyrillic word like
        // "ии" is correctly treated as too short, while a 3-letter Cyrillic
        // word like "код" passes. STOPWORDS only catches English fillers; for
        // non-English words it's a no-op (none of them match).
        .filter(|w| w.chars().count() > 2 && !STOPWORDS.contains(w.as_str()))
        .collect();

    if keywords.is_empty() {
        return;
    }

    let boost = max_score * STEM_BOOST_MULTIPLIER;
    // Cache path-parts per file_path so we don't re-split for every chunk in the same file.
    let mut path_cache: AHashMap<&str, AHashSet<String>> = AHashMap::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if scores[i] <= 0.0 || scores[i].is_nan() {
            continue;
        }
        let parts = path_cache
            .entry(chunk.file_path.as_str())
            .or_insert_with(|| build_path_parts(&chunk.file_path));

        let n_matches = count_keyword_matches(&keywords, parts);
        if n_matches > 0 {
            let match_ratio = n_matches as f64 / keywords.len() as f64;
            if match_ratio >= 0.10 {
                scores[i] += boost * match_ratio;
            }
        }
    }
}

fn build_path_parts(file_path: &str) -> AHashSet<String> {
    let path = Path::new(file_path);
    let mut parts: AHashSet<String> = AHashSet::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        parts.extend(split_identifier(stem));
    }
    if let Some(parent_name) = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        && ![".", "/", ".."].contains(&parent_name)
    {
        parts.extend(split_identifier(parent_name));
    }
    parts
}

fn count_keyword_matches(keywords: &AHashSet<String>, parts: &AHashSet<String>) -> usize {
    let mut n_matches = 0usize;
    let mut residual: Vec<&String> = Vec::with_capacity(keywords.len());
    for kw in keywords {
        if parts.contains(kw) {
            n_matches += 1;
        } else {
            residual.push(kw);
        }
    }
    if residual.is_empty() {
        return n_matches;
    }
    for keyword in residual {
        for part in parts {
            let (shorter, longer) = if keyword.len() <= part.len() {
                (keyword.as_str(), part.as_str())
            } else {
                (part.as_str(), keyword.as_str())
            };
            if shorter.len() >= 3 && longer.starts_with(shorter) {
                n_matches += 1;
                break;
            }
        }
    }
    n_matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_queries_recognised() {
        // Bare identifiers and conventional code-shaped names should all be
        // detected as "symbol-y" so that hybrid search leans BM25 and the
        // definition-boost path runs.
        for q in [
            "Manifest",
            "TopK",
            "CamelCase",
            "parse_config",
            "PREVIEW_FILE_CACHE",
            "_private_thing",
            "foo::bar",
            "module::Type",
            "obj->method",
            "Foo.bar",
        ] {
            assert!(is_symbol_query(q), "expected symbol query: {q:?}");
        }
    }

    #[test]
    fn natural_language_not_a_symbol_query() {
        for q in [
            "parse the config file",
            "how does auth work",
            "rate limiting middleware",
            "fn parse_config", // multi-token: not a bare symbol
        ] {
            assert!(!is_symbol_query(q), "did not expect symbol query: {q:?}");
        }
    }

    #[test]
    fn embedded_camelcase_is_extracted() {
        let hits: Vec<&str> = EMBEDDED_SYMBOL_RE
            .find_iter("how does FooBar interact with bazQux today")
            .map(|m| m.as_str())
            .collect();
        assert!(hits.contains(&"FooBar"), "FooBar not found in {hits:?}");
        assert!(hits.contains(&"bazQux"), "bazQux not found in {hits:?}");
    }

    #[test]
    fn symbol_def_boost_lifts_a_buried_definition() {
        // Realistic scenario: many chunks contain references to a constant
        // (e.g. `top_k: TOP_K,` as a function arg), so BM25 ranks them
        // above the chunk that defines it. The non-candidate scan must
        // still surface the definition site.
        use crate::types::Chunk;
        let chunks = vec![
            Chunk {
                content: "const TOP_K: usize = 50;".into(),
                file_path: "src/app.rs".into(),
                start_line: 1,
                end_line: 1,
                language: Some("rust".into()),
            },
            Chunk {
                content: "fn search(top_k: usize) { call(top_k); }".into(),
                file_path: "src/search.rs".into(),
                start_line: 1,
                end_line: 1,
                language: Some("rust".into()),
            },
        ];
        // Simulate: only the second chunk made the BM25 candidate pool.
        let mut scores = vec![0.0, 1.0];
        apply_query_boost(&mut scores, "TOP_K", &chunks);
        assert!(
            scores[0] > scores[1],
            "expected the const definition to outrank a reference chunk: scores={scores:?}"
        );
    }

    #[test]
    fn definition_pattern_recognises_const_and_static() {
        // Regression: before adding `const`/`static` to DEFINITION_KEYWORDS,
        // searching for a top-level constant by name would never trigger
        // the definition boost, so its definition site got buried under
        // semantic noise even though BM25 ranked it correctly.
        let p = definition_pattern("PREVIEW_FILE_CACHE");
        assert!(p.matches("const PREVIEW_FILE_CACHE: usize = 8;"));
        assert!(p.matches("static PREVIEW_FILE_CACHE: usize = 8;"));
    }

    #[test]
    fn definition_pattern_compiles_and_matches() {
        // Regression test: the previous regex used `(?<=\s)` lookbehind,
        // which the `regex` crate doesn't support. Compilation panicked
        // the moment a symbol query reached this code path.
        let p = definition_pattern("Manifest");
        assert!(p.matches("pub struct Manifest {"));
        assert!(p.matches("    fn Manifest() {}"));
        // Word-boundary semantics: should not match "Manifest" inside an
        // unrelated keyword-like prefix.
        assert!(!p.matches("classManifest {"));
        // SQL pattern is case-insensitive on the keyword.
        let sql_lower = definition_pattern("users");
        assert!(sql_lower.matches("create table users ("));
    }

    #[test]
    fn definition_pattern_cache_returns_same_arc() {
        // Two lookups for the same name must hit the cache and return
        // the same `Arc`. Validates §1.3: we don't pay the regex
        // compile cost twice for the same symbol.
        let a = definition_pattern("CachedSymbol");
        let b = definition_pattern("CachedSymbol");
        assert!(Arc::ptr_eq(&a, &b), "cache miss — expected same Arc");
    }
}
