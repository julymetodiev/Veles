//! Tokenizer for BM25 indexing — splits identifiers into sub-tokens.
//!
//! Hot path: indexing tokenises every chunk (~50 lines of code each), so a few
//! micro-optimisations matter here:
//!   * `tokenize_into` lets callers reuse an output buffer across docs.
//!   * `split_identifier_into` only emits the lowercased original when there
//!     are no sub-tokens (no Vec allocation in the simple-identifier case).
//!   * ASCII fast-path lowercase avoids the Unicode case-folding allocation
//!     for source code, which is overwhelmingly ASCII.

use regex::Regex;
use std::sync::LazyLock;

// Match identifier-like tokens in source text.
//
// Unicode-aware: any letter (including Cyrillic, Greek, CJK, Arabic, …) starts
// a token; it then continues with letters, digits, or underscores. ASCII is a
// strict subset of `\p{L}`/`\p{N}`, so this stays correct for English source.
//
// Note: camelCase splitting in `split_camel_into` operates on ASCII bytes and
// simply emits the whole-token form for non-ASCII identifiers (we don't try to
// split scripts whose case-folding semantics differ from ASCII).
static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}_][\p{L}\p{N}_]*").unwrap());

/// Split an ASCII identifier into camelCase/PascalCase sub-tokens.
///
/// Mirrors the Python tokeniser's intent (the equivalent regex used a
/// look-ahead, which the `regex` crate doesn't support):
///
///   `HandlerStack`     → `["Handler", "Stack"]`
///   `getHTTPResponse`  → `["get", "HTTP", "Response"]`
///   `XMLParser`        → `["XML", "Parser"]`
///   `parse2Things`     → `["parse", "2", "Things"]`
///
/// Handwritten in plain Rust to avoid both the regex look-around and the
/// per-call allocation of regex iteration.
fn split_camel_into(token: &str, out: &mut Vec<String>) {
    let bytes = token.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return;
    }

    let mut i = 0usize;
    while i < n {
        let c = bytes[i];

        if c.is_ascii_uppercase() {
            // Consume the run of consecutive uppercase letters.
            let run_start = i;
            while i < n && bytes[i].is_ascii_uppercase() {
                i += 1;
            }
            let upper_end = i;
            let run_len = upper_end - run_start;

            if i < n && bytes[i].is_ascii_lowercase() {
                // Uppercase run followed by lowercase letters.
                if run_len >= 2 {
                    // Acronym boundary: the *last* uppercase belongs to the next word.
                    //   "HTTPResponse" → "HTTP" + "Response"
                    let acronym_end = upper_end - 1;
                    push_slice(token, run_start, acronym_end, out);
                    let word_start = acronym_end;
                    while i < n && bytes[i].is_ascii_lowercase() {
                        i += 1;
                    }
                    push_slice(token, word_start, i, out);
                } else {
                    // Single uppercase + lowercase: "Handler", "Config".
                    while i < n && bytes[i].is_ascii_lowercase() {
                        i += 1;
                    }
                    push_slice(token, run_start, i, out);
                }
            } else {
                // Pure uppercase block: "HTTP", "XML", or trailing acronym.
                push_slice(token, run_start, upper_end, out);
            }
        } else if c.is_ascii_lowercase() {
            let s = i;
            while i < n && bytes[i].is_ascii_lowercase() {
                i += 1;
            }
            push_slice(token, s, i, out);
        } else if c.is_ascii_digit() {
            let s = i;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
            }
            push_slice(token, s, i, out);
        } else {
            // Anything else (e.g. underscore not handled here) — skip.
            i += 1;
        }
    }
}

#[inline]
fn push_slice(token: &str, start: usize, end: usize, out: &mut Vec<String>) {
    // Token is ASCII (TOKEN_RE only matches `[a-zA-Z_][a-zA-Z0-9_]*`), so byte
    // slicing is safe and we can lowercase per-byte.
    let slice = &token[start..end];
    out.push(lowercase(slice));
}

/// Lowercase an ASCII-mostly string. Falls back to `to_lowercase` if any
/// non-ASCII byte is seen, keeping correctness for source containing Unicode.
#[inline]
fn lowercase(s: &str) -> String {
    if s.bytes().all(|b| b < 0x80) {
        // ASCII-only: simple per-byte lowercase, exactly one allocation.
        let mut out = String::with_capacity(s.len());
        for b in s.bytes() {
            out.push((if b.is_ascii_uppercase() { b + 32 } else { b }) as char);
        }
        out
    } else {
        s.to_lowercase()
    }
}

/// Append sub-tokens of a single identifier (lowercased) to `out`.
///
/// Always emits the lowercased whole token. If the identifier splits into
/// multiple parts (snake_case or camelCase), each sub-token is appended too.
#[inline]
fn split_identifier_into(token: &str, out: &mut Vec<String>) {
    let lower = lowercase(token);

    if token.contains('_') {
        let parts: Vec<&str> = lower.split('_').filter(|p| !p.is_empty()).collect();
        if parts.len() >= 2 {
            let part_strings: Vec<String> = parts.iter().map(|p| (*p).to_string()).collect();
            out.push(lower);
            out.extend(part_strings);
            return;
        }
        out.push(lower);
        return;
    }

    // camelCase / PascalCase / digit boundaries — handwritten splitter.
    // We split into a scratch buffer first so we can decide whether to also
    // emit the whole-token form.
    let len_before = out.len();
    split_camel_into(token, out);
    let n_parts = out.len() - len_before;

    if n_parts >= 2 {
        // Insert the whole token *before* the sub-tokens to match the prior order:
        //   `["handlerstack", "handler", "stack"]`.
        out.insert(len_before, lower);
    } else {
        // 0 or 1 sub-tokens — keep just the whole-token form.
        out.truncate(len_before);
        out.push(lower);
    }
}

/// Split a single identifier into sub-tokens.
///
/// Convenience wrapper retained for API compatibility with ranking code.
pub fn split_identifier(token: &str) -> Vec<String> {
    let mut out = Vec::new();
    split_identifier_into(token, &mut out);
    out
}

/// Tokenize text into the provided output buffer.
///
/// The buffer is appended to (not cleared); callers can reuse a single Vec
/// across many documents to amortise allocation.
pub fn tokenize_into(text: &str, out: &mut Vec<String>) {
    for tok in TOKEN_RE.find_iter(text) {
        split_identifier_into(tok.as_str(), out);
    }
}

/// Split text into lowercase identifier-like tokens for BM25 indexing.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    tokenize_into(text, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_identifier_snake_case() {
        let parts = split_identifier("my_func");
        assert_eq!(parts, vec!["my_func", "my", "func"]);
    }

    #[test]
    fn test_split_identifier_camel_case() {
        let parts = split_identifier("HandlerStack");
        assert_eq!(parts, vec!["handlerstack", "handler", "stack"]);
    }

    #[test]
    fn test_split_identifier_simple() {
        let parts = split_identifier("simple");
        assert_eq!(parts, vec!["simple"]);
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokens = tokenize("parseConfig handler");
        assert!(tokens.contains(&"parseconfig".to_string()));
        assert!(tokens.contains(&"parse".to_string()));
        assert!(tokens.contains(&"config".to_string()));
        assert!(tokens.contains(&"handler".to_string()));
    }

    #[test]
    fn test_tokenize_cyrillic() {
        // Cyrillic words should be captured as whole tokens (no camelCase
        // splitting), and lowercased correctly.
        let tokens = tokenize("Как работи токенизаторът");
        assert!(tokens.contains(&"как".to_string()));
        assert!(tokens.contains(&"работи".to_string()));
        assert!(tokens.contains(&"токенизаторът".to_string()));
    }

    #[test]
    fn test_tokenize_mixed_scripts() {
        // Mixed ASCII and Cyrillic should each be captured as their own token.
        let tokens = tokenize("parseConfig функция handler");
        assert!(tokens.contains(&"parseconfig".to_string()));
        assert!(tokens.contains(&"parse".to_string()));
        assert!(tokens.contains(&"config".to_string()));
        assert!(tokens.contains(&"функция".to_string()));
        assert!(tokens.contains(&"handler".to_string()));
    }

    #[test]
    fn test_tokenize_cjk() {
        // CJK characters should also be captured (no spaces between chars in
        // Chinese — the regex matches each contiguous run of letters).
        let tokens = tokenize("函数 search 関数");
        assert!(tokens.contains(&"函数".to_string()));
        assert!(tokens.contains(&"search".to_string()));
        assert!(tokens.contains(&"関数".to_string()));
    }

    #[test]
    fn ascii_fast_path_matches_unicode() {
        // Make sure both code paths agree on a mixed input.
        let s_ascii = "FooBar123";
        let s_unicode = "FooBär123";
        assert_eq!(lowercase(s_ascii), s_ascii.to_lowercase());
        assert_eq!(lowercase(s_unicode), s_unicode.to_lowercase());
    }
}
