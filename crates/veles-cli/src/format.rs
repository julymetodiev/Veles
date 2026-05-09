//! Output formatters for search results.
//!
//! Each renderer takes a slice of [`SearchResult`] and writes to a `String`.
//! `pretty` is the human-friendly default; everything else is pipe-friendly
//! (no decorative header, stable per-line layout).

use std::collections::BTreeSet;
use std::str::FromStr;

use serde::Serialize;
use veles_core::symbols::Symbol;
use veles_core::types::SearchResult;

/// Output format for `search` and `find-related`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-friendly markdown with fenced code blocks (default).
    Pretty,
    /// `path:start-end  [score=X.XXX]  <first non-blank line>` — one line per result.
    Compact,
    /// Ripgrep-style: every source line as `path:lineno:content`.
    Ripgrep,
    /// Just unique paths, one per line, sorted.
    Paths,
    /// Single JSON object: `{"results": [...]}`.
    Json,
    /// JSON Lines: one result object per line.
    Jsonl,
}

impl FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pretty" | "default" | "md" | "markdown" => Ok(Self::Pretty),
            "compact" => Ok(Self::Compact),
            "ripgrep" | "rg" => Ok(Self::Ripgrep),
            "paths" | "files" => Ok(Self::Paths),
            "json" => Ok(Self::Json),
            "jsonl" | "ndjson" => Ok(Self::Jsonl),
            other => Err(format!(
                "unknown format {other:?} (try: pretty, compact, ripgrep, paths, json, jsonl)"
            )),
        }
    }
}

/// Render results in the requested format.
pub fn render(format: OutputFormat, header: &str, results: &[SearchResult]) -> String {
    match format {
        OutputFormat::Pretty => render_pretty(header, results),
        OutputFormat::Compact => render_compact(results),
        OutputFormat::Ripgrep => render_ripgrep(results),
        OutputFormat::Paths => render_paths(results),
        OutputFormat::Json => render_json(header, results, false),
        OutputFormat::Jsonl => render_json(header, results, true),
    }
}

/// Header text to emit when no results are returned.
pub fn empty_message(format: OutputFormat, what: &str) -> String {
    match format {
        OutputFormat::Pretty | OutputFormat::Compact | OutputFormat::Ripgrep => {
            format!("No {what} found.")
        }
        OutputFormat::Paths => String::new(),
        OutputFormat::Json => "{\"results\":[]}".to_string(),
        OutputFormat::Jsonl => String::new(),
    }
}

// ── Renderers ────────────────────────────────────────────────────────────

fn render_pretty(header: &str, results: &[SearchResult]) -> String {
    let mut lines: Vec<String> = vec![header.to_string(), String::new()];
    for (i, r) in results.iter().enumerate() {
        lines.push(format!(
            "## {}. {}  [score={:.3}]",
            i + 1,
            r.chunk.location(),
            r.score
        ));
        lines.push("```".to_string());
        lines.push(r.chunk.content.trim().to_string());
        lines.push("```".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

fn render_compact(results: &[SearchResult]) -> String {
    let mut out = String::new();
    for r in results {
        let snippet = first_nonblank_line(&r.chunk.content);
        out.push_str(&format!(
            "{}:{}-{}  [score={:.3}]  {}\n",
            r.chunk.file_path, r.chunk.start_line, r.chunk.end_line, r.score, snippet,
        ));
    }
    out
}

fn render_ripgrep(results: &[SearchResult]) -> String {
    let mut out = String::new();
    for r in results {
        for (line_no, line) in (r.chunk.start_line..).zip(r.chunk.content.lines()) {
            out.push_str(&format!("{}:{}:{}\n", r.chunk.file_path, line_no, line));
        }
    }
    out
}

fn render_paths(results: &[SearchResult]) -> String {
    // Unique paths, sorted, in result order otherwise.
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut out = String::new();
    for r in results {
        if seen.insert(r.chunk.file_path.as_str()) {
            out.push_str(&r.chunk.file_path);
            out.push('\n');
        }
    }
    out
}

#[derive(Serialize)]
struct JsonResult<'a> {
    file_path: &'a str,
    start_line: usize,
    end_line: usize,
    score: f64,
    source: &'a str,
    language: Option<&'a str>,
    content: &'a str,
}

#[derive(Serialize)]
struct JsonEnvelope<'a> {
    header: &'a str,
    count: usize,
    results: Vec<JsonResult<'a>>,
}

fn to_json_result(r: &SearchResult) -> JsonResult<'_> {
    JsonResult {
        file_path: &r.chunk.file_path,
        start_line: r.chunk.start_line,
        end_line: r.chunk.end_line,
        score: r.score,
        source: r.source.as_str(),
        language: r.chunk.language.as_deref(),
        content: &r.chunk.content,
    }
}

fn render_json(header: &str, results: &[SearchResult], lines: bool) -> String {
    if lines {
        let mut out = String::new();
        for r in results {
            let jr = to_json_result(r);
            // serde_json::to_string handles escaping; one line per record.
            out.push_str(&serde_json::to_string(&jr).unwrap_or_default());
            out.push('\n');
        }
        out
    } else {
        let env = JsonEnvelope {
            header,
            count: results.len(),
            results: results.iter().map(to_json_result).collect(),
        };
        serde_json::to_string(&env).unwrap_or_default()
    }
}

// ── Symbol renderers ─────────────────────────────────────────────────────

/// Render a list of symbols using the same format taxonomy as search results.
pub fn render_symbols(format: OutputFormat, header: &str, symbols: &[&Symbol]) -> String {
    match format {
        OutputFormat::Pretty => render_symbols_pretty(header, symbols),
        OutputFormat::Compact | OutputFormat::Ripgrep => render_symbols_compact(symbols),
        OutputFormat::Paths => render_symbols_paths(symbols),
        OutputFormat::Json => render_symbols_json(header, symbols, false),
        OutputFormat::Jsonl => render_symbols_json(header, symbols, true),
    }
}

fn render_symbols_pretty(header: &str, symbols: &[&Symbol]) -> String {
    let mut lines: Vec<String> = vec![header.to_string(), String::new()];
    for s in symbols {
        lines.push(format!(
            "  {:9}  {:30}  {}:{}",
            s.kind.as_str(),
            s.name,
            s.file_path,
            s.start_line
        ));
    }
    lines.join("\n")
}

fn render_symbols_compact(symbols: &[&Symbol]) -> String {
    let mut out = String::new();
    for s in symbols {
        out.push_str(&format!(
            "{}:{}\t{}\t{}\n",
            s.file_path,
            s.start_line,
            s.kind.as_str(),
            s.name,
        ));
    }
    out
}

fn render_symbols_paths(symbols: &[&Symbol]) -> String {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut out = String::new();
    for s in symbols {
        if seen.insert(s.file_path.as_str()) {
            out.push_str(&s.file_path);
            out.push('\n');
        }
    }
    out
}

#[derive(Serialize)]
struct JsonSymbol<'a> {
    name: &'a str,
    kind: &'a str,
    file_path: &'a str,
    start_line: usize,
    end_line: usize,
    language: &'a str,
}

#[derive(Serialize)]
struct JsonSymbolEnvelope<'a> {
    header: &'a str,
    count: usize,
    symbols: Vec<JsonSymbol<'a>>,
}

fn to_json_symbol(s: &Symbol) -> JsonSymbol<'_> {
    JsonSymbol {
        name: &s.name,
        kind: s.kind.as_str(),
        file_path: &s.file_path,
        start_line: s.start_line,
        end_line: s.end_line,
        language: &s.language,
    }
}

fn render_symbols_json(header: &str, symbols: &[&Symbol], lines: bool) -> String {
    if lines {
        let mut out = String::new();
        for s in symbols {
            out.push_str(&serde_json::to_string(&to_json_symbol(s)).unwrap_or_default());
            out.push('\n');
        }
        out
    } else {
        let env = JsonSymbolEnvelope {
            header,
            count: symbols.len(),
            symbols: symbols.iter().map(|s| to_json_symbol(s)).collect(),
        };
        serde_json::to_string(&env).unwrap_or_default()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn first_nonblank_line(content: &str) -> &str {
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    ""
}

#[cfg(test)]
mod tests {
    use super::*;
    use veles_core::types::{Chunk, SearchMode};

    fn r(path: &str, start: usize, end: usize, score: f64, content: &str) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                file_path: path.to_string(),
                start_line: start,
                end_line: end,
                content: content.to_string(),
                language: Some("rust".to_string()),
            },
            score,
            source: SearchMode::Hybrid,
        }
    }

    #[test]
    fn compact_one_line_per_result() {
        let results = vec![
            r("a.rs", 1, 5, 0.91, "  \nfn foo() {}\nbody\n"),
            r("b.rs", 10, 20, 0.42, "fn bar() {}"),
        ];
        let out = render(OutputFormat::Compact, "h", &results);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("a.rs:1-5"));
        assert!(lines[0].contains("fn foo() {}"));
        assert!(lines[1].starts_with("b.rs:10-20"));
    }

    #[test]
    fn ripgrep_emits_line_per_source_line() {
        let results = vec![r("x.rs", 4, 5, 0.5, "line one\nline two")];
        let out = render(OutputFormat::Ripgrep, "h", &results);
        assert_eq!(out, "x.rs:4:line one\nx.rs:5:line two\n");
    }

    #[test]
    fn paths_dedupe() {
        let results = vec![
            r("a.rs", 1, 5, 0.9, "x"),
            r("a.rs", 6, 10, 0.8, "y"),
            r("b.rs", 1, 5, 0.7, "z"),
        ];
        let out = render(OutputFormat::Paths, "h", &results);
        assert_eq!(out, "a.rs\nb.rs\n");
    }

    #[test]
    fn json_envelope_parses() {
        let results = vec![r("a.rs", 1, 5, 0.9, "fn x() {}")];
        let out = render(OutputFormat::Json, "header", &results);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["count"], 1);
        assert_eq!(v["results"][0]["file_path"], "a.rs");
        assert_eq!(v["results"][0]["score"], 0.9);
    }

    #[test]
    fn jsonl_one_object_per_line() {
        let results = vec![r("a.rs", 1, 5, 0.9, "x"), r("b.rs", 10, 20, 0.4, "y")];
        let out = render(OutputFormat::Jsonl, "h", &results);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn parse_aliases() {
        assert_eq!("rg".parse::<OutputFormat>().unwrap(), OutputFormat::Ripgrep);
        assert_eq!("md".parse::<OutputFormat>().unwrap(), OutputFormat::Pretty);
        assert_eq!(
            "ndjson".parse::<OutputFormat>().unwrap(),
            OutputFormat::Jsonl
        );
        assert!("xml".parse::<OutputFormat>().is_err());
    }
}
