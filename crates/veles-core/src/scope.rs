//! Scope-label heuristics for chunks.
//!
//! Given the index's tree-sitter symbol table and a chunk, produce a
//! short human-readable label answering "what does this chunk show?".
//!
//! Used by formatters (CLI, MCP) to enrich result headers — an agent or
//! human reading the line `crates/foo/bar.rs:46-95  [score=0.025]` gets
//! a much faster answer to "is this relevant?" when the line ends with
//! ``defines `Manifest``` or ``in `fn handle_search```.
//!
//! Two entry points:
//!
//! - [`chunk_scope_label`] — one-shot label lookup. Iterates the full
//!   symbol slice; fine for one or two labels per call (CLI format,
//!   MCP refs handler).
//! - [`ScopeIndex`] — pre-builds a `file_path → symbol indices` map so
//!   each subsequent label call is `O(symbols_in_file)` instead of
//!   `O(total_symbols)`. The right choice for the TUI render loop,
//!   which resolves ~30 labels per redraw.

use ahash::AHashMap;

use crate::symbols::Symbol;
use crate::types::Chunk;

/// Pick a short scope label for a chunk so a reader can route on the
/// result header without reading the body.
///
/// Two-tier heuristic:
/// 1. If any symbols *start* inside the chunk, the chunk is showing
///    those definitions — return ``defines `name` `` (or
///    ``defines `name` (+N more) `` when several definitions appear).
/// 2. Else find the most specific symbol whose range strictly contains
///    `chunk.start_line` (the chunk is mid-body) — return ``in `name` ``.
///
/// Returns `None` for chunks that neither define nor live inside any
/// tree-sitter-recognised symbol (typical for module-level prelude
/// before the first definition, or files in unsupported languages).
pub fn chunk_scope_label(symbols: &[Symbol], chunk: &Chunk) -> Option<String> {
    let same_file = || symbols.iter().filter(|s| s.file_path == chunk.file_path);

    let defined: Vec<&Symbol> = same_file()
        .filter(|s| s.start_line >= chunk.start_line && s.start_line <= chunk.end_line)
        .collect();
    if let Some(first) = defined.first() {
        return Some(if defined.len() == 1 {
            format!("defines `{}`", first.name)
        } else {
            format!("defines `{}` (+{} more)", first.name, defined.len() - 1)
        });
    }

    same_file()
        .filter(|s| s.start_line < chunk.start_line && chunk.start_line <= s.end_line)
        .min_by_key(|s| s.end_line.saturating_sub(s.start_line))
        .map(|s| format!("in `{}`", s.name))
}

/// Pre-indexed `file_path → symbol indices` map for fast repeated
/// scope-label lookups (the TUI render loop).
///
/// Built once from an immutable symbol slice; each `label()` call then
/// scans only the symbols of the chunk's file rather than the entire
/// index. For a 200K-symbol repo this turns each redraw's O(N × rows)
/// scan into O(symbols_per_file × rows) — typically a handful per row.
///
/// Stores `u32` indices into the original symbol slice, so the slice
/// the caller passes to `label()` must be the same one that was used
/// to build the index (length and order preserved). Pass the
/// `VelesIndex::symbols()` slice and it just works.
#[derive(Debug, Default)]
pub struct ScopeIndex {
    by_file: AHashMap<String, Vec<u32>>,
}

impl ScopeIndex {
    /// Build the lookup map. O(symbols).
    pub fn new(symbols: &[Symbol]) -> Self {
        let mut by_file: AHashMap<String, Vec<u32>> = AHashMap::new();
        for (i, s) in symbols.iter().enumerate() {
            by_file
                .entry(s.file_path.clone())
                .or_default()
                .push(i as u32);
        }
        Self { by_file }
    }

    /// Same semantics as [`chunk_scope_label`] but O(symbols_in_file).
    pub fn label(&self, symbols: &[Symbol], chunk: &Chunk) -> Option<String> {
        let indices = self.by_file.get(chunk.file_path.as_str())?;

        // Tier 1: definitions whose start line falls inside the chunk.
        let mut first_defined: Option<&Symbol> = None;
        let mut defined_count: usize = 0;
        for &i in indices {
            let s = symbols.get(i as usize)?;
            if s.start_line >= chunk.start_line && s.start_line <= chunk.end_line {
                if first_defined.is_none() {
                    first_defined = Some(s);
                }
                defined_count += 1;
            }
        }
        if let Some(first) = first_defined {
            return Some(if defined_count == 1 {
                format!("defines `{}`", first.name)
            } else {
                format!("defines `{}` (+{} more)", first.name, defined_count - 1)
            });
        }

        // Tier 2: innermost enclosing symbol whose range strictly contains chunk.start_line.
        let mut best: Option<&Symbol> = None;
        let mut best_span: usize = usize::MAX;
        for &i in indices {
            let s = symbols.get(i as usize)?;
            if s.start_line < chunk.start_line && chunk.start_line <= s.end_line {
                let span = s.end_line.saturating_sub(s.start_line);
                if span < best_span {
                    best_span = span;
                    best = Some(s);
                }
            }
        }
        best.map(|s| format!("in `{}`", s.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::SymbolKind;

    fn sym(name: &str, kind: SymbolKind, file: &str, start: usize, end: usize) -> Symbol {
        Symbol {
            name: name.to_string(),
            kind,
            file_path: file.to_string(),
            start_line: start,
            end_line: end,
            language: "rust".to_string(),
        }
    }

    fn chunk(file: &str, start: usize, end: usize) -> Chunk {
        Chunk {
            content: String::new(),
            file_path: file.to_string(),
            start_line: start,
            end_line: end,
            language: Some("rust".to_string()),
        }
    }

    #[test]
    fn defines_one_symbol() {
        let symbols = vec![sym("foo", SymbolKind::Function, "a.rs", 5, 8)];
        let label = chunk_scope_label(&symbols, &chunk("a.rs", 1, 50));
        assert_eq!(label.as_deref(), Some("defines `foo`"));
    }

    #[test]
    fn defines_with_more() {
        let symbols = vec![
            sym("foo", SymbolKind::Function, "a.rs", 5, 8),
            sym("bar", SymbolKind::Function, "a.rs", 10, 12),
            sym("baz", SymbolKind::Struct, "a.rs", 14, 20),
        ];
        let label = chunk_scope_label(&symbols, &chunk("a.rs", 1, 50));
        assert_eq!(label.as_deref(), Some("defines `foo` (+2 more)"));
    }

    #[test]
    fn picks_innermost_enclosing_when_no_def_inside() {
        // Outer fn covers 1-100, inner method covers 30-60 inside it.
        // A chunk starting at line 40 should be tagged with the inner one.
        let symbols = vec![
            sym("outer", SymbolKind::Function, "a.rs", 1, 100),
            sym("inner", SymbolKind::Function, "a.rs", 30, 60),
        ];
        let label = chunk_scope_label(&symbols, &chunk("a.rs", 40, 50));
        assert_eq!(label.as_deref(), Some("in `inner`"));
    }

    #[test]
    fn other_files_ignored() {
        let symbols = vec![sym("foo", SymbolKind::Function, "b.rs", 5, 8)];
        let label = chunk_scope_label(&symbols, &chunk("a.rs", 1, 50));
        assert_eq!(label, None);
    }

    #[test]
    fn no_match_returns_none() {
        let symbols = vec![sym("foo", SymbolKind::Function, "a.rs", 100, 110)];
        let label = chunk_scope_label(&symbols, &chunk("a.rs", 1, 50));
        assert_eq!(label, None);
    }

    #[test]
    fn scope_index_matches_one_shot() {
        // The ScopeIndex must return the same labels as chunk_scope_label
        // for every chunk — it's a strict optimisation, not a behaviour change.
        let symbols = vec![
            sym("outer", SymbolKind::Function, "a.rs", 1, 100),
            sym("inner", SymbolKind::Function, "a.rs", 30, 60),
            sym("other", SymbolKind::Function, "b.rs", 5, 8),
        ];
        let idx = ScopeIndex::new(&symbols);
        let chunks = [
            chunk("a.rs", 1, 50),    // defines outer
            chunk("a.rs", 40, 50),   // in inner
            chunk("b.rs", 1, 10),    // defines other
            chunk("a.rs", 200, 250), // nothing
            chunk("nonexistent.rs", 1, 5),
        ];
        for c in &chunks {
            let one_shot = chunk_scope_label(&symbols, c);
            let indexed = idx.label(&symbols, c);
            assert_eq!(
                one_shot, indexed,
                "ScopeIndex diverged from chunk_scope_label for chunk {c:?}"
            );
        }
    }
}
