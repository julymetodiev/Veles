//! Scope-label heuristics for chunks.
//!
//! Given the index's tree-sitter symbol table and a chunk, produce a
//! short human-readable label answering "what does this chunk show?".
//!
//! Used by formatters (CLI, MCP) to enrich result headers — an agent or
//! human reading the line `crates/foo/bar.rs:46-95  [score=0.025]` gets
//! a much faster answer to "is this relevant?" when the line ends with
//! ``defines `Manifest``` or ``in `fn handle_search```.

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
}
