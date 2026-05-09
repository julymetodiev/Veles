//! Tree-sitter-backed symbol extraction.
//!
//! For each supported language we run a small `(name) @def` query that pulls
//! out function / struct / class / etc. definitions. Symbols are stored as a
//! flat list keyed by file path; downstream code uses them for the
//! `symbols`, `defs`, and `refs` CLI commands.

use serde::{Deserialize, Serialize};
use tree_sitter::{Parser, Query, QueryCursor};

/// Kind of a definition. Kept coarse — the goal is "is this a thing you'd
/// `cmd-click` to" rather than a full IDE taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Struct,
    Class,
    Enum,
    Trait,
    Interface,
    Type,
    Const,
    Static,
    Var,
    Module,
    Macro,
}

impl SymbolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Struct => "struct",
            Self::Class => "class",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::Type => "type",
            Self::Const => "const",
            Self::Static => "static",
            Self::Var => "var",
            Self::Module => "module",
            Self::Macro => "macro",
        }
    }
}

/// A definition extracted from source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub file_path: String,
    /// 1-indexed start line.
    pub start_line: usize,
    /// 1-indexed inclusive end line.
    pub end_line: usize,
    pub language: String,
}

impl Symbol {
    pub fn location(&self) -> String {
        format!("{}:{}", self.file_path, self.start_line)
    }
}

/// True if we have a tree-sitter parser + query for this language.
pub fn supports(language: &str) -> bool {
    matches!(
        language,
        "rust" | "python" | "javascript" | "typescript" | "go"
    )
}

/// Extract definitions from a source file. Unknown languages return empty.
///
/// `file_path` is stored verbatim in the resulting symbols (no path
/// normalisation here — caller is responsible for using a stable form,
/// usually the same relative path as the corresponding [`crate::types::Chunk`]).
pub fn extract_symbols(content: &str, file_path: &str, language: &str) -> Vec<Symbol> {
    let cfg = match language_config(language) {
        Some(c) => c,
        None => return Vec::new(),
    };

    let mut parser = Parser::new();
    if parser.set_language(&cfg.ts_language).is_err() {
        return Vec::new();
    }
    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let query = match Query::new(&cfg.ts_language, cfg.query_source) {
        Ok(q) => q,
        Err(_) => return Vec::new(),
    };

    let name_idx = query
        .capture_index_for_name("name")
        .expect("query must capture @name");

    let source_bytes = content.as_bytes();
    let mut cursor = QueryCursor::new();
    let iter = cursor.matches(&query, tree.root_node(), source_bytes);
    let mut out: Vec<Symbol> = Vec::new();

    for m in iter {
        let pattern = m.pattern_index;
        let kind = match cfg.kinds.get(pattern) {
            Some(Some(k)) => *k,
            _ => continue,
        };

        // Find the @name capture in this match.
        let name_node = m
            .captures
            .iter()
            .find(|c| c.index == name_idx)
            .map(|c| c.node);
        // The "anchor" capture for line range — first capture that isn't @name.
        let def_node = m
            .captures
            .iter()
            .find(|c| c.index != name_idx)
            .map(|c| c.node)
            .or(name_node);

        let (Some(name_node), Some(def_node)) = (name_node, def_node) else {
            continue;
        };

        let name = match name_node.utf8_text(source_bytes) {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };
        let start_line = def_node.start_position().row + 1;
        let end_line = def_node.end_position().row + 1;

        out.push(Symbol {
            name,
            kind,
            file_path: file_path.to_string(),
            start_line,
            end_line,
            language: language.to_string(),
        });
    }

    out
}

// ── Per-language config ──────────────────────────────────────────────────

struct LanguageConfig {
    ts_language: tree_sitter::Language,
    query_source: &'static str,
    /// `kinds[pattern_index]` → SymbolKind for that pattern (None = skip).
    kinds: &'static [Option<SymbolKind>],
}

fn language_config(language: &str) -> Option<LanguageConfig> {
    match language {
        "rust" => Some(LanguageConfig {
            ts_language: tree_sitter_rust::LANGUAGE.into(),
            query_source: RUST_QUERY,
            kinds: RUST_KINDS,
        }),
        "python" => Some(LanguageConfig {
            ts_language: tree_sitter_python::LANGUAGE.into(),
            query_source: PYTHON_QUERY,
            kinds: PYTHON_KINDS,
        }),
        "javascript" => Some(LanguageConfig {
            ts_language: tree_sitter_javascript::LANGUAGE.into(),
            query_source: JAVASCRIPT_QUERY,
            kinds: JAVASCRIPT_KINDS,
        }),
        "typescript" => Some(LanguageConfig {
            ts_language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            query_source: TYPESCRIPT_QUERY,
            kinds: TYPESCRIPT_KINDS,
        }),
        "go" => Some(LanguageConfig {
            ts_language: tree_sitter_go::LANGUAGE.into(),
            query_source: GO_QUERY,
            kinds: GO_KINDS,
        }),
        _ => None,
    }
}

// Patterns are 1:1 with `*_KINDS` — order MUST match.
const RUST_QUERY: &str = r#"
(function_item name: (identifier) @name) @def
(struct_item name: (type_identifier) @name) @def
(enum_item name: (type_identifier) @name) @def
(trait_item name: (type_identifier) @name) @def
(type_item name: (type_identifier) @name) @def
(const_item name: (identifier) @name) @def
(static_item name: (identifier) @name) @def
(macro_definition name: (identifier) @name) @def
(mod_item name: (identifier) @name) @def
"#;
const RUST_KINDS: &[Option<SymbolKind>] = &[
    Some(SymbolKind::Function),
    Some(SymbolKind::Struct),
    Some(SymbolKind::Enum),
    Some(SymbolKind::Trait),
    Some(SymbolKind::Type),
    Some(SymbolKind::Const),
    Some(SymbolKind::Static),
    Some(SymbolKind::Macro),
    Some(SymbolKind::Module),
];

const PYTHON_QUERY: &str = r#"
(function_definition name: (identifier) @name) @def
(class_definition name: (identifier) @name) @def
"#;
const PYTHON_KINDS: &[Option<SymbolKind>] =
    &[Some(SymbolKind::Function), Some(SymbolKind::Class)];

const JAVASCRIPT_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @def
(class_declaration name: (identifier) @name) @def
(method_definition name: (property_identifier) @name) @def
(variable_declarator name: (identifier) @name value: (arrow_function)) @def
(variable_declarator name: (identifier) @name value: (function_expression)) @def
"#;
const JAVASCRIPT_KINDS: &[Option<SymbolKind>] = &[
    Some(SymbolKind::Function),
    Some(SymbolKind::Class),
    Some(SymbolKind::Method),
    Some(SymbolKind::Function),
    Some(SymbolKind::Function),
];

const TYPESCRIPT_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @def
(class_declaration name: (type_identifier) @name) @def
(method_definition name: (property_identifier) @name) @def
(interface_declaration name: (type_identifier) @name) @def
(type_alias_declaration name: (type_identifier) @name) @def
(enum_declaration name: (identifier) @name) @def
(variable_declarator name: (identifier) @name value: (arrow_function)) @def
"#;
const TYPESCRIPT_KINDS: &[Option<SymbolKind>] = &[
    Some(SymbolKind::Function),
    Some(SymbolKind::Class),
    Some(SymbolKind::Method),
    Some(SymbolKind::Interface),
    Some(SymbolKind::Type),
    Some(SymbolKind::Enum),
    Some(SymbolKind::Function),
];

const GO_QUERY: &str = r#"
(function_declaration name: (identifier) @name) @def
(method_declaration name: (field_identifier) @name) @def
(type_spec name: (type_identifier) @name) @def
(const_spec name: (identifier) @name) @def
(var_spec name: (identifier) @name) @def
"#;
const GO_KINDS: &[Option<SymbolKind>] = &[
    Some(SymbolKind::Function),
    Some(SymbolKind::Method),
    Some(SymbolKind::Type),
    Some(SymbolKind::Const),
    Some(SymbolKind::Var),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_extracts_function_struct_enum_trait_const() {
        let src = r#"
fn hello() {}
struct Foo { x: i32 }
enum Color { Red, Blue }
trait Shape { fn area(&self) -> f64; }
const MAX: u32 = 10;
static LOG: &str = "log";
type Pair = (i32, i32);
mod inner {}
macro_rules! mymac { () => {} }
"#;
        let s = extract_symbols(src, "test.rs", "rust");
        let by_name: std::collections::HashMap<&str, SymbolKind> =
            s.iter().map(|s| (s.name.as_str(), s.kind)).collect();
        assert_eq!(by_name.get("hello"), Some(&SymbolKind::Function));
        assert_eq!(by_name.get("Foo"), Some(&SymbolKind::Struct));
        assert_eq!(by_name.get("Color"), Some(&SymbolKind::Enum));
        assert_eq!(by_name.get("Shape"), Some(&SymbolKind::Trait));
        assert_eq!(by_name.get("MAX"), Some(&SymbolKind::Const));
        assert_eq!(by_name.get("LOG"), Some(&SymbolKind::Static));
        assert_eq!(by_name.get("Pair"), Some(&SymbolKind::Type));
        assert_eq!(by_name.get("inner"), Some(&SymbolKind::Module));
        assert_eq!(by_name.get("mymac"), Some(&SymbolKind::Macro));
    }

    #[test]
    fn python_extracts_function_and_class() {
        let src = "def parse(x):\n    pass\n\nclass Parser:\n    def parse(self):\n        pass\n";
        let s = extract_symbols(src, "p.py", "python");
        let names: Vec<&str> = s.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"parse"));
        assert!(names.contains(&"Parser"));
        let class_kind = s
            .iter()
            .find(|s| s.name == "Parser")
            .map(|s| s.kind)
            .unwrap();
        assert_eq!(class_kind, SymbolKind::Class);
    }

    #[test]
    fn typescript_interface_and_type() {
        let src = "interface Foo { x: number; }\ntype Bar = string;\nclass Baz {}\n";
        let s = extract_symbols(src, "t.ts", "typescript");
        let kinds: std::collections::HashMap<&str, SymbolKind> =
            s.iter().map(|s| (s.name.as_str(), s.kind)).collect();
        assert_eq!(kinds.get("Foo"), Some(&SymbolKind::Interface));
        assert_eq!(kinds.get("Bar"), Some(&SymbolKind::Type));
        assert_eq!(kinds.get("Baz"), Some(&SymbolKind::Class));
    }

    #[test]
    fn go_function_method_type() {
        let src = "package x\nfunc hello() {}\nfunc (s *Server) Run() {}\ntype Server struct{}\n";
        let s = extract_symbols(src, "g.go", "go");
        let kinds: std::collections::HashMap<&str, SymbolKind> =
            s.iter().map(|s| (s.name.as_str(), s.kind)).collect();
        assert_eq!(kinds.get("hello"), Some(&SymbolKind::Function));
        assert_eq!(kinds.get("Run"), Some(&SymbolKind::Method));
        assert_eq!(kinds.get("Server"), Some(&SymbolKind::Type));
    }

    #[test]
    fn unknown_language_returns_empty() {
        let s = extract_symbols("blah", "x.txt", "klingon");
        assert!(s.is_empty());
    }

    #[test]
    fn lines_are_one_indexed_and_correct() {
        let src = "// header\nfn foo() {\n  let x = 1;\n}\n";
        let s = extract_symbols(src, "x.rs", "rust");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "foo");
        assert_eq!(s[0].start_line, 2);
        assert_eq!(s[0].end_line, 4);
    }
}
