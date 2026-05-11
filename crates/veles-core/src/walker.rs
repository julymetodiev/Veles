//! File walker — walks directories, filters by extension, respects .gitignore.

use ahash::AHashMap;
use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Supported file types with their language and category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileCategory {
    Code,
    Document,
}

#[derive(Debug, Clone, Copy)]
pub struct FileType {
    pub language: &'static str,
    pub category: FileCategory,
}

/// Map of file extension (lowercase, with dot) to file type info.
pub static FILE_TYPES: &[(&str, FileType)] = &[
    (
        ".py",
        FileType {
            language: "python",
            category: FileCategory::Code,
        },
    ),
    (
        ".js",
        FileType {
            language: "javascript",
            category: FileCategory::Code,
        },
    ),
    (
        ".jsx",
        FileType {
            language: "javascript",
            category: FileCategory::Code,
        },
    ),
    (
        ".ts",
        FileType {
            language: "typescript",
            category: FileCategory::Code,
        },
    ),
    (
        ".tsx",
        FileType {
            language: "typescript",
            category: FileCategory::Code,
        },
    ),
    (
        ".go",
        FileType {
            language: "go",
            category: FileCategory::Code,
        },
    ),
    (
        ".rs",
        FileType {
            language: "rust",
            category: FileCategory::Code,
        },
    ),
    (
        ".java",
        FileType {
            language: "java",
            category: FileCategory::Code,
        },
    ),
    (
        ".kt",
        FileType {
            language: "kotlin",
            category: FileCategory::Code,
        },
    ),
    (
        ".kts",
        FileType {
            language: "kotlin",
            category: FileCategory::Code,
        },
    ),
    (
        ".rb",
        FileType {
            language: "ruby",
            category: FileCategory::Code,
        },
    ),
    (
        ".php",
        FileType {
            language: "php",
            category: FileCategory::Code,
        },
    ),
    (
        ".c",
        FileType {
            language: "c",
            category: FileCategory::Code,
        },
    ),
    (
        ".h",
        FileType {
            language: "c",
            category: FileCategory::Code,
        },
    ),
    (
        ".cpp",
        FileType {
            language: "cpp",
            category: FileCategory::Code,
        },
    ),
    (
        ".hpp",
        FileType {
            language: "cpp",
            category: FileCategory::Code,
        },
    ),
    (
        ".cs",
        FileType {
            language: "csharp",
            category: FileCategory::Code,
        },
    ),
    (
        ".swift",
        FileType {
            language: "swift",
            category: FileCategory::Code,
        },
    ),
    (
        ".scala",
        FileType {
            language: "scala",
            category: FileCategory::Code,
        },
    ),
    (
        ".sbt",
        FileType {
            language: "scala",
            category: FileCategory::Code,
        },
    ),
    (
        ".ex",
        FileType {
            language: "elixir",
            category: FileCategory::Code,
        },
    ),
    (
        ".exs",
        FileType {
            language: "elixir",
            category: FileCategory::Code,
        },
    ),
    (
        ".dart",
        FileType {
            language: "dart",
            category: FileCategory::Code,
        },
    ),
    (
        ".lua",
        FileType {
            language: "lua",
            category: FileCategory::Code,
        },
    ),
    (
        ".sql",
        FileType {
            language: "sql",
            category: FileCategory::Code,
        },
    ),
    (
        ".sh",
        FileType {
            language: "bash",
            category: FileCategory::Code,
        },
    ),
    (
        ".bash",
        FileType {
            language: "bash",
            category: FileCategory::Code,
        },
    ),
    (
        ".zig",
        FileType {
            language: "zig",
            category: FileCategory::Code,
        },
    ),
    (
        ".hs",
        FileType {
            language: "haskell",
            category: FileCategory::Code,
        },
    ),
    // Document types
    (
        ".md",
        FileType {
            language: "markdown",
            category: FileCategory::Document,
        },
    ),
    (
        ".yaml",
        FileType {
            language: "yaml",
            category: FileCategory::Document,
        },
    ),
    (
        ".yml",
        FileType {
            language: "yaml",
            category: FileCategory::Document,
        },
    ),
    (
        ".toml",
        FileType {
            language: "toml",
            category: FileCategory::Document,
        },
    ),
    (
        ".json",
        FileType {
            language: "json",
            category: FileCategory::Document,
        },
    ),
];

/// Default ignored directory names.
pub static DEFAULT_IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".veles",
    "dist",
    "build",
    ".eggs",
    "target",
    ".cargo",
    ".next",
    ".nuxt",
];

/// Lookup table from extension (without dot, lowercased) to language.
static EXT_LANG_MAP: LazyLock<AHashMap<&'static str, &'static str>> = LazyLock::new(|| {
    FILE_TYPES
        .iter()
        .map(|(ext, ft)| {
            // Strip the leading "." once at table-build time.
            let trimmed = ext.strip_prefix('.').unwrap_or(*ext);
            (trimmed, ft.language)
        })
        .collect()
});

/// Return the language for a file path based on its extension.
pub fn language_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    if ext
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        EXT_LANG_MAP.get(ext).copied()
    } else {
        // Slow path: handle uppercase / mixed-case extensions.
        let lower = ext.to_ascii_lowercase();
        EXT_LANG_MAP.get(lower.as_str()).copied()
    }
}

/// Pre-built extension set for "code only" — the default and most-used path.
static CODE_EXTENSIONS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    FILE_TYPES
        .iter()
        .filter(|(_, ft)| ft.category == FileCategory::Code)
        .map(|(ext, _)| (*ext).to_string())
        .collect()
});

/// Pre-built extension set for code + text documents.
static CODE_AND_DOC_EXTENSIONS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    FILE_TYPES
        .iter()
        .map(|(ext, _)| (*ext).to_string())
        .collect()
});

/// Build the set of file extensions to include based on parameters.
///
/// The two no-`extensions` paths are the overwhelmingly common case
/// (CLI / MCP / gRPC all hit them) and used to rebuild the `HashSet`
/// from scratch on every call. They now return a clone of a
/// process-wide `LazyLock<HashSet<String>>` — the clone is still
/// `O(n)` over ~35 extensions but skips the `FILE_TYPES` scan + the
/// category-set construction (§5.3 of the perf plan).
pub fn filter_extensions(
    extensions: Option<&HashSet<String>>,
    include_text_files: bool,
) -> HashSet<String> {
    if let Some(exts) = extensions {
        return exts.clone();
    }
    if include_text_files {
        CODE_AND_DOC_EXTENSIONS.clone()
    } else {
        CODE_EXTENSIONS.clone()
    }
}

/// Maximum file size to read and index (1 MB).
const MAX_FILE_BYTES: u64 = 1_000_000;

/// Walk files under `root` matching the given extensions.
///
/// Uses the `ignore` crate which automatically respects `.gitignore` files,
/// and skips hidden files and common ignored directories.
pub fn walk_files<'a>(
    root: &'a Path,
    extensions: &'a HashSet<String>,
) -> impl Iterator<Item = PathBuf> + 'a {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true) // respect global gitignore
        .git_exclude(true) // respect .git/info/exclude
        .build()
        .filter_map(move |entry| {
            let entry = entry.ok()?;
            if !entry.file_type()?.is_file() {
                return None;
            }
            let path = entry.path();

            // Check extension (avoid the per-file `format!(".{ext}")` allocation).
            let ext = path.extension()?.to_str()?;
            let matched = if ext
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            {
                // Fast path: probe directly with a stack buffer.
                let mut buf = [0u8; 16];
                let n = ext.len() + 1;
                if n > buf.len() {
                    return None;
                }
                buf[0] = b'.';
                buf[1..n].copy_from_slice(ext.as_bytes());
                let s = std::str::from_utf8(&buf[..n]).ok()?;
                extensions.contains(s)
            } else {
                let lower = ext.to_ascii_lowercase();
                let ext_with_dot = format!(".{lower}");
                extensions.contains(&ext_with_dot)
            };
            if !matched {
                return None;
            }

            // Check file size before materialising the PathBuf — saves an alloc
            // when oversized files are filtered out.
            if let Ok(metadata) = entry.metadata()
                && metadata.len() > MAX_FILE_BYTES
            {
                return None;
            }
            Some(path.to_path_buf())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_for_path() {
        assert_eq!(language_for_path(Path::new("main.rs")), Some("rust"));
        assert_eq!(language_for_path(Path::new("app.py")), Some("python"));
        assert_eq!(language_for_path(Path::new("readme.md")), Some("markdown"));
        assert_eq!(language_for_path(Path::new("Makefile")), None);
    }

    #[test]
    fn test_filter_extensions_code_only() {
        let exts = filter_extensions(None, false);
        assert!(exts.contains(".rs"));
        assert!(exts.contains(".py"));
        assert!(!exts.contains(".md"));
    }

    #[test]
    fn test_filter_extensions_with_text() {
        let exts = filter_extensions(None, true);
        assert!(exts.contains(".rs"));
        assert!(exts.contains(".md"));
        assert!(exts.contains(".json"));
    }
}
