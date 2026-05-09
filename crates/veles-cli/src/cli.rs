//! Command-line interface definition (clap derives).
//!
//! Kept separate from `main.rs` so the actual dispatch loop stays small and
//! handlers can import a single `Commands` enum without pulling in the
//! handler logic.

use clap::{Parser, Subcommand};
use clap_complete::Shell;

#[derive(Parser)]
#[command(name = "veles")]
#[command(about = "Fast and Accurate Code Search for Agents")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Search a codebase with a natural-language or code query.
    Search {
        /// Natural language or code query.
        query: String,
        /// Local path or git URL (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Number of results.
        #[arg(short, long, default_value = "5")]
        top_k: usize,
        /// Search mode.
        #[arg(short, long, default_value = "hybrid")]
        mode: String,
        /// Output format: pretty (default), compact, ripgrep, paths, json, jsonl.
        #[arg(short, long, default_value = "pretty")]
        format: String,
        /// Restrict to specific languages (comma-separated, e.g. `rust,python`).
        #[arg(short = 'l', long, value_delimiter = ',')]
        lang: Vec<String>,
        /// Glob pattern of paths to include (repeatable, e.g. `--path 'src/**/*.rs'`).
        #[arg(short = 'g', long = "path", value_name = "GLOB")]
        path_glob: Vec<String>,
        /// Glob pattern of paths to exclude (repeatable, e.g. `--exclude 'tests/**'`).
        #[arg(short = 'x', long = "exclude", value_name = "GLOB")]
        exclude_glob: Vec<String>,
        /// Drop results scoring below this threshold.
        #[arg(long)]
        min_score: Option<f64>,
        /// Also index non-code text files.
        #[arg(long)]
        include_text_files: bool,
        /// Use the multilingual embedding model (potion-multilingual-128M)
        /// instead of the default English/code-focused model. Recommended for
        /// codebases or queries containing Cyrillic, CJK, Greek, Arabic, etc.
        #[arg(long)]
        multilingual: bool,
        /// Force a fresh in-memory build, ignoring any `.veles/` cache.
        #[arg(long)]
        no_cache: bool,
    },

    /// Find code similar to a specific location.
    FindRelated {
        /// File path as shown in search results.
        file_path: String,
        /// Line number (1-indexed).
        line: usize,
        /// Local path or git URL (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Number of similar chunks to return.
        #[arg(short, long, default_value = "5")]
        top_k: usize,
        /// Output format: pretty (default), compact, ripgrep, paths, json, jsonl.
        #[arg(short, long, default_value = "pretty")]
        format: String,
        /// Drop results scoring below this threshold.
        #[arg(long)]
        min_score: Option<f64>,
        /// Also index non-code text files.
        #[arg(long)]
        include_text_files: bool,
        /// Use the multilingual embedding model.
        #[arg(long)]
        multilingual: bool,
        /// Force a fresh in-memory build, ignoring any `.veles/` cache.
        #[arg(long)]
        no_cache: bool,
    },

    /// Build the index and persist it to `<path>/.veles/`.
    Index {
        /// Local path to index (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Also index non-code text files.
        #[arg(long)]
        include_text_files: bool,
        /// Use the multilingual embedding model.
        #[arg(long)]
        multilingual: bool,
        /// Rebuild from scratch even if a `.veles/` cache already exists.
        #[arg(long)]
        force: bool,
    },

    /// Incrementally update an existing index for files that changed on disk.
    Update {
        /// Local path of the indexed repo (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Use the multilingual embedding model (must match how it was built).
        #[arg(long)]
        multilingual: bool,
    },

    /// Show stats about the persisted index at `<path>/.veles/`.
    Status {
        /// Local path of the indexed repo (default: current directory).
        #[arg(default_value = ".")]
        path: String,
    },

    /// Remove the persisted index at `<path>/.veles/`.
    Clean {
        /// Local path of the indexed repo (default: current directory).
        #[arg(default_value = ".")]
        path: String,
    },

    /// Start a gRPC server.
    ServeGrpc {
        /// Address to bind to.
        #[arg(short, long, default_value = "[::1]:50051")]
        addr: String,
    },

    /// Start an MCP server over stdio.
    ServeMcp {
        /// Optional local path to pre-index at startup.
        path: Option<String>,
        /// Also index non-code text files.
        #[arg(long)]
        include_text_files: bool,
    },

    /// List definitions in a single file (functions, structs, classes, ...).
    ///
    /// Tree-sitter parses the file directly — no index required.
    Symbols {
        /// File to outline.
        file: String,
        /// Output format: pretty (default), compact, paths, json, jsonl.
        #[arg(short, long, default_value = "pretty")]
        format: String,
    },

    /// Find every definition with the given name across the indexed repo.
    Defs {
        /// Symbol name to look up (exact match).
        name: String,
        /// Local path of the indexed repo (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Restrict to specific languages (comma-separated).
        #[arg(short = 'l', long, value_delimiter = ',')]
        lang: Vec<String>,
        /// Restrict to a kind (function, struct, class, enum, trait, interface, type, const, static, var, module, method, macro).
        #[arg(short = 'k', long)]
        kind: Option<String>,
        /// Output format: pretty (default), compact, paths, json, jsonl.
        #[arg(short, long, default_value = "pretty")]
        format: String,
        /// Use the multilingual embedding model (must match how the index was built).
        #[arg(long)]
        multilingual: bool,
    },

    /// Find references to a symbol — its definitions plus BM25 hits in chunks.
    Refs {
        /// Symbol name to search for.
        name: String,
        /// Local path of the indexed repo (default: current directory).
        #[arg(default_value = ".")]
        path: String,
        /// Number of BM25 hits to include alongside definitions.
        #[arg(short, long, default_value = "20")]
        top_k: usize,
        /// Output format: pretty (default), compact, ripgrep, paths, json, jsonl.
        #[arg(short, long, default_value = "pretty")]
        format: String,
        /// Use the multilingual embedding model.
        #[arg(long)]
        multilingual: bool,
    },

    /// Print a shell completion script to stdout.
    ///
    /// Examples:
    ///   veles completions zsh > ~/.zfunc/_veles
    ///   veles completions bash > /etc/bash_completion.d/veles
    ///   veles completions fish > ~/.config/fish/completions/veles.fish
    Completions {
        /// Target shell.
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Print a roff(7) man page to stdout (suitable for `/usr/local/share/man/man1/veles.1`).
    ///
    /// Examples:
    ///   veles man > veles.1
    ///   veles man | gzip > /usr/local/share/man/man1/veles.1.gz
    Man,
}
