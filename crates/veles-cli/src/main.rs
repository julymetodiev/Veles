//! Veles CLI — Fast and Accurate Code Search for Agents.
//!
//! This file is intentionally thin: it sets up logging, parses the CLI, and
//! dispatches to per-subcommand handlers. The real logic lives in the
//! sibling modules:
//!
//! - `cli`      — clap-derived `Cli` / `Commands` definitions
//! - `handlers` — one `handle_*` function per subcommand
//! - `output`   — stdout sinks for rendered formatter output
//! - `util`     — index/model loaders and glob-filter helpers
//! - `format`   — pure formatter renderers (no I/O)

mod cli;
mod format;
mod handlers;
mod output;
mod tui;
mod util;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (to stderr so it doesn't interfere with MCP stdio).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Search {
            query,
            path,
            top_k,
            mode,
            format,
            lang,
            path_glob,
            exclude_glob,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        }) => handlers::handle_search(
            query,
            path,
            top_k,
            mode,
            format,
            lang,
            path_glob,
            exclude_glob,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        ),

        Some(Commands::FindRelated {
            file_path,
            line,
            path,
            top_k,
            format,
            lang,
            path_glob,
            exclude_glob,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        }) => handlers::handle_find_related(
            file_path,
            line,
            path,
            top_k,
            format,
            lang,
            path_glob,
            exclude_glob,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        ),

        Some(Commands::Index {
            path,
            include_text_files,
            multilingual,
            force,
        }) => handlers::handle_index(path, include_text_files, multilingual, force),

        Some(Commands::Update { path, multilingual }) => {
            handlers::handle_update(path, multilingual)
        }

        Some(Commands::Status { path }) => handlers::handle_status(path),

        Some(Commands::Clean { path }) => handlers::handle_clean(path),

        Some(Commands::Symbols { file, format }) => handlers::handle_symbols(file, format),

        Some(Commands::Defs {
            name,
            path,
            lang,
            kind,
            format,
            multilingual,
        }) => handlers::handle_defs(name, path, lang, kind, format, multilingual),

        Some(Commands::Refs {
            name,
            path,
            top_k,
            format,
            multilingual,
        }) => handlers::handle_refs(name, path, top_k, format, multilingual),

        Some(Commands::Tui {
            path,
            multilingual,
            include_text_files,
            no_cache,
        }) => handlers::handle_tui(path, multilingual, include_text_files, no_cache),

        Some(Commands::Completions { shell }) => handlers::handle_completions(shell),

        Some(Commands::Man { out_dir }) => handlers::handle_man(out_dir),

        Some(Commands::ServeGrpc { addr }) => handlers::handle_serve_grpc(addr).await,

        Some(Commands::ServeMcp { .. }) => handlers::handle_serve_mcp().await,

        None => handlers::handle_default().await,
    }
}
