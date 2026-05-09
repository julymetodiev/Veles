//! Veles CLI — Fast and Accurate Code Search for Agents.
//!
//! Subcommands:
//! - `search` — Search a codebase (auto-loads `.veles/` cache if present)
//! - `find-related` — Find code similar to a location
//! - `index` — Build & persist the index to `.veles/`
//! - `update` — Incrementally re-index changed files
//! - `status` — Show index manifest stats and drift
//! - `clean` — Remove the on-disk index
//! - `serve-grpc` — Start a gRPC server
//! - `serve-mcp` — Start an MCP server (default if no subcommand)

mod format;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use globset::{Glob, GlobSet, GlobSetBuilder};

use veles_core::VelesIndex;
use veles_core::model;
use veles_core::persist;
use veles_core::types::{SearchMode, SearchResult};

use crate::format::OutputFormat;

#[derive(Parser)]
#[command(name = "veles")]
#[command(about = "Fast and Accurate Code Search for Agents")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
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
}

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
            format: format_str,
            lang,
            path_glob,
            exclude_glob,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        }) => {
            let format: OutputFormat = format_str
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let index = open_index(&path, multilingual, include_text_files, !no_cache)?;
            let search_mode = mode.parse::<SearchMode>().unwrap_or(SearchMode::Hybrid);

            let glob_paths =
                resolve_path_filter(&index, &path_glob, &exclude_glob)?;
            let lang_slice: Option<&[String]> = if lang.is_empty() { None } else { Some(&lang) };
            let path_slice: Option<&[String]> = glob_paths.as_deref();

            let mut results =
                index.search(&query, top_k, search_mode, None, lang_slice, path_slice);

            if let Some(threshold) = min_score {
                results.retain(|r| r.score >= threshold);
            }

            emit_results(
                format,
                &format!("Search results for: {query:?} (mode={mode})"),
                "results",
                &results,
            );
        }

        Some(Commands::FindRelated {
            file_path,
            line,
            path,
            top_k,
            format: format_str,
            min_score,
            include_text_files,
            multilingual,
            no_cache,
        }) => {
            let format: OutputFormat = format_str
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            let index = open_index(&path, multilingual, include_text_files, !no_cache)?;

            let chunk = match index.resolve_chunk(&file_path, line) {
                Some(c) => c.clone(),
                None => {
                    eprintln!("No chunk found at {file_path}:{line}.");
                    std::process::exit(1);
                }
            };

            let mut results = index.find_related(&chunk, top_k);
            if let Some(threshold) = min_score {
                results.retain(|r| r.score >= threshold);
            }

            emit_results(
                format,
                &format!("Chunks related to {file_path}:{line}"),
                "related chunks",
                &results,
            );
        }

        Some(Commands::Index {
            path,
            include_text_files,
            multilingual,
            force,
        }) => {
            let path_buf = PathBuf::from(&path);
            if !path_buf.is_dir() {
                bail!("Path is not a directory: {path}");
            }

            if persist::index_exists(&path_buf) && !force {
                eprintln!(
                    "Index already exists at {}/.veles. Use `veles update` to refresh, or `--force` to rebuild.",
                    path_buf.display()
                );
                std::process::exit(1);
            }

            let mdl = load_model(multilingual)?;
            eprintln!("Indexing {} ...", path_buf.display());
            let started = std::time::Instant::now();
            let index = VelesIndex::from_path(&path_buf, Some(mdl), None, include_text_files)?;
            let build_secs = started.elapsed().as_secs_f64();

            index.save(&path_buf)?;
            let stats = index.stats();
            println!(
                "Indexed {} files / {} chunks in {build_secs:.2}s — saved to {}/.veles",
                stats.indexed_files,
                stats.total_chunks,
                path_buf.display()
            );
        }

        Some(Commands::Update { path, multilingual }) => {
            let path_buf = PathBuf::from(&path);
            if !path_buf.is_dir() {
                bail!("Path is not a directory: {path}");
            }
            if !persist::index_exists(&path_buf) {
                bail!(
                    "No index at {}/.veles. Run `veles index {path}` first.",
                    path_buf.display()
                );
            }

            let mdl = load_model(multilingual)?;
            let mut index = VelesIndex::load(&path_buf, mdl)?;

            let started = std::time::Instant::now();
            let report = index.update_from_path(&path_buf)?;
            let secs = started.elapsed().as_secs_f64();

            if report.is_noop() {
                println!("Index is up to date ({} chunks, no changes).", report.total_chunks);
                return Ok(());
            }

            index.save(&path_buf)?;
            println!(
                "Updated in {secs:.2}s — +{} added, ~{} modified, -{} removed (kept {} chunks, embedded {} new, total {})",
                report.added_files,
                report.modified_files,
                report.removed_files,
                report.kept_chunks,
                report.new_chunks,
                report.total_chunks,
            );
        }

        Some(Commands::Status { path }) => {
            let path_buf = PathBuf::from(&path);
            if !persist::index_exists(&path_buf) {
                println!("No index found at {}/.veles", path_buf.display());
                return Ok(());
            }
            let manifest = persist::load_manifest(&path_buf)?;

            // Compute drift without loading chunks/embeddings.
            let exts = veles_core::walker::filter_extensions(None, manifest.include_text_files);
            let mut on_disk_files = 0usize;
            let mut added = 0usize;
            let mut modified = 0usize;
            let on_disk: std::collections::HashMap<String, (u64, i64)> =
                veles_core::walker::walk_files(&path_buf, &exts)
                    .filter_map(|abs| {
                        let rel = abs.strip_prefix(&path_buf).ok()?.to_string_lossy().into_owned();
                        let meta = std::fs::metadata(&abs).ok()?;
                        let mtime = meta
                            .modified()
                            .ok()?
                            .duration_since(std::time::UNIX_EPOCH)
                            .ok()
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);
                        Some((rel, (meta.len(), mtime)))
                    })
                    .collect();
            on_disk_files = on_disk.len().max(on_disk_files);
            for (rel, (size, mtime)) in &on_disk {
                match manifest.files.get(rel) {
                    Some(prev) if prev.size == *size && prev.mtime_secs == *mtime => {}
                    Some(_) => modified += 1,
                    None => added += 1,
                }
            }
            let removed = manifest
                .files
                .keys()
                .filter(|k| !on_disk.contains_key(*k))
                .count();

            println!("Index at {}/.veles", path_buf.display());
            println!("  veles version    : {}", manifest.veles_version);
            println!("  format version   : {}", manifest.format_version);
            println!("  model            : {}", manifest.model_name);
            println!("  embedding dim    : {}", manifest.embedding_dim);
            println!("  text files       : {}", manifest.include_text_files);
            println!("  indexed at       : {} (unix)", manifest.indexed_at);
            println!("  files in manifest: {}", manifest.files.len());
            println!("  total chunks     : {}", manifest.total_chunks);
            println!();
            println!("On-disk diff:");
            println!("  files seen now   : {on_disk_files}");
            println!("  added            : {added}");
            println!("  modified         : {modified}");
            println!("  removed          : {removed}");
            if added + modified + removed == 0 {
                println!("\nUp to date.");
            } else {
                println!("\nRun `veles update {path}` to refresh.");
            }
        }

        Some(Commands::Clean { path }) => {
            let path_buf = PathBuf::from(&path);
            if persist::clean(&path_buf)? {
                println!("Removed {}/.veles", path_buf.display());
            } else {
                println!("No index to remove at {}/.veles", path_buf.display());
            }
        }

        Some(Commands::ServeGrpc { addr }) => {
            let mdl = model::load_model(None)?;
            veles_grpc::serve(&addr, mdl).await?;
        }

        Some(Commands::ServeMcp {
            path: _,
            include_text_files: _,
        }) => {
            let mdl = model::load_model(None)?;
            let server = veles_mcp::McpServer::new(mdl);
            server.run().await?;
        }

        None => {
            // Default: start MCP server.
            let mdl = model::load_model(None)?;
            let server = veles_mcp::McpServer::new(mdl);
            server.run().await?;
        }
    }

    Ok(())
}

/// Resolve a path/git-URL into a `VelesIndex`, preferring a `.veles/` cache
/// for local paths when `use_cache` is true.
fn open_index(
    path: &str,
    multilingual: bool,
    include_text_files: bool,
    use_cache: bool,
) -> Result<VelesIndex> {
    let model = load_model(multilingual)?;

    if is_git_url(path) {
        return VelesIndex::from_git(path, None, Some(model), include_text_files);
    }

    let path_buf = PathBuf::from(path);
    if use_cache && persist::index_exists(&path_buf) {
        match VelesIndex::load(&path_buf, model.clone()) {
            Ok(idx) => {
                tracing::info!("Loaded persisted index from {}/.veles", path_buf.display());
                return Ok(idx);
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load persisted index ({e}). Falling back to in-memory build."
                );
            }
        }
    }

    VelesIndex::from_path(Path::new(path), Some(model), None, include_text_files)
}

fn load_model(multilingual: bool) -> Result<model::StaticModel> {
    if multilingual {
        model::load_multilingual_model()
    } else {
        model::load_model(None)
    }
}

/// Render results in the chosen format and write to stdout.
fn emit_results(
    format: OutputFormat,
    header: &str,
    what: &str,
    results: &[SearchResult],
) {
    if results.is_empty() {
        let msg = format::empty_message(format, what);
        if !msg.is_empty() {
            println!("{msg}");
        }
        return;
    }
    let rendered = format::render(format, header, results);
    // `render_pretty` produces a trailing newline naturally via its joined
    // lines; line-oriented formats also self-terminate. Use `print!` so we
    // don't double up.
    if rendered.ends_with('\n') {
        print!("{rendered}");
    } else {
        println!("{rendered}");
    }
}

/// Build a list of file paths matching the include/exclude globs.
///
/// Returns `None` when no globs are supplied (caller should pass
/// `None` for `filter_paths` so the search is unrestricted).
fn resolve_path_filter(
    index: &VelesIndex,
    include: &[String],
    exclude: &[String],
) -> Result<Option<Vec<String>>> {
    if include.is_empty() && exclude.is_empty() {
        return Ok(None);
    }

    let include_set = build_globset(include).context("invalid --path glob")?;
    let exclude_set = build_globset(exclude).context("invalid --exclude glob")?;

    // Collect unique file paths from the index, then filter.
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut matched: Vec<String> = Vec::new();
    for chunk in index.chunks() {
        if !seen.insert(chunk.file_path.as_str()) {
            continue;
        }
        let p = chunk.file_path.as_str();
        let included = match &include_set {
            Some(s) => s.is_match(p),
            None => true,
        };
        let excluded = match &exclude_set {
            Some(s) => s.is_match(p),
            None => false,
        };
        if included && !excluded {
            matched.push(p.to_string());
        }
    }

    if matched.is_empty() {
        bail!("No indexed files matched the given --path / --exclude globs");
    }
    Ok(Some(matched))
}

fn build_globset(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        let glob = Glob::new(p).with_context(|| format!("bad glob pattern {p:?}"))?;
        builder.add(glob);
    }
    Ok(Some(builder.build()?))
}

/// Check if a path looks like a git URL.
fn is_git_url(path: &str) -> bool {
    path.starts_with("https://")
        || path.starts_with("http://")
        || path.starts_with("ssh://")
        || path.starts_with("git://")
        || path.starts_with("git+ssh://")
}
