//! One handler function per subcommand.
//!
//! Each `handle_*` takes the destructured arguments from its `Commands`
//! variant and returns `anyhow::Result<()>`. Handlers may print to stdout
//! and may call `std::process::exit` for hard failures whose error
//! semantics differ from `anyhow::Error` (e.g. "no chunk at file:line").

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::CommandFactory;
use clap_complete::{Shell, generate};

use veles_core::VelesIndex;
use veles_core::model;
use veles_core::persist;
use veles_core::symbols::Symbol;
use veles_core::types::{SearchMode, SearchResult};

use crate::cli::Cli;
use crate::format::{self, OutputFormat};
use crate::output::{emit_results, emit_symbols};
use crate::util::{load_model, open_index, parse_format, resolve_path_filter};

#[allow(clippy::too_many_arguments)]
pub fn handle_search(
    query: String,
    path: String,
    top_k: usize,
    mode: String,
    format_str: String,
    lang: Vec<String>,
    path_glob: Vec<String>,
    exclude_glob: Vec<String>,
    min_score: Option<f64>,
    include_text_files: bool,
    multilingual: bool,
    no_cache: bool,
) -> Result<()> {
    let format = parse_format(&format_str)?;
    let index = open_index(&path, multilingual, include_text_files, !no_cache)?;
    let search_mode = mode.parse::<SearchMode>().unwrap_or(SearchMode::Hybrid);

    let glob_paths = resolve_path_filter(&index, &path_glob, &exclude_glob)?;
    let lang_slice: Option<&[String]> = if lang.is_empty() { None } else { Some(&lang) };
    let path_slice: Option<&[String]> = glob_paths.as_deref();

    let mut results = index.search(&query, top_k, search_mode, None, lang_slice, path_slice);

    if let Some(threshold) = min_score {
        results.retain(|r| r.score >= threshold);
    }

    emit_results(
        format,
        &format!("Search results for: {query:?} (mode={mode})"),
        "results",
        &results,
        Some(index.symbols()),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn handle_find_related(
    file_path: String,
    line: usize,
    path: String,
    top_k: usize,
    format_str: String,
    lang: Vec<String>,
    path_glob: Vec<String>,
    exclude_glob: Vec<String>,
    min_score: Option<f64>,
    include_text_files: bool,
    multilingual: bool,
    no_cache: bool,
) -> Result<()> {
    let format = parse_format(&format_str)?;
    let index = open_index(&path, multilingual, include_text_files, !no_cache)?;

    let chunk = match index.resolve_chunk(&file_path, line) {
        Some(c) => c.clone(),
        None => {
            eprintln!("No chunk found at {file_path}:{line}.");
            std::process::exit(1);
        }
    };

    let glob_paths = resolve_path_filter(&index, &path_glob, &exclude_glob)?;
    let lang_slice: Option<&[String]> = if lang.is_empty() { None } else { Some(&lang) };
    let path_slice: Option<&[String]> = glob_paths.as_deref();

    let mut results = index.find_related(&chunk, top_k, lang_slice, path_slice);
    if let Some(threshold) = min_score {
        results.retain(|r| r.score >= threshold);
    }

    emit_results(
        format,
        &format!("Chunks related to {file_path}:{line}"),
        "related chunks",
        &results,
        Some(index.symbols()),
    );
    Ok(())
}

pub fn handle_index(
    path: String,
    include_text_files: bool,
    multilingual: bool,
    force: bool,
) -> Result<()> {
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
    Ok(())
}

pub fn handle_update(path: String, multilingual: bool) -> Result<()> {
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
        println!(
            "Index is up to date ({} chunks, no changes).",
            report.total_chunks
        );
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
    Ok(())
}

pub fn handle_status(path: String) -> Result<()> {
    let path_buf = PathBuf::from(&path);
    if !persist::index_exists(&path_buf) {
        println!("No index found at {}/.veles", path_buf.display());
        return Ok(());
    }
    let manifest = persist::load_manifest(&path_buf)?;

    // Compute drift without loading chunks/embeddings.
    let exts = veles_core::walker::filter_extensions(None, manifest.include_text_files);
    let mut added = 0usize;
    let mut modified = 0usize;
    let on_disk: std::collections::HashMap<String, (u64, i64)> =
        veles_core::walker::walk_files(&path_buf, &exts)
            .filter_map(|abs| {
                let rel = abs
                    .strip_prefix(&path_buf)
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
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
    let on_disk_files = on_disk.len();
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
    Ok(())
}

pub fn handle_clean(path: String) -> Result<()> {
    let path_buf = PathBuf::from(&path);
    if persist::clean(&path_buf)? {
        println!("Removed {}/.veles", path_buf.display());
    } else {
        println!("No index to remove at {}/.veles", path_buf.display());
    }
    Ok(())
}

pub fn handle_symbols(file: String, format_str: String) -> Result<()> {
    let format = parse_format(&format_str)?;
    let path = PathBuf::from(&file);
    if !path.is_file() {
        bail!("File not found: {file}");
    }
    let language = veles_core::walker::language_for_path(&path)
        .ok_or_else(|| anyhow::anyhow!("Unsupported file type: {file}"))?;
    if !veles_core::symbols::supports(language) {
        eprintln!(
            "No tree-sitter parser for {language} files yet. Supported: rust, python, javascript, typescript, go."
        );
        std::process::exit(1);
    }
    let content = std::fs::read_to_string(&path).with_context(|| format!("read {file}"))?;
    let syms = veles_core::symbols::extract_symbols(&content, &file, language);
    let refs: Vec<&Symbol> = syms.iter().collect();
    emit_symbols(format, &format!("Symbols in {file}"), "symbols", &refs);
    Ok(())
}

pub fn handle_defs(
    name: String,
    path: String,
    lang: Vec<String>,
    kind: Option<String>,
    format_str: String,
    multilingual: bool,
) -> Result<()> {
    let format = parse_format(&format_str)?;
    let index = open_index(&path, multilingual, false, true)?;

    let kind_filter = kind.as_ref().map(|k| k.to_ascii_lowercase());

    let mut hits: Vec<&Symbol> = index
        .symbols()
        .iter()
        .filter(|s| s.name == name)
        .filter(|s| lang.is_empty() || lang.iter().any(|l| l == &s.language))
        .filter(|s| match &kind_filter {
            Some(k) => s.kind.as_str() == k,
            None => true,
        })
        .collect();
    hits.sort_by(|a, b| {
        a.file_path
            .cmp(&b.file_path)
            .then(a.start_line.cmp(&b.start_line))
    });

    emit_symbols(
        format,
        &format!("Definitions of {name:?}"),
        "definitions",
        &hits,
    );
    Ok(())
}

pub fn handle_refs(
    name: String,
    path: String,
    top_k: usize,
    format_str: String,
    multilingual: bool,
) -> Result<()> {
    let format = parse_format(&format_str)?;
    let index = open_index(&path, multilingual, false, true)?;

    let defs: Vec<&Symbol> = index.symbols().iter().filter(|s| s.name == name).collect();

    // Pull a few extra BM25 hits so dropping chunks that overlap a
    // definition site still leaves the caller with roughly the requested
    // count.
    let bm25_overshoot = top_k + (top_k / 2).max(1);
    let bm25_hits: Vec<SearchResult> = index
        .search(&name, bm25_overshoot, SearchMode::Bm25, None, None, None)
        .into_iter()
        .filter(|hit| {
            !defs.iter().any(|d| {
                d.file_path == hit.chunk.file_path
                    && d.start_line >= hit.chunk.start_line
                    && d.start_line <= hit.chunk.end_line
            })
        })
        .take(top_k)
        .collect();

    match format {
        OutputFormat::Pretty => {
            println!("References to {name:?}\n");
            if defs.is_empty() {
                println!("(no definitions found)");
            } else {
                println!("## Definitions");
                for s in &defs {
                    println!(
                        "  {:9}  {:30}  {}:{}",
                        s.kind.as_str(),
                        s.name,
                        s.file_path,
                        s.start_line
                    );
                }
            }
            println!();
            if bm25_hits.is_empty() {
                println!("(no BM25 hits)");
            } else {
                println!("## Other matches (BM25)");
                let rendered = format::render(
                    OutputFormat::Pretty,
                    &format!("{} BM25 result(s)", bm25_hits.len()),
                    &bm25_hits,
                    Some(index.symbols()),
                );
                println!("{rendered}");
            }
        }
        _ => {
            // Line-oriented: defs first, then hits.
            if !defs.is_empty() {
                let rendered =
                    format::render_symbols(format, &format!("Definitions of {name:?}"), &defs);
                write_or_fall_through(&rendered);
            }
            if !bm25_hits.is_empty() {
                let rendered = format::render(
                    format,
                    &format!("BM25 hits for {name:?}"),
                    &bm25_hits,
                    Some(index.symbols()),
                );
                write_or_fall_through(&rendered);
            }
            if defs.is_empty() && bm25_hits.is_empty() {
                eprintln!("No matches for {name:?}.");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn write_or_fall_through(rendered: &str) {
    if rendered.ends_with('\n') {
        print!("{rendered}");
    } else if !rendered.is_empty() {
        println!("{rendered}");
    }
}

pub fn handle_tui(
    path: String,
    multilingual: bool,
    include_text_files: bool,
    no_cache: bool,
) -> Result<()> {
    crate::tui::run(path, multilingual, include_text_files, !no_cache)
}

pub async fn handle_serve_grpc(addr: String) -> Result<()> {
    let mdl = model::load_model(None)?;
    veles_grpc::serve(&addr, mdl).await?;
    Ok(())
}

pub async fn handle_serve_mcp() -> Result<()> {
    let mdl = model::load_model(None)?;
    let server = veles_mcp::McpServer::new(mdl);
    server.run().await?;
    Ok(())
}

/// Default behaviour when no subcommand is given.
///
/// On an interactive terminal, print `--help` so a user who just ran
/// `cargo install veles-cli && veles` sees what's on offer instead of a
/// silent process that's actually waiting on JSON-RPC. When stdin is
/// piped (e.g. an MCP client like Claude Desktop launching the binary),
/// fall through to the MCP server — this keeps existing integrations
/// working unchanged.
pub async fn handle_default() -> Result<()> {
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        let mut cmd = Cli::command();
        cmd.print_help()?;
        println!();
        Ok(())
    } else {
        handle_serve_mcp().await
    }
}

pub fn handle_completions(shell: Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
    Ok(())
}

pub fn handle_man(out_dir: Option<PathBuf>) -> Result<()> {
    let cmd = Cli::command();
    match out_dir {
        Some(dir) => {
            std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
            let written = write_man_pages(cmd, &dir, None)?;
            eprintln!("Wrote {written} man page(s) to {}", dir.display());
        }
        None => {
            let man = clap_mangen::Man::new(cmd);
            man.render(&mut std::io::stdout())
                .context("render man page")?;
        }
    }
    Ok(())
}

/// Recursively write `veles.1`, `veles-<sub>.1`, `veles-<sub>-<subsub>.1`
/// into `dir`. Returns the number of pages written.
fn write_man_pages(
    cmd: clap::Command,
    dir: &std::path::Path,
    parent_name: Option<&str>,
) -> Result<usize> {
    use std::fs::File;

    let leaf_name = cmd.get_name();
    let full_name = match parent_name {
        Some(parent) => format!("{parent}-{leaf_name}"),
        None => leaf_name.to_string(),
    };

    let mut written = 1;
    let path = dir.join(format!("{full_name}.1"));
    let mut file = File::create(&path).with_context(|| format!("create {}", path.display()))?;

    // Render with the full name in `.TH` so the page header reads
    // `veles-search(1)` instead of just `search(1)`.
    // clap's `Command::name` wants `impl Into<Str>` where `Str` is roughly
    // `Cow<'static, str>`. The cheapest way to materialise a runtime
    // `&'static str` is `Box::leak` — the cost is one allocation per page
    // (a handful of pages per `--out-dir` invocation, never repeated).
    let full_name_static: &'static str = Box::leak(full_name.clone().into_boxed_str());
    let renamed = cmd
        .clone()
        .name(full_name_static)
        .bin_name(full_name_static);
    clap_mangen::Man::new(renamed)
        .render(&mut file)
        .with_context(|| format!("render {}", path.display()))?;

    for sub in cmd.get_subcommands() {
        // Skip the auto-generated `help` subcommand — it has no useful
        // page of its own and clutters MANPATH.
        if sub.get_name() == "help" {
            continue;
        }
        written += write_man_pages(sub.clone(), dir, Some(&full_name))?;
    }
    Ok(written)
}
