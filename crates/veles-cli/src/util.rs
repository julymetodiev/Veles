//! Glue helpers shared between handlers — index loading, model loading,
//! glob filters, git-URL detection.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};

use veles_core::VelesIndex;
use veles_core::model;
use veles_core::persist;

/// Resolve a path/git-URL into a `VelesIndex`, preferring a `.veles/` cache
/// for local paths when `use_cache` is true.
pub fn open_index(
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

pub fn load_model(multilingual: bool) -> Result<model::StaticModel> {
    if multilingual {
        model::load_multilingual_model()
    } else {
        model::load_model(None)
    }
}

/// Build a list of file paths matching the include/exclude globs.
///
/// Returns `None` when no globs are supplied (caller should pass
/// `None` for `filter_paths` so the search is unrestricted).
pub fn resolve_path_filter(
    index: &VelesIndex,
    include: &[String],
    exclude: &[String],
) -> Result<Option<Vec<String>>> {
    if include.is_empty() && exclude.is_empty() {
        return Ok(None);
    }

    let include_set = build_globset(include).context("invalid --path glob")?;
    let exclude_set = build_globset(exclude).context("invalid --exclude glob")?;

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
pub fn is_git_url(path: &str) -> bool {
    path.starts_with("https://")
        || path.starts_with("http://")
        || path.starts_with("ssh://")
        || path.starts_with("git://")
        || path.starts_with("git+ssh://")
}

/// Parse a `--format` string into an `OutputFormat`, mapping the parser's
/// `String` error into an `anyhow` error.
pub fn parse_format(s: &str) -> Result<crate::format::OutputFormat> {
    s.parse::<crate::format::OutputFormat>()
        .map_err(|e| anyhow::anyhow!(e))
}
