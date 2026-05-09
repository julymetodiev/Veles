//! Path-glob filtering for search queries.
//!
//! Used by the CLI, MCP, and gRPC servers to translate user-supplied
//! `--path` / `--exclude` glob patterns into the concrete `filter_paths`
//! slice that `VelesIndex::search` understands.

use anyhow::{Context, Result, bail};
use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::veles_index::VelesIndex;

/// Resolve include/exclude glob patterns into a list of indexed file paths.
///
/// Returns `Ok(None)` when both lists are empty — callers should pass
/// `None` to `VelesIndex::search` so the search runs unrestricted. Returns
/// `Err` when patterns are syntactically invalid or when no indexed file
/// matches (so the caller can surface a clear "nothing matched" error
/// rather than silently returning zero results).
pub fn resolve_path_filter(
    index: &VelesIndex,
    include: &[String],
    exclude: &[String],
) -> Result<Option<Vec<String>>> {
    if include.is_empty() && exclude.is_empty() {
        return Ok(None);
    }

    let include_set = build_globset(include).context("invalid include glob")?;
    let exclude_set = build_globset(exclude).context("invalid exclude glob")?;

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
        bail!("No indexed files matched the given path / exclude globs");
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
