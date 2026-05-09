//! Glue helpers shared between handlers — index loading, model loading,
//! glob filters, git-URL detection.

use std::path::{Path, PathBuf};

use anyhow::Result;

use veles_core::VelesIndex;
use veles_core::filter;
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

/// Resolve `--path` / `--exclude` globs into the file list `VelesIndex::search` expects.
///
/// Thin wrapper over [`veles_core::filter::resolve_path_filter`] kept here
/// for backwards compatibility with external callers of `veles-cli::util`.
pub fn resolve_path_filter(
    index: &VelesIndex,
    include: &[String],
    exclude: &[String],
) -> Result<Option<Vec<String>>> {
    filter::resolve_path_filter(index, include, exclude)
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
