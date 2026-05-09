//! Model loading wrapper around model2vec-rs.

use anyhow::Result;
pub use model2vec_rs::model::StaticModel;

/// Default model for code search (English-focused, code-specialised, ~16M params).
pub const DEFAULT_MODEL_NAME: &str = "minishlab/potion-code-16M";

/// Multilingual model — larger but covers Cyrillic, CJK, Greek, Arabic, …
/// Use this for codebases or queries with non-English natural language.
pub const MULTILINGUAL_MODEL_NAME: &str = "minishlab/potion-multilingual-128M";

/// Load a model from a HuggingFace model ID or local path.
pub fn load_model(model_path: Option<&str>) -> Result<StaticModel> {
    let path = model_path.unwrap_or(DEFAULT_MODEL_NAME);
    let model = StaticModel::from_pretrained(path, None, None, None)?;
    Ok(model)
}

/// Convenience: load the multilingual model.
pub fn load_multilingual_model() -> Result<StaticModel> {
    load_model(Some(MULTILINGUAL_MODEL_NAME))
}
