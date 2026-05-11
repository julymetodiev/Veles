//! Shared fixture + model loading for Criterion benches.
//!
//! Each bench binary is a separate crate, so the helpers live here and
//! get pulled in with `#[path = "common/mod.rs"] mod common;`. Each
//! bench only uses a subset of the helpers — `#![allow(dead_code)]`
//! suppresses the spurious warnings from the unused ones.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use model2vec_rs::model::StaticModel;
use tempfile::TempDir;
use veles_core::VelesIndex;
use veles_core::model;

/// Load the default embedding model once per process. Bench binaries are
/// separate processes, so this fires once per `cargo bench` invocation
/// of each binary.
pub fn shared_model() -> &'static StaticModel {
    static MODEL: OnceLock<StaticModel> = OnceLock::new();
    MODEL.get_or_init(|| {
        model::load_model(None).expect("model load — first run downloads ~64MB from HuggingFace")
    })
}

/// Deterministic LCG so corpus generation has no external rand dep.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[(self.next() as usize) % xs.len()]
    }
}

/// Build a synthetic Rust-ish corpus in a tempdir.
///
/// `n_files` controls scale. Each file holds ~`lines_per_file` lines of
/// pseudo-code containing recognisable identifiers, definitions, and
/// natural-language doc comments so BM25 / semantic / hybrid all have
/// something to score against.
///
/// Returned `TempDir` must be kept alive for the duration of the
/// benchmark — drop deletes the directory.
pub struct Corpus {
    pub dir: TempDir,
    pub root: PathBuf,
}

pub fn make_corpus(n_files: usize, lines_per_file: usize, seed: u64) -> Corpus {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();

    let modules = [
        "search",
        "index",
        "parser",
        "ranker",
        "tokenizer",
        "model",
        "store",
        "client",
        "server",
        "config",
    ];
    let verbs = [
        "parse", "build", "compute", "load", "save", "encode", "decode", "rank", "score",
        "tokenize", "merge", "split", "resolve", "fetch", "render",
    ];
    let nouns = [
        "Index", "Chunk", "Token", "Query", "Result", "Cache", "Config", "Manifest", "Vector",
        "Matrix", "Scorer", "Filter", "Walker", "Stream", "Buffer",
    ];
    let kw_struct = ["struct", "enum"];

    let mut lcg = Lcg::new(seed);
    let mut files_written = 0usize;

    while files_written < n_files {
        // Group files into module-like subdirectories.
        let module = lcg.pick(&modules);
        let sub = format!("crate_{}/src/{}", files_written / 20, module);
        let sub_path = root.join(&sub);
        std::fs::create_dir_all(&sub_path).expect("create module dir");

        let file_basename = format!("{}_{}.rs", lcg.pick(&verbs), files_written);
        let file_path = sub_path.join(&file_basename);

        let mut body = String::with_capacity(lines_per_file * 60);
        // Doc comment header — natural-language signal for semantic mode.
        body.push_str(&format!(
            "//! Module `{module}` — {verb} {noun}s for the {module_alt} layer.\n\
             //!\n\
             //! This file is part of the synthetic benchmark corpus.\n\n",
            module = module,
            module_alt = lcg.pick(&modules),
            verb = lcg.pick(&verbs),
            noun = lcg.pick(&nouns),
        ));

        let mut lines_written = body.lines().count();
        while lines_written < lines_per_file {
            // Emit a definition block (function or struct) every ~20 lines.
            let kind = lcg.next() % 4;
            match kind {
                0 => {
                    let name = format!("{}_{}", lcg.pick(&verbs), lcg.next() % 1000);
                    let arg = format!("{}_{}", lcg.pick(&nouns), lcg.next() % 100).to_lowercase();
                    let verb = lcg.pick(&verbs);
                    let noun = lcg.pick(&nouns);
                    body.push_str(&format!("/// {verb} the given {noun}.\n"));
                    body.push_str(&format!("pub fn {name}({arg}: &str) -> usize {{\n"));
                    body.push_str("    let mut n = 0usize;\n");
                    body.push_str(&format!("    for _byte in {arg}.bytes() {{\n"));
                    body.push_str("        n = n.wrapping_add(1);\n");
                    body.push_str("    }\n");
                    body.push_str("    n\n");
                    body.push_str("}\n\n");
                    lines_written += 8;
                }
                1 => {
                    let kw = lcg.pick(&kw_struct);
                    let name = format!("{}{}", lcg.pick(&nouns), lcg.next() % 1000);
                    let noun_lower = lcg.pick(&nouns).to_lowercase();
                    body.push_str(&format!("/// A {noun_lower} record.\n"));
                    body.push_str(&format!("pub {kw} {name} {{\n"));
                    body.push_str("    pub id: u32,\n");
                    body.push_str("    pub name: String,\n");
                    body.push_str("    pub kind: u8,\n");
                    body.push_str("}\n\n");
                    lines_written += 7;
                }
                2 => {
                    let name = format!(
                        "MAX_{}_{}",
                        lcg.pick(&nouns).to_uppercase(),
                        lcg.next() % 100
                    );
                    body.push_str(&format!(
                        "pub const {name}: usize = {};\n\n",
                        lcg.next() % 10_000
                    ));
                    lines_written += 2;
                }
                _ => {
                    // A comment / control-flow filler line — adds BM25 token noise.
                    body.push_str(&format!(
                        "// note: {verb} the {noun} via the {module_alt} pipeline.\n",
                        verb = lcg.pick(&verbs),
                        noun = lcg.pick(&nouns),
                        module_alt = lcg.pick(&modules),
                    ));
                    lines_written += 1;
                }
            }
        }

        std::fs::write(&file_path, body).expect("write fixture file");
        files_written += 1;
    }

    Corpus { dir, root }
}

/// Build a fully-loaded `VelesIndex` over a synthetic corpus. Expensive
/// (runs the embedding model), so callers should cache the result for
/// the duration of a benchmark.
pub fn build_index(corpus: &Corpus) -> VelesIndex {
    let model = shared_model().clone();
    VelesIndex::from_path(&corpus.root, Some(model), None, false)
        .expect("build VelesIndex from synthetic corpus")
}

/// Representative query set covering symbol, snake_case, camelCase, and
/// natural-language patterns so we exercise all branches of the
/// query-type detector.
pub const QUERIES: &[&str] = &[
    "parse the query string",
    "load model from disk",
    "Index",
    "MAX_CHUNK_42",
    "buildIndex",
    "compute score for ranker",
    "tokenize identifier",
    "resolve config path",
    "rank top results",
    "encode chunk content",
];
