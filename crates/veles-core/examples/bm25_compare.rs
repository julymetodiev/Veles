//! Head-to-head: new inverted-index BM25 vs the previous HashMap-per-doc impl.
//!
//! Both implementations share the same corpus and query set; we report
//! build time and per-query latency for each.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use veles_core::index::sparse::Bm25Index as NewBm25;
use veles_core::tokenizer::tokenize;

/// Reference implementation matching the previous `Bm25Index`:
/// `tf` is a `HashMap<String, f64>` per document, scoring iterates every doc.
struct LegacyBm25 {
    num_docs: usize,
    avg_dl: f64,
    idf: HashMap<String, f64>,
    tf: Vec<HashMap<String, f64>>,
    doc_lengths: Vec<f64>,
}

impl LegacyBm25 {
    fn new(tokenized_docs: &[Vec<String>]) -> Self {
        let num_docs = tokenized_docs.len();
        if num_docs == 0 {
            return Self {
                num_docs: 0,
                avg_dl: 0.0,
                idf: HashMap::new(),
                tf: Vec::new(),
                doc_lengths: Vec::new(),
            };
        }
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut tf: Vec<HashMap<String, f64>> = Vec::with_capacity(num_docs);
        let mut doc_lengths: Vec<f64> = Vec::with_capacity(num_docs);
        for doc_tokens in tokenized_docs {
            let mut doc_tf: HashMap<String, f64> = HashMap::new();
            for tok in doc_tokens {
                *doc_tf.entry(tok.clone()).or_insert(0.0) += 1.0;
            }
            doc_lengths.push(doc_tokens.len() as f64);
            for term in doc_tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            tf.push(doc_tf);
        }
        let total_len: f64 = doc_lengths.iter().sum();
        let avg_dl = total_len / num_docs as f64;
        let n = num_docs as f64;
        let idf: HashMap<String, f64> = df
            .iter()
            .map(|(term, freq)| {
                let dfv = *freq as f64;
                let v = ((n - dfv + 0.5) / (dfv + 0.5) + 1.0).ln();
                (term.clone(), v)
            })
            .collect();
        Self {
            num_docs,
            avg_dl,
            idf,
            tf,
            doc_lengths,
        }
    }

    fn get_scores(&self, query_tokens: &[String]) -> Vec<f64> {
        let mut scores = vec![0.0; self.num_docs];
        if self.num_docs == 0 || query_tokens.is_empty() {
            return scores;
        }
        const K1: f64 = 1.5;
        const B: f64 = 0.75;
        for tok in query_tokens {
            let idf_val = match self.idf.get(tok) {
                Some(&v) => v,
                None => continue,
            };
            for (doc_idx, doc_tf) in self.tf.iter().enumerate() {
                let tf_val = match doc_tf.get(tok) {
                    Some(&v) => v,
                    None => continue,
                };
                let dl = self.doc_lengths[doc_idx];
                let denom = tf_val + K1 * (1.0 - B + B * dl / self.avg_dl);
                let tf_component = (tf_val * (K1 + 1.0)) / denom;
                scores[doc_idx] += idf_val * tf_component;
            }
        }
        scores
    }
}

fn collect_docs(root: &Path) -> Vec<Vec<String>> {
    let mut docs = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&p) {
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|s| s.to_str())
                        && (name.starts_with('.')
                            || name == "node_modules"
                            || name == "target"
                            || name == "dist"
                            || name == "build"
                            || name == "__pycache__"
                            || name == "venv"
                            || name == ".venv")
                    {
                        continue;
                    }
                    stack.push(path);
                } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ok = matches!(
                        ext,
                        "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "java"
                    );
                    if !ok {
                        continue;
                    }
                    if let Ok(meta) = std::fs::metadata(&path)
                        && meta.len() > 1_000_000
                    {
                        continue;
                    }
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        // Chunk into 50-line blocks (loose match for benchmark purposes).
                        let lines: Vec<&str> = text.lines().collect();
                        for chunk in lines.chunks(50) {
                            let body = chunk.join("\n");
                            docs.push(tokenize(&body));
                        }
                    }
                }
            }
        }
    }
    docs
}

fn time_query<F: Fn() -> Vec<f64>>(f: F, runs: usize) -> (u128, u128) {
    // Warm-up.
    for _ in 0..3 {
        let _ = f();
    }
    let mut samples: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed().as_micros());
    }
    samples.sort();
    let p50 = samples[samples.len() / 2];
    let mean: u128 = samples.iter().sum::<u128>() / runs as u128;
    (p50, mean)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bm25_compare <repo_path> [n_runs]");
        std::process::exit(2);
    }
    let repo = Path::new(&args[1]);
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!("Collecting & tokenizing corpus from {}", repo.display());
    let t = Instant::now();
    let docs = collect_docs(repo);
    eprintln!(
        "  {} docs ({} tokens total) in {:?}",
        docs.len(),
        docs.iter().map(|d| d.len()).sum::<usize>(),
        t.elapsed()
    );

    let queries: Vec<Vec<String>> = [
        "load model from disk",
        "BM25 inverted index",
        "tokenize identifier",
        "save_pretrained",
        "rerank top-k results",
        "cosine similarity",
        "git clone repo",
        "parseConfig handler",
        "boost multi chunk files",
        "walk gitignore directory",
    ]
    .iter()
    .map(|q| tokenize(q))
    .collect();

    eprintln!("\nBuild time:");
    let t = Instant::now();
    let new = NewBm25::new(&docs);
    eprintln!("  new (inverted, parallel):  {:?}", t.elapsed());
    let t = Instant::now();
    let legacy = LegacyBm25::new(&docs);
    eprintln!("  legacy (HashMap-per-doc):  {:?}", t.elapsed());

    eprintln!("\nQuery latency (per-query, averaged across query set):");
    let (new_p50, new_mean) = time_query(
        || {
            let mut last = Vec::new();
            for q in &queries {
                last = new.get_scores(q, None);
            }
            last
        },
        runs,
    );
    let (leg_p50, leg_mean) = time_query(
        || {
            let mut last = Vec::new();
            for q in &queries {
                last = legacy.get_scores(q);
            }
            last
        },
        runs,
    );
    eprintln!(
        "  new     p50 {:>7} µs   mean {:>7} µs   ({} runs over {} queries)",
        new_p50,
        new_mean,
        runs,
        queries.len()
    );
    eprintln!(
        "  legacy  p50 {:>7} µs   mean {:>7} µs   ({} runs over {} queries)",
        leg_p50,
        leg_mean,
        runs,
        queries.len()
    );
    if new_p50 > 0 {
        eprintln!(
            "\n  speedup: {:.2}× p50, {:.2}× mean",
            leg_p50 as f64 / new_p50 as f64,
            leg_mean as f64 / new_mean as f64
        );
    }

    Ok(())
}
