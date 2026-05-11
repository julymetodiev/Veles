# veles-core

[![Crates.io](https://img.shields.io/crates/v/veles-core.svg)](https://crates.io/crates/veles-core)
[![docs.rs](https://docs.rs/veles-core/badge.svg)](https://docs.rs/veles-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Core library for [Veles](https://github.com/julymetodiev/Veles) — fast,
hybrid (BM25 + semantic) local code search for AI agents and humans,
written in pure Rust.

`veles-core` is the indexing and search engine. It walks a directory,
chunks source files, builds a BM25 inverted index plus a dense
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) embedding
index, and serves hybrid queries with Reciprocal Rank Fusion.
Tree-sitter is used to extract definitions for symbol-level lookups.

- **No GPU, no transformer forward pass at query time.** Static
  embeddings keep query latency in the tens of milliseconds on CPU.
- **Persistent on-disk index** under `<repo>/.veles/`, with incremental
  updates that reuse embeddings of unchanged files.
- **Identifier-aware tokeniser** that splits camelCase, snake_case, and
  mixed-script identifiers; multilingual model option for Cyrillic, CJK,
  Arabic, etc.
- **Pure Rust** — no Python, no `protoc`, no native ML runtime required.

## Install

```toml
[dependencies]
veles-core = "0.2"
```

## Quick start

```rust
use std::path::Path;
use veles_core::{SearchMode, VelesIndex};

# fn main() -> anyhow::Result<()> {
let index = VelesIndex::from_path(Path::new("."), None, None, false)?;

let results = index.search(
    "parse config file",
    5,
    SearchMode::Hybrid,
    None,  // alpha — auto-detect from query type
    None,  // language filter
    None,  // path filter
);

for r in results {
    println!("{} [{:.3}]", r.chunk.location(), r.score);
}
# Ok(())
# }
```

The first build downloads the default embedding model (~64 MB) into the
HuggingFace cache (`~/.cache/huggingface/hub/`).

## Persistence and incremental updates

```rust
use std::path::Path;
use veles_core::VelesIndex;

# fn main() -> anyhow::Result<()> {
let repo = Path::new(".");
let index = VelesIndex::from_path(repo, None, None, false)?;
index.save(repo)?;

// Later, reload without re-embedding:
let model = veles_core::model::load_model(None)?;
let mut reloaded = VelesIndex::load(repo, model)?;

// Files whose (size, mtime) fingerprint hasn't changed keep their
// embeddings. When mtime drifts but the BLAKE3 content hash still
// matches, we do a manifest-only refresh (no re-embed). Only files
// with genuinely different bytes are re-embedded.
let report = reloaded.update_from_path(repo)?;
eprintln!("{} added, {} modified, {} removed, {} mtime-only",
    report.added_files, report.modified_files,
    report.removed_files, report.mtime_refreshed_files);
# Ok(())
# }
```

## What's in the box

| Module       | Purpose                                                    |
|--------------|------------------------------------------------------------|
| `veles_index`| The main `VelesIndex` (index + search + persist + symbols).|
| `chunker`    | Line-based source chunking with overlap.                   |
| `tokenizer`  | Identifier-aware tokeniser (camelCase / snake_case / Unicode). |
| `index::sparse` | BM25 inverted index over a corpus of tokenised docs.    |
| `index::dense`  | Brute-force cosine similarity (rayon-parallel, top-k via min-heap). |
| `index::search` | Semantic / BM25 / hybrid (RRF) search entry points.     |
| `ranking`    | Definition boosts, file-saturation decay, path penalties.  |
| `symbols`    | Tree-sitter symbols for Rust, Python, JavaScript, TypeScript, Go. |
| `persist`    | On-disk format under `.veles/` (manifest + bincode).       |
| `walker`     | `.gitignore`-aware file walker.                            |
| `model`      | model2vec-rs loader (default + multilingual).              |

## See also

- The [end-user CLI documentation](https://github.com/julymetodiev/Veles)
  and [USAGE.md](https://github.com/julymetodiev/Veles/blob/main/USAGE.md)
  reference for the `veles` binary.
- [`veles-grpc`](https://crates.io/crates/veles-grpc) — tonic-based gRPC
  service wrapping `veles-core`.
- [`veles-mcp`](https://crates.io/crates/veles-mcp) — MCP server exposing
  `search` and `find_related` over JSON-RPC for AI agents.

## License

MIT
