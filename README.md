# Veles

Fast local code search for AI agents written in pure Rust

Inspired by [Semble](https://github.com/MinishLab/semble), Veles is a Rust reimplementation of the same hybrid retrieval approach. It uses the same [potion](https://huggingface.co/minishlab) static embedding models via [model2vec-rs](https://github.com/MinishLab/model2vec-rs) - no transformer forward pass at query time, everything runs in milliseconds on CPU.

## Interfaces

- **CLI** — `veles search "query" ./my-repo`
- **MCP server** — stdio JSON-RPC for AI agent integration (Claude, Cursor, etc.)
- **gRPC** — tonic-based service with `Index`, `Search`, `FindRelated`, `GetStats` RPCs

## Features

- **Persistent index** under `<repo>/.veles/` — searches reuse the cache and finish in tens of milliseconds. Incremental `update` keeps embeddings of unchanged files.
- **Hybrid search** with Reciprocal Rank Fusion (RRF) blending BM25 and semantic scores, using the same potion-code-16M / potion-multilingual-128M models as Semble
- **Identifier-aware tokenizer** — splits camelCase, snake_case, and mixed-script names
- **Query-type detection** — symbol queries lean BM25, natural language leans semantic
- **Definition boosting** — promotes chunks that define the queried symbol
- **Path penalties** — demotes test files, compat dirs, re-export files
- **File saturation** — avoids stacking all results from one file
- **Multilingual model** option for Cyrillic, CJK, Arabic, etc.
- **Pipe-friendly output** — `pretty`, `compact`, `ripgrep`, `paths`, `json`, `jsonl`
- **Filter flags** — `--lang`, `--path` and `--exclude` glob patterns, `--min-score`
- **Symbol commands** — tree-sitter `symbols` / `defs` / `refs` for Rust, Python, JavaScript, TypeScript, Go

## Install

```sh
# Linux / macOS
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/julymetodiev/Veles/releases/latest/download/veles-cli-installer.sh | sh

# macOS / Linux Homebrew
brew install julymetodiev/veles/veles-cli

# From source (any platform)
cargo install veles-cli
```

See **[INSTALL.md](INSTALL.md)** for Windows, manual download, and verification.

## Quickstart

```sh
# Index this repo (one-off, ~milliseconds for small repos)
veles index .

# Search — reuses the cache automatically
veles search "parse config file"

# Refresh after editing files
veles update .
```

See **[USAGE.md](USAGE.md)** for the full reference.

## Usage at a glance

```sh
# Persistent index lifecycle
veles index .                  # build & save to .veles/
veles update .                 # incremental refresh
veles status .                 # manifest stats + drift
veles clean .                  # remove .veles/

# Search modes and output formats
veles search "handler" .                       # hybrid (default)
veles search "auth flow" . --mode semantic
veles search "TokenStream" . --mode bm25
veles search "BM25" -f compact                 # one-line per result
veles search "BM25" -f rg                      # ripgrep-style path:line:content
veles search "BM25" -f paths | xargs $EDITOR   # open all matches
veles search "BM25" -f json | jq '.results[].file_path'

# Filters
veles search "auth" -l rust,python
veles search "X" -g 'src/**/*.rs' -x 'src/legacy/**'
veles search "BM25" --min-score 0.4

# Find code related to a specific location
veles find-related src/main.rs 42

# Symbol-aware (tree-sitter)
veles symbols crates/veles-core/src/persist.rs
veles defs Manifest -k struct
veles refs save_index -t 30 -f compact

# Remote repo (cloned to temp, no persistent cache)
veles search "BM25 inverted index" https://github.com/julymetodiev/Veles

# Multilingual model
veles search "функция обработка" . --multilingual
```

## MCP server

```sh
# Start MCP server (default if no subcommand given)
veles serve-mcp
veles
```

Exposed tools: `search`, `find_related`.

## gRPC server

```sh
veles serve-grpc --addr "[::1]:50051"
```

## Build

```sh
cargo build --release
```

## Architecture

```
Veles/
  crates/
    veles-core/    indexing, chunking, BM25, dense search, ranking, persistence
    veles-grpc/    gRPC service (tonic + prost)
    veles-mcp/     MCP server over stdio
    veles-cli/     CLI binary
  proto/
    veles.proto    gRPC service definition
```

The persistent index lives under `<repo>/.veles/`:

```
.veles/
  manifest.json   # model, dim, per-file (size, mtime, chunk_count)
  chunks.bin      # bincode Vec<Chunk>
  bm25.bin        # bincode BM25 inverted index
  dense.bin       # bincode dense matrix
```

`update` reuses embeddings of files whose `(size, mtime)` fingerprint hasn't changed, so refreshing after a small edit is near-instant on large repos.

## License

MIT
