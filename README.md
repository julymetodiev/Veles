# Veles

Fast, hybrid (BM25 + semantic) local code search for AI agents and humans, written in pure Rust.

Veles runs entirely on CPU — no GPU, no transformer forward pass at query time. Queries return in tens of milliseconds against a persistent on-disk index, with tree-sitter-aware symbol lookups, pipe-friendly output formats, and built-in MCP / gRPC servers for integration with Claude, Cursor, or anything else that speaks JSON-RPC. Static embeddings come from the [potion](https://huggingface.co/minishlab) family via [model2vec-rs](https://github.com/MinishLab/model2vec-rs).

Originally inspired by [Semble](https://github.com/MinishLab/semble) — Veles started as a Rust port of the same hybrid retrieval recipe and has grown to add persistent + incremental indexing, tree-sitter `symbols` / `defs` / `refs`, six pipe-friendly output formats, glob/language filters, gRPC, and shell completions.

## Interfaces

- **CLI** — `veles search "query" ./my-repo`
- **MCP server** — stdio JSON-RPC for AI agent integration (Claude, Cursor, etc.)
- **gRPC** — tonic-based service with `Index`, `Search`, `FindRelated`, `GetStats` RPCs

## Features

- **Persistent index** under `<repo>/.veles/` — searches reuse the cache and finish in tens of milliseconds. Incremental `update` keeps embeddings of unchanged files.
- **Hybrid search** with Reciprocal Rank Fusion (RRF) blending BM25 and semantic scores
- **Tree-sitter symbol commands** — `symbols` / `defs` / `refs` for Rust, Python, JavaScript, TypeScript, Go
- **Identifier-aware tokenizer** — splits camelCase, snake_case, and mixed-script names
- **Query-type detection** — symbol queries lean BM25, natural language leans semantic
- **Definition boosting** — promotes chunks that define the queried symbol
- **Path penalties** — demotes test files, compat dirs, re-export files
- **File saturation** — avoids stacking all results from one file
- **Multilingual model** option for Cyrillic, CJK, Arabic, etc.
- **Pipe-friendly output** — `pretty`, `compact`, `ripgrep`, `paths`, `json`, `jsonl`
- **Filter flags** — `--lang`, `--path` and `--exclude` glob patterns, `--min-score`
- **Prebuilt binaries** for macOS (Intel/ARM), Linux x86_64/ARM64 (musl), Windows x86_64

## Install

```sh
# Linux / macOS — prebuilt binary
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/julymetodiev/Veles/releases/latest/download/veles-cli-installer.sh | sh

# From crates.io (compiles locally; no protoc / extra deps needed)
cargo install veles-cli
```

See **[INSTALL.md](INSTALL.md)** for Windows, manual download, and verification.

## Quickstart

```sh
veles index .                     # one-off, builds .veles/
veles search "parse config file"  # auto-loads the cache
veles update .                    # refresh after edits
```

The first `search` downloads the embedding model from Hugging Face (~64 MB, cached at `~/.cache/huggingface/hub/`).

## Most-used commands

### Search

```sh
veles search "rate limiting"                          # hybrid (default)
veles search "rate limiting" -t 10 -f compact         # 10 results, 1 line each
veles search "rate limiting" -f rg                    # ripgrep-style path:line:content
veles search "rate limiting" -f json | jq '.results'  # structured for scripting
veles search "rate limiting" -f paths | xargs $EDITOR # open every matching file
```

```sh
veles search "TokenStream" -m bm25                    # exact identifier
veles search "auth flow"    -m semantic               # fuzzy concept
veles search "auth"  -l rust,python                   # language filter
veles search "X"     -g 'src/**/*.rs' -x 'src/legacy/**'   # glob include / exclude
veles search "BM25"  --min-score 0.4                  # drop weak hits
```

### Symbols (tree-sitter)

```sh
veles symbols crates/veles-core/src/persist.rs        # outline a single file
veles defs Manifest                                   # every definition named "Manifest"
veles defs save -k function -l rust                   # filter by kind + language
veles refs save_index -t 30                           # defs + BM25 references
```

### Related code

```sh
veles find-related src/main.rs 42                     # semantically similar chunks
```

### Index lifecycle

```sh
veles index .              # bootstrap
veles index . --force      # rebuild from scratch
veles update .             # incremental refresh
veles status .             # manifest + drift
veles clean .              # remove .veles/
```

### Servers

```sh
veles serve-mcp                                # MCP over stdio (default if no args)
veles serve-grpc --addr "[::1]:50051"          # gRPC
```

### Shell integration

```sh
mkdir -p ~/.zfunc ~/.local/share/man/man1
veles completions zsh > ~/.zfunc/_veles
veles man              > ~/.local/share/man/man1/veles.1
```

Then once in `~/.zshrc`:

```sh
fpath=(~/.zfunc $fpath)
autoload -Uz compinit && compinit
export MANPATH="$HOME/.local/share/man:$MANPATH"
```

### Remote repos

```sh
veles search "BM25 inverted index" https://github.com/julymetodiev/Veles
```

See **[USAGE.md](USAGE.md)** for the full reference, recipes (fzf, vim quickfix, jq), and troubleshooting.

## MCP server

```sh
veles serve-mcp     # explicit
veles               # equivalent — bare `veles` starts MCP when stdin is piped
```

Exposed tools: `search`, `find_related`.

## Build from source

```sh
cargo build --release
```

`tonic-build` ships a vendored `protoc` via `protoc-bin-vendored`, so no system-wide protobuf compiler is required.

## Architecture

```
Veles/
  crates/
    veles-core/    indexing, chunking, BM25, dense search, ranking, symbols
    veles-grpc/    gRPC service (tonic + prost)
      proto/
        veles.proto  gRPC schema
    veles-mcp/     MCP server over stdio
    veles-cli/     CLI binary
```

The persistent index lives under `<repo>/.veles/`:

```
.veles/
  manifest.json   # model, dim, per-file (size, mtime, chunk_count)
  chunks.bin      # bincode Vec<Chunk>
  bm25.bin        # bincode BM25 inverted index
  dense.bin       # bincode dense matrix
  symbols.bin     # bincode tree-sitter symbols
```

`update` reuses embeddings of files whose `(size, mtime)` fingerprint hasn't changed, so refreshing after a small edit is near-instant on large repos.

## Acknowledgments

Veles owes its initial design to [Semble](https://github.com/MinishLab/semble) and uses the same [potion](https://huggingface.co/minishlab) static-embedding models via [model2vec-rs](https://github.com/MinishLab/model2vec-rs). Many thanks to the [MinishLab](https://github.com/MinishLab) team.

## License

MIT
