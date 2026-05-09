# Veles Usage Guide

A practical reference for the `veles` CLI. For architecture and design notes
see the [README](README.md).

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [The persistent index](#the-persistent-index)
- [Searching](#searching)
- [Output formats](#output-formats)
- [Filters](#filters)
- [Finding related code](#finding-related-code)
- [Symbols, defs, refs](#symbols-defs-refs)
- [Working with remote repos](#working-with-remote-repos)
- [Multilingual content](#multilingual-content)
- [Recipes](#recipes)
- [MCP / gRPC servers](#mcp--grpc-servers)
- [Shell completions and man page](#shell-completions-and-man-page)
- [Troubleshooting](#troubleshooting)

## Install

```sh
cargo build --release
# binary at ./target/release/veles
```

Optionally symlink it onto your `$PATH`:

```sh
ln -s "$(pwd)/target/release/veles" ~/.local/bin/veles
```

## Quickstart

```sh
# 1. Build the cache once. Subsequent searches reuse it.
veles index .

# 2. Search.
veles search "parse config file"

# 3. After editing files, refresh.
veles update .

# 4. Inspect index health.
veles status .
```

Without an index, `veles search` still works — it builds an in-memory index
on every run. With an index, searches finish in tens of milliseconds.

## The persistent index

Veles stores an index under `<repo>/.veles/`. The directory holds:

```
.veles/
  manifest.json   ← model, dim, per-file (size, mtime, chunk_count)
  chunks.bin      ← bincode-encoded Vec<Chunk>
  bm25.bin        ← bincode-encoded BM25 inverted index
  dense.bin       ← bincode-encoded dense matrix
```

Add `.veles/` to your `.gitignore`.

### Build / rebuild

```sh
veles index .                  # build, refuses if .veles/ already exists
veles index . --force          # rebuild from scratch
veles index . --include-text-files
veles index . --multilingual
```

### Incremental update

```sh
veles update .
```

`update` keeps the embeddings of unchanged files (the expensive part) and only
re-chunks + re-embeds files whose `(size, mtime)` changed. Reports look like:

```
Updated in 0.01s — +0 added, ~1 modified, -0 removed
                   (kept 114 chunks, embedded 1 new, total 115)
```

### Status / drift

```sh
veles status .
```

Compares the manifest to the current filesystem and prints how many files were
added / modified / removed since the last `index` or `update`. Use this as a
fast sanity check before searching.

### Remove

```sh
veles clean .
```

### Auto-loading and `--no-cache`

`search` and `find-related` automatically load `.veles/` if it exists. Pass
`--no-cache` to force a fresh in-memory build (useful for debugging).

## Searching

```sh
veles search "QUERY" [PATH]
```

Common flags:

| Flag                     | Description                                                     |
|--------------------------|-----------------------------------------------------------------|
| `-t, --top-k N`          | Number of results (default 5)                                   |
| `-m, --mode MODE`        | `hybrid` (default), `semantic`, `bm25`                          |
| `-f, --format FORMAT`    | See [Output formats](#output-formats)                           |
| `-l, --lang LANGS`       | Comma-separated language filter (e.g. `rust,python`)            |
| `-g, --path GLOB`        | Include glob, repeatable                                        |
| `-x, --exclude GLOB`     | Exclude glob, repeatable                                        |
| `--min-score F`          | Drop results scoring below `F`                                  |
| `--include-text-files`   | Index Markdown / TOML / JSON / YAML too                         |
| `--multilingual`         | Use the multilingual embedding model                            |
| `--no-cache`             | Bypass any `.veles/` cache                                      |

### Modes

- **`hybrid`** (default) — RRF blend of BM25 and semantic, with
  query-type detection that leans BM25 for symbol-like queries and
  semantic for natural language. Best for most queries.
- **`bm25`** — pure lexical. Fastest. Use when you know the literal
  identifier or substring you're after.
- **`semantic`** — pure dense vector search. Use for fuzzy concept
  queries when you don't know the exact terms.

```sh
veles search "TokenStream" .   --mode bm25
veles search "rate limiting" . --mode semantic
```

## Output formats

Pick a format with `-f` / `--format`. Pretty is the human view; everything
else is line-oriented and pipe-friendly.

### `pretty` (default)

Markdown with fenced code blocks. The original Veles output. Best in a
terminal, worst for piping.

### `compact`

One line per result.

```
crates/veles-core/src/index/sparse.rs:1-50  [score=0.019]  //! BM25 sparse index — inverted-index implementation with token interning.
crates/veles-core/src/veles_index.rs:496-545  [score=0.017]  let chunks: Vec<Chunk> = files
```

Good for terminal use when you want a list view, or for quick inspection
in editor pickers.

### `ripgrep` (alias `rg`)

Each source line of a matched chunk emitted as `path:line:content`.

```
crates/veles-core/src/index/sparse.rs:1://! BM25 sparse index — inverted-index implementation with token interning.
crates/veles-core/src/index/sparse.rs:2://!
```

Drops straight into editor / quickfix workflows that already understand
ripgrep output.

### `paths` (alias `files`)

Unique paths, in result order, deduped.

```
crates/veles-core/src/index/sparse.rs
crates/veles-core/src/veles_index.rs
crates/veles-core/src/index/search.rs
```

### `json`

Single envelope object:

```json
{
  "header": "Search results for: \"BM25\" (mode=hybrid)",
  "count": 3,
  "results": [
    {
      "file_path": "crates/veles-core/src/index/sparse.rs",
      "start_line": 1, "end_line": 50,
      "score": 0.0189,
      "source": "hybrid",
      "language": "rust",
      "content": "..."
    }
  ]
}
```

### `jsonl` (alias `ndjson`)

One JSON object per line. Friendlier for streaming consumers and `jq -c`.

## Filters

### Languages

```sh
veles search "auth" . -l rust,python
```

Languages are inferred from extensions. See `crates/veles-core/src/walker.rs`
for the full list.

### Path globs

`--path` (`-g`) and `--exclude` (`-x`) are both repeatable. Globs match
against indexed file paths (relative to the indexed root). They are applied
**before** scoring, so top-k is computed on the matching pool — this matters
when the include set is small.

```sh
# Only the core crate
veles search "search hybrid" -g 'crates/veles-core/**'

# Skip the test suite
veles search "auth" -x 'tests/**' -x '**/*_test.rs'

# Combine
veles search "X" -g 'src/**/*.rs' -x 'src/legacy/**'
```

A bad glob fails fast with a clear error.

### Score threshold

```sh
veles search "BM25" --min-score 0.4 -f compact
```

## Finding related code

Given a file:line, return the most semantically similar chunks elsewhere.

```sh
veles find-related crates/veles-core/src/index/sparse.rs 50
veles find-related src/auth.rs 120 -t 10 -f compact
```

Works on the same persistent cache, supports `--format` and `--min-score`.

## Symbols, defs, refs

Veles parses each indexed file with tree-sitter and stores the
definitions in `.veles/symbols.bin`. This unlocks three commands
that ripgrep can't easily replicate.

Supported languages: **Rust, Python, JavaScript, TypeScript, Go**.
Other languages are still indexed for search; they just don't
contribute symbols.

### `veles symbols <file>`

List every definition in a single file. No index required —
parses the file directly.

```sh
veles symbols src/main.rs
veles symbols crates/veles-core/src/persist.rs -f compact
```

Output kinds: `function`, `method`, `struct`, `class`, `enum`,
`trait`, `interface`, `type`, `const`, `static`, `var`,
`module`, `macro`.

### `veles defs <name>`

Find every definition with the given exact name across the
indexed repo. Reads from the symbol cache, so it's near-instant.

```sh
veles defs Manifest
veles defs save -k function           # only functions named "save"
veles defs parse -l rust,python       # restrict by language
veles defs MyType -f json | jq '.symbols[].file_path'
```

Flags:

| Flag             | Description                                                  |
|------------------|--------------------------------------------------------------|
| `-l, --lang`     | Comma-separated language filter                              |
| `-k, --kind`     | Filter by kind (function, struct, class, enum, trait, type, …) |
| `-f, --format`   | `pretty` / `compact` / `paths` / `json` / `jsonl`            |

### `veles refs <name>`

References to a symbol — combines definitions (high confidence)
with BM25 hits in the rest of the corpus (best-effort identifier
matches).

```sh
veles refs Manifest
veles refs parse_config -t 30 -f compact
```

In `pretty`, the output has two clear sections (`Definitions` and
`Other matches (BM25)`). In line-oriented formats both streams
are concatenated, defs first.

> Note: tree-sitter symbol caching bumps the on-disk format
> version. Existing indexes will be rejected with
> "format version 1 is incompatible (expected 2)". Run
> `veles index . --force` to rebuild.

## Working with remote repos

Pass any git URL where you would pass a path. Veles shallow-clones into a
temp dir, indexes it in memory, and discards the clone on exit (no `.veles/`
is persisted for remote repos).

```sh
veles search "BM25 inverted index" https://github.com/julymetodiev/Veles
```

## Multilingual content

The default model (`potion-code-16M`) is English/code-focused. For codebases
or queries with Cyrillic / CJK / Greek / Arabic content, use:

```sh
veles index . --multilingual
veles search "функция обработка" . --multilingual
```

The flag must match between `index` / `update` and `search` — the model
choice is recorded in the manifest.

## Recipes

### Open every match in `$EDITOR`

```sh
veles search "TODO: deprecate" -f paths -t 50 | xargs $EDITOR
```

### Pipe into `fzf` with previews

```sh
veles search "auth flow" -t 50 -f compact \
  | fzf --delimiter ' ' --preview 'bat --color=always {1}'
```

### Count distinct files per query

```sh
veles search "rate limit" -t 100 -f paths | wc -l
```

### Programmatic post-processing

```sh
veles search "BM25" -f json \
  | jq -r '.results[] | "\(.score)\t\(.file_path):\(.start_line)"'
```

### Stream to a long-running tool

```sh
veles search "panic" -t 200 -f jsonl \
  | while read -r line; do echo "$line" | jq -r '.file_path'; done
```

### Quickfix list for vim/neovim

```sh
veles search "deprecated" -f rg > /tmp/qf.txt
# in vim: :cfile /tmp/qf.txt
```

## MCP / gRPC servers

For agent integration:

```sh
# MCP over stdio (default if no subcommand is given)
veles serve-mcp
veles                          # equivalent

# gRPC
veles serve-grpc --addr "[::1]:50051"
```

The MCP server exposes `search` and `find_related` as tools. See
`crates/veles-mcp/src/lib.rs` for the JSON-RPC schema.

## Shell completions and man page

`veles` ships its own completion generators and a `man(7)` page,
so there are no extra files to download.

### Zsh

```sh
mkdir -p ~/.zfunc
veles completions zsh > ~/.zfunc/_veles
# Once, in ~/.zshrc:
fpath=(~/.zfunc $fpath)
autoload -Uz compinit && compinit
```

### Bash

```sh
veles completions bash | sudo tee /etc/bash_completion.d/veles >/dev/null
# Or for current user only:
veles completions bash > ~/.local/share/bash-completion/completions/veles
```

### Fish

```sh
veles completions fish > ~/.config/fish/completions/veles.fish
```

### PowerShell

```powershell
veles completions powershell | Out-String | Invoke-Expression
# To persist, append the output to your $PROFILE.
```

### Elvish

```sh
veles completions elvish > ~/.config/elvish/lib/veles.elv
# Then in rc.elv: use veles
```

### Man page

```sh
# Project-local install (no sudo):
mkdir -p ~/.local/share/man/man1
veles man > ~/.local/share/man/man1/veles.1
# Make sure ~/.local/share/man is on $MANPATH.

# System-wide (Linux):
veles man | sudo tee /usr/local/share/man/man1/veles.1 >/dev/null

# View it:
man veles
```

The top-level page lists every subcommand. For per-subcommand
help, use `veles <subcommand> --help`.

## Troubleshooting

### `Index format version N is incompatible (expected M)`

The on-disk format was bumped. Run `veles index . --force` to rebuild.

### `No indexed files matched the given --path / --exclude globs`

Your glob filtered out every file in the index. Check the glob syntax —
`globset` uses standard shell globbing with `**` for any depth. Quote
patterns to avoid shell expansion: `-g 'src/**/*.rs'`.

### Slow searches on a large repo

Make sure `.veles/` exists and is up to date:

```sh
veles status .
```

If `--no-cache` is faster than the default path, your cache is missing
and Veles is falling back to in-memory builds — check that the directory
isn't being deleted by a build / CI step.

### Embedding model download

The first run downloads the embedding model from Hugging Face into the
`hf-hub` cache (`~/.cache/huggingface/hub/`). Subsequent runs reuse it.
