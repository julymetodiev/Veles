# veles-cli

[![Crates.io](https://img.shields.io/crates/v/veles-cli.svg)](https://crates.io/crates/veles-cli)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The `veles` command-line binary —
[Veles](https://github.com/julymetodiev/Veles) is fast, hybrid
(BM25 + semantic) local code search for AI agents and humans, written
in pure Rust.

```sh
# crates.io (compiles locally — no protoc, no extra system deps)
cargo install veles-cli

# Or grab a prebuilt binary
#   https://github.com/julymetodiev/Veles/releases
```

## What it does

Hybrid search (BM25 + dense embeddings via
[model2vec-rs](https://github.com/MinishLab/model2vec-rs)) over a local
or remote repo, with a persistent on-disk index, tree-sitter symbol
extraction, pipe-friendly output, and built-in MCP / gRPC servers.

## Most-used commands

```sh
veles index .                                # build & save .veles/
veles search "parse config file"             # hybrid search (default)
veles search "BM25" -f compact -t 3          # one line per result
veles search "auth" -f paths | xargs $EDITOR # open all matches
veles defs Manifest -k struct                # tree-sitter defs lookup
veles refs save_index -t 30                  # defs + BM25 hits
veles update .                               # incremental refresh
veles tui                                    # live hybrid search TUI
```

Subcommands: `search`, `find-related`, `index`, `update`, `status`,
`clean`, `symbols`, `defs`, `refs`, `tui`, `serve-grpc`, `serve-mcp`,
`completions`, `man`. Run `veles <SUB> --help` for per-command details.

## Output formats

`-f pretty` (default), `compact`, `ripgrep`, `paths`, `json`, `jsonl`.
Stable line-oriented formats are designed to compose with `xargs`,
`fzf`, vim quickfix, `jq`, etc.

## Servers

```sh
veles serve-mcp                          # MCP over stdio for Claude / Cursor
veles serve-grpc --addr "[::1]:50051"    # gRPC service
```

## Shell integration

```sh
veles completions zsh > ~/.zfunc/_veles
veles man --out-dir ~/.local/share/man/man1
```

`veles man --out-dir DIR` writes one page per subcommand
(`veles.1`, `veles-search.1`, `veles-defs.1`, ...) so
`man veles-search` resolves the same way as `man git-commit`.

## See also

- The [project README](https://github.com/julymetodiev/Veles) and the
  full [USAGE.md](https://github.com/julymetodiev/Veles/blob/main/USAGE.md)
  reference.
- [`veles-core`](https://crates.io/crates/veles-core),
  [`veles-grpc`](https://crates.io/crates/veles-grpc), and
  [`veles-mcp`](https://crates.io/crates/veles-mcp) for embedding in
  your own Rust project.

## License

MIT
