# veles-cli

The command-line binary for [Veles](https://github.com/julymetodiev/Veles) —
fast and accurate local code search for AI agents.

```sh
cargo install veles-cli
```

Or grab a prebuilt binary from
[Releases](https://github.com/julymetodiev/Veles/releases).

## What it does

Hybrid search (BM25 + dense embeddings via
[model2vec-rs](https://github.com/MinishLab/model2vec-rs)) over a local
or remote repo, with a persistent on-disk index, tree-sitter symbol
extraction, and pipe-friendly output.

## Quick taste

```sh
veles index .                                # build & save .veles/
veles search "parse config file"             # hybrid search (default)
veles search "BM25" -f compact -t 3          # one line per result
veles search "auth" -f paths | xargs $EDITOR # open all matches
veles defs Manifest -k struct                # tree-sitter defs lookup
veles refs save_index -t 30                  # defs + BM25 hits
veles update .                               # incremental refresh
```

Subcommands: `search`, `find-related`, `index`, `update`, `status`,
`clean`, `symbols`, `defs`, `refs`, `serve-grpc`, `serve-mcp`,
`completions`, `man`.

For the full reference see the
[project USAGE guide](https://github.com/julymetodiev/Veles/blob/main/USAGE.md)
and the [main README](https://github.com/julymetodiev/Veles).

## License

MIT
