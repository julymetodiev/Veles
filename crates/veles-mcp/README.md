# veles-mcp

[![Crates.io](https://img.shields.io/crates/v/veles-mcp.svg)](https://crates.io/crates/veles-mcp)
[![docs.rs](https://docs.rs/veles-mcp/badge.svg)](https://docs.rs/veles-mcp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Model Context Protocol (MCP) server for
[Veles](https://github.com/julymetodiev/Veles) — fast and accurate
local code search for AI agents.

`veles-mcp` speaks JSON-RPC 2.0 over stdio so MCP-aware clients
(Claude Desktop, Cursor, etc.) can query a codebase as a tool call.
Indexes are cached in-process across calls — repeat queries against
the same repo skip the re-index cost.

## Tools exposed to the agent

### Discovery

| Tool           | Use it for                                                                                  |
|----------------|---------------------------------------------------------------------------------------------|
| `search`       | Natural-language or code query against a repo (hybrid by default). Optional `lang` / `path` / `exclude` glob filters and a `min_score` threshold narrow noisy queries. |
| `defs`         | Every tree-sitter definition with the given name (Rust, Python, JavaScript, TypeScript, Go). More precise than `search` when you already know the symbol name. |
| `refs`         | Definitions plus BM25 hits for a symbol name. One call to answer both "where is X defined" and "where is X used". BM25 chunks that overlap a definition site are deduped out automatically. |
| `find_related` | Semantically similar chunks for a `(file_path, line)` from an earlier `search`. Accepts the same `lang` / `path` / `exclude` filters as `search`. |
| `list_symbols` | Every tree-sitter definition across the index, with optional `kind` / `lang` / `path` / `exclude` filters. The exploration tool for "every struct in crates/foo/" kind of questions. |
| `symbols`      | The tree-sitter outline of a single file — a cheap alternative to reading the whole file when only the structure matters. |
| `scope_at`     | Innermost tree-sitter symbol containing a given `file:line`. Cheap "what function does this line live in?" without scanning the file's full symbol list. |
| `files`        | Distinct file paths the index knows about, with the same glob / language filters as `search`. Orientation tool for an unfamiliar repo. |
| `read`         | Line range from an indexed file (capped at 500 lines, refuses absolute / `..`-escaping paths). Use the relative paths returned by `defs` / `search` directly. |

### Index lifecycle

| Tool           | Use it for                                                                                  |
|----------------|---------------------------------------------------------------------------------------------|
| `stats`        | What the index knows about a repo: file count, chunk count, model metadata, per-language breakdown. |
| `status`       | Non-mutating drift check: compare the persisted `.veles/manifest.json` against current disk state. Distinguishes real content changes from bare `touch` (mtime drift with matching BLAKE3). |
| `update`       | Incrementally refresh a local repo's `.veles/` index. Files whose BLAKE3 content hash changed get re-embedded; mtime drift on unchanged bytes is a manifest-only refresh (no embedding). |

The `repo` argument (defaults to `.`) may be a local directory path **or** an `https://` git URL. Remote repos are shallow-cloned into a temp directory the first time they're searched, then cached in-process. `update`, `status`, and `read` are local-only — re-run `search` against an https:// URL to re-clone it.

### Result formats

`search`, `find_related`, and `refs` accept a `format` argument:

| Format         | Output                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|
| `default`      | Scored, fenced code blocks. Each header carries a tree-sitter scope label (``defines `Foo` `` or ``in `bar` ``) so you can route on the header alone without reading the body. |
| `paths`        | Flat per-line list. `search` / `find_related` emit `path:start-end` (the chunk range); `refs` emits `path:line` — one row per word-boundary occurrence of the symbol inside each hit. Token-cheap shortlist. |
| `unique_paths` | One `path` line per file, deduped — for "which files matter" workflows.                |

## Run the server

From the CLI (the default if no subcommand is given):

```sh
veles serve-mcp
veles            # equivalent — bare `veles` starts MCP on a piped stdin
```

From code:

```rust,no_run
# async fn run() -> anyhow::Result<()> {
let model = veles_core::model::load_model(None)?;
veles_mcp::McpServer::new(model).run().await?;
# Ok(())
# }
```

## Wiring it into Claude Desktop

Add an entry to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "veles": {
      "command": "veles",
      "args": ["serve-mcp"]
    }
  }
}
```

The same shape works for any other MCP-aware client.

## See also

- [`veles-core`](https://crates.io/crates/veles-core) — indexing and
  search engine wrapped by this crate.
- [`veles-grpc`](https://crates.io/crates/veles-grpc) — gRPC flavour of
  the same surface.
- The [project README](https://github.com/julymetodiev/Veles).

## License

MIT
