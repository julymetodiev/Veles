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

| Tool          | Use it for                                                      |
|---------------|-----------------------------------------------------------------|
| `search`      | Natural-language or code query against a repo (hybrid by default). |
| `find_related`| Semantically similar chunks for a `(file_path, line)` from an earlier `search`. |

The `repo` argument may be a local directory path **or** an `https://`
git URL. Remote repos are shallow-cloned into a temp directory the
first time they're searched, then cached in-process.

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
