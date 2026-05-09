# veles-mcp

MCP (Model Context Protocol) server for
[Veles](https://github.com/julymetodiev/Veles) — fast and accurate
local code search for AI agents.

`veles-mcp` speaks JSON-RPC 2.0 over stdio and exposes two tools to
clients like Claude, Cursor, or any MCP-aware assistant:

- `search`        — natural-language or code query against a repo
- `find_related`  — semantically similar chunks for a `(file, line)`

Indexes are cached in-process across calls, so repeat queries against
the same repo are fast.

```sh
veles serve-mcp     # default if no subcommand is given
```

For the project as a whole see the
[main README](https://github.com/julymetodiev/Veles).

## License

MIT
