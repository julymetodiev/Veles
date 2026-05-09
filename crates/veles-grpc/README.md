# veles-grpc

[![Crates.io](https://img.shields.io/crates/v/veles-grpc.svg)](https://crates.io/crates/veles-grpc)
[![docs.rs](https://docs.rs/veles-grpc/badge.svg)](https://docs.rs/veles-grpc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[tonic](https://github.com/hyperium/tonic)-based gRPC service for
[Veles](https://github.com/julymetodiev/Veles) — fast and accurate
local code search for AI agents.

`veles-grpc` exposes [`veles-core`](https://crates.io/crates/veles-core)
over a small RPC surface, with a process-local cache of `VelesIndex`
instances so repeat calls against the same repo skip the re-index cost.

## RPCs

| RPC           | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `Index`       | Build / refresh an index for a path or `https://` git URL.        |
| `Search`      | Hybrid / BM25 / semantic search with optional language/path filters. |
| `FindRelated` | Semantically similar chunks for a `(file, line)` pair.            |
| `GetStats`    | Index size and per-language counts.                               |

The wire schema lives at
[`proto/veles.proto`](https://github.com/julymetodiev/Veles/blob/main/crates/veles-grpc/proto/veles.proto).
Generated types are re-exported under [`veles_grpc::proto`](https://docs.rs/veles-grpc/latest/veles_grpc/proto/).

## Build dependency: `protoc`

Building this crate runs `tonic-build`, which needs the protobuf
compiler. We bundle one via
[`protoc-bin-vendored`](https://crates.io/crates/protoc-bin-vendored),
so `cargo install veles-grpc` (or any downstream `cargo build`) works
without `protoc` installed system-wide.

## Run the server

From the CLI:

```sh
veles serve-grpc --addr "[::1]:50051"
```

From code:

```rust,no_run
# async fn run() -> anyhow::Result<()> {
let model = veles_core::model::load_model(None)?;
veles_grpc::serve("[::1]:50051", model).await?;
# Ok(())
# }
```

## See also

- [`veles-core`](https://crates.io/crates/veles-core) — indexing and
  search engine wrapped by this crate.
- [`veles-mcp`](https://crates.io/crates/veles-mcp) — MCP/JSON-RPC
  server flavour for AI-agent integration over stdio.
- The [project README](https://github.com/julymetodiev/Veles).

## License

MIT
