# veles-grpc

gRPC service for [Veles](https://github.com/julymetodiev/Veles) — fast
and accurate local code search for AI agents.

`veles-grpc` exposes [`veles-core`](https://crates.io/crates/veles-core)
over [tonic](https://github.com/hyperium/tonic). RPCs:

- `Index`        — build / refresh an index for a path
- `Search`       — hybrid / BM25 / semantic search
- `FindRelated`  — semantically similar chunks for a `(file, line)`
- `GetStats`     — index size and per-language counts

The `.proto` schema lives at `proto/veles.proto`.

```sh
veles serve-grpc --addr "[::1]:50051"
```

For the project as a whole see the
[main README](https://github.com/julymetodiev/Veles).

## License

MIT
