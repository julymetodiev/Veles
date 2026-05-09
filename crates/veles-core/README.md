# veles-core

Core library for [Veles](https://github.com/julymetodiev/Veles) — fast and
accurate local code search for AI agents.

`veles-core` is the indexing and search engine. It walks a directory,
chunks source files, builds a BM25 inverted index plus a dense
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) embedding index,
and serves hybrid queries with Reciprocal Rank Fusion. Tree-sitter is
used to extract symbols (functions, structs, classes, …) for
definition-level lookups.

```rust
use veles_core::{VelesIndex, SearchMode};
use std::path::Path;

let index = VelesIndex::from_path(Path::new("."), None, None, false)?;
let results = index.search("parse config file", 5, SearchMode::Hybrid, None, None, None);
for r in results {
    println!("{} [{:.3}]", r.chunk.location(), r.score);
}
```

The on-disk index is persisted under `<repo>/.veles/` and supports
incremental updates that reuse embeddings of unchanged files.

For end-user CLI documentation see the [project README](https://github.com/julymetodiev/Veles)
and [USAGE.md](https://github.com/julymetodiev/Veles/blob/main/USAGE.md).

## License

MIT
