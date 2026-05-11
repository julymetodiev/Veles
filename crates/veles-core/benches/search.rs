//! Per-query search latency for hybrid / BM25 / semantic modes.
//!
//! The index is built once (expensive — embeds the whole synthetic
//! corpus), then each `bench_function` times only the `index.search`
//! call. Query strings rotate through `QUERIES` so the timing reflects
//! a realistic mix of symbol / NL queries.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use veles_core::types::SearchMode;

#[path = "common/mod.rs"]
mod common;

fn bench_search(c: &mut Criterion) {
    // ~2000 chunks. Small enough to keep build under ~30s on CPU, large
    // enough that scoring N-vector allocations show up in flamegraphs.
    let corpus = common::make_corpus(150, 200, 0xC0DE);
    let index = common::build_index(&corpus);
    let stats = index.stats();
    eprintln!(
        "fixture: {} files, {} chunks",
        stats.indexed_files, stats.total_chunks
    );

    for mode in [SearchMode::Hybrid, SearchMode::Bm25, SearchMode::Semantic] {
        let mut group = c.benchmark_group(format!("search/{mode}"));
        group.throughput(Throughput::Elements(1));
        for (i, query) in common::QUERIES.iter().enumerate() {
            group.bench_with_input(BenchmarkId::from_parameter(i), query, |b, q| {
                b.iter(|| {
                    let results = index.search(black_box(q), 5, mode, None, None, None);
                    black_box(results);
                });
            });
        }
        group.finish();
    }

    // Aggregate "mixed" workload — round-robin through the query set so
    // a single number summarises end-to-end search throughput per mode.
    for mode in [SearchMode::Hybrid, SearchMode::Bm25, SearchMode::Semantic] {
        let label = format!("search_mixed/{mode}");
        c.bench_function(&label, |b| {
            let mut i = 0usize;
            b.iter(|| {
                let q = common::QUERIES[i % common::QUERIES.len()];
                i = i.wrapping_add(1);
                let r = index.search(black_box(q), 5, mode, None, None, None);
                black_box(r);
            });
        });
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
