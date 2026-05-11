//! Full index build wall-time on a synthetic corpus.
//!
//! Smaller fixture than `search.rs` because each iteration rebuilds the
//! whole index (file walk → chunking → embedding → BM25). With criterion's
//! default 100 samples that would take forever on the larger fixture, so
//! we shrink `sample_size` and corpus size.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use veles_core::VelesIndex;

#[path = "common/mod.rs"]
mod common;

fn bench_build(c: &mut Criterion) {
    // Pre-load the model so the timed section measures only build cost.
    let _ = common::shared_model();
    // 30 files × 150 lines ≈ 100-150 chunks. Keeps each iteration under
    // a few seconds on CPU; sample_size = 10 keeps the bench wall time
    // reasonable.
    let corpus = common::make_corpus(30, 150, 0xC0DE);

    let mut group = c.benchmark_group("build");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("from_path/small", |b| {
        b.iter(|| {
            let model = common::shared_model().clone();
            let index =
                VelesIndex::from_path(&corpus.root, Some(model), None, false).expect("build");
            black_box(index);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
