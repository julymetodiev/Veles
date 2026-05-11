//! Incremental update latency — a single file changes between iterations.
//!
//! We build a 2K-chunk fixture once, persist it, then for each iteration:
//!   1. Touch one file's mtime (so the cheap fingerprint diverges).
//!   2. Call `update_from_path`, which re-hashes that one file, sees the
//!      content is identical, and does the manifest-only fast path.
//!
//! This bench targets the §2.3 "round-trip flat→Vec<Vec>→flat" hot path
//! plus the fingerprint classification logic.

use criterion::{Criterion, criterion_group, criterion_main};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, SystemTime};

#[path = "common/mod.rs"]
mod common;

/// Pick one fixture file and rewrite it with a small content delta so
/// `update_from_path` takes the "modified file" path (re-chunk +
/// re-embed). Returns the path so subsequent iterations can keep editing
/// the same file.
fn pick_and_modify(root: &Path) -> std::path::PathBuf {
    for entry in walkdir_files(root) {
        if entry.extension().and_then(|s| s.to_str()) == Some("rs") {
            // Append a tiny comment line to force a size + content change.
            let mut content = fs::read_to_string(&entry).expect("read");
            content.push_str("// bench edit\n");
            fs::write(&entry, content).expect("write edit");
            // Bump mtime so the cheap fingerprint sees a diff.
            let _ = filetime_set(&entry);
            return entry;
        }
    }
    panic!("no .rs files in fixture");
}

fn walkdir_files(root: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if let Ok(rd) = fs::read_dir(&p) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

/// Bump the file's mtime to "now" — simulates an editor save.
fn filetime_set(path: &Path) -> std::io::Result<()> {
    // No filetime crate dep: just write the same bytes back; on most
    // filesystems write() refreshes mtime.
    let buf = fs::read(path)?;
    fs::write(path, buf)?;
    let _ = SystemTime::now();
    Ok(())
}

fn bench_update(c: &mut Criterion) {
    let corpus = common::make_corpus(60, 150, 0xC0DE);
    // Build once outside the timed section. Each iteration mutates one
    // file and calls update_from_path.
    let mut index = common::build_index(&corpus);
    // Persist so the manifest exists for subsequent loads (not strictly
    // needed by update_from_path, but mirrors realistic usage).
    index.save(&corpus.root).expect("save");

    let mut group = c.benchmark_group("update");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("single_file_edit", |b| {
        b.iter(|| {
            let _path = pick_and_modify(&corpus.root);
            let report = index.update_from_path(&corpus.root).expect("update");
            black_box(report);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_update);
criterion_main!(benches);
