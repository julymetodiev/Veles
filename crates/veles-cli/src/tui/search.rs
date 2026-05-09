//! Background search worker for the TUI.
//!
//! Runs on its own OS thread and owns an `Arc<VelesIndex>`. The UI thread
//! posts `WorkerCmd::Search { gen, ... }` whenever the query changes; the
//! worker drains any backlog and only services the most recent command,
//! so a fast typist doesn't queue up dozens of doomed searches behind a
//! 50ms one. Each result carries the `gen` it originated from, and the UI
//! discards anything older than the latest dispatched generation.

use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use veles_core::VelesIndex;
use veles_core::types::{Chunk, SearchMode, SearchResult};

pub enum WorkerCmd {
    Search {
        seq: u64,
        query: String,
        mode: SearchMode,
        top_k: usize,
    },
    Related {
        seq: u64,
        source: Box<Chunk>,
        top_k: usize,
    },
    Shutdown,
}

pub struct SearchDone {
    pub seq: u64,
    pub query: String,
    pub results: Vec<SearchResult>,
    pub elapsed_ms: u64,
    pub kind: ResultKind,
}

pub enum ResultKind {
    Query,
    Related { anchor: String },
}

pub enum WorkerMsg {
    SearchDone(SearchDone),
}

pub fn spawn_worker(
    index: Arc<VelesIndex>,
    cmd_rx: Receiver<WorkerCmd>,
    msg_tx: Sender<WorkerMsg>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("veles-tui-search".into())
        .spawn(move || worker_loop(index, cmd_rx, msg_tx))
        .expect("spawn veles-tui-search thread")
}

fn worker_loop(index: Arc<VelesIndex>, cmd_rx: Receiver<WorkerCmd>, msg_tx: Sender<WorkerMsg>) {
    while let Ok(mut cmd) = cmd_rx.recv() {
        // Coalesce: pull any commands queued behind us and keep only the
        // newest. A `Shutdown` always wins so the worker tears down quickly.
        loop {
            match cmd_rx.try_recv() {
                Ok(WorkerCmd::Shutdown) => return,
                Ok(newer) => cmd = newer,
                Err(_) => break,
            }
        }

        match cmd {
            WorkerCmd::Shutdown => return,
            WorkerCmd::Search {
                seq,
                query,
                mode,
                top_k,
            } => {
                let started = Instant::now();
                let results = if query.trim().is_empty() {
                    Vec::new()
                } else {
                    index.search(&query, top_k, mode, None, None, None)
                };
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let _ = msg_tx.send(WorkerMsg::SearchDone(SearchDone {
                    seq,
                    query,
                    results,
                    elapsed_ms,
                    kind: ResultKind::Query,
                }));
            }
            WorkerCmd::Related { seq, source, top_k } => {
                let started = Instant::now();
                let anchor = format!("{}:{}", source.file_path, source.start_line);
                let results = index.find_related(&source, top_k);
                let elapsed_ms = started.elapsed().as_millis() as u64;
                let _ = msg_tx.send(WorkerMsg::SearchDone(SearchDone {
                    seq,
                    query: String::new(),
                    results,
                    elapsed_ms,
                    kind: ResultKind::Related { anchor },
                }));
            }
        }
    }
}
