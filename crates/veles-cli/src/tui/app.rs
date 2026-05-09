//! TUI app state, input handling, event loop.
//!
//! The [`App`] owns *all* mutable UI state. It also owns the channel
//! handles for talking to the background search worker, which means the
//! UI thread never blocks on the index — it just dispatches generation-
//! tagged commands and discards stale results.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::Backend;
use veles_core::types::{SearchMode, SearchResult};

use crate::tui::search::{ResultKind, SearchDone, WorkerCmd, WorkerMsg};

/// Wait this long after a query change before kicking the worker.
/// Veles searches finish in tens of ms even on large repos, so this can
/// be tiny — just enough to coalesce a fast keystroke burst.
const DEBOUNCE_MS: u64 = 20;

/// How many results to fetch from the worker. The viewport rarely shows
/// more than ~30 rows so 50 is plenty of overscroll.
const TOP_K: usize = 50;

/// Cap how often the event loop wakes up when nothing is happening.
const TICK_TIMEOUT_MS: u64 = 250;

/// LRU bound for cached file contents used to render previews.
const PREVIEW_FILE_CACHE: usize = 8;

/// What kind of result list is currently being displayed.
pub enum ResultsKind {
    Query { query: String },
    Related { anchor: String },
}

/// Action to perform after the TUI exits cleanly.
pub enum ExitAction {
    /// Print `path:line` to stdout. Pipe-friendly:
    ///   `$EDITOR $(veles tui)` or `veles tui | xargs -r $EDITOR`.
    Print(String),
    /// Spawn `$EDITOR` (or `$VISUAL`) on the selected file:line.
    Editor { file: PathBuf, line: usize },
}

pub struct App {
    pub repo_path: PathBuf,
    pub total_chunks: usize,
    pub total_files: usize,

    // Query input.
    pub query: String,
    /// Cursor position counted in chars, not bytes — handles Cyrillic / CJK
    /// queries cleanly. Always in `0..=query.chars().count()`.
    pub cursor_chars: usize,

    // Search dispatch state.
    pub mode: SearchMode,
    pub seq: u64,
    pub displayed_seq: u64,
    pub searching: bool,
    pub pending_query: bool,
    pub next_dispatch_at: Option<Instant>,

    // Results.
    pub results: Vec<SearchResult>,
    pub results_kind: ResultsKind,
    pub elapsed_ms: u64,
    pub selected: usize,
    pub list_offset: usize,

    // Preview file cache (LRU). Files are usually small; we cache their
    // line splits so navigating between results in the same file is free.
    pub current_preview: Option<Arc<Vec<String>>>,
    current_preview_path: Option<String>,
    preview_cache: HashMap<String, Arc<Vec<String>>>,
    preview_order: VecDeque<String>,

    // UI state.
    pub help_open: bool,
    pub spinner_tick: u64,
    pub status_msg: Option<(String, Instant)>,
    pub quit: bool,
    pub exit_action: Option<ExitAction>,

    // Channels.
    cmd_tx: Sender<WorkerCmd>,
    msg_rx: Receiver<WorkerMsg>,
}

impl App {
    pub fn new(
        repo_path: PathBuf,
        total_files: usize,
        total_chunks: usize,
        cmd_tx: Sender<WorkerCmd>,
        msg_rx: Receiver<WorkerMsg>,
    ) -> Self {
        Self {
            repo_path,
            total_files,
            total_chunks,
            query: String::new(),
            cursor_chars: 0,
            mode: SearchMode::Hybrid,
            seq: 0,
            displayed_seq: 0,
            searching: false,
            pending_query: false,
            next_dispatch_at: None,
            results: Vec::new(),
            results_kind: ResultsKind::Query {
                query: String::new(),
            },
            elapsed_ms: 0,
            selected: 0,
            list_offset: 0,
            current_preview: None,
            current_preview_path: None,
            preview_cache: HashMap::new(),
            preview_order: VecDeque::new(),
            help_open: false,
            spinner_tick: 0,
            status_msg: None,
            quit: false,
            exit_action: None,
            cmd_tx,
            msg_rx,
        }
    }

    pub fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        loop {
            // 1. Drain worker messages without blocking.
            loop {
                match self.msg_rx.try_recv() {
                    Ok(msg) => self.handle_worker_msg(msg),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.quit = true;
                        break;
                    }
                }
            }

            // 2. Maybe kick off a debounced search.
            if let Some(cmd) = self.maybe_dispatch_query() {
                let _ = self.cmd_tx.send(cmd);
            }

            // 3. Refresh preview cache for current selection.
            self.refresh_preview();

            // 4. Render.
            terminal.draw(|f| crate::tui::ui::render(f, self))?;

            // 5. Wait for input or short tick.
            let timeout = self.pick_timeout();
            if event::poll(timeout)? {
                match event::read()? {
                    Event::Key(k) if k.kind == KeyEventKind::Press => self.handle_key(k),
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }

            self.spinner_tick = self.spinner_tick.wrapping_add(1);
            if self.quit {
                break;
            }
        }
        Ok(())
    }

    fn pick_timeout(&self) -> Duration {
        let mut ms = TICK_TIMEOUT_MS;
        if let Some(t) = self.next_dispatch_at {
            let now = Instant::now();
            if t <= now {
                ms = 0;
            } else {
                ms = ms.min((t - now).as_millis() as u64);
            }
        }
        if self.searching {
            ms = ms.min(80);
        }
        if let Some((_, until)) = &self.status_msg {
            let now = Instant::now();
            if *until > now {
                ms = ms.min((*until - now).as_millis() as u64);
            }
        }
        Duration::from_millis(ms)
    }

    // ── Worker plumbing ──────────────────────────────────────────────

    fn handle_worker_msg(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::SearchDone(done) => self.apply_results(done),
        }
    }

    fn apply_results(&mut self, done: SearchDone) {
        // Discard out-of-order completions — the channel preserves order
        // but a Related dispatch interleaved with a Query can race, and
        // the seq counter is the source of truth.
        if done.seq < self.displayed_seq {
            return;
        }
        self.displayed_seq = done.seq;
        self.results = done.results;
        self.elapsed_ms = done.elapsed_ms;
        self.selected = 0;
        self.list_offset = 0;
        self.results_kind = match done.kind {
            ResultKind::Query => ResultsKind::Query { query: done.query },
            ResultKind::Related { anchor } => ResultsKind::Related { anchor },
        };
        if done.seq == self.seq {
            self.searching = false;
        }
    }

    fn maybe_dispatch_query(&mut self) -> Option<WorkerCmd> {
        if !self.pending_query {
            return None;
        }
        if let Some(t) = self.next_dispatch_at
            && Instant::now() < t
        {
            return None;
        }
        self.pending_query = false;
        self.next_dispatch_at = None;
        self.seq += 1;
        self.searching = true;
        Some(WorkerCmd::Search {
            seq: self.seq,
            query: self.query.clone(),
            mode: self.mode,
            top_k: TOP_K,
        })
    }

    fn mark_query_changed(&mut self) {
        self.pending_query = true;
        self.next_dispatch_at = Some(Instant::now() + Duration::from_millis(DEBOUNCE_MS));
    }

    fn dispatch_related(&mut self) {
        let Some(sel) = self.results.get(self.selected) else {
            return;
        };
        let chunk = sel.chunk.clone();
        self.seq += 1;
        self.searching = true;
        // Cancel any pending query dispatch — Ctrl-R is a clear user intent.
        self.pending_query = false;
        self.next_dispatch_at = None;
        let _ = self.cmd_tx.send(WorkerCmd::Related {
            seq: self.seq,
            source: Box::new(chunk),
            top_k: TOP_K,
        });
    }

    // ── Preview cache ────────────────────────────────────────────────

    fn refresh_preview(&mut self) {
        let path = self
            .results
            .get(self.selected)
            .map(|r| r.chunk.file_path.clone());
        if path == self.current_preview_path {
            return;
        }
        self.current_preview_path = path.clone();
        self.current_preview = path.and_then(|p| self.load_file(&p));
    }

    fn load_file(&mut self, rel: &str) -> Option<Arc<Vec<String>>> {
        if let Some(arc) = self.preview_cache.get(rel) {
            return Some(arc.clone());
        }
        let abs = self.repo_path.join(rel);
        let raw = std::fs::read_to_string(&abs).ok()?;
        let lines: Vec<String> = raw.lines().map(|s| s.to_string()).collect();
        let arc = Arc::new(lines);
        self.preview_cache.insert(rel.to_string(), arc.clone());
        self.preview_order.push_back(rel.to_string());
        while self.preview_order.len() > PREVIEW_FILE_CACHE {
            if let Some(old) = self.preview_order.pop_front() {
                self.preview_cache.remove(&old);
            }
        }
        Some(arc)
    }

    // ── Key handling ─────────────────────────────────────────────────

    fn handle_key(&mut self, k: KeyEvent) {
        if self.help_open {
            self.help_open = false;
            return;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Esc => self.quit = true,
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Char('g') if ctrl => self.quit = true,

            KeyCode::Enter => self.action_print(),
            KeyCode::Char('o') if ctrl => self.action_editor(),
            KeyCode::Char('r') if ctrl => self.dispatch_related(),

            // Result navigation.
            KeyCode::Up => self.move_selected(-1),
            KeyCode::Down => self.move_selected(1),
            KeyCode::Char('p') if ctrl => self.move_selected(-1),
            KeyCode::Char('n') if ctrl => self.move_selected(1),
            KeyCode::PageUp => self.move_selected(-10),
            KeyCode::PageDown => self.move_selected(10),

            // Search mode.
            KeyCode::Tab => self.cycle_mode(false),
            KeyCode::BackTab => self.cycle_mode(true),

            // Help overlay.
            KeyCode::Char('?') if !ctrl => self.help_open = true,

            // Query editing.
            KeyCode::Backspace => self.backspace(),
            KeyCode::Delete => self.delete_forward(),
            KeyCode::Left => self.cursor_left(ctrl),
            KeyCode::Right => self.cursor_right(ctrl),
            KeyCode::Home => self.cursor_chars = 0,
            KeyCode::End => self.cursor_chars = self.query.chars().count(),
            KeyCode::Char('a') if ctrl => self.cursor_chars = 0,
            KeyCode::Char('e') if ctrl => self.cursor_chars = self.query.chars().count(),
            KeyCode::Char('u') if ctrl => self.clear_query(),
            KeyCode::Char('w') if ctrl => self.delete_word_back(),
            KeyCode::Char('k') if ctrl => self.kill_to_end(),

            KeyCode::Char(c) if !ctrl => self.insert_char(c),
            _ => {}
        }
    }

    fn move_selected(&mut self, delta: i32) {
        if self.results.is_empty() {
            return;
        }
        let max = self.results.len() as i32 - 1;
        let new = (self.selected as i32 + delta).clamp(0, max);
        self.selected = new as usize;
    }

    fn cycle_mode(&mut self, reverse: bool) {
        self.mode = if reverse {
            match self.mode {
                SearchMode::Hybrid => SearchMode::Semantic,
                SearchMode::Semantic => SearchMode::Bm25,
                SearchMode::Bm25 => SearchMode::Hybrid,
            }
        } else {
            match self.mode {
                SearchMode::Hybrid => SearchMode::Bm25,
                SearchMode::Bm25 => SearchMode::Semantic,
                SearchMode::Semantic => SearchMode::Hybrid,
            }
        };
        // Re-run with the new mode immediately.
        self.mark_query_changed();
        self.next_dispatch_at = Some(Instant::now()); // skip debounce on Tab
    }

    fn insert_char(&mut self, c: char) {
        let byte = char_idx_to_byte(&self.query, self.cursor_chars);
        self.query.insert(byte, c);
        self.cursor_chars += 1;
        self.mark_query_changed();
    }

    fn backspace(&mut self) {
        if self.cursor_chars == 0 {
            return;
        }
        let to = char_idx_to_byte(&self.query, self.cursor_chars);
        let from = char_idx_to_byte(&self.query, self.cursor_chars - 1);
        self.query.replace_range(from..to, "");
        self.cursor_chars -= 1;
        self.mark_query_changed();
    }

    fn delete_forward(&mut self) {
        let total = self.query.chars().count();
        if self.cursor_chars >= total {
            return;
        }
        let from = char_idx_to_byte(&self.query, self.cursor_chars);
        let to = char_idx_to_byte(&self.query, self.cursor_chars + 1);
        self.query.replace_range(from..to, "");
        self.mark_query_changed();
    }

    fn cursor_left(&mut self, by_word: bool) {
        if self.cursor_chars == 0 {
            return;
        }
        if by_word {
            let chars: Vec<char> = self.query.chars().collect();
            let mut i = self.cursor_chars;
            while i > 0 && chars[i - 1].is_whitespace() {
                i -= 1;
            }
            while i > 0 && !chars[i - 1].is_whitespace() {
                i -= 1;
            }
            self.cursor_chars = i;
        } else {
            self.cursor_chars -= 1;
        }
    }

    fn cursor_right(&mut self, by_word: bool) {
        let total = self.query.chars().count();
        if self.cursor_chars >= total {
            return;
        }
        if by_word {
            let chars: Vec<char> = self.query.chars().collect();
            let mut i = self.cursor_chars;
            while i < total && chars[i].is_whitespace() {
                i += 1;
            }
            while i < total && !chars[i].is_whitespace() {
                i += 1;
            }
            self.cursor_chars = i;
        } else {
            self.cursor_chars += 1;
        }
    }

    fn clear_query(&mut self) {
        self.query.clear();
        self.cursor_chars = 0;
        self.mark_query_changed();
    }

    fn delete_word_back(&mut self) {
        if self.cursor_chars == 0 {
            return;
        }
        let chars: Vec<char> = self.query.chars().collect();
        let mut i = self.cursor_chars;
        while i > 0 && chars[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].is_whitespace() {
            i -= 1;
        }
        let from = char_idx_to_byte(&self.query, i);
        let to = char_idx_to_byte(&self.query, self.cursor_chars);
        self.query.replace_range(from..to, "");
        self.cursor_chars = i;
        self.mark_query_changed();
    }

    fn kill_to_end(&mut self) {
        let total = self.query.chars().count();
        if self.cursor_chars >= total {
            return;
        }
        let from = char_idx_to_byte(&self.query, self.cursor_chars);
        self.query.truncate(from);
        self.mark_query_changed();
    }

    fn action_print(&mut self) {
        let Some(sel) = self.results.get(self.selected) else {
            return;
        };
        let s = format!("{}:{}", sel.chunk.file_path, sel.chunk.start_line);
        self.exit_action = Some(ExitAction::Print(s));
        self.quit = true;
    }

    fn action_editor(&mut self) {
        let Some(sel) = self.results.get(self.selected) else {
            return;
        };
        let abs = self.repo_path.join(&sel.chunk.file_path);
        self.exit_action = Some(ExitAction::Editor {
            file: abs,
            line: sel.chunk.start_line,
        });
        self.quit = true;
    }

    /// Query terms split on whitespace, used for highlighting in the
    /// preview pane. Empty for the Related view (we don't have a query).
    pub fn query_terms(&self) -> Vec<String> {
        match &self.results_kind {
            ResultsKind::Query { query, .. } => query
                .split_whitespace()
                .filter(|t| !t.is_empty())
                .map(|s| s.to_string())
                .collect(),
            ResultsKind::Related { .. } => Vec::new(),
        }
    }
}

fn char_idx_to_byte(s: &str, idx: usize) -> usize {
    s.char_indices().nth(idx).map(|(b, _)| b).unwrap_or(s.len())
}
