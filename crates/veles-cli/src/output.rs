//! Stdout sinks for rendered formatter output.
//!
//! Each helper handles the empty-result case (using format-aware empty
//! messaging) and avoids double newlines for line-oriented formats.

use veles_core::symbols::Symbol;
use veles_core::types::SearchResult;

use crate::format::{self, OutputFormat};

/// Render search results in the chosen format and write to stdout.
///
/// `symbols`, when supplied, lets the `pretty` and `compact` formats
/// append a scope label per hit. Pass `None` for renderers that don't
/// have access to the index's symbol table.
pub fn emit_results(
    format: OutputFormat,
    header: &str,
    what: &str,
    results: &[SearchResult],
    symbols: Option<&[Symbol]>,
) {
    if results.is_empty() {
        emit_empty(format, what);
        return;
    }
    let rendered = format::render(format, header, results, symbols);
    write_rendered(rendered);
}

/// Render symbols in the chosen format and write to stdout.
pub fn emit_symbols(format: OutputFormat, header: &str, what: &str, symbols: &[&Symbol]) {
    if symbols.is_empty() {
        emit_empty(format, what);
        return;
    }
    let rendered = format::render_symbols(format, header, symbols);
    write_rendered(rendered);
}

fn emit_empty(format: OutputFormat, what: &str) {
    let msg = format::empty_message(format, what);
    if !msg.is_empty() {
        println!("{msg}");
    }
}

fn write_rendered(rendered: String) {
    // `render_pretty` produces a trailing newline naturally via its joined
    // lines; line-oriented formats also self-terminate. Use `print!` so we
    // don't double up.
    if rendered.ends_with('\n') {
        print!("{rendered}");
    } else {
        println!("{rendered}");
    }
}
