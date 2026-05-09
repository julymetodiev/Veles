//! `veles-mcp` — Model Context Protocol server for [Veles] code search.
//!
//! Speaks JSON-RPC 2.0 over stdio so AI agents (Claude Desktop, Cursor,
//! anything else MCP-aware) can search a codebase without leaving their
//! tool-call loop. Indexes are cached in process across calls, so
//! repeat queries against the same repo skip the re-index cost.
//!
//! # Tools exposed to the agent
//!
//! - `search` — natural-language or code query against a local
//!   directory or `https://` git URL. Optional `lang` / `path` /
//!   `exclude` glob filters and a `min_score` threshold narrow
//!   noisy queries.
//! - `defs` — every tree-sitter definition with the given name
//!   (Rust, Python, JavaScript, TypeScript, Go). More precise than
//!   `search` when the symbol name is known.
//! - `symbols` — the tree-sitter outline of a single file. A cheap
//!   alternative to reading the whole file when only the structure
//!   matters.
//! - `refs` — definitions plus BM25 hits for a symbol name. One call
//!   to answer both "where is X defined" and "where is X used".
//! - `stats` — file count, chunk count, model metadata, and per-language
//!   chunk breakdown for the indexed repo.
//! - `update` — incrementally refresh the index against the current
//!   state of disk (re-embed only fingerprint-changed files) and
//!   persist to `<repo>/.veles/`. Local repos only.
//! - `find_related` — semantically similar chunks for a `(file, line)`
//!   pair returned by an earlier `search`.
//!
//! The supported transport is line-delimited JSON-RPC on stdin/stdout
//! per the [MCP 2024-11-05] revision, with `tools/list` and
//! `tools/call` as the only entry points beyond `initialize`.
//!
//! # Running the server
//!
//! From code:
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! let model = veles_core::model::load_model(None)?;
//! veles_mcp::McpServer::new(model).run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! From the CLI (the default if no subcommand is given):
//!
//! ```sh
//! veles serve-mcp
//! veles            # equivalent — bare `veles` starts MCP on a piped stdin
//! ```
//!
//! [Veles]: https://github.com/julymetodiev/Veles
//! [MCP 2024-11-05]: https://modelcontextprotocol.io/specification/2024-11-05

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use veles_core::VelesIndex;
use veles_core::filter;
use veles_core::symbols::Symbol;
use veles_core::types::SearchMode;

// ── JSON-RPC Types ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// ── MCP Tool Definitions ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "search".into(),
            description: "Search a codebase with a natural-language or code query. Pass `repo` as a local directory path or an https:// git URL. The index is cached after the first call, so repeat queries are fast. Use `lang`, `path`, `exclude`, and `min_score` to narrow noisy queries.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code query."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL to index and search."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "bm25"],
                        "description": "Search mode. 'hybrid' is best for most queries."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": 5
                    },
                    "lang": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict results to these languages (e.g. ['rust', 'python'])."
                    },
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to include (e.g. ['src/**/*.rs'])."
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to exclude (e.g. ['tests/**', '**/legacy/**'])."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Drop results whose score is below this threshold."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["default", "paths"],
                        "description": "'default' returns scored, fenced code blocks with the enclosing scope. 'paths' returns just `path:start-end` per line — token-cheap shortlist for downstream processing."
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "defs".into(),
            description: "Find every tree-sitter definition with the given name (functions, structs, classes, ...) across the indexed repo. More precise than `search` when you already know the symbol name. Supported languages: Rust, Python, JavaScript, TypeScript, Go.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Exact symbol name to look up."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    },
                    "lang": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict results to these languages."
                    },
                    "kind": {
                        "type": "string",
                        "description": "Restrict to a single symbol kind (e.g. 'function', 'struct', 'class', 'enum', 'trait', 'method')."
                    }
                },
                "required": ["name"]
            }),
        },
        Tool {
            name: "symbols".into(),
            description: "List every tree-sitter definition in a single file (functions, structs, classes, methods, ...). Cheap alternative to reading a whole file when you only need its outline. Supported languages: Rust, Python, JavaScript, TypeScript, Go.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file as stored in the index (relative to `repo`)."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    }
                },
                "required": ["file_path"]
            }),
        },
        Tool {
            name: "refs".into(),
            description: "Find references to a symbol — its tree-sitter definitions plus BM25 hits in chunks that mention the name. Use when you want both 'where is X defined' and 'where is X used' in one call.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Symbol name to look up."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of BM25 hits to return (definitions are always included).",
                        "default": 10
                    },
                    "format": {
                        "type": "string",
                        "enum": ["default", "paths"],
                        "description": "'default' splits into a Definitions section and a BM25 section with code blocks. 'paths' flattens both into a single `path:line` / `path:start-end` list."
                    }
                },
                "required": ["name"]
            }),
        },
        Tool {
            name: "stats".into(),
            description: "Show what the index knows about a repo: total files and chunks, model and embedding dim, plus a per-language chunk breakdown. Useful for self-diagnosis when search results look thin.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    }
                }
            }),
        },
        Tool {
            name: "update".into(),
            description: "Refresh the index for `repo` against the current state of disk: re-embed only files whose (size, mtime) fingerprint changed, drop removed files, pick up new ones, and persist the result under `<repo>/.veles/`. Cheap after small edits. Not supported for https:// git URLs (re-run `search` to re-clone instead).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Local directory path."
                    }
                }
            }),
        },
        Tool {
            name: "find_related".into(),
            description: "Find code chunks semantically similar to a specific location in a file. Use after `search` to explore related implementations or callers.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file as stored in the index (use file_path from a search result)."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number (1-indexed)."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar chunks to return.",
                        "default": 5
                    },
                    "format": {
                        "type": "string",
                        "enum": ["default", "paths"],
                        "description": "'default' returns scored code blocks with the enclosing scope; 'paths' returns just `path:start-end` per line."
                    }
                },
                "required": ["file_path", "line"]
            }),
        },
    ]
}

// ── Index Cache ───────────────────────────────────────────────────────────

const CACHE_MAX_SIZE: usize = 10;

struct IndexCache {
    entries: HashMap<String, VelesIndex>,
    model: model2vec_rs::model::StaticModel,
}

impl IndexCache {
    fn new(model: model2vec_rs::model::StaticModel) -> Self {
        Self {
            entries: HashMap::new(),
            model,
        }
    }

    fn get_or_index(&mut self, repo: &str, include_text_files: bool) -> Result<&VelesIndex> {
        if self.entries.contains_key(repo) {
            return Ok(self.entries.get(repo).unwrap());
        }

        // Evict LRU if at capacity.
        if self.entries.len() >= CACHE_MAX_SIZE {
            // Simple eviction: remove the first entry.
            if let Some(key) = self.entries.keys().next().cloned() {
                self.entries.remove(&key);
            }
        }

        let model = self.model.clone();
        let path = Path::new(repo);
        let index = if path.is_dir() {
            VelesIndex::from_path(path, Some(model), None, include_text_files)?
        } else if repo.starts_with("https://") || repo.starts_with("http://") {
            VelesIndex::from_git(repo, None, Some(model), include_text_files)?
        } else {
            bail!("Invalid repo: must be a local directory or https:// URL");
        };

        self.entries.insert(repo.to_string(), index);
        Ok(self.entries.get(repo).unwrap())
    }

    /// Like [`Self::get_or_index`] but returns an exclusive borrow so the
    /// caller can mutate the index (e.g. via `update_from_path`).
    fn get_or_index_mut(
        &mut self,
        repo: &str,
        include_text_files: bool,
    ) -> Result<&mut VelesIndex> {
        // Reuse the immutable path to populate the cache, then re-borrow mut.
        let _ = self.get_or_index(repo, include_text_files)?;
        Ok(self.entries.get_mut(repo).unwrap())
    }
}

// ── MCP Server ───────────────────────────────────────────────────────────

pub struct McpServer {
    cache: Arc<Mutex<IndexCache>>,
    server_info: Value,
}

impl McpServer {
    pub fn new(model: model2vec_rs::model::StaticModel) -> Self {
        Self {
            cache: Arc::new(Mutex::new(IndexCache::new(model))),
            server_info: json!({
                "name": "veles",
                "version": env!("CARGO_PKG_VERSION"),
            }),
        }
    }

    /// Run the MCP server, reading JSON-RPC from stdin and writing to stdout.
    pub async fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        // Send an initialization notification to signal readiness.
        // MCP servers are expected to just respond to requests.

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {e}"),
                        }),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&resp)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(request).await;
            let response_str = serde_json::to_string(&response)?;
            writeln!(stdout, "{response_str}")?;
            stdout.flush()?;
        }

        Ok(())
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone();

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params),
            "notifications/initialized" => {
                // Client confirmed initialization — no response needed for notifications.
                return JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: Some(Value::Null),
                    error: None,
                };
            }
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(request.params).await,
            _ => Err(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
            }),
        };

        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(value),
                error: None,
            },
            Err(error) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(error),
            },
        }
    }

    fn handle_initialize(&self, _params: Option<Value>) -> Result<Value, JsonRpcError> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": self.server_info,
        }))
    }

    fn handle_tools_list(&self) -> Result<Value, JsonRpcError> {
        Ok(json!({
            "tools": tools()
        }))
    }

    async fn handle_tools_call(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let params = params.ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing params".into(),
        })?;

        let tool_name = params["name"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing tool name".into(),
        })?;

        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match tool_name {
            "search" => self.handle_search(arguments).await,
            "find_related" => self.handle_find_related(arguments).await,
            "defs" => self.handle_defs(arguments).await,
            "symbols" => self.handle_symbols(arguments).await,
            "refs" => self.handle_refs(arguments).await,
            "stats" => self.handle_stats(arguments).await,
            "update" => self.handle_update(arguments).await,
            _ => Err(JsonRpcError {
                code: -32602,
                message: format!("Unknown tool: {tool_name}"),
            }),
        }
    }

    async fn handle_search(&self, args: Value) -> Result<Value, JsonRpcError> {
        let query = args["query"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'query' parameter".into(),
        })?;

        let repo = args["repo"].as_str().unwrap_or(".");

        let mode_str = args["mode"].as_str().unwrap_or("hybrid");
        let mode = mode_str.parse::<SearchMode>().unwrap_or(SearchMode::Hybrid);

        let top_k = args["top_k"].as_u64().unwrap_or(5) as usize;

        let lang = string_array(&args, "lang");
        let path_globs = string_array(&args, "path");
        let exclude_globs = string_array(&args, "exclude");
        let min_score = args["min_score"].as_f64();
        let format = args["format"].as_str().unwrap_or("default");

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let glob_paths =
            filter::resolve_path_filter(index, &path_globs, &exclude_globs).map_err(|e| {
                JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }
            })?;
        let lang_slice: Option<&[String]> = if lang.is_empty() { None } else { Some(&lang) };
        let path_slice: Option<&[String]> = glob_paths.as_deref();

        let mut results = index.search(query, top_k, mode, None, lang_slice, path_slice);
        if let Some(threshold) = min_score {
            results.retain(|r| r.score >= threshold);
        }

        if results.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": "No results found."}]
            }));
        }

        let text = if format == "paths" {
            format_results_paths(&results)
        } else {
            let header = format!("Search results for: {query:?} (mode={mode_str})");
            format_results(&header, &results, Some(index.symbols()))
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_defs(&self, args: Value) -> Result<Value, JsonRpcError> {
        let name = args["name"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'name' parameter".into(),
        })?;

        let repo = args["repo"].as_str().unwrap_or(".");
        let lang = string_array(&args, "lang");
        let kind_filter = args["kind"].as_str().map(|s| s.to_ascii_lowercase());

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let mut hits: Vec<&Symbol> = index
            .symbols()
            .iter()
            .filter(|s| s.name == name)
            .filter(|s| lang.is_empty() || lang.iter().any(|l| l == &s.language))
            .filter(|s| match &kind_filter {
                Some(k) => s.kind.as_str() == k,
                None => true,
            })
            .collect();
        hits.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.start_line.cmp(&b.start_line))
        });

        if hits.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No definitions named {name:?} found.")}]
            }));
        }

        let header = format!("Definitions of {name:?}");
        let text = format_symbols(&header, &hits);

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_symbols(&self, args: Value) -> Result<Value, JsonRpcError> {
        let file_path = args["file_path"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'file_path' parameter".into(),
        })?;
        let repo = args["repo"].as_str().unwrap_or(".");

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let mut hits = index.symbols_for_file(file_path);
        hits.sort_by_key(|s| s.start_line);

        if hits.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No tree-sitter symbols found in {file_path}. The file may be in an unsupported language (supported: rust, python, javascript, typescript, go) or not part of the indexed repo.")}]
            }));
        }

        let header = format!("Symbols in {file_path}");
        let text = format_symbols(&header, &hits);

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_refs(&self, args: Value) -> Result<Value, JsonRpcError> {
        let name = args["name"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'name' parameter".into(),
        })?;
        let repo = args["repo"].as_str().unwrap_or(".");
        let top_k = args["top_k"].as_u64().unwrap_or(10) as usize;
        let format = args["format"].as_str().unwrap_or("default");

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let mut defs: Vec<&Symbol> = index.symbols().iter().filter(|s| s.name == name).collect();
        defs.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.start_line.cmp(&b.start_line))
        });

        let bm25_hits = index.search(name, top_k, SearchMode::Bm25, None, None, None);

        if defs.is_empty() && bm25_hits.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No definitions or BM25 hits found for {name:?}.")}]
            }));
        }

        let text = if format == "paths" {
            // Flat list: defs as `path:line`, BM25 hits as `path:start-end`.
            let mut lines: Vec<String> = Vec::with_capacity(defs.len() + bm25_hits.len());
            for s in &defs {
                lines.push(format!("{}:{}", s.file_path, s.start_line));
            }
            for r in &bm25_hits {
                lines.push(r.chunk.location());
            }
            lines.join("\n")
        } else {
            let mut lines: Vec<String> = vec![format!("References to {name:?}"), String::new()];
            if defs.is_empty() {
                lines.push("## Definitions".to_string());
                lines.push("(none)".to_string());
            } else {
                lines.push("## Definitions".to_string());
                for s in &defs {
                    lines.push(format!(
                        "- {kind} {name} ({lang}) — {file}:{line}",
                        kind = s.kind.as_str(),
                        name = s.name,
                        lang = s.language,
                        file = s.file_path,
                        line = s.start_line,
                    ));
                }
            }
            lines.push(String::new());
            if bm25_hits.is_empty() {
                lines.push("## Other matches (BM25)".to_string());
                lines.push("(none)".to_string());
            } else {
                let header = format!("## Other matches (BM25) — {} hit(s)", bm25_hits.len());
                lines.push(format_results(&header, &bm25_hits, Some(index.symbols())));
            }
            lines.join("\n")
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_stats(&self, args: Value) -> Result<Value, JsonRpcError> {
        let repo = args["repo"].as_str().unwrap_or(".");

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let stats = index.stats();
        let manifest = index.manifest();

        let mut lines: Vec<String> = vec![format!("Index for {repo}"), String::new()];
        lines.push(format!("Files:         {}", stats.indexed_files));
        lines.push(format!("Chunks:        {}", stats.total_chunks));
        if let Some(m) = manifest {
            lines.push(format!("Model:         {}", m.model_name));
            lines.push(format!("Embedding dim: {}", m.embedding_dim));
            lines.push(format!("Format:        v{}", m.format_version));
            lines.push(format!("Text files:    {}", m.include_text_files));
        }

        if !stats.languages.is_empty() {
            lines.push(String::new());
            lines.push("Languages (chunks):".to_string());
            let mut langs: Vec<(&String, &usize)> = stats.languages.iter().collect();
            langs.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let width = langs.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
            for (lang, count) in langs {
                lines.push(format!("  {lang:<width$}  {count}"));
            }
        }

        Ok(json!({
            "content": [{"type": "text", "text": lines.join("\n")}]
        }))
    }

    async fn handle_update(&self, args: Value) -> Result<Value, JsonRpcError> {
        let repo = args["repo"].as_str().unwrap_or(".");
        if repo.starts_with("https://") || repo.starts_with("http://") {
            return Err(JsonRpcError {
                code: -32000,
                message: "Cannot update a remote git URL — re-run `search` against it to re-clone."
                    .into(),
            });
        }
        let path = Path::new(repo);
        if !path.is_dir() {
            return Err(JsonRpcError {
                code: -32000,
                message: format!("Path is not a directory: {repo}"),
            });
        }
        let path_buf = path.to_path_buf();

        let mut cache = self.cache.lock().await;
        let index = cache
            .get_or_index_mut(repo, false)
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;

        let started = std::time::Instant::now();
        let report = index
            .update_from_path(&path_buf)
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let secs = started.elapsed().as_secs_f64();

        let text = if report.is_noop() {
            format!(
                "Index is up to date in {secs:.2}s ({} chunks, no file changes detected).",
                report.total_chunks
            )
        } else {
            // Persist only when something actually changed.
            index.save(&path_buf).map_err(|e| JsonRpcError {
                code: -32000,
                message: format!("update applied in memory but save failed: {e}"),
            })?;
            format!(
                "Updated in {secs:.2}s — +{} added, ~{} modified, -{} removed (kept {} chunks, embedded {} new, total {}). Persisted to {}/.veles.",
                report.added_files,
                report.modified_files,
                report.removed_files,
                report.kept_chunks,
                report.new_chunks,
                report.total_chunks,
                repo,
            )
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_find_related(&self, args: Value) -> Result<Value, JsonRpcError> {
        let file_path = args["file_path"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'file_path' parameter".into(),
        })?;

        let line = args["line"].as_u64().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'line' parameter".into(),
        })? as usize;

        let repo = args["repo"].as_str().unwrap_or(".");
        let top_k = args["top_k"].as_u64().unwrap_or(5) as usize;
        let format = args["format"].as_str().unwrap_or("default");

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let chunk = index
            .resolve_chunk(file_path, line)
            .ok_or_else(|| JsonRpcError {
                code: -32000,
                message: format!("No chunk found at {file_path}:{line}"),
            })?
            .clone();

        let results = index.find_related(&chunk, top_k);

        if results.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No related chunks found for {file_path}:{line}")}]
            }));
        }

        let text = if format == "paths" {
            format_results_paths(&results)
        } else {
            let header = format!("Chunks related to {file_path}:{line}");
            format_results(&header, &results, Some(index.symbols()))
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }
}

/// Pull a `Vec<String>` from a JSON arg that's either an array of strings or absent.
/// A single string value is also accepted and wrapped in a one-element vec, so
/// callers that send `"lang": "rust"` instead of `"lang": ["rust"]` still work.
fn string_array(args: &Value, key: &str) -> Vec<String> {
    match args.get(key) {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Format symbol hits as `kind name (lang) — file:line` lines under a header.
fn format_symbols(header: &str, symbols: &[&Symbol]) -> String {
    let mut lines: Vec<String> = vec![header.to_string(), String::new()];
    for s in symbols {
        lines.push(format!(
            "- {kind} {name} ({lang}) — {file}:{line}",
            kind = s.kind.as_str(),
            name = s.name,
            lang = s.language,
            file = s.file_path,
            line = s.start_line,
        ));
    }
    lines.join("\n")
}

/// Pick a short scope label for a chunk so an agent can route on the
/// result header without reading the body.
///
/// Two-tier heuristic:
/// 1. If any symbols *start* inside the chunk, the chunk is showing
///    those definitions — return ``defines `name` ``.
/// 2. Else find the most specific symbol whose range strictly contains
///    `chunk.start_line` (the chunk is mid-body) — return ``in `name` ``.
///
/// Returns `None` for chunks that neither define nor live inside any
/// tree-sitter-recognised symbol (typical for module-level prelude
/// before the first definition, or files in unsupported languages).
fn chunk_scope_label(
    symbols: &[veles_core::symbols::Symbol],
    chunk: &veles_core::types::Chunk,
) -> Option<String> {
    let same_file = || symbols.iter().filter(|s| s.file_path == chunk.file_path);

    let defined: Vec<&veles_core::symbols::Symbol> = same_file()
        .filter(|s| s.start_line >= chunk.start_line && s.start_line <= chunk.end_line)
        .collect();
    if let Some(first) = defined.first() {
        return Some(if defined.len() == 1 {
            format!("defines `{}`", first.name)
        } else {
            format!("defines `{}` (+{} more)", first.name, defined.len() - 1)
        });
    }

    same_file()
        .filter(|s| s.start_line < chunk.start_line && chunk.start_line <= s.end_line)
        .min_by_key(|s| s.end_line.saturating_sub(s.start_line))
        .map(|s| format!("in `{}`", s.name))
}

/// Format search results as numbered, fenced code blocks. When `symbols`
/// is `Some`, each header is suffixed with a scope label (e.g.
/// ``` defines `Manifest` ``` or ``` in `fn search_hybrid` ```).
fn format_results(
    header: &str,
    results: &[veles_core::types::SearchResult],
    symbols: Option<&[veles_core::symbols::Symbol]>,
) -> String {
    let mut lines: Vec<String> = vec![header.to_string(), String::new()];
    for (i, r) in results.iter().enumerate() {
        let scope_suffix = symbols
            .and_then(|syms| chunk_scope_label(syms, &r.chunk))
            .map(|label| format!("  {label}"))
            .unwrap_or_default();
        lines.push(format!(
            "## {}. {}  [score={:.3}]{scope_suffix}",
            i + 1,
            r.chunk.location(),
            r.score,
        ));
        lines.push("```".to_string());
        lines.push(r.chunk.content.trim().to_string());
        lines.push("```".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Flat `path:start-end` per line — no header, no score, no chunk body.
/// Optimised for agents that just want a shortlist of files / line ranges
/// to act on.
fn format_results_paths(results: &[veles_core::types::SearchResult]) -> String {
    results
        .iter()
        .map(|r| r.chunk.location())
        .collect::<Vec<_>>()
        .join("\n")
}
