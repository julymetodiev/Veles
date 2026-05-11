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
//! - `list_symbols` — every tree-sitter definition in the index with
//!   optional kind / language / path-glob filters. Exploration tool
//!   for "every struct in crates/foo/" kind of questions.
//! - `refs` — definitions plus BM25 hits for a symbol name. One call
//!   to answer both "where is X defined" and "where is X used".
//! - `find_related` — semantically similar chunks for a `(file, line)`
//!   pair returned by an earlier `search`.
//! - `scope_at` — innermost tree-sitter symbol containing a given
//!   `file:line`. Cheaper than scanning the file's full symbol list.
//! - `files` — distinct file paths in the index with optional
//!   language / path / exclude filters. Orientation tool.
//! - `read` — read a line range from an indexed file (capped at 500
//!   lines, refuses absolute / `..`-escaping paths).
//! - `stats` — file count, chunk count, model metadata, and per-language
//!   chunk breakdown for the indexed repo.
//! - `status` — non-mutating drift check: compare the persisted
//!   manifest against current disk state. Useful before deciding
//!   whether to call `update`.
//! - `update` — incrementally refresh the index against the current
//!   state of disk (re-embed only files whose BLAKE3 content hash
//!   changed; mtime drift on unchanged bytes is a manifest-only
//!   refresh) and persist to `<repo>/.veles/`. Local repos only.
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

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

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
                        "enum": ["default", "paths", "unique_paths"],
                        "description": "'default' returns scored, fenced code blocks with the enclosing scope. 'paths' returns just `path:start-end` per line — token-cheap shortlist for downstream processing. 'unique_paths' collapses multiple hits in the same file to a single `path` line; use it when you want a shortlist of *which files* matter, not *which chunks*."
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
                        "enum": ["default", "paths", "unique_paths"],
                        "description": "'default' splits into a Definitions section and a BM25 section with code blocks. 'paths' flattens both into a single `path:line` / `path:start-end` list. 'unique_paths' collapses everything to one `path` line per file."
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
            description: "Refresh the index for `repo` against the current state of disk: re-embed only files whose content actually changed, drop removed files, pick up new ones, and persist under `<repo>/.veles/`. Files whose mtime drifted but content (BLAKE3) still matches do a manifest-only refresh — no re-embed. Cheap after small edits. Not supported for https:// git URLs (re-run `search` to re-clone instead).".into(),
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
            description: "Find code chunks semantically similar to a specific location in a file. Use after `search` to explore related implementations or callers. Defaults to the source chunk's language; pass `lang` / `path` / `exclude` to override.".into(),
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
                    "lang": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict results to these languages. Overrides the default (same language as the source chunk)."
                    },
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to include (e.g. ['src/**/*.rs'])."
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to exclude (e.g. ['tests/**'])."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["default", "paths", "unique_paths"],
                        "description": "'default' returns scored code blocks with the enclosing scope; 'paths' returns just `path:start-end` per line; 'unique_paths' collapses to one `path` per file."
                    }
                },
                "required": ["file_path", "line"]
            }),
        },
        Tool {
            name: "list_symbols".into(),
            description: "List tree-sitter definitions across the index with optional filters by kind, language, and path globs. Unlike `defs` (which requires an exact name) this is the exploration tool — answer questions like 'every struct in crates/foo/' or 'all public functions in Python files'. Returns `kind name (lang) — file:line` per match, sorted by path then start line.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    },
                    "kind": {
                        "type": "string",
                        "description": "Restrict to a single symbol kind (e.g. 'function', 'struct', 'class', 'enum', 'trait', 'method', 'const')."
                    },
                    "lang": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these languages (e.g. ['rust', 'python'])."
                    },
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to include (e.g. ['crates/**/src/**/*.rs'])."
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to exclude (e.g. ['tests/**'])."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of symbols to return. Default 200; raise for full-repo enumerations.",
                        "default": 200
                    }
                }
            }),
        },
        Tool {
            name: "files".into(),
            description: "List distinct file paths known to the index, with optional language and path-glob filters. Use to orient yourself in an unfamiliar repo or to feed downstream tools that expect a file shortlist. Returns one path per line, sorted alphabetically.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    },
                    "lang": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to files in these languages (matched against the chunk's tree-sitter language tag — 'rust', 'python', 'javascript', 'typescript', 'go', etc.)."
                    },
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to include (e.g. ['src/**/*.rs'])."
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns of paths to exclude."
                    }
                }
            }),
        },
        Tool {
            name: "scope_at".into(),
            description: "Return the innermost tree-sitter symbol whose range contains the given file:line. Use to answer 'which function / struct does this line live in?' without scanning the whole file's symbol list. Returns symbol kind, name, language, and start/end lines.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file as stored in the index (relative to `repo`)."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number (1-indexed)."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path or https:// git URL."
                    }
                },
                "required": ["file_path", "line"]
            }),
        },
        Tool {
            name: "read".into(),
            description: "Read a line range from a file in the indexed repo. Resolves `file_path` against the repo root, refuses paths that escape the repo (no '..' or absolute paths), and caps each call at 500 lines. Use when you already know the location (e.g. from `defs`, `search`, or `scope_at`) and want the actual source bytes. Local repos only.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file relative to the repo root (no leading '/' or '..')."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to include (1-indexed)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to include (1-indexed, inclusive). Capped at start_line + 499."
                    },
                    "repo": {
                        "type": "string",
                        "description": "Local directory path. Git URLs are not supported — clone locally and pass the path."
                    }
                },
                "required": ["file_path", "start_line", "end_line"]
            }),
        },
        Tool {
            name: "status".into(),
            description: "Non-mutating drift check: compare the persisted `.veles/manifest.json` against the current state of disk and report file counts (added / modified / removed). Useful before deciding whether to call `update`. Local repos only.".into(),
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
    ]
}

// ── MCP Server ───────────────────────────────────────────────────────────

pub struct McpServer {
    cache: Arc<veles_core::cache::IndexCache>,
    server_info: Value,
}

impl McpServer {
    pub fn new(model: model2vec_rs::model::StaticModel) -> Self {
        Self {
            cache: Arc::new(veles_core::cache::IndexCache::new(model)),
            server_info: json!({
                "name": "veles",
                "version": env!("CARGO_PKG_VERSION"),
            }),
        }
    }

    /// Run the MCP server, reading JSON-RPC from stdin and writing to stdout.
    ///
    /// Uses tokio's async stdin/stdout so the runtime stays responsive
    /// while we await the parse / dispatch of each request (§4.3 of the
    /// perf plan). The previous sync `BufRead.lines()` loop pinned a
    /// worker thread on `read(2)`; with async I/O the worker is free to
    /// run other tasks while we wait for input.
    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();
        let mut stdout = tokio::io::stdout();

        while let Some(line) = reader.next_line().await? {
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
                    let line = serde_json::to_string(&resp)?;
                    stdout.write_all(line.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    continue;
                }
            };

            let response = self.handle_request(request).await;
            let response_str = serde_json::to_string(&response)?;
            stdout.write_all(response_str.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
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
            "list_symbols" => self.handle_list_symbols(arguments).await,
            "files" => self.handle_files(arguments).await,
            "scope_at" => self.handle_scope_at(arguments).await,
            "read" => self.handle_read(arguments).await,
            "status" => self.handle_status(arguments).await,
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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        let glob_paths =
            filter::resolve_path_filter(&index, &path_globs, &exclude_globs).map_err(|e| {
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

        let text = match format {
            "paths" => format_results_paths(&results),
            "unique_paths" => format_results_unique_paths(&results),
            _ => {
                let header = format!("Search results for: {query:?} (mode={mode_str})");
                format_results(&header, &results, Some(index.symbols()))
            }
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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        let mut defs: Vec<&Symbol> = index.symbols().iter().filter(|s| s.name == name).collect();
        defs.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then(a.start_line.cmp(&b.start_line))
        });

        // Pull a few extra BM25 hits so that dropping chunks that overlap a
        // definition site still leaves the caller with roughly the requested
        // count. Two-thirds extra is enough for most realistic queries
        // without ballooning the round-trip.
        let bm25_overshoot = top_k + (top_k / 2).max(1);
        let bm25_hits: Vec<veles_core::types::SearchResult> = index
            .search(name, bm25_overshoot, SearchMode::Bm25, None, None, None)
            .into_iter()
            .filter(|hit| {
                // Drop hits whose chunk *contains* a definition's start line —
                // those are the same code the def section already pointed at.
                !defs.iter().any(|d| {
                    d.file_path == hit.chunk.file_path
                        && d.start_line >= hit.chunk.start_line
                        && d.start_line <= hit.chunk.end_line
                })
            })
            .take(top_k)
            .collect();

        if defs.is_empty() && bm25_hits.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No definitions or BM25 hits found for {name:?}.")}]
            }));
        }

        let text = if format == "paths" {
            // Flat list, line-precise: defs at their start line, BM25 hits
            // expanded to one `path:line` per word-boundary occurrence of
            // the symbol inside the chunk. Falls back to the chunk range
            // if no word-boundary match lands inside (BM25 sometimes picks
            // chunks via path tokens or partial-stem hits).
            let needle = regex::Regex::new(&format!(r"\b{}\b", regex::escape(name))).ok();
            let mut lines: Vec<String> = Vec::with_capacity(defs.len() + bm25_hits.len());
            for s in &defs {
                lines.push(format!("{}:{}", s.file_path, s.start_line));
            }
            for r in &bm25_hits {
                let mut emitted = 0usize;
                if let Some(ref re) = needle {
                    for (i, line) in r.chunk.content.lines().enumerate() {
                        if re.is_match(line) {
                            lines.push(format!("{}:{}", r.chunk.file_path, r.chunk.start_line + i));
                            emitted += 1;
                        }
                    }
                }
                if emitted == 0 {
                    lines.push(r.chunk.location());
                }
            }
            lines.join("\n")
        } else if format == "unique_paths" {
            // Collapse defs + BM25 hits to one path per file. Defs come
            // first since they're authoritative; BM25 hits then add any
            // additional files where the symbol is referenced.
            let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
            let mut out: Vec<String> = Vec::new();
            for s in &defs {
                if seen.insert(s.file_path.as_str()) {
                    out.push(s.file_path.clone());
                }
            }
            for r in &bm25_hits {
                if seen.insert(r.chunk.file_path.as_str()) {
                    out.push(r.chunk.file_path.clone());
                }
            }
            out.join("\n")
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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

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

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let mut index = index_arc.write().await;

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
            // Persist only when something actually changed (chunk-level edits
            // or a manifest-only mtime refresh).
            index.save(&path_buf).map_err(|e| JsonRpcError {
                code: -32000,
                message: format!("update applied in memory but save failed: {e}"),
            })?;
            if report.added_files + report.modified_files + report.removed_files == 0 {
                format!(
                    "Refreshed manifest in {secs:.2}s — {} file(s) had stale mtime but unchanged content (kept {} chunks). Persisted to {}/.veles.",
                    report.mtime_refreshed_files, report.total_chunks, repo,
                )
            } else {
                format!(
                    "Updated in {secs:.2}s — +{} added, ~{} modified, -{} removed, ⟳{} mtime-only (kept {} chunks, embedded {} new, total {}). Persisted to {}/.veles.",
                    report.added_files,
                    report.modified_files,
                    report.removed_files,
                    report.mtime_refreshed_files,
                    report.kept_chunks,
                    report.new_chunks,
                    report.total_chunks,
                    repo,
                )
            }
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_list_symbols(&self, args: Value) -> Result<Value, JsonRpcError> {
        let repo = args["repo"].as_str().unwrap_or(".");
        let kind_filter = args["kind"].as_str().map(|s| s.to_ascii_lowercase());
        let lang = string_array(&args, "lang");
        let path_globs = string_array(&args, "path");
        let exclude_globs = string_array(&args, "exclude");
        let limit = args["limit"].as_u64().unwrap_or(200) as usize;

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        // Resolve globs against the index's known file set. None means
        // no glob filter; Some(empty) is impossible since `resolve_path_filter`
        // errors when nothing matches.
        let glob_paths =
            filter::resolve_path_filter(&index, &path_globs, &exclude_globs).map_err(|e| {
                JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }
            })?;
        let path_set: Option<std::collections::HashSet<&str>> = glob_paths
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let mut hits: Vec<&Symbol> = index
            .symbols()
            .iter()
            .filter(|s| match &kind_filter {
                Some(k) => s.kind.as_str() == k,
                None => true,
            })
            .filter(|s| lang.is_empty() || lang.iter().any(|l| l == &s.language))
            .filter(|s| match &path_set {
                Some(set) => set.contains(s.file_path.as_str()),
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
                "content": [{"type": "text", "text": "No symbols matched the given filters."}]
            }));
        }

        let total = hits.len();
        let truncated = total > limit;
        let shown: Vec<&Symbol> = hits.into_iter().take(limit).collect();

        let header = if truncated {
            format!("Symbols ({} shown of {total})", shown.len())
        } else {
            format!("Symbols ({total})")
        };
        let mut text = format_symbols(&header, &shown);
        if truncated {
            text.push_str(&format!(
                "\n\n(Showing first {limit}. Raise `limit` to see more.)"
            ));
        }

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_files(&self, args: Value) -> Result<Value, JsonRpcError> {
        let repo = args["repo"].as_str().unwrap_or(".");
        let lang = string_array(&args, "lang");
        let path_globs = string_array(&args, "path");
        let exclude_globs = string_array(&args, "exclude");

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        let glob_paths =
            filter::resolve_path_filter(&index, &path_globs, &exclude_globs).map_err(|e| {
                JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }
            })?;

        // Collect distinct file paths + their dominant language tag. A file
        // can technically have chunks with no language (text fallback), so
        // we keep the first non-empty tag seen.
        let mut seen: std::collections::HashMap<&str, Option<&str>> =
            std::collections::HashMap::new();
        for chunk in index.chunks() {
            let entry = seen.entry(chunk.file_path.as_str()).or_insert(None);
            if entry.is_none() {
                *entry = chunk.language.as_deref();
            }
        }

        let allowed: Option<std::collections::HashSet<&str>> = glob_paths
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let mut files: Vec<&str> = seen
            .iter()
            .filter(|(path, _)| match &allowed {
                Some(set) => set.contains(*path),
                None => true,
            })
            .filter(|(_, lang_opt)| {
                if lang.is_empty() {
                    return true;
                }
                match lang_opt {
                    Some(l) => lang.iter().any(|wanted| wanted == l),
                    None => false,
                }
            })
            .map(|(p, _)| *p)
            .collect();
        files.sort();

        if files.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": "No files matched the given filters."}]
            }));
        }

        Ok(json!({
            "content": [{"type": "text", "text": files.join("\n")}]
        }))
    }

    async fn handle_scope_at(&self, args: Value) -> Result<Value, JsonRpcError> {
        let file_path = args["file_path"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'file_path' parameter".into(),
        })?;
        let line = args["line"].as_u64().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'line' parameter".into(),
        })? as usize;
        let repo = args["repo"].as_str().unwrap_or(".");

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        // Innermost = symbol with the smallest range that still contains `line`.
        // Ties broken by the later start_line (more specific). Both `start_line`
        // and `end_line` are 1-indexed and inclusive.
        let innermost = index
            .symbols()
            .iter()
            .filter(|s| s.file_path == file_path)
            .filter(|s| s.start_line <= line && line <= s.end_line)
            .min_by(|a, b| {
                let a_span = a.end_line.saturating_sub(a.start_line);
                let b_span = b.end_line.saturating_sub(b.start_line);
                a_span.cmp(&b_span).then(b.start_line.cmp(&a.start_line))
            });

        let text = match innermost {
            Some(s) => format!(
                "{kind} {name} ({lang}) — {file}:{start}-{end}\n  query line {line} is inside this scope.",
                kind = s.kind.as_str(),
                name = s.name,
                lang = s.language,
                file = s.file_path,
                start = s.start_line,
                end = s.end_line,
            ),
            None => format!(
                "No tree-sitter symbol contains {file_path}:{line}. Either the file is in an unsupported language, the line is module-level prelude before the first definition, or the path is not in the index."
            ),
        };

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_read(&self, args: Value) -> Result<Value, JsonRpcError> {
        const MAX_LINES_PER_CALL: usize = 500;

        let file_path = args["file_path"].as_str().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'file_path' parameter".into(),
        })?;
        let start_line = args["start_line"].as_u64().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'start_line' parameter".into(),
        })? as usize;
        let end_line = args["end_line"].as_u64().ok_or_else(|| JsonRpcError {
            code: -32602,
            message: "Missing 'end_line' parameter".into(),
        })? as usize;
        let repo = args["repo"].as_str().unwrap_or(".");

        if start_line == 0 {
            return Err(JsonRpcError {
                code: -32602,
                message: "start_line must be >= 1 (1-indexed).".into(),
            });
        }
        if end_line < start_line {
            return Err(JsonRpcError {
                code: -32602,
                message: format!("end_line ({end_line}) must be >= start_line ({start_line})."),
            });
        }
        if repo.starts_with("https://") || repo.starts_with("http://") {
            return Err(JsonRpcError {
                code: -32000,
                message: "`read` is not supported for git URLs — clone locally and pass the path."
                    .into(),
            });
        }

        // Reject any path that could escape the repo. We check the raw
        // string before joining: absolute paths, parent traversal, and
        // backslash separators (Windows-style) are all rejected.
        let rejected = file_path.starts_with('/')
            || file_path.starts_with('\\')
            || file_path.contains("..")
            || file_path.contains('\0');
        if rejected {
            return Err(JsonRpcError {
                code: -32602,
                message: format!(
                    "Refusing unsafe file_path {file_path:?} — must be relative, no '..', no leading '/'."
                ),
            });
        }

        let repo_path = Path::new(repo);
        if !repo_path.is_dir() {
            return Err(JsonRpcError {
                code: -32000,
                message: format!("Repo path is not a directory: {repo}"),
            });
        }
        let abs = repo_path.join(file_path);

        // Cap the range before reading so a 5000-line request doesn't
        // pull the whole file into memory.
        let effective_end = end_line.min(start_line + MAX_LINES_PER_CALL - 1);
        let truncated = effective_end < end_line;

        let raw = std::fs::read_to_string(&abs).map_err(|e| JsonRpcError {
            code: -32000,
            message: format!("read {}: {e}", abs.display()),
        })?;
        let lines: Vec<&str> = raw.lines().collect();
        let total = lines.len();
        if start_line > total {
            return Err(JsonRpcError {
                code: -32602,
                message: format!(
                    "start_line {start_line} is past end of file (total lines: {total})."
                ),
            });
        }
        let end_clamped = effective_end.min(total);
        let slice = &lines[start_line - 1..end_clamped];

        let body = slice
            .iter()
            .enumerate()
            .map(|(i, ln)| format!("{:>5}  {}", start_line + i, ln))
            .collect::<Vec<_>>()
            .join("\n");

        let mut header_parts = vec![format!("{file_path}:{start_line}-{end_clamped}")];
        if truncated {
            header_parts.push(format!(
                "(capped at {MAX_LINES_PER_CALL} lines; requested up to {end_line})"
            ));
        }
        let text = format!("{}\n```\n{body}\n```", header_parts.join("  "));

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }

    async fn handle_status(&self, args: Value) -> Result<Value, JsonRpcError> {
        let repo = args["repo"].as_str().unwrap_or(".");
        if repo.starts_with("https://") || repo.starts_with("http://") {
            return Err(JsonRpcError {
                code: -32000,
                message: "`status` is not supported for git URLs — they don't have a persisted on-disk index to compare against."
                    .into(),
            });
        }
        let repo_path = Path::new(repo);
        if !repo_path.is_dir() {
            return Err(JsonRpcError {
                code: -32000,
                message: format!("Repo path is not a directory: {repo}"),
            });
        }

        if !veles_core::persist::index_exists(repo_path) {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No index found at {}/.veles. Run `search` once to create it.", repo_path.display())}]
            }));
        }

        let manifest = veles_core::persist::load_manifest(repo_path).map_err(|e| JsonRpcError {
            code: -32000,
            message: format!("load manifest: {e}"),
        })?;

        let exts = veles_core::walker::filter_extensions(None, manifest.include_text_files);
        // Shared classification (§3.3) — same code path as
        // VelesIndex::update_from_path, so `status` and `update` are
        // guaranteed to agree on counts.
        let state = veles_core::persist::classify_disk(repo_path, &manifest, &exts);

        let added = state.count_added();
        let modified = state.count_modified();
        let mtime_only = state.count_mtime_only();
        let removed = state.count_removed();

        let mut lines: Vec<String> = vec![
            format!("Index at {}/.veles", repo_path.display()),
            format!("  veles version    : {}", manifest.veles_version),
            format!("  format version   : {}", manifest.format_version),
            format!("  model            : {}", manifest.model_name),
            format!("  embedding dim    : {}", manifest.embedding_dim),
            format!("  text files       : {}", manifest.include_text_files),
            format!("  files in manifest: {}", manifest.files.len()),
            format!("  total chunks     : {}", manifest.total_chunks),
            String::new(),
            "On-disk diff:".to_string(),
            format!("  files seen now   : {}", state.seen_now()),
            format!("  added            : {added}"),
            format!("  modified         : {modified}"),
            format!("  removed          : {removed}"),
            format!("  mtime-only       : {mtime_only}"),
        ];
        if added + modified + removed + mtime_only == 0 {
            lines.push(String::new());
            lines.push("Up to date.".to_string());
        } else if added + modified + removed == 0 {
            lines.push(String::new());
            lines.push(format!(
                "{mtime_only} file(s) had mtime drift but unchanged content. \
                 Run `update` (repo={repo}) to refresh fingerprints (no re-embed)."
            ));
        } else {
            lines.push(String::new());
            lines.push(format!("Run `update` (repo={repo}) to refresh."));
        }

        Ok(json!({
            "content": [{"type": "text", "text": lines.join("\n")}]
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
        let lang = string_array(&args, "lang");
        let path_globs = string_array(&args, "path");
        let exclude_globs = string_array(&args, "exclude");

        let index_arc = self
            .cache
            .get_or_load(repo, false)
            .await
            .map_err(|e| JsonRpcError {
                code: -32000,
                message: e.to_string(),
            })?;
        let index = index_arc.read().await;

        let chunk = index
            .resolve_chunk(file_path, line)
            .ok_or_else(|| JsonRpcError {
                code: -32000,
                message: format!("No chunk found at {file_path}:{line}"),
            })?
            .clone();

        let glob_paths =
            filter::resolve_path_filter(&index, &path_globs, &exclude_globs).map_err(|e| {
                JsonRpcError {
                    code: -32000,
                    message: e.to_string(),
                }
            })?;
        let lang_slice: Option<&[String]> = if lang.is_empty() { None } else { Some(&lang) };
        let path_slice: Option<&[String]> = glob_paths.as_deref();

        let results = index.find_related(&chunk, top_k, lang_slice, path_slice);

        if results.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("No related chunks found for {file_path}:{line}")}]
            }));
        }

        let text = match format {
            "paths" => format_results_paths(&results),
            "unique_paths" => format_results_unique_paths(&results),
            _ => {
                let header = format!("Chunks related to {file_path}:{line}");
                format_results(&header, &results, Some(index.symbols()))
            }
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

// Scope-label heuristic lives in `veles_core::scope` so the CLI and MCP
// share exactly the same policy. Re-import locally for the formatter.
use veles_core::scope::chunk_scope_label;

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

/// Like [`format_results_paths`] but collapses multiple hits in the same
/// file to a single line. Order is preserved by best-scoring hit, so the
/// most relevant file appears first. Useful when the agent wants a
/// shortlist of *which files* to consider, not *which chunks*.
fn format_results_unique_paths(results: &[veles_core::types::SearchResult]) -> String {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for r in results {
        if seen.insert(r.chunk.file_path.as_str()) {
            out.push(r.chunk.file_path.clone());
        }
    }
    out.join("\n")
}
