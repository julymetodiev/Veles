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
//!   directory or `https://` git URL.
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
            description: "Search a codebase with a natural-language or code query. Pass `repo` as a local directory path or an https:// git URL. The index is cached after the first call, so repeat queries are fast.".into(),
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
                    }
                },
                "required": ["query"]
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

        let mut cache = self.cache.lock().await;
        let index = cache.get_or_index(repo, false).map_err(|e| JsonRpcError {
            code: -32000,
            message: e.to_string(),
        })?;

        let results = index.search(query, top_k, mode, None, None, None);

        if results.is_empty() {
            return Ok(json!({
                "content": [{"type": "text", "text": "No results found."}]
            }));
        }

        let header = format!("Search results for: {query:?} (mode={mode_str})");
        let text = format_results(&header, &results);

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

        let header = format!("Chunks related to {file_path}:{line}");
        let text = format_results(&header, &results);

        Ok(json!({
            "content": [{"type": "text", "text": text}]
        }))
    }
}

/// Format search results as numbered, fenced code blocks (same format as Python version).
fn format_results(header: &str, results: &[veles_core::types::SearchResult]) -> String {
    let mut lines: Vec<String> = vec![header.to_string(), String::new()];
    for (i, r) in results.iter().enumerate() {
        lines.push(format!(
            "## {}. {}  [score={:.3}]",
            i + 1,
            r.chunk.location(),
            r.score
        ));
        lines.push("```".to_string());
        lines.push(r.chunk.content.trim().to_string());
        lines.push("```".to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}
