//! gRPC service for Veles code search.
//!
//! Exposes `Index`, `Search`, `FindRelated`, and `GetStats` RPCs backed by
//! an in-memory cache of `VelesIndex` instances.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use veles_core::VelesIndex;
use veles_core::types::SearchMode;

// Include the generated protobuf code.
pub mod proto {
    tonic::include_proto!("veles");
}

use proto::veles_service_server::{VelesService, VelesServiceServer};
use proto::*;

// ── Index Cache ───────────────────────────────────────────────────────────

const DEFAULT_CACHE_SIZE: usize = 10;

struct IndexCache {
    indexes: HashMap<String, VelesIndex>,
    max_size: usize,
    model: model2vec_rs::model::StaticModel,
}

impl IndexCache {
    fn new(model: model2vec_rs::model::StaticModel) -> Self {
        Self {
            indexes: HashMap::new(),
            max_size: DEFAULT_CACHE_SIZE,
            model,
        }
    }

    /// Ensure the repo is cached (builds the index if not present).
    fn ensure_cached(&mut self, repo: &str, include_text_files: bool) -> Result<(), String> {
        if self.indexes.contains_key(repo) {
            return Ok(());
        }

        // Evict first entry if at capacity.
        if self.indexes.len() >= self.max_size {
            if let Some(key) = self.indexes.keys().next().cloned() {
                self.indexes.remove(&key);
            }
        }

        let repo_owned = repo.to_string();
        let model = self.model.clone();
        let path = Path::new(&repo_owned);

        let index = if path.is_dir() {
            VelesIndex::from_path(path, Some(model), None, include_text_files)
        } else if repo.starts_with("https://") || repo.starts_with("http://") {
            VelesIndex::from_git(repo, None, Some(model), include_text_files)
        } else {
            return Err(format!(
                "Invalid repo: must be a local directory or https:// URL"
            ));
        }
        .map_err(|e| format!("Failed to index {repo}: {e}"))?;

        self.indexes.insert(repo.to_string(), index);
        Ok(())
    }

    fn get(&self, repo: &str) -> Option<&VelesIndex> {
        self.indexes.get(repo)
    }
}

// ── gRPC Service Implementation ───────────────────────────────────────────

#[derive(Clone)]
pub struct VelesServiceImpl {
    cache: Arc<Mutex<IndexCache>>,
}

impl VelesServiceImpl {
    pub fn new(model: model2vec_rs::model::StaticModel) -> Self {
        Self {
            cache: Arc::new(Mutex::new(IndexCache::new(model))),
        }
    }

    /// Create a tonic Server ready to serve.
    pub fn into_server(self) -> VelesServiceServer<Self> {
        VelesServiceServer::new(self)
    }
}

fn convert_stats(stats: veles_core::types::IndexStats) -> Option<proto::IndexStats> {
    Some(proto::IndexStats {
        indexed_files: stats.indexed_files as i32,
        total_chunks: stats.total_chunks as i32,
        languages: stats
            .languages
            .into_iter()
            .map(|(k, v)| (k, v as i32))
            .collect(),
    })
}

fn convert_result(r: veles_core::types::SearchResult) -> proto::SearchResult {
    proto::SearchResult {
        chunk: Some(proto::Chunk {
            content: r.chunk.content,
            file_path: r.chunk.file_path,
            start_line: r.chunk.start_line as i32,
            end_line: r.chunk.end_line as i32,
            language: r.chunk.language,
        }),
        score: r.score,
        source: r.source.to_string(),
    }
}

#[tonic::async_trait]
impl VelesService for VelesServiceImpl {
    async fn index(
        &self,
        request: Request<IndexRequest>,
    ) -> Result<Response<IndexResponse>, Status> {
        let req = request.into_inner();
        let mut cache = self.cache.lock().await;
        cache
            .ensure_cached(&req.repo, req.include_text_files)
            .map_err(Status::internal)?;

        let stats = cache.get(&req.repo).unwrap().stats();
        Ok(Response::new(IndexResponse {
            stats: convert_stats(stats),
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let top_k = if req.top_k > 0 { req.top_k as usize } else { 5 };
        let mode = req.mode.parse::<SearchMode>().unwrap_or(SearchMode::Hybrid);

        let mut cache = self.cache.lock().await;
        cache
            .ensure_cached(&req.repo, req.include_text_files)
            .map_err(Status::internal)?;

        let index = cache.get(&req.repo).unwrap();
        let results = index.search(
            &req.query,
            top_k,
            mode,
            req.alpha,
            if req.filter_languages.is_empty() {
                None
            } else {
                Some(req.filter_languages.as_slice())
            },
            if req.filter_paths.is_empty() {
                None
            } else {
                Some(req.filter_paths.as_slice())
            },
        );

        Ok(Response::new(SearchResponse {
            results: results.into_iter().map(convert_result).collect(),
        }))
    }

    async fn find_related(
        &self,
        request: Request<FindRelatedRequest>,
    ) -> Result<Response<FindRelatedResponse>, Status> {
        let req = request.into_inner();
        let top_k = if req.top_k > 0 { req.top_k as usize } else { 5 };

        let mut cache = self.cache.lock().await;
        cache
            .ensure_cached(&req.repo, req.include_text_files)
            .map_err(Status::internal)?;

        let index = cache.get(&req.repo).unwrap();
        let chunk = index
            .resolve_chunk(&req.file_path, req.line as usize)
            .ok_or_else(|| {
                Status::not_found(format!("No chunk found at {}:{}", req.file_path, req.line))
            })?
            .clone();

        let results = index.find_related(&chunk, top_k);

        Ok(Response::new(FindRelatedResponse {
            results: results.into_iter().map(convert_result).collect(),
        }))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        let req = request.into_inner();
        let cache = self.cache.lock().await;
        let index = cache
            .get(&req.repo)
            .ok_or_else(|| Status::not_found(format!("Repo not indexed: {}", req.repo)))?;

        let stats = index.stats();
        Ok(Response::new(GetStatsResponse {
            stats: convert_stats(stats),
        }))
    }
}

/// Run the gRPC server on the given address.
pub async fn serve(addr: &str, model: model2vec_rs::model::StaticModel) -> anyhow::Result<()> {
    let service = VelesServiceImpl::new(model);
    let addr: std::net::SocketAddr = addr.parse()?;

    println!("Veles gRPC server listening on {addr}");

    tonic::transport::Server::builder()
        .add_service(service.into_server())
        .serve(addr)
        .await?;

    Ok(())
}
