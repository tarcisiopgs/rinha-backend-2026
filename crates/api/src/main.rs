//! Servidor da API em tokio + hyper. Single OS thread (multi_thread runtime
//! com `worker_threads=1`) — compatível com o orçamento de 1 CPU do desafio.
//! O cgroup do compose enforça o limite real; tokio só decide concorrência
//! cooperativa em cima desse worker.

#![allow(unreachable_pub)]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use common::Dataset;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, UnixListener};

mod handler;
mod json;
mod knn;
#[cfg(target_arch = "x86_64")]
mod knn_avx2;
mod responses;

#[derive(Debug, Clone)]
enum Listen {
    Uds(PathBuf),
    Tcp(String),
}

#[derive(Debug)]
struct Config {
    listen: Listen,
    dataset_path: PathBuf,
}

impl Config {
    fn from_env() -> Self {
        let raw = std::env::var("API_LISTEN").unwrap_or_else(|_| "tcp:0.0.0.0:9000".into());
        let listen = if let Some(path) = raw.strip_prefix("unix:") {
            Listen::Uds(PathBuf::from(path))
        } else if let Some(addr) = raw.strip_prefix("tcp:") {
            Listen::Tcp(addr.into())
        } else {
            Listen::Tcp(raw)
        };
        let dataset_path = std::env::var("DATASET_PATH")
            .unwrap_or_else(|_| "/data/references.bin".into())
            .into();
        Self {
            listen,
            dataset_path,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .compact()
        .init();

    let cfg = Config::from_env();
    tracing::info!(?cfg, "boot");

    let dataset = Dataset::open(&cfg.dataset_path)
        .with_context(|| format!("abrir dataset em {}", cfg.dataset_path.display()))?;
    tracing::info!(n = dataset.len(), "dataset carregado");
    let dataset = Arc::new(dataset);
    let responses = Arc::new(responses::ResponseTable::new());

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(1)
        .enable_all()
        .build()
        .context("init tokio runtime")?;

    runtime.block_on(serve(cfg, dataset, responses))
}

async fn serve(
    cfg: Config,
    dataset: Arc<Dataset>,
    responses: Arc<responses::ResponseTable>,
) -> Result<()> {
    match cfg.listen {
        Listen::Tcp(addr) => {
            let listener = TcpListener::bind(&addr)
                .await
                .with_context(|| format!("bind tcp {addr}"))?;
            tracing::info!(tcp = %addr, "ouvindo");
            accept_tcp(listener, dataset, responses).await
        }
        Listen::Uds(path) => {
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("remover socket stale {}", path.display()))?;
            }
            let listener = UnixListener::bind(&path)
                .with_context(|| format!("bind UDS em {}", path.display()))?;
            tracing::info!(uds = %path.display(), "ouvindo");
            accept_uds(listener, dataset, responses).await
        }
    }
}

async fn accept_tcp(
    listener: TcpListener,
    dataset: Arc<Dataset>,
    responses: Arc<responses::ResponseTable>,
) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await.context("accept tcp")?;
        let _ = stream.set_nodelay(true);
        let dataset = Arc::clone(&dataset);
        let responses = Arc::clone(&responses);
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req| {
                handler::handle(req, Arc::clone(&dataset), Arc::clone(&responses))
            });
            if let Err(err) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!(?err, "conn ended");
            }
        });
    }
}

async fn accept_uds(
    listener: UnixListener,
    dataset: Arc<Dataset>,
    responses: Arc<responses::ResponseTable>,
) -> Result<()> {
    loop {
        let (stream, _) = listener.accept().await.context("accept uds")?;
        let dataset = Arc::clone(&dataset);
        let responses = Arc::clone(&responses);
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req| {
                handler::handle(req, Arc::clone(&dataset), Arc::clone(&responses))
            });
            if let Err(err) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!(?err, "conn ended");
            }
        });
    }
}
