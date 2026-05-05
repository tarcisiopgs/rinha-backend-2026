//! Servidor da API. Single-threaded por instância, monoio + io_uring,
//! escuta em UDS pra eliminar overhead de TCP loopback contra o LB.

#![allow(unreachable_pub)]

use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Context, Result};
use common::Dataset;

mod handler;
mod http;
mod json;
mod knn;

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
        let raw = std::env::var("API_LISTEN").unwrap_or_else(|_| "unix:/sockets/api.sock".into());
        let listen = if let Some(path) = raw.strip_prefix("unix:") {
            Listen::Uds(PathBuf::from(path))
        } else if let Some(addr) = raw.strip_prefix("tcp:") {
            Listen::Tcp(addr.into())
        } else {
            Listen::Uds(PathBuf::from(raw))
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

    rt_block_on(serve(cfg, Rc::new(dataset)))
}

#[cfg(target_os = "linux")]
fn rt_block_on<F: std::future::Future<Output = Result<()>>>(fut: F) -> Result<()> {
    let driver = std::env::var("MONOIO_DRIVER").unwrap_or_default();
    if driver != "legacy" {
        if let Ok(mut rt) = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
            .enable_timer()
            .build()
        {
            tracing::info!("monoio: io_uring");
            return rt.block_on(fut);
        }
        tracing::warn!("io_uring indisponível, caindo pro legacy driver");
    }
    monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
        .enable_timer()
        .build()
        .context("init monoio legacy runtime")?
        .block_on(fut)
}

#[cfg(not(target_os = "linux"))]
fn rt_block_on<F: std::future::Future<Output = Result<()>>>(fut: F) -> Result<()> {
    monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
        .enable_timer()
        .build()
        .context("init monoio legacy runtime")?
        .block_on(fut)
}

async fn serve(cfg: Config, dataset: Rc<Dataset>) -> Result<()> {
    match cfg.listen {
        Listen::Uds(path) => serve_uds(path, dataset).await,
        Listen::Tcp(addr) => serve_tcp(addr, dataset).await,
    }
}

async fn serve_uds(path: PathBuf, dataset: Rc<Dataset>) -> Result<()> {
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("remover socket stale {}", path.display()))?;
    }
    let listener = monoio::net::UnixListener::bind(&path)
        .with_context(|| format!("bind UDS em {}", path.display()))?;
    tracing::info!(uds = %path.display(), "ouvindo");

    loop {
        let (stream, _) = listener.accept().await.context("accept")?;
        let dataset = Rc::clone(&dataset);
        monoio::spawn(async move {
            if let Err(err) = handler::serve_uds(stream, dataset).await {
                tracing::debug!(?err, "conexão encerrada com erro");
            }
        });
    }
}

async fn serve_tcp(addr: String, dataset: Rc<Dataset>) -> Result<()> {
    let listener =
        monoio::net::TcpListener::bind(&addr).with_context(|| format!("bind TCP em {addr}"))?;
    tracing::info!(tcp = %addr, "ouvindo");

    loop {
        let (stream, _) = listener.accept().await.context("accept")?;
        let dataset = Rc::clone(&dataset);
        monoio::spawn(async move {
            if let Err(err) = handler::serve_tcp(stream, dataset).await {
                tracing::debug!(?err, "conexão encerrada com erro");
            }
        });
    }
}
