//! Servidor da API. Single-threaded por instância, monoio + io_uring,
//! escuta em UDS pra eliminar overhead de TCP loopback contra o LB.

use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Context, Result};
use common::Dataset;

mod handler;
mod http;
mod knn;

#[derive(Debug)]
struct Config {
    socket_path: PathBuf,
    dataset_path: PathBuf,
}

impl Config {
    fn from_env() -> Self {
        let socket_path = std::env::var("API_SOCKET")
            .unwrap_or_else(|_| "/tmp/api.sock".into())
            .into();
        let dataset_path = std::env::var("DATASET_PATH")
            .unwrap_or_else(|_| "/data/references.bin".into())
            .into();
        Self {
            socket_path,
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

    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_timer()
        .build()
        .context("inicializar runtime monoio")?;

    rt.block_on(serve(cfg, Rc::new(dataset)))
}

async fn serve(cfg: Config, dataset: Rc<Dataset>) -> Result<()> {
    if cfg.socket_path.exists() {
        std::fs::remove_file(&cfg.socket_path)
            .with_context(|| format!("remover socket stale {}", cfg.socket_path.display()))?;
    }

    let listener = monoio::net::UnixListener::bind(&cfg.socket_path)
        .with_context(|| format!("bind UDS em {}", cfg.socket_path.display()))?;
    tracing::info!(socket = %cfg.socket_path.display(), "ouvindo");

    loop {
        let (stream, _) = listener.accept().await.context("accept")?;
        let dataset = Rc::clone(&dataset);
        monoio::spawn(async move {
            if let Err(err) = handler::serve_connection(stream, dataset).await {
                tracing::debug!(?err, "conexão encerrada com erro");
            }
        });
    }
}
