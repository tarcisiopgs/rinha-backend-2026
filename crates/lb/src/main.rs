//! Load balancer dedicado: TCP :9999 → 2x UDS de upstream, round-robin estrito.
//!
//! Não interpreta HTTP — splice/copy bidirecional de bytes entre cliente e
//! upstream escolhido. Round-robin atômico, sem health-check (rinha exige LB
//! "burro" sem lógica).

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Context, Result};
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{TcpListener, TcpStream, UnixStream};

#[derive(Debug)]
struct Config {
    listen: String,
    upstreams: Vec<PathBuf>,
}

impl Config {
    fn from_env() -> Self {
        let listen = std::env::var("LB_LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".into());
        let upstreams = std::env::var("LB_UPSTREAMS")
            .unwrap_or_else(|_| "/tmp/api1.sock,/tmp/api2.sock".into())
            .split(',')
            .map(|s| PathBuf::from(s.trim()))
            .collect();
        Self { listen, upstreams }
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
    tracing::info!(?cfg, "boot lb");

    let mut rt = monoio::RuntimeBuilder::<monoio::IoUringDriver>::new()
        .enable_timer()
        .build()
        .context("init runtime")?;

    rt.block_on(serve(cfg))
}

async fn serve(cfg: Config) -> Result<()> {
    let listener = TcpListener::bind(&cfg.listen)
        .with_context(|| format!("bind tcp {}", cfg.listen))?;
    tracing::info!(listen = %cfg.listen, upstreams = ?cfg.upstreams, "ouvindo");

    let counter = Rc::new(Cell::new(0_usize));
    let upstreams: Rc<[PathBuf]> = cfg.upstreams.into();

    loop {
        let (client, _) = listener.accept().await.context("accept")?;
        let counter = Rc::clone(&counter);
        let upstreams = Rc::clone(&upstreams);

        monoio::spawn(async move {
            let idx = counter.get() % upstreams.len();
            counter.set(counter.get().wrapping_add(1));
            if let Err(err) = proxy(client, &upstreams[idx]).await {
                tracing::debug!(?err, "proxy encerrado com erro");
            }
        });
    }
}

async fn proxy(client: TcpStream, upstream: &std::path::Path) -> Result<()> {
    let upstream_stream = UnixStream::connect(upstream)
        .await
        .with_context(|| format!("conectar upstream {}", upstream.display()))?;

    let (client_r, client_w) = client.into_split();
    let (up_r, up_w) = upstream_stream.into_split();

    let c2u = monoio::spawn(copy(client_r, up_w));
    let u2c = monoio::spawn(copy(up_r, client_w));

    let _ = c2u.await;
    let _ = u2c.await;
    Ok(())
}

async fn copy<R, W>(mut r: R, mut w: W) -> Result<()>
where
    R: AsyncReadRent,
    W: AsyncWriteRentExt,
{
    let mut buf = bytes::BytesMut::with_capacity(8192);
    loop {
        let take = std::mem::replace(&mut buf, bytes::BytesMut::new());
        let (res, returned) = r.read(take).await;
        buf = returned;
        let n = res?;
        if n == 0 {
            return Ok(());
        }
        let chunk = buf.split().freeze();
        let (res, _) = w.write_all(chunk).await;
        res?;
    }
}
