//! Load balancer dedicado: TCP :9999 → 2x UDS de upstream, round-robin estrito.
//!
//! Não interpreta HTTP — splice/copy bidirecional de bytes entre cliente e
//! upstream escolhido. Round-robin atômico, sem health-check (rinha exige LB
//! "burro" sem lógica).

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Context, Result};
use monoio::io::{AsyncReadRent, AsyncWriteRentExt, Splitable};
use monoio::net::{TcpListener, TcpStream, UnixStream};

#[derive(Debug, Clone)]
enum Upstream {
    Uds(PathBuf),
    Tcp(String),
}

impl Upstream {
    fn parse(s: &str) -> Self {
        if let Some(path) = s.strip_prefix("unix:") {
            Self::Uds(PathBuf::from(path))
        } else if let Some(addr) = s.strip_prefix("tcp:") {
            Self::Tcp(addr.into())
        } else {
            Self::Uds(PathBuf::from(s))
        }
    }
}

#[derive(Debug)]
struct Config {
    listen: String,
    upstreams: Vec<Upstream>,
}

impl Config {
    fn from_env() -> Self {
        let listen = std::env::var("LB_LISTEN").unwrap_or_else(|_| "0.0.0.0:9999".into());
        let upstreams = std::env::var("LB_UPSTREAMS")
            .unwrap_or_else(|_| "unix:/sockets/api1.sock,unix:/sockets/api2.sock".into())
            .split(',')
            .map(|s| Upstream::parse(s.trim()))
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

    rt_block_on(serve(cfg))
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

async fn serve(cfg: Config) -> Result<()> {
    let listener =
        TcpListener::bind(&cfg.listen).with_context(|| format!("bind tcp {}", cfg.listen))?;
    tracing::info!(listen = %cfg.listen, upstreams = ?cfg.upstreams, "ouvindo");

    let counter = Rc::new(Cell::new(0_usize));
    let upstreams: Rc<[Upstream]> = cfg.upstreams.into();

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

async fn proxy(client: TcpStream, upstream: &Upstream) -> Result<()> {
    match upstream {
        Upstream::Uds(path) => {
            let up = UnixStream::connect(path)
                .await
                .with_context(|| format!("conectar upstream {}", path.display()))?;
            pump(client, up).await
        }
        Upstream::Tcp(addr) => {
            let up = TcpStream::connect(addr)
                .await
                .with_context(|| format!("conectar upstream {addr}"))?;
            pump(client, up).await
        }
    }
}

async fn pump<U>(client: TcpStream, upstream: U) -> Result<()>
where
    U: monoio::io::Splitable,
    U::OwnedRead: AsyncReadRent + 'static,
    U::OwnedWrite: AsyncWriteRentExt + 'static,
{
    let (client_r, client_w) = client.into_split();
    let (up_r, up_w) = upstream.into_split();

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
    // Vec<u8> + clear por iteração mantém a capacity intacta. O padrão antigo
    // com `BytesMut::split().freeze()` movia a alocação inteira pro chunk de
    // saída e deixava `buf` com capacity 0 — o `read` seguinte retornava 0
    // (sem espaço) e o LB interpretava como EOF, fechando conexões keep-alive
    // de cliente ↔ LB ↔ upstream prematuramente.
    const BUF: usize = 8192;
    let mut buf: Vec<u8> = Vec::with_capacity(BUF);
    loop {
        let take = std::mem::take(&mut buf);
        let (res, returned) = r.read(take).await;
        buf = returned;
        let n = res?;
        if n == 0 {
            return Ok(());
        }
        let to_write = std::mem::take(&mut buf);
        let (res, returned) = w.write_all(to_write).await;
        res?;
        buf = returned;
        buf.clear();
    }
}
