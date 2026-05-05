//! Loop por conexão: lê → parseia → normaliza → KNN → resposta pré-montada.
//!
//! Estratégia de erro: em qualquer falha (parse, normalização) devolvemos
//! `200 OK approved=true fraud_score=0.0`. HTTP 5xx pesa 5× na pontuação;
//! falso negativo pesa 3×; falso positivo pesa 1×. Devolver "aprovado" em
//! caso de erro minimiza o pior caso esperado.

use std::rc::Rc;

use anyhow::{Context, Result};
use common::normalize::NormalizationConfig;
use common::{Dataset, McCRiskTable};
use monoio::buf::{IoBuf, IoBufMut};
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{TcpStream, UnixStream};

use crate::http::{self, ResponseTable, Route};
use crate::{json, knn};

/// Buffer com semântica de append. monoio's `IoBufMut for Vec<u8>` escreve a
/// partir do offset 0 ignorando `len()` — bytes não consumidos da iter
/// anterior são sobrescritos quando o request chega fragmentado em múltiplos
/// pacotes TCP. Esta wrapper aponta `write_ptr` pra `len()` e adiciona ao
/// len ao invés de zerar.
struct AppendBuf(Vec<u8>);

impl AppendBuf {
    fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    fn as_slice(&self) -> &[u8] {
        &self.0
    }

    fn drain_front(&mut self, n: usize) {
        self.0.drain(..n);
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn ensure_room(&mut self, n: usize) {
        if self.0.capacity() - self.0.len() < n {
            self.0.reserve(n);
        }
    }
}

// SAFETY: write_ptr aponta pra `len()`, dentro da alocação; bytes_total
// devolve apenas o espaço livre; set_init incrementa len em `init_len`,
// mantendo o conteúdo já válido [0, len) intocado.
unsafe impl IoBufMut for AppendBuf {
    fn write_ptr(&mut self) -> *mut u8 {
        // SAFETY: len ≤ capacity; ponteiro+len ainda dentro da alocação.
        unsafe { self.0.as_mut_ptr().add(self.0.len()) }
    }

    fn bytes_total(&mut self) -> usize {
        self.0.capacity() - self.0.len()
    }

    unsafe fn set_init(&mut self, init_len: usize) {
        let new_len = self.0.len() + init_len;
        debug_assert!(new_len <= self.0.capacity());
        // SAFETY: init_len bytes acabam de ser escritos pelo kernel a partir
        // de `len()` (write_ptr); ficam inicializados.
        unsafe { self.0.set_len(new_len) };
    }
}

unsafe impl IoBuf for AppendBuf {
    fn read_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.0.len()
    }
}

thread_local! {
    static RESPONSES: ResponseTable = ResponseTable::new();
    static MCC: McCRiskTable = McCRiskTable::default();
    static CFG: NormalizationConfig = NormalizationConfig::default();
}

const READ_BUF_INITIAL: usize = 4096;

pub async fn serve_uds(stream: UnixStream, dataset: Rc<Dataset>) -> Result<()> {
    serve_loop(stream, dataset).await
}

pub async fn serve_tcp(stream: TcpStream, dataset: Rc<Dataset>) -> Result<()> {
    serve_loop(stream, dataset).await
}

async fn serve_loop<S>(mut stream: S, dataset: Rc<Dataset>) -> Result<()>
where
    S: AsyncReadRent + AsyncWriteRentExt,
{
    let mut buf = AppendBuf::with_capacity(READ_BUF_INITIAL);

    loop {
        buf.ensure_room(1024);
        let take = std::mem::replace(&mut buf, AppendBuf::with_capacity(0));
        let (res, returned) = stream.read(take).await;
        buf = returned;
        let n = res.context("read")?;
        if n == 0 {
            return Ok(());
        }

        loop {
            match http::parse(buf.as_slice()) {
                Ok((req, consumed)) => {
                    let response = match req.route {
                        Route::FraudScore => {
                            handle_fraud_score(&buf.as_slice()[req.body], &dataset)
                        }
                        Route::Ready => RESPONSES.with(ResponseTable::ready),
                    };
                    let keep_alive = req.keep_alive;
                    let (res, _) = stream.write_all(response).await;
                    res.context("write")?;
                    if !keep_alive {
                        return Ok(());
                    }
                    buf.drain_front(consumed);
                    if buf.is_empty() {
                        break;
                    }
                }
                Err(http::ParseError::Incomplete) => break,
                Err(http::ParseError::Unsupported) => {
                    let response = RESPONSES.with(ResponseTable::not_found);
                    let (res, _) = stream.write_all(response).await;
                    res.context("write")?;
                    return Ok(());
                }
                Err(_) => {
                    let response = RESPONSES.with(ResponseTable::fallback_approved);
                    let (res, _) = stream.write_all(response).await;
                    res.context("write")?;
                    return Ok(());
                }
            }
        }
    }
}

fn handle_fraud_score(body: &[u8], dataset: &Dataset) -> bytes::Bytes {
    let mut known = Vec::new();
    let parse_result = json::parse(body, &mut known);

    let count = match parse_result {
        Ok(view) => MCC.with(|mcc| {
            CFG.with(|cfg| {
                let raw = common::normalize::normalize(&view, cfg, mcc);
                knn::count_fraud_neighbors(&raw, dataset)
            })
        }),
        Err(_) => return RESPONSES.with(ResponseTable::fallback_approved),
    };

    let bucket = (count.min(5)) as usize;
    let approved = f32::from(count) / 5.0 < common::APPROVED_THRESHOLD;
    RESPONSES.with(|r| r.fraud_score(bucket, approved))
}
