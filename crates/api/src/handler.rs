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
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::{TcpStream, UnixStream};

use crate::http::{self, ResponseTable, Route};
use crate::{json, knn};

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
    // Reuso o mesmo `Vec<u8>` pra todos os reads desta conexão. `drain(..consumed)`
    // remove os bytes já processados sem mexer na capacity da alocação. O bug
    // anterior usava `BytesMut::split_to`, que faz advance no start ptr e reduz
    // a capacity disponível a cada request — em keep-alive longo, capacity
    // chegava a zero, `read` retornava 0 e o handler interpretava como EOF.
    let mut buf: Vec<u8> = Vec::with_capacity(READ_BUF_INITIAL);

    loop {
        if buf.capacity() - buf.len() < 1024 {
            buf.reserve(1024);
        }

        let take = std::mem::take(&mut buf);
        let (res, returned) = stream.read(take).await;
        buf = returned;
        let n = res.context("read")?;
        if n == 0 {
            return Ok(());
        }

        loop {
            match http::parse(&buf) {
                Ok((req, consumed)) => {
                    let response = match req.route {
                        Route::FraudScore => handle_fraud_score(&buf[req.body], &dataset),
                        Route::Ready => RESPONSES.with(ResponseTable::ready),
                    };
                    let (res, _) = stream.write_all(response).await;
                    res.context("write")?;
                    buf.drain(..consumed);
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
