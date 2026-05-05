//! Loop por conexão: lê → parseia → normaliza → KNN → resposta pré-montada.
//!
//! Estratégia de erro: em qualquer falha (parse, normalização) devolvemos
//! `200 OK approved=true fraud_score=0.0`. HTTP 5xx pesa 5× na pontuação;
//! falso negativo pesa 3×; falso positivo pesa 1×. Devolver "aprovado" em
//! caso de erro minimiza o pior caso esperado.

use std::rc::Rc;

use anyhow::{Context, Result};
use bytes::BytesMut;
use common::normalize::NormalizationConfig;
use common::{simd, Dataset, McCRiskTable};
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::UnixStream;

use crate::http::{self, ResponseTable, Route};
use crate::{json, knn};

thread_local! {
    static RESPONSES: ResponseTable = ResponseTable::new();
    static MCC: McCRiskTable = McCRiskTable::default();
    static CFG: NormalizationConfig = NormalizationConfig::default();
}

const READ_BUF_INITIAL: usize = 4096;

pub async fn serve_connection(mut stream: UnixStream, dataset: Rc<Dataset>) -> Result<()> {
    let mut buf = BytesMut::with_capacity(READ_BUF_INITIAL);

    loop {
        let take = std::mem::replace(&mut buf, BytesMut::new());
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
                    let _ = buf.split_to(consumed);
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
                let q = simd::quantize(&raw);
                knn::count_fraud_neighbors(&q, dataset)
            })
        }),
        Err(_) => return RESPONSES.with(ResponseTable::fallback_approved),
    };

    let bucket = (count.min(5)) as usize;
    let approved = f32::from(count) / 5.0 < common::APPROVED_THRESHOLD;
    RESPONSES.with(|r| r.fraud_score(bucket, approved))
}
