//! Hyper service handler. Recebe um `Request<Incoming>`, despacha pra rota
//! correta e devolve `Response<Full<Bytes>>`. Hyper cuida de keep-alive,
//! Connection header, parsing HTTP, partial reads e Content-Length.
//!
//! Estratégia de erro: em qualquer falha (parse, normalização) devolvemos
//! `200 OK approved=true fraud_score=0.0`. HTTP 5xx pesa 5× na pontuação;
//! falso negativo pesa 3×; falso positivo pesa 1×.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use common::normalize::NormalizationConfig;
use common::{Dataset, McCRiskTable};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{CONNECTION, CONTENT_TYPE};
use hyper::{Method, Request, Response, StatusCode};

use crate::responses::ResponseTable;
use crate::{json, knn};

thread_local! {
    static MCC: McCRiskTable = McCRiskTable::default();
    static CFG: NormalizationConfig = NormalizationConfig::default();
}

pub async fn handle(
    req: Request<Incoming>,
    dataset: Arc<Dataset>,
    responses: Arc<ResponseTable>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let resp = match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => empty_ok(),
        (&Method::POST, "/fraud-score") => match req.collect().await {
            Ok(collected) => {
                let body = collected.to_bytes();
                handle_fraud_score(&body, &dataset, &responses)
            }
            Err(_) => json_ok(responses.fallback_body()),
        },
        _ => not_found(),
    };
    Ok(resp)
}

fn handle_fraud_score(
    body: &[u8],
    dataset: &Dataset,
    responses: &ResponseTable,
) -> Response<Full<Bytes>> {
    let mut known = Vec::with_capacity(8);
    let parse_result = json::parse(body, &mut known);

    let count = match parse_result {
        Ok(view) => MCC.with(|mcc| {
            CFG.with(|cfg| {
                let raw = common::normalize::normalize(&view, cfg, mcc);
                knn::count_fraud_neighbors(&raw, dataset)
            })
        }),
        Err(_) => return json_ok(responses.fallback_body()),
    };

    let bucket = (count.min(5)) as usize;
    let approved = f32::from(count) / 5.0 < common::APPROVED_THRESHOLD;
    json_ok(responses.fraud_body(bucket, approved))
}

#[inline]
fn json_ok(body: Bytes) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .header(CONNECTION, "keep-alive")
        .body(Full::new(body))
        .expect("static header values are valid")
}

#[inline]
fn empty_ok() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONNECTION, "keep-alive")
        .body(Full::new(Bytes::new()))
        .expect("static header values are valid")
}

#[inline]
fn not_found() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header(CONNECTION, "keep-alive")
        .body(Full::new(Bytes::new()))
        .expect("static header values are valid")
}
