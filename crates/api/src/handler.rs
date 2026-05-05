//! Loop por conexão: lê → parseia → resolve → escreve. Buffer reusável.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use common::Dataset;
use monoio::io::{AsyncReadRent, AsyncWriteRentExt};
use monoio::net::UnixStream;

use crate::http::{self, ResponseTable, Route};
use crate::knn;

thread_local! {
    static RESPONSES: ResponseTable = ResponseTable::new();
}

const READ_BUF_INITIAL: usize = 4096;

pub async fn serve_connection(stream: UnixStream, dataset: Rc<Dataset>) -> Result<()> {
    let mut buf = BytesMut::with_capacity(READ_BUF_INITIAL);
    let stream = RefCell::new(stream);

    loop {
        // Lê dados em bytes::BytesMut emprestado de volta pela API do monoio.
        let read_buf = std::mem::replace(&mut buf, BytesMut::new());
        let (res, returned) = stream.borrow_mut().read(read_buf).await;
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
                    write_all(&stream, response).await?;
                    let _ = buf.split_to(consumed);
                    if buf.is_empty() {
                        break;
                    }
                }
                Err(http::ParseError::Incomplete) => break,
                Err(http::ParseError::Unsupported) => {
                    let response = RESPONSES.with(ResponseTable::not_found);
                    write_all(&stream, response).await?;
                    return Ok(());
                }
                Err(_) => {
                    let response = RESPONSES.with(ResponseTable::bad_request);
                    write_all(&stream, response).await?;
                    return Ok(());
                }
            }
        }
    }
}

async fn write_all(stream: &RefCell<UnixStream>, response: Bytes) -> Result<()> {
    let (res, _) = stream.borrow_mut().write_all(response).await;
    res.context("write")?;
    Ok(())
}

fn handle_fraud_score(body: &[u8], dataset: &Dataset) -> Bytes {
    let Some(query) = parse_vector(body) else {
        return RESPONSES.with(ResponseTable::bad_request);
    };

    let bucket_idx = knn::predict_bucket(&query, dataset);
    let approved = bucket_idx < 3; // 0.0, 0.2, 0.4 aprovam

    RESPONSES.with(|r| r.fraud_score(bucket_idx, approved))
}

/// Parser JSON manual: extrai array `vector` de 14 floats.
/// Aceita formato `{"vector":[v0,v1,...,v13]}` (whitespace tolerado).
fn parse_vector(body: &[u8]) -> Option<[i16; common::DIM]> {
    let bracket = memchr::memchr(b'[', body)?;
    let close = memchr::memchr(b']', &body[bracket..])? + bracket;
    let inner = &body[bracket + 1..close];

    let mut out = [0_i16; common::DIM];
    let mut idx = 0;
    let mut start = 0;
    let mut i = 0;
    let scale = 8192.0_f32;

    while i <= inner.len() {
        if i == inner.len() || inner[i] == b',' {
            let token = inner[start..i].trim_ascii();
            if token.is_empty() {
                return None;
            }
            let s = std::str::from_utf8(token).ok()?;
            let v: f32 = s.parse().ok()?;
            if idx >= common::DIM {
                return None;
            }
            let q = (v * scale).round();
            out[idx] = q.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            idx += 1;
            start = i + 1;
        }
        i += 1;
    }

    (idx == common::DIM).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_vector() {
        let body = br#"{"vector":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}"#;
        let v = parse_vector(body).unwrap();
        assert_eq!(v, [4096_i16; 14]);
    }

    #[test]
    fn rejects_short_vector() {
        let body = br#"{"vector":[0.5,0.5]}"#;
        assert!(parse_vector(body).is_none());
    }
}
