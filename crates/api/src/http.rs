//! Parser HTTP minimalista pro hot path. Cobre apenas o necessário pro
//! desafio: `POST /fraud-score` com `Content-Length` e `GET /ready`.

use std::ops::Range;

use bytes::Bytes;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("incompleto")]
    Incomplete,

    #[error("malformado")]
    Malformed,

    #[error("sem Content-Length em POST")]
    MissingContentLength,

    #[error("método ou rota não suportada")]
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    FraudScore,
    Ready,
}

#[derive(Debug)]
pub struct Request {
    pub route: Route,
    pub body: Range<usize>,
}

/// Tenta parsear uma request a partir do buffer. Retorna `Incomplete` se
/// faltar dados — chamador deve ler mais do socket.
pub fn parse(buf: &[u8]) -> Result<(Request, usize), ParseError> {
    let header_end = memchr::memmem::find(buf, b"\r\n\r\n").ok_or(ParseError::Incomplete)?;
    let header = &buf[..header_end];

    let line_end = memchr::memchr(b'\r', header).ok_or(ParseError::Malformed)?;
    let request_line = &header[..line_end];

    let mut parts = request_line.splitn(3, |&b| b == b' ');
    let method = parts.next().ok_or(ParseError::Malformed)?;
    let path = parts.next().ok_or(ParseError::Malformed)?;
    let _version = parts.next().ok_or(ParseError::Malformed)?;

    let route = match (method, path) {
        (b"POST", b"/fraud-score") => Route::FraudScore,
        (b"GET", b"/ready") => Route::Ready,
        _ => return Err(ParseError::Unsupported),
    };

    let content_length = match route {
        Route::FraudScore => parse_content_length(header)?.ok_or(ParseError::MissingContentLength)?,
        Route::Ready => 0,
    };

    let body_start = header_end + 4;
    let body_end = body_start + content_length;
    if buf.len() < body_end {
        return Err(ParseError::Incomplete);
    }

    Ok((
        Request {
            route,
            body: body_start..body_end,
        },
        body_end,
    ))
}

fn parse_content_length(header: &[u8]) -> Result<Option<usize>, ParseError> {
    const KEY: &[u8] = b"content-length:";
    let mut start = 0;
    while let Some(eol) = memchr::memmem::find(&header[start..], b"\r\n") {
        let line_start = start;
        let line = &header[line_start..line_start + eol];
        start += eol + 2;

        if line.len() < KEY.len() {
            continue;
        }
        let key = &line[..KEY.len()];
        if !key.eq_ignore_ascii_case(KEY) {
            continue;
        }

        let value = line[KEY.len()..].trim_ascii();
        let s = std::str::from_utf8(value).map_err(|_| ParseError::Malformed)?;
        let n: usize = s.parse().map_err(|_| ParseError::Malformed)?;
        return Ok(Some(n));
    }
    Ok(None)
}

/// Respostas HTTP pré-montadas. Score só pode assumir 6 valores (`SCORE_BUCKETS`),
/// então pré-construímos os 12 buffers (6 buckets × {approved, denied}).
pub struct ResponseTable {
    fraud_score: [Bytes; 12],
    ready: Bytes,
    not_found: Bytes,
    bad_request: Bytes,
}

impl std::fmt::Debug for ResponseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseTable").finish_non_exhaustive()
    }
}

impl ResponseTable {
    pub fn new() -> Self {
        let mut fraud_score: [Bytes; 12] = std::array::from_fn(|_| Bytes::new());
        for bucket in 0..6_usize {
            for &approved in &[false, true] {
                let score = common::proto::SCORE_BUCKETS[bucket];
                let body = format!(
                    "{{\"approved\":{},\"fraud_score\":{}}}",
                    approved, score
                );
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                    body.len(),
                    body
                );
                let idx = bucket * 2 + usize::from(approved);
                fraud_score[idx] = Bytes::from(resp.into_bytes());
            }
        }

        Self {
            fraud_score,
            ready: Bytes::from_static(
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            ),
            not_found: Bytes::from_static(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            ),
            bad_request: Bytes::from_static(
                b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
        }
    }

    #[inline]
    pub fn fraud_score(&self, bucket: usize, approved: bool) -> Bytes {
        let idx = bucket * 2 + usize::from(approved);
        self.fraud_score[idx].clone()
    }

    #[inline]
    pub fn ready(&self) -> Bytes {
        self.ready.clone()
    }

    #[inline]
    pub fn not_found(&self) -> Bytes {
        self.not_found.clone()
    }

    #[inline]
    pub fn bad_request(&self) -> Bytes {
        self.bad_request.clone()
    }
}

impl Default for ResponseTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_post_fraud_score() {
        let raw = b"POST /fraud-score HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
        let (req, n) = parse(raw).unwrap();
        assert_eq!(req.route, Route::FraudScore);
        assert_eq!(&raw[req.body], b"hello");
        assert_eq!(n, raw.len());
    }

    #[test]
    fn parse_get_ready() {
        let raw = b"GET /ready HTTP/1.1\r\nHost: x\r\n\r\n";
        let (req, n) = parse(raw).unwrap();
        assert_eq!(req.route, Route::Ready);
        assert_eq!(req.body.len(), 0);
        assert_eq!(n, raw.len());
    }

    #[test]
    fn parse_incomplete_returns_err() {
        let raw = b"POST /fraud-score HTTP/1.1\r\nContent-Length:";
        assert!(matches!(parse(raw), Err(ParseError::Incomplete)));
    }
}
