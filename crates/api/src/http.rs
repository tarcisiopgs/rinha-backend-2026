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
    pub keep_alive: bool,
}

pub fn parse(buf: &[u8]) -> Result<(Request, usize), ParseError> {
    let header_end = memchr::memmem::find(buf, b"\r\n\r\n").ok_or(ParseError::Incomplete)?;
    let header = &buf[..header_end];

    let line_end = memchr::memchr(b'\r', header).ok_or(ParseError::Malformed)?;
    let request_line = &header[..line_end];

    let mut parts = request_line.splitn(3, |&b| b == b' ');
    let method = parts.next().ok_or(ParseError::Malformed)?;
    let path = parts.next().ok_or(ParseError::Malformed)?;
    let version = parts.next().ok_or(ParseError::Malformed)?;

    let route = match (method, path) {
        (b"POST", b"/fraud-score") => Route::FraudScore,
        (b"GET", b"/ready") => Route::Ready,
        _ => return Err(ParseError::Unsupported),
    };

    let content_length = match route {
        Route::FraudScore => {
            parse_content_length(header)?.ok_or(ParseError::MissingContentLength)?
        }
        Route::Ready => 0,
    };

    let body_start = header_end + 4;
    let body_end = body_start + content_length;
    if buf.len() < body_end {
        return Err(ParseError::Incomplete);
    }

    let keep_alive = derive_keep_alive(version, header);

    Ok((
        Request {
            route,
            body: body_start..body_end,
            keep_alive,
        },
        body_end,
    ))
}

/// HTTP/1.1 default = keep-alive a menos que tenha `Connection: close`.
/// HTTP/1.0 default = close a menos que tenha `Connection: keep-alive`.
fn derive_keep_alive(version: &[u8], header: &[u8]) -> bool {
    let conn = parse_connection_header(header);
    let is_http_1_0 = version.eq_ignore_ascii_case(b"HTTP/1.0");
    match conn {
        Some(ConnectionToken::Close) => false,
        Some(ConnectionToken::KeepAlive) => true,
        None => !is_http_1_0,
    }
}

#[derive(Debug, Clone, Copy)]
enum ConnectionToken {
    KeepAlive,
    Close,
}

fn parse_connection_header(header: &[u8]) -> Option<ConnectionToken> {
    const KEY: &[u8] = b"connection:";
    for line in header.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.len() < KEY.len() {
            continue;
        }
        if !line[..KEY.len()].eq_ignore_ascii_case(KEY) {
            continue;
        }
        let value = line[KEY.len()..].trim_ascii();
        if value.eq_ignore_ascii_case(b"close") {
            return Some(ConnectionToken::Close);
        }
        if value.eq_ignore_ascii_case(b"keep-alive") {
            return Some(ConnectionToken::KeepAlive);
        }
        return None;
    }
    None
}

fn parse_content_length(header: &[u8]) -> Result<Option<usize>, ParseError> {
    const KEY: &[u8] = b"content-length:";
    for line in header.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.len() < KEY.len() {
            continue;
        }
        if !line[..KEY.len()].eq_ignore_ascii_case(KEY) {
            continue;
        }
        let value = line[KEY.len()..].trim_ascii();
        let s = std::str::from_utf8(value).map_err(|_| ParseError::Malformed)?;
        let n: usize = s.parse().map_err(|_| ParseError::Malformed)?;
        return Ok(Some(n));
    }
    Ok(None)
}

/// Tabela de respostas HTTP pré-montadas. Score só pode assumir 6 valores
/// discretos (`SCORE_BUCKETS`), então pré-construímos os 12 buffers (6 buckets
/// × {approved, denied}).
pub struct ResponseTable {
    fraud_score: [Bytes; 12],
    ready: Bytes,
    not_found: Bytes,
    fallback_approved: Bytes,
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
                let score = common::SCORE_BUCKETS[bucket];
                let body = format!(
                    "{{\"approved\":{},\"fraud_score\":{}}}",
                    approved,
                    format_score(score)
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

        let fallback_body = "{\"approved\":true,\"fraud_score\":0.0}";
        let fallback = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
            fallback_body.len(),
            fallback_body
        );

        Self {
            fraud_score,
            ready: Bytes::from_static(
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            ),
            not_found: Bytes::from_static(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
            ),
            fallback_approved: Bytes::from(fallback.into_bytes()),
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

    /// Resposta padrão em caso de erro de parsing/normalização.
    /// Devolve `200 approved=true fraud_score=0.0` pra evitar HTTP 5xx
    /// (peso 5 na fórmula de detecção, vs FN peso 3).
    #[inline]
    pub fn fallback_approved(&self) -> Bytes {
        self.fallback_approved.clone()
    }
}

impl Default for ResponseTable {
    fn default() -> Self {
        Self::new()
    }
}

fn format_score(score: f32) -> &'static str {
    // SCORE_BUCKETS são valores discretos conhecidos — formato fixo no JSON
    // de resposta evita chamadas a `ryu`/`format!` em hot path.
    match (score * 10.0).round() as i32 {
        0 => "0.0",
        2 => "0.2",
        4 => "0.4",
        6 => "0.6",
        8 => "0.8",
        10 => "1.0",
        _ => "0.0",
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
        assert!(req.keep_alive);
    }

    #[test]
    fn parse_get_ready() {
        let raw = b"GET /ready HTTP/1.1\r\nHost: x\r\n\r\n";
        let (req, n) = parse(raw).unwrap();
        assert_eq!(req.route, Route::Ready);
        assert_eq!(req.body.len(), 0);
        assert_eq!(n, raw.len());
        assert!(req.keep_alive);
    }

    #[test]
    fn http10_default_close() {
        let raw = b"GET /ready HTTP/1.0\r\nHost: x\r\n\r\n";
        let (req, _) = parse(raw).unwrap();
        assert!(!req.keep_alive);
    }

    #[test]
    fn http10_explicit_keep_alive() {
        let raw = b"GET /ready HTTP/1.0\r\nConnection: keep-alive\r\nHost: x\r\n\r\n";
        let (req, _) = parse(raw).unwrap();
        assert!(req.keep_alive);
    }

    #[test]
    fn http11_explicit_close() {
        let raw = b"GET /ready HTTP/1.1\r\nConnection: close\r\nHost: x\r\n\r\n";
        let (req, _) = parse(raw).unwrap();
        assert!(!req.keep_alive);
    }

    #[test]
    fn parse_incomplete_returns_err() {
        let raw = b"POST /fraud-score HTTP/1.1\r\nContent-Length:";
        assert!(matches!(parse(raw), Err(ParseError::Incomplete)));
    }

    #[test]
    fn response_table_builds_all_buckets() {
        let t = ResponseTable::new();
        for bucket in 0..6 {
            for approved in [false, true] {
                let r = t.fraud_score(bucket, approved);
                assert!(r.starts_with(b"HTTP/1.1 200 OK"));
            }
        }
    }
}
