//! Tabela de respostas pré-montadas. `fraud_score` discretiza em 6 buckets
//! (n/5 para n ∈ 0..=5), então existem 12 corpos JSON possíveis (6 × {true,
//! false}). Reutilizamos os mesmos `Bytes` em `Full<Bytes>` — clone é
//! atomic-add no refcount.

use bytes::Bytes;

#[derive(Debug)]
pub struct ResponseTable {
    pub fraud_score: [Bytes; 12],
    pub fallback_approved: Bytes,
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
                let idx = bucket * 2 + usize::from(approved);
                fraud_score[idx] = Bytes::from(body.into_bytes());
            }
        }

        Self {
            fraud_score,
            fallback_approved: Bytes::from_static(b"{\"approved\":true,\"fraud_score\":0.0}"),
        }
    }

    #[inline]
    pub fn fraud_body(&self, bucket: usize, approved: bool) -> Bytes {
        let idx = bucket * 2 + usize::from(approved);
        self.fraud_score[idx].clone()
    }

    #[inline]
    pub fn fallback_body(&self) -> Bytes {
        self.fallback_approved.clone()
    }
}

impl Default for ResponseTable {
    fn default() -> Self {
        Self::new()
    }
}

fn format_score(score: f32) -> &'static str {
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
    fn builds_all_12_buckets() {
        let t = ResponseTable::new();
        for bucket in 0..6 {
            for approved in [false, true] {
                let body = t.fraud_body(bucket, approved);
                assert!(body.starts_with(b"{\"approved\":"));
                assert!(body.ends_with(b"}"));
            }
        }
    }
}
