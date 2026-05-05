//! Busca k-NN brute-force sobre dataset SoA i16 (f32 quantizado). Dispatcha
//! pra AVX2 quando disponível, fallback escalar pra outras arquiteturas.

use common::proto::{K, NULL_SENTINEL_I16, QUANT_SCALE};
use common::{Dataset, DIM};

#[derive(Debug, Clone, Copy)]
pub(crate) struct TopK {
    pub dists: [i32; K],
    pub indices: [u32; K],
    pub len: usize,
}

impl TopK {
    pub(crate) fn new() -> Self {
        Self {
            dists: [i32::MAX; K],
            indices: [0; K],
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn try_push(&mut self, dist: i32, idx: u32) {
        if self.len < K {
            self.dists[self.len] = dist;
            self.indices[self.len] = idx;
            self.len += 1;
            return;
        }
        let mut max_pos = 0;
        let mut max_val = self.dists[0];
        for i in 1..K {
            if self.dists[i] > max_val {
                max_val = self.dists[i];
                max_pos = i;
            }
        }
        if dist < max_val {
            self.dists[max_pos] = dist;
            self.indices[max_pos] = idx;
        }
    }
}

/// Quantiza vetor de query f32 ([0, 1] com -1 como sentinela) para i16
/// usando o mesmo `QUANT_SCALE` aplicado no dataset offline.
#[inline]
pub(crate) fn quantize_query(query: &[f32; DIM]) -> [i16; DIM] {
    let mut q = [0_i16; DIM];
    for d in 0..DIM {
        let v = query[d];
        q[d] = if v < 0.0 {
            NULL_SENTINEL_I16
        } else {
            let scaled = (v * QUANT_SCALE).round();
            if scaled <= f32::from(i16::MIN) {
                i16::MIN
            } else if scaled >= f32::from(i16::MAX) {
                i16::MAX
            } else {
                scaled as i16
            }
        };
    }
    q
}

/// Conta quantos dos K vizinhos mais próximos são `fraud`.
///
/// Returns valor em `0..=K` que mapeia diretamente em
/// `fraud_score = count / 5.0` (= bucket index em `SCORE_BUCKETS`).
pub fn count_fraud_neighbors(query: &[f32; DIM], dataset: &Dataset) -> u8 {
    let q = quantize_query(query);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detectado em runtime.
            return unsafe { crate::knn_avx2::count_fraud_neighbors_avx2(&q, dataset) };
        }
    }
    count_fraud_neighbors_scalar(&q, dataset)
}

pub(crate) fn count_fraud_neighbors_scalar(query: &[i16; DIM], dataset: &Dataset) -> u8 {
    let mut top = TopK::new();
    let n = dataset.len();

    let columns: [&[i16]; DIM] = std::array::from_fn(|d| dataset.dim_column(d));

    for i in 0..n {
        let mut acc: i32 = 0;
        for d in 0..DIM {
            // SAFETY: cada coluna tem len = n_padded ≥ n.
            let v = unsafe { *columns[d].get_unchecked(i) };
            let diff = i32::from(query[d]) - i32::from(v);
            acc += diff * diff;
        }
        top.try_push(acc, i as u32);
    }

    let mut count = 0_u8;
    for slot in 0..top.len {
        if dataset.is_fraud(top.indices[slot] as usize) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_keeps_smallest_k() {
        let mut t = TopK::new();
        for (i, d) in [10_i32, 1, 50, 3, 8, 2, 100].iter().enumerate() {
            t.try_push(*d, i as u32);
        }
        let mut dists = t.dists;
        dists.sort();
        assert_eq!(dists, [1, 2, 3, 8, 10]);
    }
}
