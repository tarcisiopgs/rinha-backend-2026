//! Busca k-NN brute-force sobre dataset SoA f32. Dispatcha pra AVX2 quando
//! disponível, fallback escalar pra outras arquiteturas.

use common::proto::K;
use common::{Dataset, DIM};

#[derive(Debug, Clone, Copy)]
pub(crate) struct TopK {
    pub dists: [f32; K],
    pub indices: [u32; K],
    pub len: usize,
}

impl TopK {
    pub(crate) fn new() -> Self {
        Self {
            dists: [f32::INFINITY; K],
            indices: [0; K],
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn try_push(&mut self, dist: f32, idx: u32) {
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

/// Conta quantos dos K vizinhos mais próximos são `fraud`.
///
/// Returns valor em `0..=K` que mapeia diretamente em
/// `fraud_score = count / 5.0` (= bucket index em `SCORE_BUCKETS`).
pub fn count_fraud_neighbors(query: &[f32; DIM], dataset: &Dataset) -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 + FMA detectados em runtime.
            return unsafe { crate::knn_avx2::count_fraud_neighbors_avx2(query, dataset) };
        }
    }
    count_fraud_neighbors_scalar(query, dataset)
}

pub(crate) fn count_fraud_neighbors_scalar(query: &[f32; DIM], dataset: &Dataset) -> u8 {
    let mut top = TopK::new();
    let n = dataset.len();

    let columns: [&[f32]; DIM] = std::array::from_fn(|d| dataset.dim_column(d));

    for i in 0..n {
        let mut acc = 0.0_f32;
        for d in 0..DIM {
            // SAFETY: cada coluna tem len = n_padded ≥ n.
            let v = unsafe { *columns[d].get_unchecked(i) };
            let diff = query[d] - v;
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
        for (i, d) in [10.0_f32, 1.0, 50.0, 3.0, 8.0, 2.0, 100.0].iter().enumerate() {
            t.try_push(*d, i as u32);
        }
        let mut dists = t.dists;
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, [1.0, 2.0, 3.0, 8.0, 10.0]);
    }
}
