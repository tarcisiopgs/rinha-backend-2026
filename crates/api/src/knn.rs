//! Busca k-NN brute-force sobre dataset SoA. Retorna count de fraudes entre
//! os K mais próximos.
//!
//! Brute-force varre n=3M vetores. Baseline escalar — substituir por AVX2
//! ou IVF index quando profile confirmar bottleneck.

use common::proto::K;
use common::{Dataset, DIM};

#[derive(Debug, Clone, Copy)]
struct TopK {
    dists: [i64; K],
    indices: [u32; K],
    len: usize,
}

impl TopK {
    fn new() -> Self {
        Self {
            dists: [i64::MAX; K],
            indices: [0; K],
            len: 0,
        }
    }

    #[inline]
    fn try_push(&mut self, dist: i64, idx: u32) {
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
pub fn count_fraud_neighbors(query: &[i16; DIM], dataset: &Dataset) -> u8 {
    let mut top = TopK::new();
    let n = dataset.len();

    let columns: [&[i16]; DIM] = std::array::from_fn(|d| dataset.dim_column(d));

    for i in 0..n {
        let mut acc: i64 = 0;
        for d in 0..DIM {
            // SAFETY: cada coluna tem len = n; i < n.
            let v = unsafe { *columns[d].get_unchecked(i) };
            let diff = i32::from(query[d]) - i32::from(v);
            acc += i64::from(diff) * i64::from(diff);
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
        for (i, d) in [10_i64, 1, 50, 3, 8, 2, 100].iter().enumerate() {
            t.try_push(*d, i as u32);
        }
        let mut dists = t.dists;
        dists.sort_unstable();
        assert_eq!(dists, [1, 2, 3, 8, 10]);
    }
}
