//! Busca k-NN brute-force sobre dataset SoA. Retorna bucket de score agregado.
//!
//! Brute-force varre n=3M vetores por request. AVX2 + SoA permite ~16 dot
//! products i16 por ciclo. Para cair abaixo de 1ms precisamos provavelmente
//! migrar pra IVF (ver `roadmap` no README) — esta primeira versão é o baseline.

use common::proto::{K, SCORE_BUCKETS};
use common::{Dataset, DIM};

/// Heap min de tamanho fixo K (top-K menores distâncias).
#[derive(Debug, Clone, Copy)]
struct TopK {
    dists: [i64; K],
    indices: [usize; K],
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

    /// Insere `(dist, idx)` se for menor que o maior atual. O(K) por insert,
    /// K=5 fixo => competitivo vs heap binário pra K pequeno.
    #[inline]
    fn try_push(&mut self, dist: i64, idx: usize) {
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

/// Calcula bucket de score do query (índice em `SCORE_BUCKETS`).
///
/// Estratégia atual: brute-force escalar — substituir por AVX2 quando profile
/// confirmar bottleneck no dot product.
pub fn predict_bucket(query: &[i16; DIM], dataset: &Dataset) -> usize {
    let mut top = TopK::new();
    let n = dataset.len();

    let columns: [&[i16]; DIM] = std::array::from_fn(|d| dataset.dim_column(d));

    for i in 0..n {
        let mut acc: i64 = 0;
        for d in 0..DIM {
            // SAFETY: i < n e cada coluna tem len = n.
            let v = unsafe { *columns[d].get_unchecked(i) };
            let diff = i32::from(query[d]) - i32::from(v);
            acc += i64::from(diff * diff);
        }
        top.try_push(acc, i);
    }

    let mut sum_score = 0_u32;
    for i in 0..top.len {
        sum_score += u32::from(dataset.score_u8(top.indices[i]));
    }
    let mean = (sum_score as f32) / (top.len.max(1) as f32) / 255.0;

    nearest_bucket(mean)
}

#[inline]
fn nearest_bucket(score: f32) -> usize {
    let mut best = 0_usize;
    let mut best_dist = (score - SCORE_BUCKETS[0]).abs();
    for (i, &b) in SCORE_BUCKETS.iter().enumerate().skip(1) {
        let d = (score - b).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_keeps_smallest_k() {
        let mut t = TopK::new();
        for (i, d) in [10_i64, 1, 50, 3, 8, 2, 100].iter().enumerate() {
            t.try_push(*d, i);
        }
        let mut dists = t.dists;
        dists.sort_unstable();
        assert_eq!(dists, [1, 2, 3, 8, 10]);
    }

    #[test]
    fn nearest_bucket_snaps_correctly() {
        assert_eq!(nearest_bucket(0.0), 0);
        assert_eq!(nearest_bucket(0.19), 1);
        assert_eq!(nearest_bucket(0.5), 2); // empate vai pro primeiro encontrado (0.4)
        assert_eq!(nearest_bucket(1.0), 5);
    }
}
