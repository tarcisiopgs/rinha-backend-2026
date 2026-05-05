//! Busca k-NN com índice IVF (inverted file). Para cada query:
//!   1. Calcula distância contra os `NLIST` centroides;
//!   2. Seleciona os `N_PROBES` mais próximos;
//!   3. Faz brute-force só dentro desses clusters (~24k vetores em vez de 3M);
//!   4. TopK final + contagem de fraudes.
//!
//! AVX2 quando disponível; fallback escalar pra outras arquiteturas.

use common::proto::{K, NLIST, NULL_SENTINEL_I16, N_PROBES, QUANT_SCALE};
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
pub fn count_fraud_neighbors(query: &[f32; DIM], dataset: &Dataset) -> u8 {
    let q = quantize_query(query);

    // 1. Selecciona os N_PROBES centroides mais próximos.
    let probes = select_probes(&q, dataset);

    // 2. Brute-force interno em cada cluster.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detectado em runtime.
            return unsafe { crate::knn_avx2::count_fraud_in_probes_avx2(&q, dataset, &probes) };
        }
    }
    count_fraud_in_probes_scalar(&q, dataset, &probes)
}

/// Calcula L2² query × cada centroide e retorna os `N_PROBES` ids mais perto,
/// ordenados arbitrariamente (a ordem dentro do conjunto não importa).
fn select_probes(query: &[i16; DIM], dataset: &Dataset) -> [u32; N_PROBES] {
    let columns: [&[i16]; DIM] = std::array::from_fn(|d| dataset.centroid_column(d));
    debug_assert_eq!(columns[0].len(), NLIST);

    let mut top = [(i32::MAX, 0_u32); N_PROBES];
    let mut len = 0_usize;

    for c in 0..NLIST {
        let mut acc: i32 = 0;
        for d in 0..DIM {
            // SAFETY: c < NLIST = column.len().
            let v = unsafe { *columns[d].get_unchecked(c) };
            let diff = i32::from(query[d]) - i32::from(v);
            acc += diff * diff;
        }

        if len < N_PROBES {
            top[len] = (acc, c as u32);
            len += 1;
            continue;
        }
        // Substitui o pior caso se o atual for menor.
        let mut max_pos = 0;
        let mut max_val = top[0].0;
        for (i, entry) in top.iter().enumerate().take(N_PROBES).skip(1) {
            if entry.0 > max_val {
                max_val = entry.0;
                max_pos = i;
            }
        }
        if acc < max_val {
            top[max_pos] = (acc, c as u32);
        }
    }

    let mut out = [0_u32; N_PROBES];
    for (i, t) in top.iter().enumerate() {
        out[i] = t.1;
    }
    out
}

fn count_fraud_in_probes_scalar(
    query: &[i16; DIM],
    dataset: &Dataset,
    probes: &[u32; N_PROBES],
) -> u8 {
    let mut top = TopK::new();
    let columns: [&[i16]; DIM] = std::array::from_fn(|d| dataset.dim_column(d));

    for &c in probes {
        let (start, end) = dataset.cluster_range(c as usize);
        for i in start..end {
            let mut acc: i32 = 0;
            for d in 0..DIM {
                // SAFETY: i < end ≤ n ≤ n_padded = column.len().
                let v = unsafe { *columns[d].get_unchecked(i) };
                let diff = i32::from(query[d]) - i32::from(v);
                acc += diff * diff;
            }
            top.try_push(acc, i as u32);
        }
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
