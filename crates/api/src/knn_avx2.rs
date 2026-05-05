//! Implementação AVX2 + FMA do k-NN brute-force.
//!
//! Estratégia: SoA f32, processa 8 vetores em paralelo por iteração com
//! 4 acumuladores independentes (4-way unrolling) pra mascarar latência da
//! cadeia FMA. 14 dimensões × 8 lanes = scan completo de 8 vetores em ~17
//! ciclos efetivos no Haswell.
//!
//! O módulo é cfg-gated em `target_arch = "x86_64"` no `main.rs`.

use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps, _mm256_sub_ps,
};

use common::proto::K;
use common::{Dataset, DIM};

use crate::knn::TopK;

const LANES: usize = 8;

/// Conta fraudes entre os K vizinhos mais próximos. Caller garante AVX2+FMA.
///
/// # Safety
/// Chamador deve ter validado `is_x86_feature_detected!("avx2")` e
/// `is_x86_feature_detected!("fma")` antes de invocar.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn count_fraud_neighbors_avx2(query: &[f32; DIM], dataset: &Dataset) -> u8 {
    let mut top = TopK::new();
    let n = dataset.len();
    let n_padded = dataset.n_padded();

    let columns: [*const f32; DIM] = std::array::from_fn(|d| dataset.dim_column(d).as_ptr());

    // SAFETY: target_feature avx2 garantida pelo chamador.
    let q: [__m256; DIM] = std::array::from_fn(|d| unsafe { _mm256_set1_ps(query[d]) });

    let mut chunk = 0;
    while chunk < n_padded {
        // SAFETY: chunk + LANES ≤ n_padded por construção do dataset.
        let dists = unsafe { chunk_dists(&q, &columns, chunk) };
        let mut buf = [0.0_f32; LANES];
        // SAFETY: buf tem LANES f32 contíguos.
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), dists) };
        for (lane, &dist) in buf.iter().enumerate() {
            let idx = chunk + lane;
            if idx >= n {
                break;
            }
            top.try_push(dist, idx as u32);
        }
        chunk += LANES;
    }

    let mut count = 0_u8;
    for slot in 0..top.len {
        if dataset.is_fraud(top.indices[slot] as usize) {
            count += 1;
        }
    }
    debug_assert!(count <= K as u8);
    count
}

/// Calcula L2² entre `query` e os 8 vetores do dataset começando em `chunk`.
/// Usa 4 acumuladores independentes pra paralelismo no pipeline FMA.
///
/// # Safety
/// `chunk + LANES ≤ n_padded` em cada `dim_column`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn chunk_dists(q: &[__m256; DIM], columns: &[*const f32; DIM], chunk: usize) -> __m256 {
    // SAFETY: target_feature avx2,fma; ponteiros alinhados a 8 lanes f32 dentro de n_padded.
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        macro_rules! step {
            ($acc:ident, $d:expr) => {{
                let v = _mm256_loadu_ps(columns[$d].add(chunk));
                let diff = _mm256_sub_ps(q[$d], v);
                $acc = _mm256_fmadd_ps(diff, diff, $acc);
            }};
        }

        step!(acc0, 0);
        step!(acc1, 1);
        step!(acc2, 2);
        step!(acc3, 3);
        step!(acc0, 4);
        step!(acc1, 5);
        step!(acc2, 6);
        step!(acc3, 7);
        step!(acc0, 8);
        step!(acc1, 9);
        step!(acc2, 10);
        step!(acc3, 11);
        step!(acc0, 12);
        step!(acc1, 13);

        let s01 = _mm256_add_ps(acc0, acc1);
        let s23 = _mm256_add_ps(acc2, acc3);
        _mm256_add_ps(s01, s23)
    }
}
