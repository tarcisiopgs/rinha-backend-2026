//! Implementação AVX2 do k-NN sobre IVF (clusters reordenados em SoA i16).
//!
//! O scan brute-force é restrito aos clusters retornados pela seleção de
//! probes em `knn::select_probes`. Cada cluster é varrido em chunks de 8
//! vetores (LANES=8), com 4 acumuladores i32 independentes pra paralelizar
//! `mullo_epi32`. Lanes que ultrapassam o fim do cluster são filtradas via
//! `idx >= cluster_end` no caller.

use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_cvtepi16_epi32, _mm256_mullo_epi32,
    _mm256_setzero_si256, _mm256_storeu_si256, _mm_loadu_si128, _mm_set1_epi16, _mm_sub_epi16,
};

use common::proto::{K, N_PROBES};
use common::{Dataset, DIM};

use crate::knn::TopK;

const LANES: usize = 8;

/// Conta fraudes entre os K vizinhos mais próximos limitados aos clusters
/// listados em `probes`. Caller garante AVX2.
///
/// # Safety
/// Chamador deve ter validado `is_x86_feature_detected!("avx2")` antes de invocar.
#[target_feature(enable = "avx2")]
pub unsafe fn count_fraud_in_probes_avx2(
    query: &[i16; DIM],
    dataset: &Dataset,
    probes: &[u32; N_PROBES],
) -> u8 {
    let mut top = TopK::new();
    let n_padded = dataset.n_padded();
    let columns: [*const i16; DIM] = std::array::from_fn(|d| dataset.dim_column(d).as_ptr());

    // SAFETY: target_feature avx2 garantida pelo chamador.
    let q: [__m128i; DIM] = std::array::from_fn(|d| unsafe { _mm_set1_epi16(query[d]) });

    let mut buf = [0_i32; LANES];

    for &c in probes {
        let (start, end) = dataset.cluster_range(c as usize);
        let mut chunk = start;
        while chunk < end {
            // Garante que chunk + LANES não passa do fim do array global.
            // n_padded ≥ n; o último cluster termina em n; logo chunk + LANES ≤ n_padded.
            debug_assert!(chunk + LANES <= n_padded);
            // SAFETY: chunk + LANES ≤ n_padded.
            let dists = unsafe { chunk_dists(&q, &columns, chunk) };
            // SAFETY: buf tem LANES i32 contíguos.
            unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast::<__m256i>(), dists) };
            for (lane, &dist) in buf.iter().enumerate() {
                let idx = chunk + lane;
                if idx >= end {
                    break;
                }
                top.try_push(dist, idx as u32);
            }
            chunk += LANES;
        }
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
/// 4 acumuladores i32 independentes pra paralelizar `mullo_epi32`.
///
/// # Safety
/// `chunk + LANES ≤ n_padded` em cada `dim_column`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn chunk_dists(q: &[__m128i; DIM], columns: &[*const i16; DIM], chunk: usize) -> __m256i {
    // SAFETY: target_feature avx2; ponteiros alinhados a 8 lanes i16 dentro de n_padded.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        macro_rules! step {
            ($acc:ident, $d:expr) => {{
                let v = _mm_loadu_si128(columns[$d].add(chunk).cast::<__m128i>());
                let diff = _mm_sub_epi16(q[$d], v);
                let diff32 = _mm256_cvtepi16_epi32(diff);
                let sq = _mm256_mullo_epi32(diff32, diff32);
                $acc = _mm256_add_epi32($acc, sq);
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

        let s01 = _mm256_add_epi32(acc0, acc1);
        let s23 = _mm256_add_epi32(acc2, acc3);
        _mm256_add_epi32(s01, s23)
    }
}
