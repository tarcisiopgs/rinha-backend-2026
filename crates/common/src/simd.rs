//! Operações SIMD pro hot path da busca k-NN.
//!
//! Strategy: dot product i16 x i16 → i32 acumulado. AVX2 processa 16 i16/ciclo
//! via `_mm256_madd_epi16`. Fallback escalar pra arquiteturas sem AVX2.

use crate::dataset::DIM;

/// Distância L2² entre `query` (i16; len = DIM) e o vetor `i` no dataset SoA.
///
/// Layout SoA permite varrer múltiplos vetores em batch tocando cache linha
/// a linha. Esta função opera em um único vetor — para batch, ver `l2_batch`.
#[inline]
#[must_use]
pub fn l2_squared_scalar(query: &[i16; DIM], dataset_vec: &[i16; DIM]) -> i64 {
    let mut acc: i64 = 0;
    for d in 0..DIM {
        let diff = i32::from(query[d]) - i32::from(dataset_vec[d]);
        acc += i64::from(diff * diff);
    }
    acc
}

/// Quantiza vetor f32 → i16 com escala dada. Satura em [`i16::MIN`, `i16::MAX`].
#[inline]
pub fn quantize(input: &[f32], scale: f32, out: &mut [i16]) {
    debug_assert_eq!(input.len(), out.len());
    for (i, &v) in input.iter().enumerate() {
        let q = (v * scale).round();
        out[i] = q.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_squared_zero_for_identical() {
        let v = [1_i16; DIM];
        assert_eq!(l2_squared_scalar(&v, &v), 0);
    }

    #[test]
    fn l2_squared_known_distance() {
        let mut a = [0_i16; DIM];
        let mut b = [0_i16; DIM];
        a[0] = 3;
        b[0] = 7;
        // (7 - 3)^2 = 16
        assert_eq!(l2_squared_scalar(&a, &b), 16);
    }

    #[test]
    fn quantize_round_trip() {
        let input = [0.5_f32; DIM];
        let mut out = [0_i16; DIM];
        quantize(&input, 8192.0, &mut out);
        assert_eq!(out, [4096_i16; DIM]);
    }
}
