//! Quantização e operações SIMD pro hot path.
//!
//! Hot path real (k-NN brute-force) está em `api::knn` — aqui ficam apenas
//! helpers reutilizáveis (quantização e baseline escalar pra testes/profile).

use crate::proto::{DIM, NULL_SENTINEL, QUANT_SCALE};

/// Quantiza vetor f32 → i16 com escala `QUANT_SCALE`. Sentinelas `-1.0`
/// (ausência de dado nas posições 5/6) viram `NULL_SENTINEL` (`i16::MIN`).
#[inline]
#[must_use]
pub fn quantize(input: &[f32; DIM]) -> [i16; DIM] {
    let mut out = [0_i16; DIM];
    for i in 0..DIM {
        out[i] = quantize_one(input[i]);
    }
    out
}

#[inline]
#[must_use]
pub fn quantize_one(x: f32) -> i16 {
    if x < 0.0 {
        NULL_SENTINEL
    } else {
        let q = (x * QUANT_SCALE).round();
        q.clamp(0.0, f32::from(i16::MAX)) as i16
    }
}

/// L2² escalar entre dois vetores quantizados. Não otimizado — referência
/// pra teste e fallback.
#[inline]
#[must_use]
pub fn l2_squared_scalar(a: &[i16; DIM], b: &[i16; DIM]) -> i64 {
    let mut acc: i64 = 0;
    for d in 0..DIM {
        let diff = i32::from(a[d]) - i32::from(b[d]);
        acc += i64::from(diff) * i64::from(diff);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_normal_value() {
        assert_eq!(quantize_one(0.5), 4096);
        assert_eq!(quantize_one(1.0), 8192);
        assert_eq!(quantize_one(0.0), 0);
    }

    #[test]
    fn quantize_negative_becomes_sentinel() {
        assert_eq!(quantize_one(-1.0), NULL_SENTINEL);
        assert_eq!(quantize_one(-0.5), NULL_SENTINEL);
    }

    #[test]
    fn l2_zero_for_identical() {
        let v = [1_i16; DIM];
        assert_eq!(l2_squared_scalar(&v, &v), 0);
    }
}
