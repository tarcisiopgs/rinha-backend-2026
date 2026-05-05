//! Operações vetoriais escalares (referência). AVX2 fica em `api::knn_avx2`.

use crate::proto::DIM;

/// Distância L2² escalar entre dois vetores f32.
///
/// Sentinela `-1.0` é tratada como valor regular — `(query[d] - v[d])²`
/// produz separação grande quando apenas um lado é null, e zero quando
/// ambos são null, exatamente o comportamento descrito na spec.
#[inline]
#[must_use]
pub fn l2_squared(query: &[f32; DIM], v: &[f32; DIM]) -> f32 {
    let mut acc = 0.0_f32;
    for d in 0..DIM {
        let diff = query[d] - v[d];
        acc += diff * diff;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_zero_for_identical() {
        let v = [0.5_f32; DIM];
        assert!(l2_squared(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn l2_known_distance() {
        let mut a = [0.0_f32; DIM];
        let mut b = [0.0_f32; DIM];
        a[0] = 0.5;
        b[0] = 0.2;
        // (0.5 - 0.2)^2 = 0.09
        assert!((l2_squared(&a, &b) - 0.09).abs() < 1e-6);
    }

    #[test]
    fn l2_null_pair_zero() {
        let null_vec = [-1.0_f32; DIM];
        assert!(l2_squared(&null_vec, &null_vec).abs() < 1e-6);
    }
}
