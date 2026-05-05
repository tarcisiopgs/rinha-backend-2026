//! Tabela de risco MCC. Fixed-size lookup com fallback `0.5` para chaves
//! desconhecidas (regra do `mcc_risk.json`).

use crate::proto::MCC_DEFAULT_RISK;

/// Mapeamento estático MCC → risco. Conteúdo padrão = `mcc_risk.json` oficial.
#[derive(Debug, Clone)]
pub struct McCRiskTable {
    entries: Vec<(Box<[u8]>, f32)>,
}

impl McCRiskTable {
    pub fn new(entries: impl IntoIterator<Item = (Vec<u8>, f32)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.into_boxed_slice(), v))
                .collect(),
        }
    }

    #[inline]
    #[must_use]
    pub fn lookup(&self, mcc: &[u8]) -> f32 {
        for (k, v) in &self.entries {
            if k.as_ref() == mcc {
                return *v;
            }
        }
        MCC_DEFAULT_RISK
    }
}

impl Default for McCRiskTable {
    fn default() -> Self {
        Self::new([
            (b"5411".to_vec(), 0.15),
            (b"5812".to_vec(), 0.30),
            (b"5912".to_vec(), 0.20),
            (b"5944".to_vec(), 0.45),
            (b"7801".to_vec(), 0.80),
            (b"7802".to_vec(), 0.75),
            (b"7995".to_vec(), 0.85),
            (b"4511".to_vec(), 0.35),
            (b"5311".to_vec(), 0.25),
            (b"5999".to_vec(), 0.50),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_mcc_returns_value() {
        let t = McCRiskTable::default();
        assert!((t.lookup(b"5411") - 0.15).abs() < 1e-6);
        assert!((t.lookup(b"7995") - 0.85).abs() < 1e-6);
    }

    #[test]
    fn unknown_mcc_returns_default() {
        let t = McCRiskTable::default();
        assert!((t.lookup(b"9999") - 0.5).abs() < 1e-6);
    }
}
