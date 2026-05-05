//! Normalização payload → vetor f32 de 14 dimensões. Implementa as fórmulas
//! definidas em [REGRAS_DE_DETECCAO.md] do desafio. Sentinela `-1` para
//! índices 5 e 6 quando `last_transaction` é null.
//!
//! [REGRAS_DE_DETECCAO.md]: https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/REGRAS_DE_DETECCAO.md

use crate::mcc::McCRiskTable;
use crate::proto::DIM;
use crate::time::DateTime;

/// Constantes do `normalization.json`. Defaults espelham o arquivo oficial.
#[derive(Debug, Clone, Copy)]
pub struct NormalizationConfig {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1_440.0,
            max_km: 1_000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }
}

/// Entrada da normalização — view sobre o payload já parseado.
#[derive(Debug, Clone, Copy)]
pub struct PayloadView<'a> {
    pub amount: f32,
    pub installments: f32,
    pub requested_at: DateTime,
    pub customer_avg_amount: f32,
    pub customer_tx_count_24h: f32,
    pub merchant_id: &'a [u8],
    pub merchant_mcc: &'a [u8],
    pub merchant_avg_amount: f32,
    pub terminal_is_online: bool,
    pub terminal_card_present: bool,
    pub terminal_km_from_home: f32,
    pub last_transaction: Option<LastTransaction>,
    pub known_merchants: &'a [&'a [u8]],
}

#[derive(Debug, Clone, Copy)]
pub struct LastTransaction {
    pub timestamp: DateTime,
    pub km_from_current: f32,
}

/// Aplica as 14 fórmulas e devolve `[f32; 14]` em ordem.
///
/// Os índices 5 e 6 recebem `-1.0` quando `last_transaction` é `None`.
#[must_use]
pub fn normalize(p: &PayloadView<'_>, cfg: &NormalizationConfig, mcc: &McCRiskTable) -> [f32; DIM] {
    let mut v = [0.0_f32; DIM];

    v[0] = clamp01(p.amount / cfg.max_amount);
    v[1] = clamp01(p.installments / cfg.max_installments);

    let avg = if p.customer_avg_amount > 0.0 {
        p.customer_avg_amount
    } else {
        1.0
    };
    v[2] = clamp01((p.amount / avg) / cfg.amount_vs_avg_ratio);

    v[3] = f32::from(p.requested_at.hour) / 23.0;
    v[4] = f32::from(p.requested_at.day_of_week_mon0()) / 6.0;

    if let Some(last) = p.last_transaction {
        let minutes = (p.requested_at.minutes_since_epoch() - last.timestamp.minutes_since_epoch())
            .max(0) as f32;
        v[5] = clamp01(minutes / cfg.max_minutes);
        v[6] = clamp01(last.km_from_current / cfg.max_km);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    v[7] = clamp01(p.terminal_km_from_home / cfg.max_km);
    v[8] = clamp01(p.customer_tx_count_24h / cfg.max_tx_count_24h);
    v[9] = if p.terminal_is_online { 1.0 } else { 0.0 };
    v[10] = if p.terminal_card_present { 1.0 } else { 0.0 };
    v[11] = if known_contains(p.known_merchants, p.merchant_id) {
        0.0
    } else {
        1.0
    };
    v[12] = mcc.lookup(p.merchant_mcc);
    v[13] = clamp01(p.merchant_avg_amount / cfg.max_merchant_avg_amount);

    v
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline]
fn known_contains(known: &[&[u8]], target: &[u8]) -> bool {
    known.iter().any(|m| *m == target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dt(s: &str) -> DateTime {
        DateTime::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn normalize_legit_example_from_docs() {
        // Exemplo da spec: tx-1329056812
        let cfg = NormalizationConfig::default();
        let mcc = McCRiskTable::default();
        let known: [&[u8]; 2] = [b"MERC-003", b"MERC-016"];
        let p = PayloadView {
            amount: 41.12,
            installments: 2.0,
            requested_at: dt("2026-03-11T18:45:53Z"),
            customer_avg_amount: 82.24,
            customer_tx_count_24h: 3.0,
            merchant_id: b"MERC-016",
            merchant_mcc: b"5411",
            merchant_avg_amount: 60.25,
            terminal_is_online: false,
            terminal_card_present: true,
            terminal_km_from_home: 29.23,
            last_transaction: None,
            known_merchants: &known,
        };
        let v = normalize(&p, &cfg, &mcc);

        // Esperado da spec:
        // [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        let expected = [
            0.0041_f32, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.0292, 0.15, 0.0, 1.0, 0.0,
            0.15, 0.006,
        ];

        for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.005,
                "índice {i}: esperado {exp}, obtido {got}"
            );
        }
    }
}
