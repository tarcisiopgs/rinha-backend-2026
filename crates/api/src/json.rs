//! Parsing do payload `POST /fraud-score` via serde_json. Estrutura segue
//! [API.md] do desafio. Otimização futura: parser manual com memchr no hot path.
//!
//! [API.md]: https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/br/API.md

use common::normalize::{LastTransaction, PayloadView};
use common::time::DateTime;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Payload<'a> {
    #[serde(borrow, default, rename = "id")]
    pub _id: &'a str,
    #[serde(borrow)]
    pub transaction: TransactionFields<'a>,
    #[serde(borrow)]
    pub customer: CustomerFields<'a>,
    #[serde(borrow)]
    pub merchant: MerchantFields<'a>,
    pub terminal: TerminalFields,
    #[serde(borrow, default)]
    pub last_transaction: Option<LastTransactionFields<'a>>,
}

#[derive(Debug, Deserialize)]
pub struct TransactionFields<'a> {
    pub amount: f32,
    pub installments: f32,
    #[serde(borrow)]
    pub requested_at: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct CustomerFields<'a> {
    pub avg_amount: f32,
    pub tx_count_24h: f32,
    #[serde(borrow, default)]
    pub known_merchants: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
pub struct MerchantFields<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(borrow)]
    pub mcc: &'a str,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct TerminalFields {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransactionFields<'a> {
    #[serde(borrow)]
    pub timestamp: &'a str,
    pub km_from_current: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("timestamp inválido")]
    BadTimestamp,
}

/// Parseia body JSON em `PayloadView`. Aloca `Vec<&[u8]>` pros known_merchants
/// — única alocação no hot path. Considerar reuso futuro.
pub fn parse<'a>(
    body: &'a [u8],
    scratch_known: &'a mut Vec<&'a [u8]>,
) -> Result<PayloadView<'a>, BuildError> {
    let p: Payload<'a> = serde_json::from_slice(body)?;

    let requested_at =
        DateTime::parse(p.transaction.requested_at.as_bytes()).ok_or(BuildError::BadTimestamp)?;

    let last_transaction = match p.last_transaction {
        Some(lt) => {
            let ts = DateTime::parse(lt.timestamp.as_bytes()).ok_or(BuildError::BadTimestamp)?;
            Some(LastTransaction {
                timestamp: ts,
                km_from_current: lt.km_from_current,
            })
        }
        None => None,
    };

    scratch_known.clear();
    scratch_known.extend(p.customer.known_merchants.iter().map(|s| s.as_bytes()));

    Ok(PayloadView {
        amount: p.transaction.amount,
        installments: p.transaction.installments,
        requested_at,
        customer_avg_amount: p.customer.avg_amount,
        customer_tx_count_24h: p.customer.tx_count_24h,
        merchant_id: p.merchant.id.as_bytes(),
        merchant_mcc: p.merchant.mcc.as_bytes(),
        merchant_avg_amount: p.merchant.avg_amount,
        terminal_is_online: p.terminal.is_online,
        terminal_card_present: p.terminal.card_present,
        terminal_km_from_home: p.terminal.km_from_home,
        last_transaction,
        known_merchants: scratch_known,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_legit_example_from_docs() {
        let body = br#"{
            "id": "tx-1329056812",
            "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
            "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
            "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
            "last_transaction": null
        }"#;
        let mut scratch = Vec::new();
        let view = parse(body, &mut scratch).unwrap();
        assert_eq!(view.merchant_id, b"MERC-016");
        assert_eq!(view.merchant_mcc, b"5411");
        assert!(view.last_transaction.is_none());
        assert_eq!(view.known_merchants.len(), 2);
    }

    #[test]
    fn parses_with_last_transaction() {
        let body = br#"{
            "id": "x",
            "transaction": { "amount": 10, "installments": 1, "requested_at": "2026-03-11T18:00:00Z" },
            "customer": { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "M", "mcc": "5411", "avg_amount": 50 },
            "terminal": { "is_online": true, "card_present": false, "km_from_home": 5.0 },
            "last_transaction": { "timestamp": "2026-03-11T17:00:00Z", "km_from_current": 1.5 }
        }"#;
        let mut scratch = Vec::new();
        let view = parse(body, &mut scratch).unwrap();
        let last = view.last_transaction.unwrap();
        assert_eq!(last.timestamp.hour, 17);
        assert!((last.km_from_current - 1.5).abs() < 1e-6);
    }
}
