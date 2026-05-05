//! Parser JSON manual focado no schema do `POST /fraud-score`. Substitui
//! `serde_json` no hot path: zero alocação no caso comum, varredura linear
//! com `memchr` pra encontrar chaves e valores.

use common::normalize::{LastTransaction, PayloadView};
use common::time::DateTime;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("json malformado")]
    Malformed,

    #[error("timestamp inválido")]
    BadTimestamp,
}

/// Parseia o corpo `POST /fraud-score` e devolve um `PayloadView`. O caller
/// passa um `Vec<&[u8]>` reutilizável (`scratch_known`) que é populado com
/// os elementos de `customer.known_merchants`. O Vec é apenas referência —
/// os bytes vivem em `body`.
pub fn parse<'a>(
    body: &'a [u8],
    scratch_known: &'a mut Vec<&'a [u8]>,
) -> Result<PayloadView<'a>, BuildError> {
    scratch_known.clear();

    // Procura `"transaction":` e parseia campos esperados na ordem
    // `amount`, `installments`, `requested_at`. Os payloads do desafio
    // sempre seguem esse esquema.
    let trans_obj = locate_object(body, b"\"transaction\"")?;
    let amount = read_number(trans_obj, b"\"amount\"")?;
    let installments = read_number(trans_obj, b"\"installments\"")?;
    let requested_at_bytes = read_string(trans_obj, b"\"requested_at\"")?;
    let requested_at = DateTime::parse(requested_at_bytes).ok_or(BuildError::BadTimestamp)?;

    let cust_obj = locate_object(body, b"\"customer\"")?;
    let customer_avg_amount = read_number(cust_obj, b"\"avg_amount\"")?;
    let customer_tx_count_24h = read_number(cust_obj, b"\"tx_count_24h\"")?;
    parse_string_array(cust_obj, b"\"known_merchants\"", scratch_known)?;

    let merch_obj = locate_object(body, b"\"merchant\"")?;
    let merchant_id = read_string(merch_obj, b"\"id\"")?;
    let merchant_mcc = read_string(merch_obj, b"\"mcc\"")?;
    let merchant_avg_amount = read_number(merch_obj, b"\"avg_amount\"")?;

    let term_obj = locate_object(body, b"\"terminal\"")?;
    let terminal_is_online = read_bool(term_obj, b"\"is_online\"")?;
    let terminal_card_present = read_bool(term_obj, b"\"card_present\"")?;
    let terminal_km_from_home = read_number(term_obj, b"\"km_from_home\"")?;

    let last_transaction = parse_last_transaction(body)?;

    Ok(PayloadView {
        amount,
        installments,
        requested_at,
        customer_avg_amount,
        customer_tx_count_24h,
        merchant_id,
        merchant_mcc,
        merchant_avg_amount,
        terminal_is_online,
        terminal_card_present,
        terminal_km_from_home,
        last_transaction,
        known_merchants: scratch_known,
    })
}

fn parse_last_transaction(body: &[u8]) -> Result<Option<LastTransaction>, BuildError> {
    let key = b"\"last_transaction\"";
    let pos = match memchr::memmem::find(body, key) {
        Some(p) => p,
        None => return Ok(None),
    };
    let after_colon = skip_to_value(body, pos + key.len())?;
    if after_colon.starts_with(b"null") {
        return Ok(None);
    }
    if !after_colon.starts_with(b"{") {
        return Err(BuildError::Malformed);
    }
    let obj = extract_object_body(after_colon)?;
    let timestamp_bytes = read_string(obj, b"\"timestamp\"")?;
    let timestamp = DateTime::parse(timestamp_bytes).ok_or(BuildError::BadTimestamp)?;
    let km_from_current = read_number(obj, b"\"km_from_current\"")?;
    Ok(Some(LastTransaction {
        timestamp,
        km_from_current,
    }))
}

/// Encontra `"chave":` em `body` e devolve o slice **entre as chaves `{}`**
/// do valor, assumindo que o valor é um objeto.
fn locate_object<'a>(body: &'a [u8], key: &[u8]) -> Result<&'a [u8], BuildError> {
    let pos = memchr::memmem::find(body, key).ok_or(BuildError::Malformed)?;
    let after = skip_to_value(body, pos + key.len())?;
    extract_object_body(after)
}

/// Recebe slice começando em `{`, devolve slice do conteúdo entre `{}`
/// (sem as chaves), respeitando aninhamento e strings.
fn extract_object_body(after: &[u8]) -> Result<&[u8], BuildError> {
    if !after.starts_with(b"{") {
        return Err(BuildError::Malformed);
    }
    let mut depth = 1_i32;
    let mut i = 1_usize;
    while i < after.len() {
        let b = after[i];
        if b == b'"' {
            i = skip_string(after, i)?;
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Ok(&after[1..i]);
            }
        }
        i += 1;
    }
    Err(BuildError::Malformed)
}

/// `i` aponta pra `"`. Avança até a `"` de fechamento, retorna índice
/// imediatamente após. Suporta escapes `\"`.
fn skip_string(buf: &[u8], i: usize) -> Result<usize, BuildError> {
    debug_assert_eq!(buf.get(i), Some(&b'"'));
    let mut j = i + 1;
    while j < buf.len() {
        let b = buf[j];
        if b == b'\\' {
            j += 2;
            continue;
        }
        if b == b'"' {
            return Ok(j + 1);
        }
        j += 1;
    }
    Err(BuildError::Malformed)
}

/// `start` aponta logo após uma chave (depois das aspas). Pula whitespace
/// e o `:`, devolve slice começando no primeiro byte do valor.
fn skip_to_value(buf: &[u8], start: usize) -> Result<&[u8], BuildError> {
    let mut i = start;
    while i < buf.len() {
        match buf[i] {
            b' ' | b'\t' | b'\n' | b'\r' | b':' => i += 1,
            _ => return Ok(&buf[i..]),
        }
    }
    Err(BuildError::Malformed)
}

fn read_number(obj: &[u8], key: &[u8]) -> Result<f32, BuildError> {
    let pos = memchr::memmem::find(obj, key).ok_or(BuildError::Malformed)?;
    let after = skip_to_value(obj, pos + key.len())?;
    let end = number_end(after);
    if end == 0 {
        return Err(BuildError::Malformed);
    }
    parse_f32(&after[..end])
}

fn read_string<'a>(obj: &'a [u8], key: &[u8]) -> Result<&'a [u8], BuildError> {
    let pos = memchr::memmem::find(obj, key).ok_or(BuildError::Malformed)?;
    let after = skip_to_value(obj, pos + key.len())?;
    if !after.starts_with(b"\"") {
        return Err(BuildError::Malformed);
    }
    // Encontra a `"` de fechamento (sem escapes — payloads do desafio são
    // ascii puro com IDs e timestamps).
    let close = memchr::memchr(b'"', &after[1..]).ok_or(BuildError::Malformed)?;
    Ok(&after[1..1 + close])
}

fn read_bool(obj: &[u8], key: &[u8]) -> Result<bool, BuildError> {
    let pos = memchr::memmem::find(obj, key).ok_or(BuildError::Malformed)?;
    let after = skip_to_value(obj, pos + key.len())?;
    if after.starts_with(b"true") {
        Ok(true)
    } else if after.starts_with(b"false") {
        Ok(false)
    } else {
        Err(BuildError::Malformed)
    }
}

fn parse_string_array<'a>(
    obj: &'a [u8],
    key: &[u8],
    out: &mut Vec<&'a [u8]>,
) -> Result<(), BuildError> {
    let pos = memchr::memmem::find(obj, key).ok_or(BuildError::Malformed)?;
    let after = skip_to_value(obj, pos + key.len())?;
    if !after.starts_with(b"[") {
        return Err(BuildError::Malformed);
    }
    let mut i = 1_usize;
    while i < after.len() {
        match after[i] {
            b' ' | b'\t' | b'\n' | b'\r' | b',' => i += 1,
            b']' => return Ok(()),
            b'"' => {
                let close = memchr::memchr(b'"', &after[i + 1..]).ok_or(BuildError::Malformed)?;
                out.push(&after[i + 1..i + 1 + close]);
                i += 1 + close + 1;
            }
            _ => return Err(BuildError::Malformed),
        }
    }
    Err(BuildError::Malformed)
}

#[inline]
fn number_end(buf: &[u8]) -> usize {
    let mut i = 0;
    while i < buf.len() {
        let b = buf[i];
        if b.is_ascii_digit() || b == b'-' || b == b'+' || b == b'.' || b == b'e' || b == b'E' {
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// Parser f32 minimalista — assume formato sem exponente (raro nos payloads
/// do desafio) ou com exponente simples. Casos edge (NaN, Infinity) não
/// aparecem nos dados oficiais.
fn parse_f32(s: &[u8]) -> Result<f32, BuildError> {
    let txt = std::str::from_utf8(s).map_err(|_| BuildError::Malformed)?;
    txt.parse::<f32>().map_err(|_| BuildError::Malformed)
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
        assert_eq!(view.known_merchants[0], b"MERC-003");
        assert_eq!(view.known_merchants[1], b"MERC-016");
        assert!((view.amount - 41.12).abs() < 1e-3);
        assert!((view.installments - 2.0).abs() < 1e-6);
        assert!(!view.terminal_is_online);
        assert!(view.terminal_card_present);
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
        assert_eq!(view.known_merchants.len(), 0);
    }

    #[test]
    fn rejects_malformed() {
        let body = b"{ not json";
        let mut scratch = Vec::new();
        assert!(parse(body, &mut scratch).is_err());
    }
}
