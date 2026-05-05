//! Constantes do protocolo da Rinha 2026.

/// Dimensão do vetor de detecção.
pub const DIM: usize = 14;

/// k vizinhos mais próximos.
pub const K: usize = 5;

/// `approved = fraud_score < APPROVED_THRESHOLD` (fixo na especificação).
pub const APPROVED_THRESHOLD: f32 = 0.6;

/// Buckets discretos de `fraud_score` (n_fraudes / 5).
pub const SCORE_BUCKETS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// Quantização f32 → i16. Range válido normalizado [0.0, 1.0] mapeia em [0, 8192].
/// Sentinela de "ausência de dado" usa `i16::MIN`.
pub const QUANT_SCALE: f32 = 8192.0;

/// Sentinela `-1` do payload (last_transaction null) é codificado como i16::MIN
/// no dataset SoA. Em distância L2², dois sentinelas ficam à distância 0; um
/// sentinela vs valor normalizado fica à distância enorme — agrupa naturalmente
/// transações sem histórico (comportamento desejado pela spec).
pub const NULL_SENTINEL: i16 = i16::MIN;

/// Default de risco MCC quando categoria não está em `mcc_risk.json`.
pub const MCC_DEFAULT_RISK: f32 = 0.5;
