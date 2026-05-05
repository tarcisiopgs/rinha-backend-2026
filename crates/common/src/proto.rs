//! Constantes do protocolo da Rinha 2026.

/// Dimensão do vetor de detecção.
pub const DIM: usize = 14;

/// k vizinhos mais próximos.
pub const K: usize = 5;

/// `approved = fraud_score < APPROVED_THRESHOLD` (fixo na especificação).
pub const APPROVED_THRESHOLD: f32 = 0.6;

/// Buckets discretos de `fraud_score` (n_fraudes / 5).
pub const SCORE_BUCKETS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// Sentinela "ausência de dado" nos índices 5/6 quando `last_transaction = null`.
/// A spec define -1.0 explicitamente; mantemos o valor literal pra que
/// transações sem histórico fiquem naturalmente agrupadas no espaço vetorial.
pub const NULL_SENTINEL: f32 = -1.0;

/// Default de risco MCC quando categoria não está em `mcc_risk.json`.
pub const MCC_DEFAULT_RISK: f32 = 0.5;
