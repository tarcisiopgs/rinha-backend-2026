//! Constantes de protocolo da Rinha 2026.

/// k vizinhos mais próximos pra agregar score.
pub const K: usize = 5;

/// Buckets discretos de score (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
pub const SCORE_BUCKETS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// Threshold de aprovação (`approved = fraud_score < THRESHOLD`).
pub const APPROVED_THRESHOLD: f32 = 0.5;
