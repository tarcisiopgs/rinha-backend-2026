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

/// Fator de quantização f32 → i16 do dataset. Range pós-normalização é
/// `[-1.0, 1.0]`; com SCALE 4096 cada valor cabe em `[-4096, 4096]`. Diff
/// entre dois pontos cabe em i16; quadrado cabe em i32; soma de DIM=14
/// quadrados também cabe em i32 (≈940M, max i32 ≈2.1G).
pub const QUANT_SCALE: f32 = 4096.0;

/// `NULL_SENTINEL` (-1.0) quantizado.
pub const NULL_SENTINEL_I16: i16 = -4096;

/// Quantidade de centroides do índice IVF (kmeans inverted file).
/// Sqrt(3M) ≈ 1732 → 1024 mantém clusters em torno de ~3000 vetores cada.
pub const NLIST: usize = 1024;

/// Quantos clusters mais próximos são escaneados em cada query (recall vs custo).
/// Em 1024 clusters de ~3k vetores cada, 4 clusters cobrem ~12k vetores
/// (~0.4% do dataset). Empiricamente o recall@5 cai poucos pontos vs N=8 e
/// a redução no compute é crítica no Mac Mini Late 2014 (2 cores) que roda
/// o bench oficial.
pub const N_PROBES: usize = 4;
