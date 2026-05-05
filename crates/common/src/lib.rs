#![doc = "Tipos e utilitários compartilhados entre API, LB e preprocess."]

pub mod dataset;
pub mod mcc;
pub mod normalize;
pub mod proto;
pub mod simd;
pub mod time;

pub use dataset::{Dataset, DatasetError};
pub use mcc::McCRiskTable;
pub use normalize::{normalize, NormalizationConfig};
pub use proto::{
    APPROVED_THRESHOLD, DIM, K, NULL_SENTINEL, NULL_SENTINEL_I16, QUANT_SCALE, SCORE_BUCKETS,
};
