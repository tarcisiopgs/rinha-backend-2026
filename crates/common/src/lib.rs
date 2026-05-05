#![doc = "Tipos e utilitários compartilhados entre API, LB e preprocess."]

pub mod dataset;
pub mod proto;
pub mod simd;

pub use dataset::{Dataset, DatasetError, DIM};
