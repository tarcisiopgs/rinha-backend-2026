//! Dataset de referência em SoA f32 + label binário.
//!
//! Formato binário (little-endian, alinhado a 64 bytes pra AVX2):
//! ```text
//! HEADER (32 bytes)
//!   magic        u32  = 0x52424B33 ("RBK3")
//!   version      u32  = 3
//!   dim          u32  = 14
//!   n            u32  = 3_000_000
//!   _reserved    [u8; 16]
//!
//! LABELS  (n bytes)             // u8: 0 = legit, 1 = fraud
//! PAD     (até múltiplo de 64)
//! VECTORS (DIM * n_padded * 4)  // f32 SoA: dim 0 todos n_padded, dim 1 todos n_padded, ...
//!                               // Sentinela `-1.0` mantida do payload (last_transaction null).
//! ```
//!
//! `n_padded = ceil(n / 8) * 8` permite carregar 8 lanes f32 (256-bit AVX2)
//! sem checagem de bounds no hot path. Posições extras recebem distância
//! infinita via valores fora do espaço normalizado (TopK descarta).

use std::path::Path;

use memmap2::Mmap;

use crate::proto::DIM;

pub const MAGIC: u32 = 0x5242_4B33;
pub const VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 32;
pub const LABEL_LEGIT: u8 = 0;
pub const LABEL_FRAUD: u8 = 1;
pub const ALIGN: usize = 64;
pub const SIMD_LANES: usize = 8;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("magic inválido: esperado 0x{MAGIC:08X}, encontrado 0x{0:08X}")]
    BadMagic(u32),

    #[error("versão não suportada: esperado {VERSION}, encontrado {0}")]
    BadVersion(u32),

    #[error("dim inesperado: esperado {DIM}, encontrado {0}")]
    BadDim(u32),

    #[error("arquivo truncado: esperado {expected} bytes, tem {actual}")]
    Truncated { expected: usize, actual: usize },
}

#[derive(Debug)]
pub struct Dataset {
    _mmap: Mmap,
    n: usize,
    n_padded: usize,
    labels: *const u8,
    vectors: *const f32,
}

// SAFETY: mmap imutável; ponteiros vivem enquanto _mmap vive. Sem mutação.
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

impl Dataset {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DatasetError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: arquivo não será modificado durante o tempo de vida do Dataset.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(DatasetError::Truncated {
                expected: HEADER_SIZE,
                actual: mmap.len(),
            });
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(DatasetError::BadMagic(magic));
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(DatasetError::BadVersion(version));
        }

        let dim = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if dim as usize != DIM {
            return Err(DatasetError::BadDim(dim));
        }

        let n = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let n_padded = padded_n(n);

        let labels_off = HEADER_SIZE;
        let vectors_off = align_up(labels_off + n, ALIGN);
        let expected = vectors_off + DIM * n_padded * std::mem::size_of::<f32>();

        if mmap.len() < expected {
            return Err(DatasetError::Truncated {
                expected,
                actual: mmap.len(),
            });
        }

        let labels = unsafe { mmap.as_ptr().add(labels_off) };
        let vectors = unsafe { mmap.as_ptr().add(vectors_off).cast::<f32>() };

        Ok(Self {
            _mmap: mmap,
            n,
            n_padded,
            labels,
            vectors,
        })
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `n` arredondado pra múltiplo de SIMD_LANES. Hot path SIMD lê
    /// `n_padded` lanes, posições além de `n` contêm valores que produzem
    /// distância grande o suficiente pra serem descartadas pelo TopK.
    #[inline]
    #[must_use]
    pub fn n_padded(&self) -> usize {
        self.n_padded
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `true` se vetor `i` é fraude. Posições além de `n` retornam `false`.
    #[inline]
    #[must_use]
    pub fn is_fraud(&self, i: usize) -> bool {
        if i >= self.n {
            return false;
        }
        // SAFETY: i < n.
        unsafe { *self.labels.add(i) == LABEL_FRAUD }
    }

    /// Slice contíguo da dimensão `d`, tamanho = `n_padded`.
    #[inline]
    #[must_use]
    pub fn dim_column(&self, d: usize) -> &[f32] {
        debug_assert!(d < DIM);
        // SAFETY: matriz SoA tem DIM colunas de tamanho n_padded.
        unsafe { std::slice::from_raw_parts(self.vectors.add(d * self.n_padded), self.n_padded) }
    }

    #[inline]
    #[must_use]
    pub fn vectors_ptr(&self) -> *const f32 {
        self.vectors
    }
}

#[inline]
#[must_use]
pub const fn padded_n(n: usize) -> usize {
    n.div_ceil(SIMD_LANES) * SIMD_LANES
}

#[inline]
#[must_use]
pub const fn align_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}
