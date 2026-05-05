//! Dataset de referência quantizado i16 SoA + label binário.
//!
//! Formato binário (little-endian):
//! ```text
//! HEADER (32 bytes)
//!   magic        u32  = 0x52424B32 ("RBK2")
//!   version      u32  = 2
//!   dim          u32  = 14
//!   n            u32  = 3_000_000
//!   scale        f32  = 8192.0
//!   _reserved    [u8; 12]
//!
//! LABELS  (n bytes)         // u8: 0 = legit, 1 = fraud
//! VECTORS (DIM * n * 2 b)   // i16 SoA: dim 0 todos n, dim 1 todos n, ...
//!                           // Sentinela `i16::MIN` = ausência de dado (-1 do payload).
//! ```
//!
//! SoA + alinhamento natural acelera SIMD AVX2 e reduz cache miss em batch
//! por dimensão. mmap deixa o kernel paginar sob demanda.

use std::path::Path;

use memmap2::Mmap;

use crate::proto::DIM;

pub const MAGIC: u32 = 0x5242_4B32;
pub const VERSION: u32 = 2;
pub const HEADER_SIZE: usize = 32;
pub const LABEL_LEGIT: u8 = 0;
pub const LABEL_FRAUD: u8 = 1;

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
    scale: f32,
    labels: *const u8,
    vectors: *const i16,
}

// SAFETY: mmap imutável compartilhada; ponteiros vivem enquanto _mmap vive.
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
        let scale = f32::from_le_bytes(mmap[16..20].try_into().unwrap());

        let labels_off = HEADER_SIZE;
        let vectors_off = labels_off + n;
        let expected = vectors_off + DIM * n * std::mem::size_of::<i16>();

        if mmap.len() < expected {
            return Err(DatasetError::Truncated {
                expected,
                actual: mmap.len(),
            });
        }

        let labels = unsafe { mmap.as_ptr().add(labels_off) };
        let vectors = unsafe { mmap.as_ptr().add(vectors_off).cast::<i16>() };

        Ok(Self {
            _mmap: mmap,
            n,
            scale,
            labels,
            vectors,
        })
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// `true` se vetor `i` é fraude.
    #[inline]
    #[must_use]
    pub fn is_fraud(&self, i: usize) -> bool {
        debug_assert!(i < self.n);
        // SAFETY: i validado por debug_assert.
        unsafe { *self.labels.add(i) == LABEL_FRAUD }
    }

    /// Slice contíguo da dimensão `d`. Tamanho = `n`.
    #[inline]
    #[must_use]
    pub fn dim_column(&self, d: usize) -> &[i16] {
        debug_assert!(d < DIM);
        // SAFETY: matriz SoA tem DIM colunas de tamanho n.
        unsafe { std::slice::from_raw_parts(self.vectors.add(d * self.n), self.n) }
    }

    #[inline]
    #[must_use]
    pub fn vectors_ptr(&self) -> *const i16 {
        self.vectors
    }
}
