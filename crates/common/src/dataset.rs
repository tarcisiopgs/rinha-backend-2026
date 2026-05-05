//! Dataset de referência quantizado i16 + score, layout SoA.
//!
//! Formato binário (little-endian):
//! ```text
//! HEADER (32 bytes)
//!   magic         u32  = 0x52424B32 ("RBK2")
//!   version       u32  = 1
//!   dim           u32  = 14
//!   n             u32  = 3_000_000
//!   scale         f32  = 8192.0
//!   _reserved     [u8; 12]
//!
//! SCORES (n * 1 byte)        // u8 ∈ {0, 51, 102, 153, 204, 255} (0.0..=1.0 * 255)
//! VECTORS_SOA (dim * n * 2 bytes)  // i16 column-major: dim 0 todos n, dim 1 todos n, ...
//! ```
//!
//! SoA + alinhamento 64 bytes acelera SIMD AVX2 e reduz cache miss.

use std::path::Path;

use memmap2::Mmap;

pub const DIM: usize = 14;
pub const MAGIC: u32 = 0x5242_4B32;
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 32;
pub const DEFAULT_SCALE: f32 = 8192.0;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("magic inválido: esperado 0x{MAGIC:08X}, encontrado 0x{0:08X}")]
    BadMagic(u32),

    #[error("versão não suportada: {0}")]
    BadVersion(u32),

    #[error("dim inesperado: esperado {DIM}, encontrado {0}")]
    BadDim(u32),

    #[error("arquivo truncado: esperado {expected} bytes, tem {actual}")]
    Truncated { expected: usize, actual: usize },
}

/// Dataset memory-mapped, somente leitura.
///
/// Carregamento O(1): apenas mapeia o arquivo. Páginas são paginadas sob demanda
/// pelo kernel — primeira passada de busca toca tudo, subsequentes ficam em RSS.
#[derive(Debug)]
pub struct Dataset {
    _mmap: Mmap,
    n: usize,
    scale: f32,
    scores: *const u8,
    /// Ponteiro pra início da matriz SoA. Indexação: `vectors[d * n + i]`.
    vectors: *const i16,
}

// SAFETY: ponteiros apontam pra mmap imutável que sobrevive a Self via _mmap.
// Sem mutação concorrente possível (somente leitura).
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

impl Dataset {
    /// Abre dataset binário pré-processado via mmap.
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

        let magic = u32::from_le_bytes(mmap[0..4].try_into().expect("4 bytes"));
        if magic != MAGIC {
            return Err(DatasetError::BadMagic(magic));
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().expect("4 bytes"));
        if version != VERSION {
            return Err(DatasetError::BadVersion(version));
        }

        let dim = u32::from_le_bytes(mmap[8..12].try_into().expect("4 bytes"));
        if dim as usize != DIM {
            return Err(DatasetError::BadDim(dim));
        }

        let n = u32::from_le_bytes(mmap[12..16].try_into().expect("4 bytes")) as usize;
        let scale = f32::from_le_bytes(mmap[16..20].try_into().expect("4 bytes"));

        let scores_off = HEADER_SIZE;
        let vectors_off = scores_off + n;
        let expected = vectors_off + DIM * n * std::mem::size_of::<i16>();

        if mmap.len() < expected {
            return Err(DatasetError::Truncated {
                expected,
                actual: mmap.len(),
            });
        }

        let scores = unsafe { mmap.as_ptr().add(scores_off) };
        let vectors = unsafe { mmap.as_ptr().add(vectors_off).cast::<i16>() };

        Ok(Self {
            _mmap: mmap,
            n,
            scale,
            scores,
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

    /// Score quantizado u8 ∈ [0, 255]. Multiplicar por `1.0 / 255.0` pra obter f32.
    #[inline]
    #[must_use]
    pub fn score_u8(&self, i: usize) -> u8 {
        debug_assert!(i < self.n);
        // SAFETY: i validado por debug_assert; release confia no chamador.
        unsafe { *self.scores.add(i) }
    }

    /// Slice contíguo de uma dimensão. Tamanho = n.
    #[inline]
    #[must_use]
    pub fn dim_column(&self, d: usize) -> &[i16] {
        debug_assert!(d < DIM);
        // SAFETY: matriz SoA tem DIM colunas de tamanho n; d validado.
        unsafe { std::slice::from_raw_parts(self.vectors.add(d * self.n), self.n) }
    }

    /// Ponteiro raw pra início da matriz SoA. Uso em hot path SIMD.
    #[inline]
    #[must_use]
    pub fn vectors_ptr(&self) -> *const i16 {
        self.vectors
    }
}
