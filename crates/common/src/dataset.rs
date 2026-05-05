//! Dataset de referência IVF (inverted file) sobre i16 quantizado + label binário.
//!
//! Formato binário (little-endian, alinhado a 64 bytes pra AVX2):
//! ```text
//! HEADER (32 bytes)
//!   magic        u32  = 0x52424B33 ("RBK3")
//!   version      u32  = 5
//!   dim          u32  = 14
//!   n            u32  = 3_000_000
//!   nlist        u32  = 1024
//!   _reserved    [u8; 12]
//!
//! LABELS_SORTED  (n bytes)                       // u8: ordem reordenada por cluster.
//! PAD            (até múltiplo de 64)
//! CENTROIDS_SOA  (DIM * NLIST * 2 bytes)         // i16, dim 0 todos os centroides, dim 1, ...
//! PAD            (até múltiplo de 64)
//! BOUNDARIES     ((nlist + 1) * 4 bytes)         // u32: índice inicial do cluster c.
//! PAD            (até múltiplo de 64)
//! VECTORS_SOA    (DIM * n_padded * 2 bytes)      // i16, vetores reordenados por cluster.
//! ```
//!
//! `n_padded = ceil(n / 8) * 8` permite lidar com SIMD de 8 lanes mesmo
//! escaneando ranges arbitrários — a última fração pode "vazar" pro próximo
//! cluster, mas o filtro `idx >= cluster_end` no caller descarta as lanes.

use std::path::Path;

use memmap2::Mmap;

use crate::proto::{DIM, NLIST};

pub const MAGIC: u32 = 0x5242_4B33;
pub const VERSION: u32 = 5;
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

    #[error("nlist inesperado: esperado {NLIST}, encontrado {0}")]
    BadNlist(u32),

    #[error("arquivo truncado: esperado {expected} bytes, tem {actual}")]
    Truncated { expected: usize, actual: usize },
}

#[derive(Debug)]
pub struct Dataset {
    _mmap: Mmap,
    n: usize,
    n_padded: usize,
    nlist: usize,
    labels: *const u8,
    centroids: *const i16,
    boundaries: *const u32,
    vectors: *const i16,
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

        let nlist = u32::from_le_bytes(mmap[16..20].try_into().unwrap());
        if nlist as usize != NLIST {
            return Err(DatasetError::BadNlist(nlist));
        }

        let labels_off = HEADER_SIZE;
        let centroids_off = align_up(labels_off + n, ALIGN);
        let boundaries_off = align_up(
            centroids_off + DIM * NLIST * std::mem::size_of::<i16>(),
            ALIGN,
        );
        let vectors_off = align_up(
            boundaries_off + (NLIST + 1) * std::mem::size_of::<u32>(),
            ALIGN,
        );
        let expected = vectors_off + DIM * n_padded * std::mem::size_of::<i16>();

        if mmap.len() < expected {
            return Err(DatasetError::Truncated {
                expected,
                actual: mmap.len(),
            });
        }

        // SAFETY: offsets validados via expected ≤ mmap.len().
        let labels = unsafe { mmap.as_ptr().add(labels_off) };
        let centroids = unsafe { mmap.as_ptr().add(centroids_off).cast::<i16>() };
        let boundaries = unsafe { mmap.as_ptr().add(boundaries_off).cast::<u32>() };
        let vectors = unsafe { mmap.as_ptr().add(vectors_off).cast::<i16>() };

        Ok(Self {
            _mmap: mmap,
            n,
            n_padded,
            nlist: NLIST,
            labels,
            centroids,
            boundaries,
            vectors,
        })
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `n` arredondado pra múltiplo de SIMD_LANES.
    #[inline]
    #[must_use]
    pub fn n_padded(&self) -> usize {
        self.n_padded
    }

    #[inline]
    #[must_use]
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `true` se vetor `i` (na ordem reordenada por cluster) é fraude.
    #[inline]
    #[must_use]
    pub fn is_fraud(&self, i: usize) -> bool {
        if i >= self.n {
            return false;
        }
        // SAFETY: i < n.
        unsafe { *self.labels.add(i) == LABEL_FRAUD }
    }

    /// Slice contíguo da dimensão `d` dos vetores reordenados, tamanho = `n_padded`.
    #[inline]
    #[must_use]
    pub fn dim_column(&self, d: usize) -> &[i16] {
        debug_assert!(d < DIM);
        // SAFETY: matriz SoA tem DIM colunas de tamanho n_padded.
        unsafe { std::slice::from_raw_parts(self.vectors.add(d * self.n_padded), self.n_padded) }
    }

    /// Slice contíguo da dimensão `d` dos centroides, tamanho = `nlist`.
    #[inline]
    #[must_use]
    pub fn centroid_column(&self, d: usize) -> &[i16] {
        debug_assert!(d < DIM);
        // SAFETY: matriz SoA tem DIM colunas de tamanho nlist.
        unsafe { std::slice::from_raw_parts(self.centroids.add(d * self.nlist), self.nlist) }
    }

    /// Range `[start, end)` de vetores pertencentes ao cluster `c`.
    #[inline]
    #[must_use]
    pub fn cluster_range(&self, c: usize) -> (usize, usize) {
        debug_assert!(c < self.nlist);
        // SAFETY: c < nlist; boundaries tem nlist+1 entradas.
        unsafe {
            let start = *self.boundaries.add(c) as usize;
            let end = *self.boundaries.add(c + 1) as usize;
            (start, end)
        }
    }

    #[inline]
    #[must_use]
    pub fn vectors_ptr(&self) -> *const i16 {
        self.vectors
    }

    #[inline]
    #[must_use]
    pub fn centroids_ptr(&self) -> *const i16 {
        self.centroids
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
