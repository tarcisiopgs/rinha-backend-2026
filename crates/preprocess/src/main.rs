//! Pré-processador offline. Lê `references.json.gz` (3M vetores rotulados),
//! roda kmeans pra construir índice IVF e produz `references.bin` com:
//!   - centroides i16 SoA
//!   - vetores reordenados por cluster (i16 SoA global)
//!   - boundaries `[start; nlist+1]` por cluster
//!   - labels reordenadas
//!
//! Uso: `preprocess <input.json.gz> <output.bin>`

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use common::dataset::{
    align_up, padded_n, ALIGN, HEADER_SIZE, LABEL_FRAUD, LABEL_LEGIT, MAGIC, VERSION,
};
use common::proto::{DIM, NLIST, NULL_SENTINEL_I16, QUANT_SCALE};
use flate2::read::GzDecoder;
use serde::Deserialize;

const KMEANS_ITERS: usize = 8;
const KMEANS_SAMPLE: usize = 200_000;

#[derive(Debug, Deserialize)]
struct Reference {
    vector: Vec<f32>,
    label: String,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .ok_or_else(|| anyhow!("uso: preprocess <input.json.gz> <output.bin>"))?
        .into();
    let output: PathBuf = args
        .next()
        .ok_or_else(|| anyhow!("falta arg de saída"))?
        .into();

    eprintln!("lendo {}", input.display());
    let raw = File::open(&input).with_context(|| format!("abrir {}", input.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, GzDecoder::new(raw));

    let mut buf = String::new();
    reader.read_to_string(&mut buf).context("ler json.gz")?;
    let refs: Vec<Reference> = serde_json::from_str(&buf).context("parse json")?;
    drop(buf);

    let n = refs.len();
    let n_padded = padded_n(n);
    eprintln!("carregados {n} vetores (padded={n_padded})");

    // 1. Quantiza vetores pra i16 em layout AoS provisório (n × DIM).
    let mut vec_aos = vec![0_i16; n * DIM];
    let mut labels_orig = vec![0_u8; n];
    let mut fraud_count = 0_usize;
    for (i, r) in refs.iter().enumerate() {
        if r.vector.len() != DIM {
            return Err(anyhow!(
                "vetor #{i} tem dim {}, esperado {DIM}",
                r.vector.len()
            ));
        }
        labels_orig[i] = match r.label.as_str() {
            "fraud" => {
                fraud_count += 1;
                LABEL_FRAUD
            }
            "legit" => LABEL_LEGIT,
            other => return Err(anyhow!("label desconhecida em #{i}: {other:?}")),
        };
        for d in 0..DIM {
            let v = r.vector[d];
            vec_aos[i * DIM + d] = if v < 0.0 {
                NULL_SENTINEL_I16
            } else {
                quantize(v)
            };
        }
    }
    drop(refs);
    eprintln!("fraud={fraud_count} legit={}", n - fraud_count);

    // 2. KMeans Lloyd's mini-batch: treina centroides numa amostra de
    //    KMEANS_SAMPLE vetores (uniformemente espaçada no dataset original)
    //    por KMEANS_ITERS iterações. Atribuições finais usam todos os n
    //    vetores numa única passada. Corte de custo: 3M × 1024 × DIM × ITERS
    //    seria ~430 G ops; mini-batch + assign final fica ~15× menor.
    eprintln!("kmeans: {NLIST} centroides, sample={KMEANS_SAMPLE}, {KMEANS_ITERS} iters");
    let t0 = Instant::now();

    let sample_size = KMEANS_SAMPLE.min(n);
    let sample_stride = n / sample_size.max(1);
    let mut sample_idx = Vec::with_capacity(sample_size);
    for s in 0..sample_size {
        sample_idx.push(s * sample_stride);
    }

    // Inicializa centroides com 1024 vetores espaçados uniformemente.
    let mut centroids = vec![0_i32; NLIST * DIM];
    let init_stride = sample_size / NLIST;
    for c in 0..NLIST {
        let src = sample_idx[c * init_stride];
        for d in 0..DIM {
            centroids[c * DIM + d] = i32::from(vec_aos[src * DIM + d]);
        }
    }

    let mut sample_assigns = vec![0_u32; sample_size];
    let mut cluster_sums = vec![0_i64; NLIST * DIM];
    let mut cluster_counts = vec![0_u32; NLIST];

    for it in 0..KMEANS_ITERS {
        for (s, &i) in sample_idx.iter().enumerate() {
            sample_assigns[s] = nearest_centroid(&vec_aos, i, &centroids);
        }

        cluster_sums.fill(0);
        cluster_counts.fill(0);
        for (s, &i) in sample_idx.iter().enumerate() {
            let c = sample_assigns[s] as usize;
            cluster_counts[c] += 1;
            for d in 0..DIM {
                cluster_sums[c * DIM + d] += i64::from(vec_aos[i * DIM + d]);
            }
        }
        for c in 0..NLIST {
            let count = cluster_counts[c];
            if count == 0 {
                let src = (c * 7919) % n;
                for d in 0..DIM {
                    centroids[c * DIM + d] = i32::from(vec_aos[src * DIM + d]);
                }
                continue;
            }
            for d in 0..DIM {
                centroids[c * DIM + d] = (cluster_sums[c * DIM + d] / i64::from(count)) as i32;
            }
        }
        eprintln!("  iter {it}: ok ({:?})", t0.elapsed());
    }
    drop(sample_assigns);
    drop(sample_idx);

    // Atribuições finais: todos n vetores → centroide mais próximo.
    let mut assignments = vec![0_u32; n];
    for (i, slot) in assignments.iter_mut().enumerate() {
        *slot = nearest_centroid(&vec_aos, i, &centroids);
    }
    eprintln!("kmeans concluído em {:?}", t0.elapsed());

    // 3. Boundaries: prefix sum dos counts. Ordena vetores por cluster.
    let mut boundaries = vec![0_u32; NLIST + 1];
    for &c in &assignments {
        boundaries[c as usize + 1] += 1;
    }
    for c in 1..=NLIST {
        boundaries[c] += boundaries[c - 1];
    }
    debug_assert_eq!(boundaries[NLIST] as usize, n);

    // Reordena vetores e labels usando boundaries como cursor.
    let mut cursors = boundaries[..NLIST].to_vec();
    let mut soa = vec![0_i16; DIM * n_padded];
    let mut labels_sorted = vec![0_u8; n];
    for i in 0..n {
        let c = assignments[i] as usize;
        let dst = cursors[c] as usize;
        cursors[c] += 1;
        labels_sorted[dst] = labels_orig[i];
        for d in 0..DIM {
            soa[d * n_padded + dst] = vec_aos[i * DIM + d];
        }
    }
    drop(vec_aos);
    drop(assignments);

    // Quantiza centroides pra i16 SoA.
    let mut centroids_soa = vec![0_i16; DIM * NLIST];
    for c in 0..NLIST {
        for d in 0..DIM {
            centroids_soa[d * NLIST + c] =
                centroids[c * DIM + d].clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
    }
    drop(centroids);

    // 4. Escreve arquivo binário no novo formato.
    eprintln!("escrevendo {}", output.display());
    let out = File::create(&output).with_context(|| format!("criar {}", output.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, out);

    let mut header = [0_u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(DIM as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(n as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(NLIST as u32).to_le_bytes());
    w.write_all(&header)?;
    w.write_all(&labels_sorted)?;

    let labels_end = HEADER_SIZE + n;
    let centroids_off = align_up(labels_end, ALIGN);
    pad_to(&mut w, labels_end, centroids_off)?;
    let centroids_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            centroids_soa.as_ptr().cast::<u8>(),
            centroids_soa.len() * std::mem::size_of::<i16>(),
        )
    };
    w.write_all(centroids_bytes)?;
    let centroids_end = centroids_off + centroids_soa.len() * std::mem::size_of::<i16>();

    let boundaries_off = align_up(centroids_end, ALIGN);
    pad_to(&mut w, centroids_end, boundaries_off)?;
    let boundaries_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            boundaries.as_ptr().cast::<u8>(),
            boundaries.len() * std::mem::size_of::<u32>(),
        )
    };
    w.write_all(boundaries_bytes)?;
    let boundaries_end = boundaries_off + boundaries.len() * std::mem::size_of::<u32>();

    let vectors_off = align_up(boundaries_end, ALIGN);
    pad_to(&mut w, boundaries_end, vectors_off)?;
    let vectors_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            soa.as_ptr().cast::<u8>(),
            soa.len() * std::mem::size_of::<i16>(),
        )
    };
    w.write_all(vectors_bytes)?;
    w.flush()?;

    let total = vectors_off + soa.len() * std::mem::size_of::<i16>();
    eprintln!(
        "ok ({} bytes / {:.2} MB)",
        total,
        total as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn pad_to<W: Write>(w: &mut W, current: usize, target: usize) -> Result<()> {
    let pad = target - current;
    if pad > 0 {
        w.write_all(&vec![0_u8; pad])?;
    }
    Ok(())
}

/// Quantiza f32 normalizado em `[0, 1]` para i16 multiplicando por
/// `QUANT_SCALE` e arredondando. Clampeia em `[-32768, 32767]` por segurança
/// contra valores fora do intervalo esperado.
#[inline]
fn quantize(value: f32) -> i16 {
    let scaled = (value * QUANT_SCALE).round();
    if scaled <= f32::from(i16::MIN) {
        i16::MIN
    } else if scaled >= f32::from(i16::MAX) {
        i16::MAX
    } else {
        scaled as i16
    }
}

#[inline]
fn nearest_centroid(vec_aos: &[i16], i: usize, centroids: &[i32]) -> u32 {
    let v_off = i * DIM;
    let mut best = 0_u32;
    let mut best_dist = i64::MAX;
    for c in 0..NLIST {
        let c_off = c * DIM;
        let mut d_acc: i64 = 0;
        for d in 0..DIM {
            let diff = i64::from(centroids[c_off + d]) - i64::from(vec_aos[v_off + d]);
            d_acc += diff * diff;
        }
        if d_acc < best_dist {
            best_dist = d_acc;
            best = c as u32;
        }
    }
    best
}
