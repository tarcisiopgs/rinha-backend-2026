//! Pré-processador offline. Lê `references.json.gz` (3M vetores rotulados) e
//! produz `references.bin` no formato SoA i16 (f32 quantizado por
//! `QUANT_SCALE`) esperado pela API.
//!
//! Uso: `preprocess <input.json.gz> <output.bin>`

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use common::dataset::{
    align_up, padded_n, ALIGN, HEADER_SIZE, LABEL_FRAUD, LABEL_LEGIT, MAGIC, VERSION,
};
use common::proto::{DIM, NULL_SENTINEL_I16, QUANT_SCALE};
use flate2::read::GzDecoder;
use serde::Deserialize;

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

    let mut labels = vec![0_u8; n];
    // SoA i16: posições padding (n..n_padded) ficam em 0. O hot path filtra
    // índices ≥ n após o TopK, então o valor não importa contanto que não
    // cause overflow no acumulador i32 da soma de 14 quadrados.
    let mut soa = vec![0_i16; DIM * n_padded];

    let mut fraud_count = 0_usize;
    for (i, r) in refs.iter().enumerate() {
        if r.vector.len() != DIM {
            return Err(anyhow!(
                "vetor #{i} tem dim {}, esperado {DIM}",
                r.vector.len()
            ));
        }
        labels[i] = match r.label.as_str() {
            "fraud" => {
                fraud_count += 1;
                LABEL_FRAUD
            }
            "legit" => LABEL_LEGIT,
            other => return Err(anyhow!("label desconhecida em #{i}: {other:?}")),
        };

        for d in 0..DIM {
            let value = r.vector[d];
            soa[d * n_padded + i] = if value < 0.0 {
                NULL_SENTINEL_I16
            } else {
                quantize(value)
            };
        }
    }
    drop(refs);

    eprintln!("fraud={fraud_count} legit={}", n - fraud_count);

    eprintln!("escrevendo {}", output.display());
    let out = File::create(&output).with_context(|| format!("criar {}", output.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, out);

    let mut header = [0_u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(DIM as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(n as u32).to_le_bytes());
    w.write_all(&header)?;
    w.write_all(&labels)?;

    // Padding entre LABELS e VECTORS pra alinhar SoA em ALIGN bytes.
    let labels_end = HEADER_SIZE + n;
    let vectors_off = align_up(labels_end, ALIGN);
    let pad = vectors_off - labels_end;
    if pad > 0 {
        w.write_all(&vec![0_u8; pad])?;
    }

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            soa.as_ptr().cast::<u8>(),
            soa.len() * std::mem::size_of::<i16>(),
        )
    };
    w.write_all(bytes)?;
    w.flush()?;

    let total = vectors_off + soa.len() * std::mem::size_of::<i16>();
    eprintln!(
        "ok ({} bytes / {:.2} MB)",
        total,
        total as f64 / (1024.0 * 1024.0)
    );
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
