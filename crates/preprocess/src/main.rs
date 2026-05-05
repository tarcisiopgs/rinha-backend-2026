//! Pré-processador offline. Lê `references.json.gz` (3M vetores rotulados) e
//! produz `references.bin` no formato SoA i16 esperado pela API.
//!
//! Uso: `preprocess <input.json.gz> <output.bin>`

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use common::dataset::{HEADER_SIZE, LABEL_FRAUD, LABEL_LEGIT, MAGIC, VERSION};
use common::proto::{DIM, NULL_SENTINEL, QUANT_SCALE};
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
    eprintln!("carregados {n} vetores");

    let mut labels = vec![0_u8; n];
    let mut soa = vec![0_i16; DIM * n];

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
            soa[d * n + i] = if value < 0.0 {
                NULL_SENTINEL
            } else {
                let q = (value * QUANT_SCALE).round();
                q.clamp(0.0, f32::from(i16::MAX)) as i16
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
    header[16..20].copy_from_slice(&QUANT_SCALE.to_le_bytes());
    w.write_all(&header)?;

    w.write_all(&labels)?;

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            soa.as_ptr().cast::<u8>(),
            soa.len() * std::mem::size_of::<i16>(),
        )
    };
    w.write_all(bytes)?;
    w.flush()?;

    let total = HEADER_SIZE + n + soa.len() * 2;
    eprintln!(
        "ok ({} bytes / {:.2} MB)",
        total,
        total as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}
