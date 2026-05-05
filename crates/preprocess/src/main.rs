//! Pré-processador: lê `references.json.gz` (3M entradas) e produz binário
//! quantizado i16 SoA esperado pela API. Roda offline antes do build da imagem.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use common::dataset::{DEFAULT_SCALE, DIM, HEADER_SIZE, MAGIC, VERSION};
use flate2::read::GzDecoder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Reference {
    vector: Vec<f32>,
    score: f32,
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

    let raw = File::open(&input).with_context(|| format!("abrir {}", input.display()))?;
    let mut reader = BufReader::new(GzDecoder::new(raw));

    let mut buf = String::new();
    reader.read_to_string(&mut buf).context("ler json.gz")?;
    let refs: Vec<Reference> = serde_json::from_str(&buf).context("parse json")?;
    let n = refs.len();
    eprintln!("carregados {n} vetores");

    let mut scores: Vec<u8> = Vec::with_capacity(n);
    let mut soa: Vec<i16> = vec![0_i16; DIM * n];

    for (i, r) in refs.iter().enumerate() {
        if r.vector.len() != DIM {
            return Err(anyhow!("vetor #{i} tem dim {}, esperado {DIM}", r.vector.len()));
        }
        let s = (r.score.clamp(0.0, 1.0) * 255.0).round() as u8;
        scores.push(s);

        for d in 0..DIM {
            let q = (r.vector[d] * DEFAULT_SCALE).round();
            soa[d * n + i] = q.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
        }
    }

    let out = File::create(&output).with_context(|| format!("criar {}", output.display()))?;
    let mut w = BufWriter::with_capacity(1 << 20, out);

    let mut header = [0_u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(DIM as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(n as u32).to_le_bytes());
    header[16..20].copy_from_slice(&DEFAULT_SCALE.to_le_bytes());
    w.write_all(&header)?;
    w.write_all(&scores)?;

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(soa.as_ptr().cast::<u8>(), soa.len() * std::mem::size_of::<i16>())
    };
    w.write_all(bytes)?;
    w.flush()?;

    eprintln!("escrito {}", output.display());
    Ok(())
}
