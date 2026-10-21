//! Training examples, AdamW and the adapter checkpoint.

use std::io::{Read, Write};
use std::path::Path;

use aien_inference_abi::tokenizer::ChatTokenizer;
use serde::Deserialize;

use crate::model::{Model, Param};

/// One training line: text segments (template markup included) and the weight on predicting
/// each of their tokens.
#[derive(Debug, Deserialize)]
pub struct Line {
    pub id: String,
    #[serde(default)]
    pub split: String,
    pub segs: Vec<(String, f32)>,
}

pub struct Example {
    pub id: String,
    pub tokens: Vec<u32>,
    /// `weights[i]` weighs predicting `tokens[i + 1]`.
    pub weights: Vec<f32>,
}

impl Example {
    pub fn from_line(tok: &ChatTokenizer, line: &Line) -> Result<Self, String> {
        let mut tokens = Vec::new();
        let mut target_w = Vec::new();
        for (text, w) in &line.segs {
            let ids = tok
                .encode_with_special(text, false)
                .map_err(|e| format!("{}: {e}", line.id))?;
            target_w.extend(std::iter::repeat(*w).take(ids.len()));
            tokens.extend(ids);
        }
        let t = tokens.len();
        let mut weights = vec![0f32; t];
        for i in 1..t {
            weights[i - 1] = target_w[i];
        }
        Ok(Self {
            id: line.id.clone(),
            tokens,
            weights,
        })
    }
}

pub fn read_lines(path: &Path) -> Result<Vec<Line>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", path.display())))
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct Adam {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub wd: f32,
}

/// Scales the gradients by `1 / weight`, clips their global norm at `clip`, takes one AdamW
/// step number `step` (from 1) and zeroes the gradients. Returns the norm before clipping.
pub fn step(
    params: &mut [(String, &mut Param)],
    weight: f64,
    clip: f32,
    a: Adam,
    step: u64,
) -> f32 {
    let inv = (1.0 / weight.max(1e-9)) as f32;
    let mut sq = 0f64;
    for (_, p) in params.iter_mut() {
        for g in p.g.iter_mut() {
            *g *= inv;
            sq += (*g as f64) * (*g as f64);
        }
    }
    let norm = sq.sqrt() as f32;
    let k = if norm > clip { clip / norm } else { 1.0 };
    let (c1, c2) = (1.0 - a.b1.powi(step as i32), 1.0 - a.b2.powi(step as i32));
    for (_, p) in params.iter_mut() {
        for i in 0..p.v.len() {
            let g = p.g[i] * k;
            p.m[i] = a.b1 * p.m[i] + (1.0 - a.b1) * g;
            p.s[i] = a.b2 * p.s[i] + (1.0 - a.b2) * g * g;
            let upd = (p.m[i] / c1) / ((p.s[i] / c2).sqrt() + a.eps);
            p.v[i] -= a.lr * (upd + a.wd * p.v[i]);
            p.g[i] = 0.0;
        }
    }
    norm
}

/// Writes every adapter tensor with its moments: a JSON line (names, lengths, step) then the
/// little-endian f32 values of `v`, `m`, `s` per tensor.
pub fn save_adapters(model: &mut Model, path: &Path, step: u64) -> Result<(), String> {
    let params = model.params_mut();
    let meta = serde_json::json!({
        "step": step,
        "tensors": params.iter().map(|(n, p)| serde_json::json!([n, p.v.len()])).collect::<Vec<_>>(),
    });
    let tmp = path.with_extension("tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(|e| e.to_string())?);
    writeln!(f, "{meta}").map_err(|e| e.to_string())?;
    for (_, p) in &params {
        for buf in [&p.v, &p.m, &p.s] {
            for v in buf.iter() {
                f.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
    }
    f.flush().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

/// Reads [`save_adapters`] output into a model already carrying adapters of the same shape;
/// returns the step.
pub fn load_adapters(model: &mut Model, path: &Path) -> Result<u64, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let nl = bytes
        .iter()
        .position(|&b| b == b'\n')
        .ok_or("adapters: no header")?;
    let meta: serde_json::Value =
        serde_json::from_slice(&bytes[..nl]).map_err(|e| e.to_string())?;
    let mut rest = &bytes[nl + 1..];
    let mut params = model.params_mut();
    let names = meta["tensors"].as_array().ok_or("adapters: no tensors")?;
    if names.len() != params.len() {
        return Err(format!(
            "adapters: {} tensors, model has {}",
            names.len(),
            params.len()
        ));
    }
    for (want, (name, p)) in names.iter().zip(params.iter_mut()) {
        if want[0].as_str() != Some(name.as_str()) || want[1].as_u64() != Some(p.v.len() as u64) {
            return Err(format!(
                "adapters: {want} where the model has {name} [{}]",
                p.v.len()
            ));
        }
        for buf in [&mut p.v, &mut p.m, &mut p.s] {
            for v in buf.iter_mut() {
                let mut w = [0u8; 4];
                rest.read_exact(&mut w).map_err(|e| e.to_string())?;
                *v = f32::from_le_bytes(w);
            }
        }
    }
    meta["step"]
        .as_u64()
        .ok_or_else(|| "adapters: no step".into())
}
