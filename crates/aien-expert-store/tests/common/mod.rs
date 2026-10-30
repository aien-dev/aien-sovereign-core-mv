#![allow(dead_code)]

use aien_expert_store::format::build_artifact;
use aien_expert_store::{ExpertKey, IndexEntry, Projection, QuantFormat, ScaleDtype};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MODEL: [u8; 32] = [7u8; 32];

pub fn key(layer: u32, expert: u32, projection: Projection) -> ExpertKey {
    ExpertKey {
        model_digest: MODEL,
        layer,
        expert,
        projection,
        quant: QuantFormat::Fp8E4m3Block128,
    }
}

/// Deterministic filler (no randomness).
pub fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as u8
        })
        .collect()
}

/// A canonical artifact and its index entry for a `[rows][cols]` projection.
pub fn artifact(
    k: ExpertKey,
    rows: u64,
    cols: u64,
    seed: u32,
) -> (IndexEntry, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = pattern((rows * cols) as usize, seed);
    let s = pattern(
        (rows.div_ceil(128) * cols.div_ceil(128) * 2) as usize,
        seed + 1,
    );
    let (h, bytes) = build_artifact(k, rows, cols, ScaleDtype::Bf16, &w, &s, u64::MAX).unwrap();
    let e = IndexEntry {
        key: k,
        data_digest: aien_expert_store::format::sha256(&bytes),
        object_len: h.object_len,
    };
    (e, bytes, w, s)
}

static DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh empty directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "aien-expert-store-{tag}-{}-{}",
            std::process::id(),
            DIR_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One tensor for [`write_shard`].
pub struct T {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Writes a safetensors file holding `tensors` in order.
pub fn write_shard(path: &Path, tensors: &[T]) {
    let mut header = serde_json::Map::new();
    let mut off = 0u64;
    for t in tensors {
        let end = off + t.data.len() as u64;
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [off, end]}),
        );
        off = end;
    }
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for t in tensors {
        out.extend_from_slice(&t.data);
    }
    std::fs::write(path, out).unwrap();
}

pub const H: u64 = 256;
pub const I: u64 = 192; // not a block multiple: exercises the ceil scale shape
pub const LAYERS: u32 = 2;
pub const EXPERTS: u32 = 3;

pub fn proj_shape(p: Projection) -> (u64, u64) {
    match p {
        Projection::Gate | Projection::Up => (I, H),
        Projection::Down => (H, I),
    }
}

/// A tiny Qwen3-MoE FP8 checkpoint: layer `l` in shard `l`, BF16 scales,
/// plus a non-expert tensor. Returns the expected (weights, scales) per key.
pub fn write_checkpoint(dir: &Path) -> Vec<(ExpertKey, Vec<u8>, Vec<u8>)> {
    let mut expected = Vec::new();
    let mut weight_map = serde_json::Map::new();
    for l in 0..LAYERS {
        let shard = format!("model-{:05}-of-{:05}.safetensors", l + 1, LAYERS);
        let mut ts = vec![T {
            name: format!("model.layers.{l}.mlp.gate.weight"),
            dtype: "BF16",
            shape: vec![EXPERTS as u64, H],
            data: pattern((EXPERTS as u64 * H * 2) as usize, 999 + l),
        }];
        for e in 0..EXPERTS {
            for p in Projection::ALL {
                let (r, c) = proj_shape(p);
                let seed = l * 100 + e * 10 + p as u32;
                let w = pattern((r * c) as usize, seed);
                let s = pattern(
                    (r.div_ceil(128) * c.div_ceil(128) * 2) as usize,
                    seed + 5000,
                );
                let base = format!("model.layers.{l}.mlp.experts.{e}.{}", p.checkpoint_name());
                ts.push(T {
                    name: format!("{base}.weight"),
                    dtype: "F8_E4M3",
                    shape: vec![r, c],
                    data: w.clone(),
                });
                ts.push(T {
                    name: format!("{base}.weight_scale_inv"),
                    dtype: "BF16",
                    shape: vec![r.div_ceil(128), c.div_ceil(128)],
                    data: s.clone(),
                });
                expected.push((key(l, e, p), w, s));
            }
        }
        for t in &ts {
            weight_map.insert(t.name.clone(), serde_json::Value::String(shard.clone()));
        }
        write_shard(&dir.join(&shard), &ts);
    }
    let index = serde_json::json!({"metadata": {}, "weight_map": weight_map});
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    expected
}
