//! Expert manifest: per (layer, expert) the byte span of its six tensors
//! (gate/up/down weight + weight_scale_inv) inside one safetensors shard,
//! plus a BLAKE3 content address computed once at build time.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::HeartbeatError;

pub const MANIFEST_SCHEMA: &str = "aien-heartbeat-manifest/2";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorRange {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Offset in the expert payload (segments concatenated in order).
    pub offset: u64,
    pub len: u64,
}

/// One contiguous run of an expert's bytes in one shard file. safetensors
/// groups tensors by dtype, so a Qwen3 FP8 expert is two segments (F8 weights
/// gate/up/down, then the BF16 scales elsewhere), three if it straddles a shard.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Segment {
    /// Index into [`Manifest::shards`].
    pub shard: u32,
    /// Absolute byte offset in the shard file.
    pub start: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertEntry {
    pub layer: u16,
    pub expert: u16,
    pub segments: Vec<Segment>,
    /// Payload bytes (sum of segment lengths).
    pub len: u64,
    pub tensors: Vec<TensorRange>,
    /// BLAKE3 over the segments' bytes in order, hex.
    pub digest: String,
}

impl ExpertEntry {
    pub fn digest_bytes(&self) -> Result<[u8; 32], HeartbeatError> {
        let h = blake3::Hash::from_hex(&self.digest)
            .map_err(|e| HeartbeatError::Manifest(format!("bad digest hex: {e}")))?;
        Ok(*h.as_bytes())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema: String,
    /// BLAKE3 over the index file, every expert shard's name and header bytes, and every expert's
    /// `(layer, expert, digest)` in order: names the checkpoint by content, not by path. The per-expert
    /// digests cover the weight bytes themselves, so this is not an index-only digest (sc#338).
    pub model_digest: String,
    pub model_dir: PathBuf,
    pub shards: Vec<String>,
    pub num_layers: u16,
    pub experts_per_layer: u16,
    /// Sorted by (layer, expert).
    pub experts: Vec<ExpertEntry>,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// (name, dtype, shape, absolute offset, length)
type RawTensor = (String, String, Vec<u64>, u64, u64);

/// Parse `model.layers.{L}.mlp.experts.{E}.<rest>`.
fn parse_expert_name(name: &str) -> Option<(u16, u16)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e, _) = rest.split_once('.')?;
    Some((l.parse().ok()?, e.parse().ok()?))
}

/// Read a safetensors header: returns (data section start, tensors).
#[allow(clippy::type_complexity)]
fn read_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, serde_json::Value>, Vec<u8>), HeartbeatError> {
    let f = File::open(path).map_err(|e| HeartbeatError::io(path.display().to_string(), e))?;
    let mut lenb = [0u8; 8];
    f.read_exact_at(&mut lenb, 0)
        .map_err(|e| HeartbeatError::io("header length", e))?;
    let hlen = u64::from_le_bytes(lenb);
    if hlen > 256 << 20 {
        return Err(HeartbeatError::Manifest(format!(
            "header too large: {hlen}"
        )));
    }
    let mut h = vec![0u8; hlen as usize];
    f.read_exact_at(&mut h, 8)
        .map_err(|e| HeartbeatError::io("header", e))?;
    let map: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&h)
        .map_err(|e| HeartbeatError::Manifest(format!("header json: {e}")))?;
    Ok((8 + hlen, map, h))
}

impl Manifest {
    /// Build from `<model_dir>/model.safetensors.index.json`, reading every
    /// expert span once to compute its digest. Read-only on the model.
    pub fn build(model_dir: &Path) -> Result<Self, HeartbeatError> {
        let idx_path = model_dir.join("model.safetensors.index.json");
        let idx_bytes = std::fs::read(&idx_path).map_err(|e| HeartbeatError::io("index", e))?;
        let idx: Index = serde_json::from_slice(&idx_bytes)
            .map_err(|e| HeartbeatError::Manifest(format!("index json: {e}")))?;

        let mut model_h = blake3::Hasher::new();
        model_h.update(b"aien-heartbeat-model/1\0");
        model_h.update(&(idx_bytes.len() as u64).to_le_bytes());
        model_h.update(&idx_bytes);
        let mut shards: Vec<String> = Vec::new();
        for (name, shard) in &idx.weight_map {
            if parse_expert_name(name).is_some() && !shards.contains(shard) {
                shards.push(shard.clone());
            }
        }
        shards.sort();

        // (layer, expert) -> tensors (shard, name, dtype, shape, abs offset, len)
        let mut groups: BTreeMap<(u16, u16), Vec<(u32, RawTensor)>> = BTreeMap::new();
        for (si, shard) in shards.iter().enumerate() {
            let (data_start, header, raw_header) = read_header(&model_dir.join(shard))?;
            model_h.update(&(shard.len() as u64).to_le_bytes());
            model_h.update(shard.as_bytes());
            model_h.update(&(raw_header.len() as u64).to_le_bytes());
            model_h.update(&raw_header);
            for (name, v) in header {
                let Some(key) = parse_expert_name(&name) else {
                    continue;
                };
                if idx.weight_map.get(&name) != Some(shard) {
                    continue;
                }
                let t: HeaderTensor = serde_json::from_value(v)
                    .map_err(|e| HeartbeatError::Manifest(format!("{name}: {e}")))?;
                groups.entry(key).or_default().push((
                    si as u32,
                    (
                        name,
                        t.dtype,
                        t.shape,
                        data_start + t.data_offsets[0],
                        t.data_offsets[1] - t.data_offsets[0],
                    ),
                ));
            }
        }

        let mut experts = Vec::with_capacity(groups.len());
        let mut files: Vec<File> = Vec::new();
        for s in &shards {
            files
                .push(File::open(model_dir.join(s)).map_err(|e| HeartbeatError::io(s.clone(), e))?);
        }
        let mut buf = Vec::new();
        for ((layer, expert), mut ts) in groups {
            // Shard order, then file order; adjacent tensors merge into one segment.
            ts.sort_by_key(|(s, t)| (*s, t.3));
            let mut segments: Vec<Segment> = Vec::new();
            let mut tensors = Vec::with_capacity(ts.len());
            let mut payload_off = 0u64;
            for (shard, (name, dtype, shape, off, l)) in ts {
                match segments.last_mut() {
                    Some(seg) if seg.shard == shard && seg.start + seg.len == off => seg.len += l,
                    _ => segments.push(Segment {
                        shard,
                        start: off,
                        len: l,
                    }),
                }
                tensors.push(TensorRange {
                    name,
                    dtype,
                    shape,
                    offset: payload_off,
                    len: l,
                });
                payload_off += l;
            }
            let mut h = blake3::Hasher::new();
            for seg in &segments {
                buf.resize(seg.len as usize, 0);
                files[seg.shard as usize]
                    .read_exact_at(&mut buf, seg.start)
                    .map_err(|e| {
                        HeartbeatError::io(format!("digest read ({layer},{expert})"), e)
                    })?;
                h.update(&buf);
            }
            let digest = h.finalize().to_hex().to_string();
            model_h.update(&layer.to_le_bytes());
            model_h.update(&expert.to_le_bytes());
            model_h.update(digest.as_bytes());
            experts.push(ExpertEntry {
                layer,
                expert,
                segments,
                len: payload_off,
                tensors,
                digest,
            });
        }
        let num_layers = experts.iter().map(|e| e.layer + 1).max().unwrap_or(0);
        let experts_per_layer = experts.iter().map(|e| e.expert + 1).max().unwrap_or(0);
        if experts.len() != num_layers as usize * experts_per_layer as usize {
            return Err(HeartbeatError::Manifest(format!(
                "incomplete expert grid: {} entries for {num_layers}x{experts_per_layer}",
                experts.len()
            )));
        }
        Ok(Self {
            schema: MANIFEST_SCHEMA.into(),
            model_digest: model_h.finalize().to_hex().to_string(),
            model_dir: model_dir.to_path_buf(),
            shards,
            num_layers,
            experts_per_layer,
            experts,
        })
    }

    pub fn load(path: &Path) -> Result<Self, HeartbeatError> {
        let mut s = String::new();
        File::open(path)
            .and_then(|mut f| f.read_to_string(&mut s))
            .map_err(|e| HeartbeatError::io(path.display().to_string(), e))?;
        let m: Self =
            serde_json::from_str(&s).map_err(|e| HeartbeatError::Manifest(e.to_string()))?;
        if m.schema != MANIFEST_SCHEMA {
            return Err(HeartbeatError::Manifest(format!("schema {}", m.schema)));
        }
        Ok(m)
    }

    pub fn save(&self, path: &Path) -> Result<(), HeartbeatError> {
        let s = serde_json::to_vec(self).map_err(|e| HeartbeatError::Manifest(e.to_string()))?;
        std::fs::write(path, s).map_err(|e| HeartbeatError::io(path.display().to_string(), e))
    }

    pub fn shard_paths(&self) -> Vec<PathBuf> {
        self.shards.iter().map(|s| self.model_dir.join(s)).collect()
    }

    pub fn max_expert_len(&self) -> u64 {
        self.experts.iter().map(|e| e.len).max().unwrap_or(0)
    }

    pub fn total_bytes(&self) -> u64 {
        self.experts.iter().map(|e| e.len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_expert_names() {
        assert_eq!(
            parse_expert_name("model.layers.7.mlp.experts.93.up_proj.weight"),
            Some((7, 93))
        );
        assert_eq!(parse_expert_name("model.layers.7.mlp.gate.weight"), None);
        assert_eq!(parse_expert_name("lm_head.weight"), None);
    }
}
