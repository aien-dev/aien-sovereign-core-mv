//! Deterministic, read-only checkpoint-to-artifact indexer.
//!
//! The source checkpoint (`model.safetensors.index.json` plus its shards) is
//! opened read-only and only the tensors of the requested experts are read,
//! through their header byte ranges, so one expert can be fetched without
//! scanning or decoding the other shards. The weight and scale bytes are
//! copied unchanged into artifacts (reversible: `ReadyExpert::weights` and
//! `scales` return exactly the checkpoint bytes). The same checkpoint and
//! arguments always produce the same artifacts, digests and index.
//!
//! The shard checks mirror the resident loader's reader
//! (`aien-inference-abi` `qwen3_moe::Shard`): plain shard file names only,
//! bounded header length, exact dtype, shape and byte range per tensor.

use crate::error::StoreError;
use crate::format::{ExpertKey, IndexEntry, Projection, QuantFormat, ScaleDtype, FP8_BLOCK};
use crate::store::ExpertStore;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};

/// Same bound as the resident loader.
const MAX_HEADER_BYTES: u64 = 256 << 20;
/// Bound on `model.safetensors.index.json`.
const MAX_INDEX_JSON_BYTES: u64 = 64 << 20;

/// One tensor's dtype, shape and absolute byte range in its shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub start: u64,
    pub end: u64,
}

struct Shard {
    file: File,
    payload_start: u64,
    file_len: u64,
    header: serde_json::Map<String, serde_json::Value>,
}

/// A read-only view of a safetensors checkpoint directory.
pub struct Checkpoint {
    dir: PathBuf,
    weight_map: serde_json::Map<String, serde_json::Value>,
    shards: HashMap<String, Shard>,
}

fn ck(msg: impl Into<String>) -> StoreError {
    StoreError::Checkpoint(msg.into())
}

fn shard_path(dir: &Path, filename: &str) -> Result<PathBuf, StoreError> {
    let mut c = Path::new(filename).components();
    match (c.next(), c.next()) {
        (Some(Component::Normal(name)), None) if name == filename => Ok(dir.join(name)),
        _ => Err(ck(format!(
            "shard file name {filename:?} is not a plain name"
        ))),
    }
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self, StoreError> {
        let f = File::open(dir.join("model.safetensors.index.json"))?;
        let mut raw = Vec::new();
        f.take(MAX_INDEX_JSON_BYTES + 1).read_to_end(&mut raw)?;
        if raw.len() as u64 > MAX_INDEX_JSON_BYTES {
            return Err(ck("index json too large"));
        }
        let v: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| ck(format!("index json: {e}")))?;
        let weight_map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .cloned()
            .ok_or_else(|| ck("index json: missing weight_map object"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            weight_map,
            shards: HashMap::new(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn shard_for(&mut self, name: &str) -> Result<&Shard, StoreError> {
        let filename = self
            .weight_map
            .get(name)
            .ok_or_else(|| StoreError::MissingTensor(name.to_string()))?
            .as_str()
            .ok_or_else(|| ck(format!("index: shard of {name} is not a string")))?
            .to_string();
        if !self.shards.contains_key(&filename) {
            let shard = open_shard(&shard_path(&self.dir, &filename)?)?;
            self.shards.insert(filename.clone(), shard);
        }
        Ok(&self.shards[&filename])
    }

    /// The header entry of `name`, with its byte range checked against the shard.
    pub fn info(&mut self, name: &str) -> Result<TensorInfo, StoreError> {
        let shard = self.shard_for(name)?;
        let entry = shard
            .header
            .get(name)
            .ok_or_else(|| StoreError::MissingTensor(name.to_string()))?;
        let dtype = entry
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ck(format!("{name}: no dtype")))?
            .to_string();
        let shape = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ck(format!("{name}: no shape")))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| ck(format!("{name}: bad shape"))))
            .collect::<Result<Vec<u64>, _>>()?;
        let offs = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ck(format!("{name}: no data_offsets")))?;
        let (Some(s), Some(e), 2) = (
            offs.first().and_then(|v| v.as_u64()),
            offs.get(1).and_then(|v| v.as_u64()),
            offs.len(),
        ) else {
            return Err(ck(format!("{name}: data_offsets must hold two integers")));
        };
        let start = shard
            .payload_start
            .checked_add(s)
            .ok_or(StoreError::Overflow)?;
        let end = shard
            .payload_start
            .checked_add(e)
            .ok_or(StoreError::Overflow)?;
        if s > e || end > shard.file_len {
            return Err(ck(format!("{name}: byte range outside the shard")));
        }
        Ok(TensorInfo {
            dtype,
            shape,
            start,
            end,
        })
    }

    /// Reads `name` after checking its dtype and shape. `max_bytes` bounds the read.
    pub fn read(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[u64],
        max_bytes: u64,
    ) -> Result<Vec<u8>, StoreError> {
        let info = self.info(name)?;
        if info.dtype != dtype {
            return Err(StoreError::DtypeMismatch {
                name: name.to_string(),
                found: info.dtype,
            });
        }
        if info.shape != shape {
            return Err(StoreError::ShapeMismatch {
                name: name.to_string(),
            });
        }
        let len = info.end - info.start;
        if len > max_bytes {
            return Err(StoreError::ObjectTooLarge {
                len,
                max: max_bytes,
            });
        }
        let mut out = vec![0u8; usize::try_from(len).map_err(|_| StoreError::Overflow)?];
        self.shard_for(name)?
            .file
            .read_exact_at(&mut out, info.start)?;
        Ok(out)
    }
}

fn open_shard(path: &Path) -> Result<Shard, StoreError> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact_at(&mut prefix, 0)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || header_len.saturating_add(8) > file_len {
        return Err(ck("shard header length is invalid"));
    }
    let mut raw = vec![0u8; header_len as usize];
    file.read_exact_at(&mut raw, 8)?;
    let header = match serde_json::from_slice(&raw) {
        Ok(serde_json::Value::Object(m)) => m,
        Ok(_) => return Err(ck("shard header is not a JSON object")),
        Err(e) => return Err(ck(format!("shard header json: {e}"))),
    };
    Ok(Shard {
        file,
        payload_start: 8 + header_len,
        file_len,
        header,
    })
}

/// The Qwen3-MoE expert geometry the indexer needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3MoeExperts {
    pub model_digest: [u8; 32],
    pub hidden: u64,
    pub moe_intermediate: u64,
    pub experts: u32,
}

impl Qwen3MoeExperts {
    /// `(rows, cols)` of a projection's weight: gate/up `[i][h]`, down `[h][i]`.
    pub fn shape(&self, p: Projection) -> (u64, u64) {
        match p {
            Projection::Gate | Projection::Up => (self.moe_intermediate, self.hidden),
            Projection::Down => (self.hidden, self.moe_intermediate),
        }
    }

    /// The checkpoint name of a projection's weight (scales add `_scale_inv`).
    pub fn weight_name(layer: u32, expert: u32, p: Projection) -> String {
        format!(
            "model.layers.{layer}.mlp.experts.{expert}.{}.weight",
            p.checkpoint_name()
        )
    }
}

/// Reads one projection of one expert straight from the checkpoint (weights,
/// scales, scale dtype), with full dtype and shape checks.
pub fn read_expert(
    ckpt: &mut Checkpoint,
    shape: &Qwen3MoeExperts,
    layer: u32,
    expert: u32,
    p: Projection,
    max_bytes: u64,
) -> Result<(Vec<u8>, Vec<u8>, ScaleDtype), StoreError> {
    if expert >= shape.experts {
        return Err(StoreError::MissingTensor(format!(
            "expert {expert} of {}",
            shape.experts
        )));
    }
    let (rows, cols) = shape.shape(p);
    let wname = Qwen3MoeExperts::weight_name(layer, expert, p);
    let sname = format!("{wname}_scale_inv");
    let weights = ckpt.read(&wname, "F8_E4M3", &[rows, cols], max_bytes)?;
    let sinfo = ckpt.info(&sname)?;
    let sdt = ScaleDtype::from_safetensors(&sinfo.dtype).ok_or(StoreError::DtypeMismatch {
        name: sname.clone(),
        found: sinfo.dtype.clone(),
    })?;
    let sshape = [rows.div_ceil(FP8_BLOCK), cols.div_ceil(FP8_BLOCK)];
    let scales = ckpt.read(&sname, sdt.safetensors_name(), &sshape, max_bytes)?;
    Ok((weights, scales, sdt))
}

fn inside(child: &Path, parent: &Path) -> bool {
    match (child.canonicalize(), parent.canonicalize()) {
        (Ok(c), Ok(p)) => c.starts_with(p),
        _ => false,
    }
}

/// Indexes every gate, up and down projection of `experts` in `layers` into
/// `store`, in (layer, expert, projection) order, and commits the index.
/// Refuses a store root inside the checkpoint directory. Any missing tensor,
/// dtype or shape mismatch, or a key already bound to other content aborts
/// before the index is committed.
pub fn index_qwen3_moe(
    ckpt: &mut Checkpoint,
    shape: &Qwen3MoeExperts,
    layers: core::ops::Range<u32>,
    store: &mut ExpertStore,
) -> Result<Vec<IndexEntry>, StoreError> {
    if inside(store.root(), ckpt.dir()) {
        return Err(StoreError::StoreInsideCheckpoint);
    }
    let max = store.limits().max_object_bytes;
    let mut out = Vec::new();
    for layer in layers {
        for expert in 0..shape.experts {
            for p in Projection::ALL {
                let (w, s, sdt) = read_expert(ckpt, shape, layer, expert, p, max)?;
                let (rows, cols) = shape.shape(p);
                let key = ExpertKey {
                    model_digest: shape.model_digest,
                    layer,
                    expert,
                    projection: p,
                    quant: QuantFormat::Fp8E4m3Block128,
                };
                out.push(store.put(key, rows, cols, sdt, &w, &s)?);
            }
        }
    }
    store.commit()?;
    Ok(out)
}
