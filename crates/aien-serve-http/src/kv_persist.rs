//! Persistent KV park and restore on the omega door (sc#505, ADR 0003), opt in with
//! `AIEN_KV_STORE_DIR`.
//!
//! With the directory unset nothing in this module runs: [`settings_from_lookup`] reads that one
//! variable and returns `None`, the door keeps [`PersistState::Off`], and generation, logs and
//! `/health` are what they were before persistence existed.
//!
//! With it set the door opens the sealed host store (`aien-kv-store`, the #511/#512 rules: a
//! private directory under a trusted parent, `AIEN_KV_STORE_KEY` required) and:
//!
//! - **park**: after a generation, under the GPU lock, the K/V of every chunk of the stored
//!   history that the store does not hold yet is exported (`Qwen3MoeOmegaEngine::export_kv`, the
//!   `HostPagedPoolV1` bytes of `kvio`); the lock is released and the chunks are submitted to the
//!   storage worker, which seals and writes them on its own thread.
//! - **restore**: before a generation, without the GPU lock, the longest stored prefix of the
//!   prompt (capped at `len - 1`, ADR 6.5: the last prompt token is always stepped) is looked up
//!   and loaded; only when it is longer than what the engine still holds in RAM is it imported
//!   (under the GPU lock, `restore_kv`), all or nothing.
//!
//! Every problem (store refused at start, missing or malformed key, unknown omega switch, a
//! miss, a failed or corrupt load, a timeout, an import error) ends in a plain prefill. Nothing
//! is ever partly restored: `restore_kv` resets the engine on error.
//!
//! The namespace binds the checkpoint digest, the model shape, the bf16 pool layout, the chunk
//! size, the sharing namespace and the engine identity of ADR 1.2.2 built by
//! [`OmegaKvIdentity::new`]: weight format as loaded, the build (omega commit, linked
//! `libomega_gpu.a` sha256, this crate's commit), a rounding class, and an engine config digest
//! over the MoE shape and every run switch that can move a K/V byte.
//!
//! Exactness limit (ADR 6.3 deviation): a stored tail chunk is found only when the prompt
//! continues the parked history; a prompt that diverges inside a chunk restores up to the last
//! whole chunk before the divergence, and may then restore less than the in-RAM rule would reuse
//! in an uninterrupted run (correct, only slower).

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use aien_inference_abi::qwen3_moe_omega::kvio::{self, PoolBlockGeometry};
use aien_inference_abi::qwen3_moe_omega::{
    OmegaKvSwitches, Qwen3MoeConfig, Qwen3MoeOmegaEngine, WeightFormat, KV_BLOCK_SIZE,
};
use aien_kv_store::host_fs::{Boundary, FaultAction, FaultHook, HostFsConfig, HostFsStore};
use aien_kv_store::worker::StorageWorker;
use aien_kv_store::{
    chunk_chain, chunk_key, BackendFormat, ChunkHeader, ChunkKey, ComputeConfig, EngineConfig,
    EngineIdentity, HiddenAct, HolderGuard, Incarnation, JobId, KvDtype, KvStore, LayoutStrides,
    NamespaceFields, NamespaceId, PinnedDest, PinnedSource, RecoveryProgress, StoreState, Terminal,
};
use aien_runtime::shared_kv::{KvPersistSettings, KV_STORE_DIR_ENV};
use sha2::{Digest, Sha256};

/// `ComputeConfig::arith_id` of the omega engine.
pub const ARITH_ID: &str = "omega-qwen3-moe";
/// Domain of the omega engine config digest.
const OMEGA_ENG_DOMAIN: &[u8] = b"AIEN-PKV-OMEGA-ENG-V1\0";
/// How long a restore waits for its chunks before giving up (plain prefill).
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a park waits for the storage worker's park budget to free (its earlier chunks
/// publishing) before it gives up on the rest of the chain. One 256-token chunk of the 30B model
/// is 25 MB and the default park budget is 64 MB, so a long turn always waits for credit (#514).
pub const PARK_CREDIT_WAIT: Duration = Duration::from_secs(10);

/// Omega environment switches that are read at run time and can change what the chip computes
/// (omega C `getenv` and the engine's own). Their values are part of the engine config digest.
pub const NUMERIC_ENV: &[&str] = &[
    "AIEN_OMEGA_CTA_BUDGET",
    "AIEN_OMEGA_DGRAPH_BARRIER",
    "AIEN_OMEGA_DGRAPH_PREFETCH",
    "AIEN_OMEGA_FP8_GEMV",
    "AIEN_OMEGA_FP8_WEIGHT_CACHE",
    "OWN30B_DECODE_GRAPH",
    "OWN30B_DECODE_GRAPH_STAMPS",
];
/// Omega variables that cannot move a K/V byte: timing (spin), build inputs (bound by the archive
/// digest in the build id) and test paths.
pub const NEUTRAL_ENV: &[&str] = &[
    "AIEN_OMEGA_SPIN_US",
    "AIEN_OMEGA_DIR",
    "AIEN_OMEGA_GPU",
    "AIEN_OMEGA_GPU_LIB",
    "AIEN_OMEGA_GPU_LIB_SHA256",
    "AIEN_OMEGA_PINNED_SHA",
    "AIEN_OMEGA_COMPOSE",
    "AIEN_OMEGA_COMPOSE_DIR",
    "AIEN_OMEGA_COMPOSE_LIB",
    "AIEN_OMEGA_COMPOSE_LIB_SHA256",
    "AIEN_OMEGA_COMPOSE_SHA",
    "AIEN_OMEGA_COMPOSE_EXPECTED_SHA",
    "OWN30B_MODEL_DIR",
    "OWN30B_CHECKPOINT",
];
/// Negative controls: persistence refuses while one is set.
pub const MUTANT_ENV: &[&str] = &["OWN30B_DECODE_GRAPH_MUTANT"];

/// The build of the engine that writes the K/V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmegaBuild {
    /// The omega commit `omega.lock` pins.
    pub omega_commit: String,
    /// How `libomega_gpu.a` was linked (`pinned-source`, `dev-override`, `prebuilt`, `stub`).
    pub gpu_link: String,
    /// sha256 of the linked `libomega_gpu.a`.
    pub gpu_archive_sha256: String,
    /// `locked` (the sha `native-artifacts.lock` records), `unrecorded` (no record for this
    /// commit and target), `mismatch` or `none`.
    pub gpu_digest: String,
    pub gpu_target: String,
    /// This crate's commit (`build.rs`).
    pub sc_commit: String,
    /// Built from a checkout with modified tracked files (or unknown).
    pub sc_dirty: bool,
}

impl OmegaBuild {
    /// What this binary was built from and linked.
    pub fn linked() -> Self {
        let id = aien_omega_gpu::native_identity();
        let recorded = env!("AIEN_SERVE_HTTP_GPU_RECORDED_SHA256");
        let gpu_digest = if id.archive_sha256.is_empty() {
            "none"
        } else if recorded == "unrecorded" {
            "unrecorded"
        } else if recorded == id.archive_sha256 {
            "locked"
        } else {
            "mismatch"
        };
        let pin = env!("AIEN_SERVE_HTTP_OMEGA_LOCK");
        Self {
            omega_commit: pin.to_string(),
            // A pinned-source archive built from another commit than the pin is not the pin.
            gpu_link: if id.link == "pinned-source" && id.source_sha != pin {
                "pinned-source-other-commit".into()
            } else {
                id.link.to_string()
            },
            gpu_archive_sha256: id.archive_sha256.to_string(),
            gpu_digest: gpu_digest.to_string(),
            gpu_target: id.target.to_string(),
            sc_commit: env!("AIEN_SERVE_HTTP_GIT_COMMIT").to_string(),
            sc_dirty: env!("AIEN_SERVE_HTTP_GIT_DIRTY") != "0",
        }
    }

    /// `omega:<commit>+libomega_gpu:<sha256>+sc:<commit>`. Refused when any part is unknown or
    /// cannot be trusted to name the bytes that run (fail closed).
    pub fn build_id(&self) -> Result<String, String> {
        let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
        if self.gpu_link != "pinned-source" {
            return Err(format!(
                "libomega_gpu.a is linked as {:?}, not built from the omega.lock pin",
                self.gpu_link
            ));
        }
        if !hex(&self.gpu_archive_sha256, 64) {
            return Err("the linked libomega_gpu.a has no sha256".into());
        }
        if !matches!(self.gpu_digest.as_str(), "locked" | "unrecorded") {
            return Err(format!(
                "libomega_gpu.a digest is {:?} against native-artifacts.lock",
                self.gpu_digest
            ));
        }
        if !hex(&self.omega_commit, 40) {
            return Err(format!(
                "omega.lock commit {:?} is not a 40-hex commit",
                self.omega_commit
            ));
        }
        if !(self.sc_commit.len() >= 7
            && self.sc_commit.len() <= 64
            && hex(&self.sc_commit, self.sc_commit.len()))
        {
            return Err(format!(
                "the engine commit {:?} is unknown (build without git; set AIEN_BUILD_COMMIT)",
                self.sc_commit
            ));
        }
        if self.sc_dirty {
            return Err("the binary was built from a checkout with uncommitted changes".into());
        }
        if self.gpu_target.is_empty() {
            return Err("the linked libomega_gpu.a has no target".into());
        }
        Ok(format!(
            "omega:{}+libomega_gpu:{}+sc:{}",
            self.omega_commit, self.gpu_archive_sha256, self.sc_commit
        ))
    }
}

/// Everything the omega engine identity is built from.
pub struct IdentityInputs<'a> {
    /// The served model id (namespace field 4).
    pub model_id: &'a str,
    pub model_sha256: [u8; 32],
    /// `file` or `index+shards` ([`checkpoint_digest`]).
    pub model_digest_kind: &'a str,
    pub config: &'a Qwen3MoeConfig,
    /// `config.json` has a non-null `rope_scaling` (the omega engine has no scaled rope: refused).
    pub rope_scaling: bool,
    pub weights: WeightFormat,
    pub switches: OmegaKvSwitches,
    pub build: &'a OmegaBuild,
    /// The process environment (name, value).
    pub env: &'a [(String, String)],
}

/// The owned namespace inputs of the omega door, except the chunk size and the sharing
/// namespace (from the settings).
#[derive(Debug, Clone, PartialEq)]
pub struct OmegaKvIdentity {
    pub weight_format: aien_kv_store::WeightFormat,
    pub engine_build_id: String,
    pub rounding_class: String,
    pub engine_config_digest: [u8; 32],
    model_sha256: [u8; 32],
    model_digest_kind: String,
    model_id: String,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_theta: f32,
    compute_config_digest: [u8; 32],
    geometry: PoolBlockGeometry,
}

fn store_weight_format(w: WeightFormat) -> aien_kv_store::WeightFormat {
    match w {
        WeightFormat::Bf16 => aien_kv_store::WeightFormat::Bf16,
        WeightFormat::Fp8Experts => aien_kv_store::WeightFormat::Fp8Experts,
        WeightFormat::Fp8All => aien_kv_store::WeightFormat::Fp8All,
    }
}

fn weights_label(w: WeightFormat) -> &'static str {
    match w {
        WeightFormat::Bf16 => "bf16",
        WeightFormat::Fp8Experts => "fp8-experts",
        WeightFormat::Fp8All => "fp8-all",
    }
}

fn u32_of(name: &str, v: usize) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{name} {v} does not fit 32 bits"))
}

impl OmegaKvIdentity {
    pub fn new(i: &IdentityInputs<'_>) -> Result<Self, String> {
        if i.rope_scaling {
            return Err("config.json has rope_scaling, the omega engine has no scaled rope".into());
        }
        // The omega environment: mutants refuse, unknown switches refuse (unsure = refuse), the
        // numeric ones are bound by value.
        let mut numeric: BTreeMap<&str, &str> = BTreeMap::new();
        for (k, v) in i.env {
            let omega = k.starts_with("AIEN_OMEGA_") || k.starts_with("OWN30B_");
            if !omega {
                continue;
            }
            if MUTANT_ENV.contains(&k.as_str()) {
                if !v.is_empty() {
                    return Err(format!(
                        "{k} is set (a negative control): K/V from a mutant must never be stored"
                    ));
                }
            } else if let Some(n) = NUMERIC_ENV.iter().find(|n| **n == k.as_str()) {
                numeric.insert(n, v.as_str());
            } else if !NEUTRAL_ENV.contains(&k.as_str()) {
                return Err(format!("{k} is set and its effect on the K/V is not known"));
            }
        }
        // Exhaustive on purpose: no `..`, so a new run switch has to be placed here.
        let OmegaKvSwitches {
            attention_path,
            device_routing,
            prefill_chunk,
            prefill_gemm,
            prefill_device_routing,
            whole_prompt,
            whole_prompt_device_host,
            decode_graph,
            lm_head_gemv,
            routing_mutant,
            trace_layers,
        } = i.switches;
        if routing_mutant {
            return Err("the engine runs the routing mutant (a negative control)".into());
        }
        // Exhaustive on purpose: no `..`.
        let Qwen3MoeConfig {
            layers,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            experts,
            top_k,
            moe_intermediate,
            vocab,
            rms_eps,
            rope_theta,
            norm_topk_prob,
            tie_word_embeddings,
            // Decides when a sequence stops, not what is computed.
            eos_token_ids: _,
        } = i.config;
        let engine_build_id = i.build.build_id()?;
        let base = EngineConfig {
            num_experts: u32_of("experts", *experts)?,
            experts_per_token: u32_of("top_k", *top_k)?,
            expert_intermediate_dim: *moe_intermediate as u64,
            whole_prompt_prefill: whole_prompt,
            device_router: device_routing,
            matmul_path: if prefill_gemm {
                "fp8-grouped-gemm"
            } else {
                "grouped-gemv"
            },
        }
        .digest()
        .map_err(|e| format!("engine config: {e:?}"))?;
        let mut text = format!(
            "weights={}\nnorm_topk_prob={norm_topk_prob}\nattention_path={attention_path}\nprefill_chunk={prefill_chunk}\nprefill_device_routing={prefill_device_routing}\nwhole_prompt_device_host={whole_prompt_device_host}\ndecode_graph={decode_graph}\nlm_head_gemv={lm_head_gemv}\ntrace_layers={trace_layers}\nq_heads={q_heads}\n",
            weights_label(i.weights)
        );
        for n in NUMERIC_ENV {
            match numeric.get(n) {
                Some(v) => text.push_str(&format!("env {n}={v:?}\n")),
                None => text.push_str(&format!("env {n} unset\n")),
            }
        }
        let mut h = Sha256::new();
        h.update(OMEGA_ENG_DOMAIN);
        h.update(base);
        h.update(text.as_bytes());
        let engine_config_digest: [u8; 32] = h.finalize().into();
        let rounding_class = format!(
            "omega-gpu/{}/{}/kv-bf16/attn-{attention_path}@kv-arith-epoch-{}",
            i.build.gpu_target,
            weights_label(i.weights),
            aien_inference_abi::KV_ARITH_EPOCH
        );
        let weight_format = store_weight_format(i.weights);
        let engine = EngineIdentity {
            weight_format,
            engine_build_id: &engine_build_id,
            rounding_class: &rounding_class,
            engine_config_digest,
        };
        let compute_config_digest = ComputeConfig {
            num_heads: u32_of("q_heads", *q_heads)?,
            hidden_dim: *hidden as u64,
            intermediate_dim: *moe_intermediate as u64,
            vocab_size: *vocab as u64,
            rms_norm_eps: *rms_eps,
            tie_word_embeddings: *tie_word_embeddings,
            hidden_act: HiddenAct::Silu,
            arith_id: ARITH_ID,
            arith_epoch: aien_inference_abi::KV_ARITH_EPOCH,
        }
        .digest_with_engine(&engine)
        .map_err(|e| format!("compute config: {e:?}"))?;
        Ok(Self {
            weight_format,
            engine_build_id,
            rounding_class,
            engine_config_digest,
            model_sha256: i.model_sha256,
            model_digest_kind: i.model_digest_kind.to_string(),
            model_id: i.model_id.to_string(),
            num_layers: u32_of("layers", *layers)?,
            num_kv_heads: u32_of("kv_heads", *kv_heads)?,
            head_dim: u32_of("head_dim", *head_dim)?,
            rope_theta: *rope_theta,
            compute_config_digest,
            geometry: PoolBlockGeometry::for_config(i.config),
        })
    }

    /// The namespace fields for `chunk_tokens` and `sharing_namespace`.
    pub fn fields<'a>(
        &'a self,
        chunk_tokens: u32,
        sharing_namespace: &'a str,
    ) -> NamespaceFields<'a> {
        let g = &self.geometry;
        NamespaceFields {
            model_sha256: self.model_sha256,
            model_digest_kind: &self.model_digest_kind,
            model_id: &self.model_id,
            num_layers: self.num_layers,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            block_size: KV_BLOCK_SIZE as u32,
            rope_theta: self.rope_theta,
            rope_scaling: None,
            qk_norm: true,
            dtype: KvDtype::Bf16,
            layout: LayoutStrides {
                head_stride: g.head_stride() as u64,
                token_stride: g.token_stride() as u64,
                kv_plane_stride: g.kv_plane_stride() as u64,
                layer_stride: g.layer_stride() as u64,
                block_stride: g.block_stride() as u64,
                element_bytes: 2,
            },
            backend_format: BackendFormat::HostPagedPoolV1,
            chunk_tokens,
            sharing_namespace,
            compute_config_digest: self.compute_config_digest,
        }
    }
}

/// Test-only fault hook of the chip acceptance checks (sc#507): `AIEN_KV_TEST_FAULT` =
/// `<boundary>:<stall ms>[:<count>]` makes the store sleep at one point of its protocol so a
/// fault lands deterministically: `p1` before the temp write, `p2` before the temp fsync, `p3`
/// before the rename that publishes, `p4` before the directory fsync, `p5` before the index
/// insert, `load` before a restore read. `count` stalls only the first N passes (default 0:
/// every pass). It only delays I/O: no K/V byte changes, so it is not part of the identity.
/// Unset in production; a malformed value refuses persistence (fail closed).
pub const TEST_FAULT_ENV: &str = "AIEN_KV_TEST_FAULT";

/// The parsed `AIEN_KV_TEST_FAULT` stall.
pub struct TestFault {
    boundary: Boundary,
    stall: Duration,
    /// Passes still to stall; `u64::MAX` = every pass.
    left: AtomicU64,
    spec: String,
}

impl TestFault {
    /// `None` when the variable is unset or empty.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Option<Arc<Self>>, String> {
        let Some(spec) = get(TEST_FAULT_ENV).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };
        let bad = || {
            format!(
                "{TEST_FAULT_ENV}={spec:?}: expected <p1|p2|p3|p4|p5|load>:<stall ms>[:<count>]"
            )
        };
        let parts: Vec<&str> = spec.split(':').collect();
        if !(2..=3).contains(&parts.len()) {
            return Err(bad());
        }
        let boundary = match parts[0] {
            "p1" => Boundary::P1Write,
            "p2" => Boundary::P2FsyncFile,
            "p3" => Boundary::P3Rename,
            "p4" => Boundary::P4FsyncDir,
            "p5" => Boundary::P5IndexInsert,
            "load" => Boundary::LoadRead,
            _ => return Err(bad()),
        };
        let ms: u64 = parts[1].parse().map_err(|_| bad())?;
        let count: u64 = match parts.get(2) {
            Some(c) => c.parse().map_err(|_| bad())?,
            None => 0,
        };
        Ok(Some(Arc::new(Self {
            boundary,
            stall: Duration::from_millis(ms),
            left: AtomicU64::new(if count == 0 { u64::MAX } else { count }),
            spec,
        })))
    }

    pub fn spec(&self) -> &str {
        &self.spec
    }
}

impl FaultHook for TestFault {
    fn at(&self, boundary: Boundary, _key: &ChunkKey) -> Option<FaultAction> {
        if boundary != self.boundary {
            return None;
        }
        let took = self
            .left
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| match n {
                0 => None,
                u64::MAX => Some(u64::MAX),
                n => Some(n - 1),
            })
            .is_ok();
        took.then_some(FaultAction::Stall(self.stall))
    }
}

/// Persistence settings when `AIEN_KV_STORE_DIR` is set and not empty; `None` otherwise, having
/// read nothing but that one variable (the switch-off path).
pub fn settings_from_lookup(
    get: impl Fn(&str) -> Option<String>,
) -> Option<Result<KvPersistSettings, String>> {
    let dir = get(KV_STORE_DIR_ENV)?;
    if dir.is_empty() {
        return None;
    }
    Some(KvPersistSettings::from_lookup(|n| {
        if n == KV_STORE_DIR_ENV {
            Some(dir.clone())
        } else {
            get(n)
        }
    }))
}

/// sha256 of a file, streamed.
fn sha256_file(p: &Path) -> Result<[u8; 32], String> {
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().into())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The checkpoint digest of `aien-cli` (`checkpoint_digest`): one weight file is its sha256
/// (`file`); a sharded checkpoint is the sha256 of the manifest `aien-checkpoint-digest v1`,
/// the index sha256 and one `<sha256>  <shard>` line per shard (`index+shards`).
pub fn checkpoint_digest(model_dir: &Path) -> Result<([u8; 32], &'static str), String> {
    let index = model_dir.join("model.safetensors.index.json");
    let path = if index.exists() {
        index
    } else {
        model_dir.join("model.safetensors")
    };
    let files = aien_inference_abi::checkpoint_files(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let Some(index) = files.index else {
        let (_, file) = files
            .weights
            .first()
            .ok_or("the checkpoint names no weight file")?;
        return Ok((sha256_file(file)?, "file"));
    };
    if files.weights.is_empty() {
        return Err("the shard index names no shard".into());
    }
    let mut manifest = format!(
        "aien-checkpoint-digest v1\nindex {}\n",
        hex(&sha256_file(&index)?)
    );
    for (name, file) in &files.weights {
        manifest.push_str(&format!("{}  {name}\n", hex(&sha256_file(file)?)));
    }
    Ok((Sha256::digest(manifest.as_bytes()).into(), "index+shards"))
}

/// Whether `config.json` in `model_dir` sets a non-null `rope_scaling`.
pub fn config_has_rope_scaling(model_dir: &Path) -> Result<bool, String> {
    let p = model_dir.join("config.json");
    let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(v.get("rope_scaling").is_some_and(|r| !r.is_null()))
}

/// What the door does about persistence.
pub enum PersistState {
    /// `AIEN_KV_STORE_DIR` unset: nothing runs.
    Off,
    /// Asked for, refused at start: every prompt is prefilled.
    Refused(String),
    On(Box<KvPersistence>),
}

/// Counters of the persistence path (`/health` `persistent_kv`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PersistCounters {
    /// Chunks the store published.
    pub parked_chunks: u64,
    pub park_bytes: u64,
    /// Chunks of the history the store already held.
    pub park_dedup: u64,
    /// Chunks the store refused at submit (quota, queue, state).
    pub park_refused: u64,
    /// Chunks whose export or write failed.
    pub park_failed: u64,
    /// Generations not parked because the store was not open (recovering, degraded, failed).
    pub park_skipped: u64,
    /// Generations that started from a restored prefix.
    pub restores: u64,
    pub restored_tokens: u64,
    pub restore_bytes: u64,
    /// Restore attempts that fell back to plain prefill, by reason.
    pub restore_misses: BTreeMap<String, u64>,
    /// Start refusals (0 or 1).
    pub refusals: u64,
}

/// One stored chunk of a restore: `stored` tokens on disk, the first `used` of them restored.
struct StoredChunk {
    key: ChunkKey,
    parent: Option<ChunkKey>,
    first: usize,
    stored: usize,
    used: usize,
}

/// A loaded prefix, ready to import.
pub struct Fetched {
    pub tokens: usize,
    pub payload: Vec<u8>,
}

/// One exported chunk waiting for the store.
pub struct ParkItem {
    header: ChunkHeader,
    tokens: usize,
    bytes: Vec<u8>,
}

/// The landing buffer of one load: shared by the load job (through its holder guard) and the
/// door; freed when both let go.
struct LandBuf {
    ptr: *mut u8,
    len: usize,
    _buf: Box<[u8]>,
}

// SAFETY: the bytes behind `ptr` are written only by the load job while its holder is held, and
// read by the door only after that job's terminal completion was polled (ADR 5.5).
unsafe impl Send for LandBuf {}
unsafe impl Sync for LandBuf {}

impl LandBuf {
    fn new(len: usize) -> Arc<Self> {
        let mut buf = vec![0u8; len].into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        Arc::new(Self {
            ptr,
            len,
            _buf: buf,
        })
    }
    /// # Safety
    /// Only after the load job writing this buffer has reached its terminal completion.
    unsafe fn bytes(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}

#[derive(Default)]
struct Inner {
    counters: PersistCounters,
    /// In-flight park jobs.
    stores: HashMap<JobId, u64>,
    /// In-flight load jobs: `None` until the terminal arrives.
    loads: HashMap<JobId, Option<Result<(), String>>>,
}

/// An open store with its worker, for one omega door.
pub struct KvPersistence {
    worker: Arc<StorageWorker>,
    /// The store itself, read only for its I/O and recovery counters in `/health` (#509).
    store: Arc<HostFsStore>,
    /// The last recovery step's progress (startup recovery, #509).
    recovery: Arc<Mutex<RecoveryProgress>>,
    identity: OmegaKvIdentity,
    sharing: String,
    chunk_tokens: usize,
    ns: NamespaceId,
    store_dir: PathBuf,
    park: AtomicBool,
    restore: bool,
    drain_ms: u64,
    inner: Mutex<Inner>,
    seq: AtomicU64,
    /// The armed [`TEST_FAULT_ENV`] spec, if any.
    test_fault: Option<String>,
}

fn miss_reason(e: &aien_platform::PlatformError) -> &'static str {
    use aien_platform::PlatformError as P;
    match e {
        P::ObjectNotFound(_) => "not-found",
        P::InvalidLayout => "invalid",
        P::AccessDenied => "access-denied",
        P::HardwareFault(_) => "io",
        P::QueueFull | P::OutOfMemory => "busy",
        P::DeviceUnavailable => "store-unavailable",
        _ => "error",
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

impl KvPersistence {
    fn fields(&self) -> NamespaceFields<'_> {
        self.identity
            .fields(self.chunk_tokens as u32, &self.sharing)
    }

    fn inc(&self) -> Incarnation {
        Incarnation::new(self.seq.fetch_add(1, Ordering::Relaxed) + 1, 1)
    }

    fn block_bytes(&self) -> usize {
        self.identity.geometry.block_stride()
    }

    /// The store finished recovery (or failed) within `timeout`.
    pub fn wait_ready(&self, timeout: Duration) -> bool {
        let t0 = Instant::now();
        loop {
            if self.worker.capacity().state != StoreState::Recovering {
                return true;
            }
            if t0.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Every submitted park reached its terminal within `timeout`.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let t0 = Instant::now();
        loop {
            {
                let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                self.pump(&mut g);
                if g.stores.is_empty() {
                    return true;
                }
            }
            if t0.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    pub fn counters(&self) -> PersistCounters {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.pump(&mut g);
        g.counters.clone()
    }

    pub fn status_json(&self) -> serde_json::Value {
        let c = self.counters();
        let t = aien_kv_store::seal::seal_timing();
        let r = *self.recovery.lock().unwrap_or_else(|p| p.into_inner());
        let s = self.store.stats();
        let state = match self.worker.capacity().state {
            StoreState::Recovering => "recovering",
            StoreState::Open => "open",
            StoreState::Degraded => "degraded",
            StoreState::Failed => "failed",
        };
        serde_json::json!({
            "state": "on",
            "store_state": state,
            "store_dir": self.store_dir.display().to_string(),
            "namespace": hex(&self.ns.0[..8]),
            "chunk_tokens": self.chunk_tokens,
            "park": self.park.load(Ordering::Relaxed),
            "restore": self.restore,
            "engine_build_id": self.identity.engine_build_id,
            "rounding_class": self.identity.rounding_class,
            "parked_chunks": c.parked_chunks,
            "park_bytes": c.park_bytes,
            "park_dedup": c.park_dedup,
            "park_refused": c.park_refused,
            "park_failed": c.park_failed,
            "park_skipped": c.park_skipped,
            "restores": c.restores,
            "restored_tokens": c.restored_tokens,
            "restore_bytes": c.restore_bytes,
            "restore_misses": c.restore_misses,
            "refusals": c.refusals,
            "seal": {
                "seal_calls": t.seal_calls,
                "seal_bytes": t.seal_bytes,
                "seal_ns": t.seal_ns,
                "open_calls": t.open_calls,
                "open_bytes": t.open_bytes,
                "open_ns": t.open_ns,
            },
            "recovery": {
                "done": r.done,
                "failed": r.failed,
                "indexed": r.indexed,
                "indexed_bytes": r.bytes,
                "foreign_namespace": r.foreign_namespace,
                "corrupt_deleted": r.corrupt_deleted,
                "tmp_deleted": r.tmp_deleted,
                "elapsed_us": r.elapsed_us,
                "unlink_failures": s.recovery_unlink_failures,
                "foreign_files": s.foreign_files,
            },
            "io": {
                "reads": s.reads,
                "read_bytes": s.read_bytes,
                "writes": s.writes,
                "write_bytes": s.write_bytes,
                "file_fsyncs": s.file_fsyncs,
                "published": s.published_total,
                "disk_full": s.disk_full_total,
                "queue_full": s.queue_full_total,
                "indeterminate": s.indeterminate_total,
                "failed": s.terminal_failed,
                "corrupt_deleted": s.corrupt_deleted_total,
            },
            "test_fault": self.test_fault,
        })
    }

    fn miss(&self, reason: &str) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        *g.counters
            .restore_misses
            .entry(reason.to_string())
            .or_insert(0) += 1;
    }

    /// Drains the worker's completions into the counters and the load slots.
    fn pump(&self, g: &mut Inner) {
        loop {
            let batch = self.worker.poll(64);
            if batch.is_empty() {
                return;
            }
            for c in batch {
                let Some(t) = c.terminal else { continue };
                if let Some(bytes) = g.stores.remove(&c.job) {
                    match t {
                        Terminal::Done(_) => {
                            g.counters.parked_chunks += 1;
                            g.counters.park_bytes += bytes;
                        }
                        Terminal::Failed(_) | Terminal::Cancelled => g.counters.park_failed += 1,
                    }
                } else if let Some(slot) = g.loads.get_mut(&c.job) {
                    *slot = Some(match t {
                        Terminal::Done(_) => Ok(()),
                        Terminal::Failed(e) => Err(miss_reason(&e).to_string()),
                        Terminal::Cancelled => Err("cancelled".to_string()),
                    });
                }
                // `c.holder` (a load's landing holder) drops here, after the slot is set.
            }
        }
    }

    /// Pauses or resumes parking (the chip check freezes the store after turn 1).
    pub fn set_park(&self, on: bool) {
        self.park.store(on, Ordering::Relaxed);
    }

    pub fn restore_on(&self) -> bool {
        self.restore
    }

    /// The chunks of the longest stored prefix of `prompt` (capped at `len - 1`).
    fn find(&self, prompt: &[u32]) -> Result<Vec<StoredChunk>, &'static str> {
        let cap = prompt.len().saturating_sub(1);
        let ct = self.chunk_tokens;
        if cap == 0 {
            return Err("short-prompt");
        }
        let full = cap / ct;
        let keys = if full > 0 {
            chunk_chain(&self.ns, ct as u32, &prompt[..full * ct]).map_err(|_| "key")?
        } else {
            Vec::new()
        };
        let found = self.worker.lookup(&keys);
        let f = found.iter().take_while(|x| x.is_some()).count();
        let mut out: Vec<StoredChunk> = (0..f)
            .map(|i| StoredChunk {
                key: keys[i],
                parent: i.checked_sub(1).map(|p| keys[p]),
                first: i * ct,
                stored: ct,
                used: ct,
            })
            .collect();
        // The tail: the parked history may end inside chunk `f` (a stored partial chunk).
        let first = f * ct;
        // Probe up to the prompt end, use up to the cap: a stored tail that runs to the last
        // prompt token still serves every position before it (ADR 6.5).
        let max_len = if first < cap {
            ct.min(prompt.len() - first)
        } else {
            0
        };
        let parent = f.checked_sub(1).map(|p| keys[p]);
        let lens: Vec<usize> = (1..=max_len)
            .rev()
            .filter(|&l| !(l == ct && f < full))
            .collect();
        let mut tail_keys = Vec::with_capacity(lens.len());
        for &l in &lens {
            tail_keys.push(
                chunk_key(
                    &self.ns,
                    parent.as_ref(),
                    first as u32,
                    &prompt[first..first + l],
                )
                .map_err(|_| "key")?,
            );
        }
        let tail_found = self.worker.lookup(&tail_keys);
        if let Some(j) = tail_found.iter().position(|x| x.is_some()) {
            out.push(StoredChunk {
                key: tail_keys[j],
                parent,
                first,
                stored: lens[j],
                used: lens[j].min(cap - first),
            });
        }
        Ok(out)
    }

    /// Looks up and loads the longest stored prefix of `prompt` when it beats `in_ram`. Runs
    /// without the GPU lock. `None` = plain prefill (the reason is counted).
    pub fn fetch(&self, prompt: &[u32], in_ram: usize) -> Option<Fetched> {
        if self.worker.capacity().state == StoreState::Recovering {
            self.miss("store-recovering");
            return None;
        }
        let chunks = match self.find(prompt) {
            Ok(c) => c,
            Err(r) => {
                self.miss(r);
                return None;
            }
        };
        let r: usize = chunks.iter().map(|c| c.used).sum();
        if r == 0 {
            self.miss("not-found");
            return None;
        }
        if r <= in_ram {
            self.miss("ram-prefix-as-long");
            return None;
        }
        let fields = self.fields();
        let mut jobs: Vec<(JobId, Arc<LandBuf>)> = Vec::new();
        let mut fail: Option<String> = None;
        let t_fetch = Instant::now();
        for c in &chunks {
            let (key, tokens) = (c.key, c.stored);
            let header = match fields.chunk_header(
                &self.ns,
                key,
                c.parent.as_ref(),
                c.first as u32,
                tokens as u32,
                0,
            ) {
                Ok(h) => h,
                Err(_) => {
                    fail = Some("header".into());
                    break;
                }
            };
            let len = header.payload_len as usize;
            let nb = tokens.div_ceil(KV_BLOCK_SIZE);
            if len != nb * self.block_bytes() {
                fail = Some("invalid".into());
                break;
            }
            let buf = LandBuf::new(len);
            // A prefix larger than the worker's restore budget is normal on the real model (a
            // 2079-token prefix of the 30B is 204 MB, the default restore budget 192 MB): on
            // `QueueFull` the chunk waits for the earlier loads to land and free their credit,
            // within [`LOAD_TIMEOUT`] of the fetch start (#514). The inner lock is not held
            // while waiting.
            loop {
                let keep = buf.clone();
                // SAFETY: `buf` stays allocated while the guard (held by the job) lives, and
                // nobody reads it until the job's terminal completion has been polled. A refused
                // submit drops its guard; `buf` itself stays alive for the retry.
                let dst = unsafe {
                    PinnedDest::new(
                        buf.ptr,
                        self.block_bytes(),
                        (0..nb).collect(),
                        HolderGuard::new(move || drop(keep)),
                    )
                };
                let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                self.pump(&mut g);
                match self.worker.submit_load(key, &header, dst, self.inc()) {
                    Ok(job) => {
                        g.loads.insert(job, None);
                        jobs.push((job, buf));
                        break;
                    }
                    Err(aien_platform::PlatformError::QueueFull)
                        if t_fetch.elapsed() < LOAD_TIMEOUT =>
                    {
                        drop(g);
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    Err(e) => {
                        fail = Some(miss_reason(&e).to_string());
                        break;
                    }
                }
            }
            if fail.is_some() {
                break;
            }
        }
        if fail.is_none() {
            let t0 = Instant::now();
            loop {
                let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                self.pump(&mut g);
                let mut all = true;
                for (job, _) in &jobs {
                    match g.loads.get(job) {
                        Some(Some(Ok(()))) => {}
                        Some(Some(Err(e))) => {
                            fail = Some(e.clone());
                            break;
                        }
                        _ => all = false,
                    }
                }
                if fail.is_some() || all {
                    break;
                }
                drop(g);
                if t0.elapsed() >= LOAD_TIMEOUT {
                    fail = Some("timeout".into());
                    break;
                }
                std::thread::sleep(Duration::from_micros(200));
            }
        }
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(reason) = fail {
            for (job, _) in &jobs {
                if !matches!(g.loads.get(job), Some(Some(_))) {
                    let _ = self.worker.cancel(*job);
                }
                g.loads.remove(job);
            }
            *g.counters.restore_misses.entry(reason).or_insert(0) += 1;
            // Buffers of jobs still running stay alive through their guards.
            return None;
        }
        let mut payload = Vec::with_capacity(jobs.iter().map(|(_, b)| b.len).sum());
        for (job, buf) in &jobs {
            g.loads.remove(job);
            // SAFETY: the job's terminal `Done` was polled above.
            payload.extend_from_slice(unsafe { buf.bytes() });
        }
        // A tail chunk stored past the cap: keep its first rows, zero the rest of the block.
        let g = &self.identity.geometry;
        payload.truncate(g.payload_len(r));
        kvio::zero_padding(g, r, &mut payload);
        Some(Fetched { tokens: r, payload })
    }

    /// Counts a restore that the engine imported.
    pub fn note_restored(&self, f: &Fetched) {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.counters.restores += 1;
        g.counters.restored_tokens += f.tokens as u64;
        g.counters.restore_bytes += f.payload.len() as u64;
    }

    /// Counts a fetched prefix the engine refused to import (it was reset: plain prefill).
    pub fn note_import_failed(&self) {
        self.miss("import");
    }

    /// Under the GPU lock, after a generation: exports the chunks of the engine's history that
    /// the store does not hold. Submitting them is [`Self::submit`], after the lock is released.
    pub fn snapshot(&self, e: &mut Qwen3MoeOmegaEngine) -> Vec<ParkItem> {
        if !self.park.load(Ordering::Relaxed) {
            return Vec::new();
        }
        if self.worker.capacity().state != StoreState::Open {
            self.inner
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .counters
                .park_skipped += 1;
            return Vec::new();
        }
        let hist = e.history().to_vec();
        if hist.is_empty() {
            return Vec::new();
        }
        let ct = self.chunk_tokens;
        let Ok(keys) = chunk_chain(&self.ns, ct as u32, &hist) else {
            self.inner
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .counters
                .park_failed += 1;
            return Vec::new();
        };
        let found = self.worker.lookup(&keys);
        let fields = self.fields();
        let written = now_ns();
        let mut items = Vec::new();
        let mut dedup = 0u64;
        let mut failed = 0u64;
        for (i, key) in keys.iter().enumerate() {
            if found.get(i).copied().flatten().is_some() {
                dedup += 1;
                continue;
            }
            let first = i * ct;
            let tokens = (hist.len() - first).min(ct);
            let header = match fields.chunk_header(
                &self.ns,
                *key,
                i.checked_sub(1).map(|p| &keys[p]),
                first as u32,
                tokens as u32,
                written,
            ) {
                Ok(h) => h,
                Err(_) => {
                    failed += 1;
                    break;
                }
            };
            match e.export_kv(first, tokens) {
                Ok(bytes) if bytes.len() as u64 == header.payload_len => items.push(ParkItem {
                    header,
                    tokens,
                    bytes,
                }),
                _ => {
                    failed += 1;
                    break;
                }
            }
        }
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        g.counters.park_dedup += dedup;
        g.counters.park_failed += failed;
        items
    }

    /// Hands exported chunks to the storage worker (no GPU lock held). Stops at the first
    /// refusal: a chain with a hole cannot be restored past it.
    ///
    /// A chain longer than the worker's park budget is normal on the real model (one 256-token
    /// chunk of the 30B is 25 MB, the default budget 64 MB): on `QueueFull` the chunk waits, up to
    /// [`PARK_CREDIT_WAIT`], for the earlier chunks to publish and free their credit, then retries.
    /// Runs after the GPU lock is released; the inner lock is not held while waiting.
    pub fn submit(&self, items: Vec<ParkItem>) {
        for it in items {
            let nb = it.tokens.div_ceil(KV_BLOCK_SIZE);
            let tail_valid = (it.tokens - (nb - 1) * KV_BLOCK_SIZE) as u32;
            // Shared so a refused submit (which drops its guard) leaves the bytes for the retry.
            let bytes: Arc<[u8]> = it.bytes.into();
            let len = bytes.len() as u64;
            let t0 = Instant::now();
            loop {
                let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
                self.pump(&mut g);
                let keep = bytes.clone();
                // SAFETY: the guard owns a reference to the bytes, so they live and stay unwritten
                // (nothing writes through an `Arc<[u8]>`) until the store releases the holder.
                let src = unsafe {
                    PinnedSource::new(
                        bytes.as_ptr(),
                        self.block_bytes(),
                        (0..nb).collect(),
                        tail_valid,
                        HolderGuard::new(move || drop(keep)),
                    )
                };
                match self.worker.submit_store(it.header, src, self.inc()) {
                    Ok(job) => {
                        g.stores.insert(job, len);
                        break;
                    }
                    Err(aien_platform::PlatformError::QueueFull)
                        if t0.elapsed() < PARK_CREDIT_WAIT =>
                    {
                        drop(g);
                        std::thread::sleep(Duration::from_micros(500));
                    }
                    Err(_) => {
                        g.counters.park_refused += 1;
                        // A chain with a hole cannot be restored past it.
                        return;
                    }
                }
            }
        }
    }
}

impl Drop for KvPersistence {
    fn drop(&mut self) {
        // Let the parks of the last turns publish, then stop the worker.
        let _ = self.wait_idle(Duration::from_millis(self.drain_ms));
        let _ = self.worker.shutdown(self.drain_ms);
    }
}

/// Opens the store for `identity` under `settings`. Every problem refuses (plain prefill) and
/// says why on the returned startup line.
pub fn open(
    settings: Result<KvPersistSettings, String>,
    identity: Result<OmegaKvIdentity, String>,
) -> (PersistState, String) {
    open_with_test_fault(settings, identity, Ok(None))
}

/// [`open`] with the store's test-only stall ([`TEST_FAULT_ENV`]); `Err` (a malformed value) refuses.
pub fn open_with_test_fault(
    settings: Result<KvPersistSettings, String>,
    identity: Result<OmegaKvIdentity, String>,
    fault: Result<Option<Arc<TestFault>>, String>,
) -> (PersistState, String) {
    match open_inner(settings, identity, fault) {
        Ok((p, line)) => (PersistState::On(Box::new(p)), line),
        Err(why) => {
            let line = format!("persistent KV: REFUSED ({why}); running without persistence: no park, no restore, every prompt is prefilled");
            (PersistState::Refused(why), line)
        }
    }
}

fn open_inner(
    settings: Result<KvPersistSettings, String>,
    identity: Result<OmegaKvIdentity, String>,
    fault: Result<Option<Arc<TestFault>>, String>,
) -> Result<(KvPersistence, String), String> {
    let settings = settings?;
    let fault = fault?;
    let root = settings.store_dir.clone().ok_or("no store directory")?;
    let key = match &settings.store_key {
        Some(Ok(k)) => k.clone(),
        Some(Err(r)) => {
            aien_kv_store::seal::note_key_refusal();
            return Err(format!("{r}"));
        }
        None => return Err("the store key was not read".into()),
    };
    let identity = identity.map_err(|e| format!("engine identity: {e}"))?;
    let chunk_tokens = (settings.chunk_blocks as usize)
        .checked_mul(KV_BLOCK_SIZE)
        .filter(|&t| u32::try_from(t).is_ok())
        .ok_or("AIEN_KV_CHUNK_BLOCKS: chunk size overflows")?;
    let (ns, mut cfg) = {
        let fields = identity.fields(chunk_tokens as u32, &settings.namespace);
        let ns = fields
            .namespace_id()
            .map_err(|e| format!("namespace: {e:?}"))?;
        (
            ns,
            HostFsConfig::from_fields(root.clone(), &fields)
                .map_err(|e| format!("store config: {e:?}"))?,
        )
    };
    if let Some(v) = settings.quota_bytes {
        cfg.quota_bytes = v;
    }
    if let Some(v) = settings.max_objects {
        cfg.max_objects = v;
    }
    if let Some(v) = settings.recovery_slice_ms {
        cfg.recovery_slice_ms = v;
    }
    cfg.seal_key = Some(key);
    if let Some(f) = &fault {
        cfg.faults = Some(f.clone() as Arc<dyn FaultHook>);
    }
    let slice_ms = cfg.recovery_slice_ms;
    let recovery = Arc::new(Mutex::new(RecoveryProgress::default()));
    let store = Arc::new(HostFsStore::open(cfg).map_err(|e| match e {
        // Another open store holds the directory (#521): no privacy hint.
        aien_platform::PlatformError::TaskRejected(why) => format!("store open: {why}"),
        e => format!("store open at {}: {e:?} (the directory must be private, under a parent only you can write)", root.display()),
    })?);
    let drive = {
        let store = store.clone();
        let progress = recovery.clone();
        let mut recovered = false;
        move || {
            if !recovered {
                let p = store.recover_step(slice_ms);
                *progress.lock().unwrap_or_else(|e| e.into_inner()) = p;
                recovered = p.done || p.failed;
            }
        }
    };
    let worker = Arc::new(
        StorageWorker::spawn(store.clone(), settings.worker.clone(), drive)
            .map_err(|e| format!("storage worker: {e:?}"))?,
    );
    let line = format!(
        "persistent KV: on, store {} namespace {:?} chunk {} tokens, weights {}, engine build {}, rounding {}, park {}, restore {}, drain {} ms (recovering in the background)",
        root.display(),
        settings.namespace,
        chunk_tokens,
        identity.weight_format.as_str(),
        identity.engine_build_id,
        identity.rounding_class,
        if settings.park { "on" } else { "off" },
        if settings.restore { "on" } else { "off" },
        settings.worker.drain_ms,
    );
    let line = match &fault {
        Some(f) => format!(
            "{line}; TEST FAULT {TEST_FAULT_ENV}={} armed (chip check only)",
            f.spec()
        ),
        None => line,
    };
    Ok((
        KvPersistence {
            worker,
            store,
            recovery,
            identity,
            sharing: settings.namespace.clone(),
            chunk_tokens,
            ns,
            store_dir: root,
            park: AtomicBool::new(settings.park),
            restore: settings.restore,
            drain_ms: settings.worker.drain_ms,
            inner: Mutex::new(Inner::default()),
            seq: AtomicU64::new(0),
            test_fault: fault.map(|f| f.spec().to_string()),
        },
        line,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> OmegaBuild {
        OmegaBuild {
            omega_commit: "d0889a944b02066f84fec71af7b9bbb92f5c986f".into(),
            gpu_link: "pinned-source".into(),
            gpu_archive_sha256: "ab".repeat(32),
            gpu_digest: "unrecorded".into(),
            gpu_target: "aarch64-unknown-linux-gnu".into(),
            sc_commit: "cd".repeat(20),
            sc_dirty: false,
        }
    }

    fn config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            layers: 48,
            hidden: 2048,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            experts: 128,
            top_k: 8,
            moe_intermediate: 768,
            vocab: 151936,
            rms_eps: 1e-6,
            rope_theta: 1e7,
            norm_topk_prob: true,
            tie_word_embeddings: false,
            eos_token_ids: vec![151645],
        }
    }

    fn switches() -> OmegaKvSwitches {
        OmegaKvSwitches {
            attention_path: "fused",
            device_routing: true,
            prefill_chunk: 64,
            prefill_gemm: true,
            prefill_device_routing: true,
            whole_prompt: true,
            whole_prompt_device_host: true,
            decode_graph: false,
            lm_head_gemv: true,
            routing_mutant: false,
            trace_layers: false,
        }
    }

    struct In {
        model_id: String,
        sha: [u8; 32],
        kind: String,
        config: Qwen3MoeConfig,
        weights: WeightFormat,
        switches: OmegaKvSwitches,
        build: OmegaBuild,
        env: Vec<(String, String)>,
    }

    fn base() -> In {
        In {
            model_id: "qwen3-30b".into(),
            sha: [1; 32],
            kind: "index+shards".into(),
            config: config(),
            weights: WeightFormat::Fp8All,
            switches: switches(),
            build: build(),
            env: vec![("PATH".into(), "/bin".into())],
        }
    }

    fn ns(i: &In) -> Result<[u8; 32], String> {
        let id = OmegaKvIdentity::new(&IdentityInputs {
            model_id: &i.model_id,
            model_sha256: i.sha,
            model_digest_kind: &i.kind,
            config: &i.config,
            rope_scaling: false,
            weights: i.weights,
            switches: i.switches,
            build: &i.build,
            env: &i.env,
        })?;
        Ok(id
            .fields(256, "default")
            .namespace_id()
            .map_err(|e| format!("{e:?}"))?
            .0)
    }

    /// Every identity input changed alone gives another namespace (so another chunk key: a
    /// parked prefix of the old one can never be found).
    #[test]
    fn every_identity_input_moves_the_namespace() {
        type Change = (&'static str, fn(&mut In));
        let changes: &[Change] = &[
            ("model id", |i| i.model_id.push('x')),
            ("model digest", |i| i.sha[0] ^= 1),
            ("digest kind", |i| i.kind = "file".into()),
            ("layers", |i| i.config.layers += 1),
            ("hidden", |i| i.config.hidden += 128),
            ("q heads", |i| i.config.q_heads += 1),
            ("kv heads", |i| i.config.kv_heads += 1),
            ("head dim", |i| i.config.head_dim += 1),
            ("experts", |i| i.config.experts += 1),
            ("top k", |i| i.config.top_k += 1),
            ("moe intermediate", |i| i.config.moe_intermediate += 1),
            ("vocab", |i| i.config.vocab += 1),
            ("rms eps", |i| i.config.rms_eps *= 2.0),
            ("rope theta", |i| i.config.rope_theta *= 2.0),
            ("norm topk prob", |i| i.config.norm_topk_prob ^= true),
            ("tie embeddings", |i| i.config.tie_word_embeddings ^= true),
            ("weights bf16", |i| i.weights = WeightFormat::Bf16),
            ("weights fp8 experts", |i| {
                i.weights = WeightFormat::Fp8Experts
            }),
            ("attention path", |i| i.switches.attention_path = "split"),
            ("device routing", |i| i.switches.device_routing ^= true),
            ("prefill chunk", |i| i.switches.prefill_chunk += 1),
            ("prefill gemm", |i| i.switches.prefill_gemm ^= true),
            ("prefill device routing", |i| {
                i.switches.prefill_device_routing ^= true
            }),
            ("whole prompt", |i| i.switches.whole_prompt ^= true),
            ("whole prompt device host", |i| {
                i.switches.whole_prompt_device_host ^= true
            }),
            ("decode graph", |i| i.switches.decode_graph ^= true),
            ("lm head gemv", |i| i.switches.lm_head_gemv ^= true),
            ("trace layers", |i| i.switches.trace_layers ^= true),
            ("omega commit", |i| i.build.omega_commit = "e".repeat(40)),
            ("archive digest", |i| {
                i.build.gpu_archive_sha256 = "ef".repeat(32)
            }),
            ("gpu target", |i| {
                i.build.gpu_target = "x86_64-unknown-linux-gnu".into()
            }),
            ("engine commit", |i| i.build.sc_commit = "ab".repeat(20)),
            ("env cta budget", |i| {
                i.env.push(("AIEN_OMEGA_CTA_BUDGET".into(), "4".into()))
            }),
            ("env dgraph barrier", |i| {
                i.env.push(("AIEN_OMEGA_DGRAPH_BARRIER".into(), "1".into()))
            }),
            ("env dgraph prefetch", |i| {
                i.env
                    .push(("AIEN_OMEGA_DGRAPH_PREFETCH".into(), "1".into()))
            }),
            ("env fp8 gemv", |i| {
                i.env.push(("AIEN_OMEGA_FP8_GEMV".into(), "0".into()))
            }),
            ("env fp8 weight cache", |i| {
                i.env
                    .push(("AIEN_OMEGA_FP8_WEIGHT_CACHE".into(), "1".into()))
            }),
            ("env decode graph", |i| {
                i.env.push(("OWN30B_DECODE_GRAPH".into(), "1".into()))
            }),
            ("env decode graph stamps", |i| {
                i.env
                    .push(("OWN30B_DECODE_GRAPH_STAMPS".into(), "1".into()))
            }),
        ];
        let b = ns(&base()).unwrap();
        let mut seen = vec![b];
        for (what, change) in changes {
            let mut i = base();
            change(&mut i);
            let n = ns(&i).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(!seen.contains(&n), "{what} did not move the namespace");
            seen.push(n);
        }
        // Timing and build-input variables do not move it.
        let mut i = base();
        i.env.push(("AIEN_OMEGA_SPIN_US".into(), "50".into()));
        i.env.push(("AIEN_OMEGA_DIR".into(), "/x".into()));
        assert_eq!(ns(&i).unwrap(), b);
    }

    #[test]
    fn untrusted_builds_and_negative_controls_refuse() {
        type Change = (&'static str, fn(&mut In));
        let refusals: &[Change] = &[
            ("stub link", |i| i.build.gpu_link = "stub".into()),
            ("dev override", |i| i.build.gpu_link = "dev-override".into()),
            ("other commit", |i| {
                i.build.gpu_link = "pinned-source-other-commit".into()
            }),
            ("no archive sha", |i| i.build.gpu_archive_sha256.clear()),
            ("digest mismatch", |i| {
                i.build.gpu_digest = "mismatch".into()
            }),
            ("unknown omega commit", |i| {
                i.build.omega_commit = "unknown".into()
            }),
            ("unknown engine commit", |i| {
                i.build.sc_commit = "unknown".into()
            }),
            ("dirty build", |i| i.build.sc_dirty = true),
            ("routing mutant", |i| i.switches.routing_mutant = true),
            ("decode graph mutant", |i| {
                i.env
                    .push(("OWN30B_DECODE_GRAPH_MUTANT".into(), "skip-norm".into()))
            }),
            ("unknown omega switch", |i| {
                i.env.push(("AIEN_OMEGA_NEW_THING".into(), "1".into()))
            }),
            ("unknown own30b switch", |i| {
                i.env.push(("OWN30B_SOMETHING".into(), "1".into()))
            }),
        ];
        for (what, change) in refusals {
            let mut i = base();
            change(&mut i);
            assert!(ns(&i).is_err(), "{what} must refuse persistence");
        }
    }

    #[test]
    fn the_namespace_layout_is_the_pool_layout() {
        let i = base();
        let id = OmegaKvIdentity::new(&IdentityInputs {
            model_id: &i.model_id,
            model_sha256: i.sha,
            model_digest_kind: &i.kind,
            config: &i.config,
            rope_scaling: false,
            weights: i.weights,
            switches: i.switches,
            build: &i.build,
            env: &i.env,
        })
        .unwrap();
        let f = id.fields(256, "default");
        // [block][layer][K|V][token][head][dim] bf16: 48 x 2 x 16 x 4 x 128 x 2 bytes per block.
        assert_eq!(f.layout.block_stride, 48 * 2 * 16 * 4 * 128 * 2);
        assert_eq!(
            f.expected_payload_len(17).unwrap(),
            2 * f.layout.block_stride
        );
        assert!(HostFsConfig::from_fields(PathBuf::from("/nonexistent"), &f).is_ok());
    }
}
