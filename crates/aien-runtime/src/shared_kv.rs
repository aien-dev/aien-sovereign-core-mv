// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! One KV for runtime and backend (PREFILL-E2E-0 cut C2).
//!
//! Before this module the daemon built the spine on a pool-less KV manager
//! and the backend with no KV manager at all, so block tables in the runtime
//! were bookkeeping only and the real K/V lived in the backend's private
//! dense per-request cache. `build_shared_kv_runtime` creates ONE manager with
//! a physical tensor pool sized for the loaded model and hands the same `Arc`
//! to the spine (scheduler, swarm forks, reclaim) and to the backend (which
//! writes prefill/decode K/V into the pool blocks named by the spine's block
//! tables).
//!
//! Lives in aien-runtime (not aien-cli) because this crate owns the spine and
//! already depends on aien-inference-abi and aien-kv-cache, so the assembly
//! can be exercised by integration tests next to the other spine tests.

use std::sync::Arc;

use aien_inference_abi::{NativeTransformerBackend, TensorBackend, TransformerWeights};
use aien_kv_cache::{
    create_shared_kv_manager_with_pool, create_shared_kv_manager_with_pool_buffer, KvDType,
    KvPoolConfig, SharedKvManager,
};
use aien_scheduler::SchedulerConfig;

use crate::spine::AienRuntimeSpine;

/// Sizing for the shared runtime/backend KV manager.
#[derive(Debug, Clone, Copy)]
pub struct SharedKvSizing {
    /// Sequence arena capacity of the spine.
    pub arena_capacity: usize,
    /// Number of physical KV blocks in the pool (and in the block allocator).
    pub total_blocks: usize,
}

/// Builds the pooled KV manager for `weights.config`. The block size is the
/// model config's `block_size`, because the backend's paged writes index
/// blocks with `weights.config.block_size`
/// (aien-inference-abi/src/transformer_backend.rs, prefill paged path).
pub fn build_model_kv_manager(
    weights: &TransformerWeights,
    total_blocks: usize,
) -> Result<SharedKvManager, String> {
    let cfg = &weights.config;
    if cfg.block_size == 0 {
        return Err("shared KV: model config block_size is 0".to_string());
    }
    let pool_cfg = KvPoolConfig::for_model(
        total_blocks,
        cfg.block_size,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        SHARED_KV_DTYPE,
    );
    create_shared_kv_manager_with_pool(total_blocks, cfg.block_size, pool_cfg)
}

/// `build_model_kv_manager` with the pool memory `backend` offers when it has some
/// (`TensorBackend::alloc_kv_pool`: omega's resident pool, which its paged attention reads in
/// place instead of staging the KV slices on every call); otherwise the host-mapped pool as before.
pub fn build_model_kv_manager_for(
    weights: &TransformerWeights,
    total_blocks: usize,
    backend: &dyn TensorBackend,
) -> Result<SharedKvManager, String> {
    let cfg = &weights.config;
    if cfg.block_size == 0 {
        return Err("shared KV: model config block_size is 0".to_string());
    }
    let pool_cfg = KvPoolConfig::for_model(
        total_blocks,
        cfg.block_size,
        cfg.num_layers,
        cfg.num_kv_heads,
        cfg.head_dim,
        SHARED_KV_DTYPE,
    );
    let bytes = aien_kv_cache::KvLayout::from_config(&pool_cfg)?.total_bytes;
    match backend.alloc_kv_pool(bytes) {
        Some(buffer) => create_shared_kv_manager_with_pool_buffer(
            total_blocks,
            cfg.block_size,
            pool_cfg,
            buffer,
        ),
        None => create_shared_kv_manager_with_pool(total_blocks, cfg.block_size, pool_cfg),
    }
}

/// Builds the runtime spine and the native backend over ONE pooled KV
/// manager. The returned backend's `kv_manager` is the same `Arc` as the
/// spine's `kv_manager` (`Arc::ptr_eq`).
pub fn build_shared_kv_runtime(
    weights: TransformerWeights,
    tensor_backend: Arc<dyn TensorBackend>,
    scheduler_config: SchedulerConfig,
    sizing: SharedKvSizing,
) -> Result<(AienRuntimeSpine, NativeTransformerBackend), String> {
    let kv_manager =
        build_model_kv_manager_for(&weights, sizing.total_blocks, tensor_backend.as_ref())?;
    let spine = AienRuntimeSpine::new(sizing.arena_capacity, scheduler_config, kv_manager.clone());
    let backend =
        NativeTransformerBackend::with_shared_kv_and_backend(weights, tensor_backend, kv_manager);
    Ok((spine, backend))
}

/// Element type of the shared pool. `build_model_kv_manager` lays the pool out
/// in this dtype, and [`KvPoolPlan`] prices it with the same value.
///
/// Bf16 since sc#384 (2026-10-12): the Omega GB10 backend runs its paged attention
/// kernel straight on a bf16 pool, while any other dtype makes it gather the
/// whole context on the host every call (6 ms of a 92 ms decode step at 300
/// tokens, growing with context). K/V are rounded to bf16 on write; the
/// reference CPU path reads bf16 pools too, and the pool costs half the bytes.
pub const SHARED_KV_DTYPE: KvDType = KvDType::Bf16;

/// The daemon's KV pool, sized from the loaded model (issue #239).
///
/// Before #239 the daemon asked for a fixed 8192 blocks for every model. With
/// the 16-token blocks of `model_dir.rs` that is 131072 tokens, which happens to
/// be Llama-3.2-1B's `max_position_embeddings`, but it priced every other model
/// at 131072 tokens too (SmolLM2-1.7B: 48 GiB of Fp32 KV for a model whose
/// context is 8192 tokens). The plan now takes the declared context budget from
/// the model itself (`ModelConfig::max_sequence_length`, i.e. config.json
/// `max_position_embeddings`) and derives
/// `bytes = blocks * block_size * num_layers * num_kv_heads * head_dim * 2 (K and V) * dtype bytes`,
/// with `blocks = ceil(context / block_size)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolPlan {
    /// Declared context budget in tokens (where it came from: `context_source`).
    pub context_tokens: usize,
    /// Where `context_tokens` was taken from, for the startup log.
    pub context_source: &'static str,
    pub block_size: usize,
    pub total_blocks: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype_bytes: usize,
    /// K plus V bytes for one token across all layers.
    pub bytes_per_token: usize,
    /// Bytes of the whole pool (`total_blocks * block_size * bytes_per_token`).
    pub total_bytes: usize,
}

const GIB: f64 = (1u64 << 30) as f64;

/// Bytes as GiB with two decimals, for logs and refusals.
pub fn fmt_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB)
}

impl KvPoolPlan {
    /// Plan for `cfg` at the model's own declared context (`max_sequence_length`).
    pub fn for_model(cfg: &aien_inference_abi::ModelConfig) -> Result<Self, String> {
        Self::for_context(
            cfg,
            cfg.max_sequence_length,
            "model max_sequence_length (config.json max_position_embeddings)",
        )
    }

    /// Plan for `cfg` at the model's own context, or at an operator cap that can only
    /// lower it ([`KV_CONTEXT_CAP_ENV`]). A cap above the model's declared context is
    /// refused, never used to enlarge the pool.
    pub fn for_model_capped(
        cfg: &aien_inference_abi::ModelConfig,
        cap: Option<usize>,
    ) -> Result<Self, String> {
        match cap {
            None => Self::for_model(cfg),
            Some(c) if c > cfg.max_sequence_length => Err(format!(
                "{KV_CONTEXT_CAP_ENV}={c} exceeds the model's declared context ({} tokens); the cap can only lower it",
                cfg.max_sequence_length
            )),
            Some(c) => Self::for_context(cfg, c, "operator cap (AIEN_KV_CONTEXT_TOKENS)"),
        }
    }

    /// Plan for `cfg` at an explicit context budget. Every product is checked, so a
    /// config that would overflow `usize` is refused instead of wrapping.
    pub fn for_context(
        cfg: &aien_inference_abi::ModelConfig,
        context_tokens: usize,
        context_source: &'static str,
    ) -> Result<Self, String> {
        let named = [
            ("context_tokens", context_tokens),
            ("block_size", cfg.block_size),
            ("num_layers", cfg.num_layers),
            ("num_kv_heads", cfg.num_kv_heads),
            ("head_dim", cfg.head_dim),
        ];
        if let Some((name, _)) = named.iter().find(|(_, v)| *v == 0) {
            return Err(format!("KV pool plan: {name} is 0"));
        }
        // Fp32 is 4 bytes; the f32 accessor is exact for every whole-byte dtype.
        let dtype_bytes = SHARED_KV_DTYPE.bytes_per_element() as usize;
        let overflow = || "KV pool plan: size overflows usize".to_string();
        let bytes_per_token = 2usize
            .checked_mul(cfg.num_layers)
            .and_then(|v| v.checked_mul(cfg.num_kv_heads))
            .and_then(|v| v.checked_mul(cfg.head_dim))
            .and_then(|v| v.checked_mul(dtype_bytes))
            .ok_or_else(overflow)?;
        let total_blocks = context_tokens.div_ceil(cfg.block_size);
        let total_bytes = total_blocks
            .checked_mul(cfg.block_size)
            .and_then(|v| v.checked_mul(bytes_per_token))
            .ok_or_else(overflow)?;
        Ok(Self {
            context_tokens,
            context_source,
            block_size: cfg.block_size,
            total_blocks,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            dtype_bytes,
            bytes_per_token,
            total_bytes,
        })
    }

    /// The pool config `build_model_kv_manager` builds for this plan.
    pub fn pool_config(&self) -> KvPoolConfig {
        KvPoolConfig::for_model(
            self.total_blocks,
            self.block_size,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            SHARED_KV_DTYPE,
        )
    }

    /// One startup log line: size in bytes and GiB plus every derived parameter.
    pub fn log_line(&self) -> String {
        format!(
            "KV pool: {} bytes ({}) = {} blocks x {} tokens x {} layers x {} kv heads x {} head_dim x 2 (K,V) x {} bytes; context {} tokens from {}",
            self.total_bytes,
            fmt_gib(self.total_bytes as u64),
            self.total_blocks,
            self.block_size,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.dtype_bytes,
            self.context_tokens,
            self.context_source,
        )
    }
}

/// `MemAvailable` from the text of `/proc/meminfo`, in bytes (the file reports kB).
pub fn parse_mem_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        let mut parts = rest.split_whitespace();
        let value: u64 = parts.next()?.parse().ok()?;
        match parts.next() {
            Some("kB") => value.checked_mul(1024),
            _ => None,
        }
    })
}

/// `MemAvailable` of this host in bytes, or `None` where `/proc/meminfo` is absent
/// or has no parsable `MemAvailable` line.
pub fn read_mem_available() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .as_deref()
        .and_then(parse_mem_available)
}

/// Why `MemAvailable` could not be established, as a plain message.
pub fn parse_mem_available_checked(meminfo: &str) -> Result<u64, String> {
    parse_mem_available(meminfo)
        .ok_or_else(|| "/proc/meminfo has no parsable `MemAvailable: <n> kB` line".to_string())
}

/// `MemAvailable` of this host in bytes, or the reason it cannot be established
/// (`/proc/meminfo` unreadable, or no parsable `MemAvailable` line).
pub fn read_mem_available_checked() -> Result<u64, String> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| format!("/proc/meminfo cannot be read: {e}"))?;
    parse_mem_available_checked(&text)
}

/// Operator override (documented): set to `1` to let the daemon allocate the KV pool
/// although `MemAvailable` cannot be established. Without it the daemon refuses to
/// start (fail closed). Every use is logged loudly on stderr.
pub const ALLOW_UNCHECKED_MEMORY_ENV: &str = "AIEN_ALLOW_UNCHECKED_KV_MEMORY";

/// True only when [`ALLOW_UNCHECKED_MEMORY_ENV`] is exactly `1`.
pub fn allow_unchecked_memory_from_env() -> bool {
    std::env::var(ALLOW_UNCHECKED_MEMORY_ENV).is_ok_and(|v| v == "1")
}

/// Operator cap (documented): the daemon's KV context budget in tokens, a positive
/// decimal integer no larger than the model's declared context. Unset: the model's
/// own context (unchanged behaviour). A sequence that outgrows the capped pool is
/// refused with `KV_POOL_EXHAUSTED_PREFIX` before anything is mutated.
pub const KV_CONTEXT_CAP_ENV: &str = "AIEN_KV_CONTEXT_TOKENS";

/// Parses a [`KV_CONTEXT_CAP_ENV`] value: digits only, greater than zero.
pub fn parse_kv_context_cap(value: &str) -> Result<usize, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "{KV_CONTEXT_CAP_ENV}={value:?} is not a positive decimal token count"
        ));
    }
    match value.parse::<usize>() {
        Ok(0) => Err(format!(
            "{KV_CONTEXT_CAP_ENV}=0: the cap must be at least 1 token"
        )),
        Ok(n) => Ok(n),
        Err(e) => Err(format!("{KV_CONTEXT_CAP_ENV}={value:?}: {e}")),
    }
}

/// The operator cap from the environment: `Ok(None)` when unset, a refusal when set
/// but not a valid count (including non-UTF-8).
pub fn kv_context_cap_from_env() -> Result<Option<usize>, String> {
    match std::env::var(KV_CONTEXT_CAP_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{KV_CONTEXT_CAP_ENV}: {e}")),
        Ok(v) => parse_kv_context_cap(&v).map(Some),
    }
}

/// Refuses the pool when the host reports less available memory than it needs.
/// `available` is `Err(reason)` when memory could not be established: that refuses
/// too (fail closed, #254) unless `allow_unchecked` is set, in which case the check
/// is bypassed with a loud stderr line.
pub fn check_kv_memory(
    requested: u64,
    available: Result<u64, String>,
    allow_unchecked: bool,
) -> Result<(), String> {
    match available {
        Ok(avail) if requested > avail => Err(format!(
            "KV pool needs {requested} bytes ({}) but MemAvailable is {avail} bytes ({}); refusing to allocate",
            fmt_gib(requested),
            fmt_gib(avail),
        )),
        Ok(_) => Ok(()),
        Err(reason) if allow_unchecked => {
            eprintln!(
                "WARNING: KV POOL MEMORY CHECK BYPASSED by {ALLOW_UNCHECKED_MEMORY_ENV}=1: {reason}; allocating {requested} bytes ({}) WITHOUT knowing MemAvailable",
                fmt_gib(requested)
            );
            Ok(())
        }
        Err(reason) => Err(format!(
            "KV pool memory cannot be checked: {reason}; refusing to allocate {requested} bytes ({}). Set {ALLOW_UNCHECKED_MEMORY_ENV}=1 to override (not recommended)",
            fmt_gib(requested)
        )),
    }
}

/// Daemon planning step (#239): plans the pool for `cfg`, logs the plan, and
/// checks `MemAvailable` (read through `mem_available`) before anything is
/// allocated. Allocates nothing itself. Fails closed when memory cannot be
/// established unless `allow_unchecked` (#254).
pub fn plan_checked_model_kv(
    cfg: &aien_inference_abi::ModelConfig,
    context_cap: Option<usize>,
    mem_available: &dyn Fn() -> Result<u64, String>,
    allow_unchecked: bool,
) -> Result<KvPoolPlan, String> {
    let plan = KvPoolPlan::for_model_capped(cfg, context_cap)?;
    println!("  {}", plan.log_line());
    let available = mem_available();
    if let Ok(avail) = &available {
        println!(
            "  KV pool memory check: needs {} bytes ({}), MemAvailable {} bytes ({})",
            plan.total_bytes,
            fmt_gib(plan.total_bytes as u64),
            avail,
            fmt_gib(*avail)
        );
    }
    check_kv_memory(plan.total_bytes as u64, available, allow_unchecked)?;
    Ok(plan)
}

/// Daemon path (#239): [`plan_checked_model_kv`], then builds the spine and
/// backend over the one pooled manager sized by the plan.
pub fn build_shared_kv_runtime_for_model(
    weights: TransformerWeights,
    tensor_backend: Arc<dyn TensorBackend>,
    scheduler_config: SchedulerConfig,
    arena_capacity: usize,
    context_cap: Option<usize>,
    mem_available: &dyn Fn() -> Result<u64, String>,
    allow_unchecked: bool,
) -> Result<(AienRuntimeSpine, NativeTransformerBackend, KvPoolPlan), String> {
    let plan = plan_checked_model_kv(&weights.config, context_cap, mem_available, allow_unchecked)?;
    let (spine, backend) = build_shared_kv_runtime(
        weights,
        tensor_backend,
        scheduler_config,
        SharedKvSizing {
            arena_capacity,
            total_blocks: plan.total_blocks,
        },
    )?;
    Ok((spine, backend, plan))
}

// ---------------------------------------------------------------------------
// Persistent KV settings and store construction (ADR 0003, cuts C6 and C7;
// docs/persistent-kv/METRICS_AND_SETTINGS.md)
// ---------------------------------------------------------------------------

/// Root of the host object store. Unset: persistence off, no behaviour change.
pub const KV_STORE_DIR_ENV: &str = "AIEN_KV_STORE_DIR";
pub const KV_STORE_NAMESPACE_ENV: &str = "AIEN_KV_STORE_NAMESPACE";
pub const KV_CHUNK_BLOCKS_ENV: &str = "AIEN_KV_CHUNK_BLOCKS";
pub const KV_RESTORE_MAX_BLOCKS_ENV: &str = "AIEN_KV_RESTORE_MAX_BLOCKS";
pub const KV_PARK_ENV: &str = "AIEN_KV_PARK";
pub const KV_RESTORE_ENV: &str = "AIEN_KV_RESTORE";
pub const KV_RESTORE_RETRY_WINDOW_MS_ENV: &str = "AIEN_KV_RESTORE_RETRY_WINDOW_MS";
/// Applied to the host store (`host_fs.rs` names); checked here so a bad value
/// stops the start with its name.
pub const KV_STORE_QUOTA_BYTES_ENV: &str = "AIEN_KV_STORE_QUOTA_BYTES";
pub const KV_STORE_MAX_OBJECTS_ENV: &str = "AIEN_KV_STORE_MAX_OBJECTS";
pub const KV_STORE_RECOVERY_SLICE_MS_ENV: &str = "AIEN_KV_STORE_RECOVERY_SLICE_MS";

/// Default of `AIEN_KV_RESTORE_RETRY_WINDOW_MS`.
pub const DEFAULT_RESTORE_RETRY_WINDOW_MS: u64 = 60_000;

/// Persistent KV settings read from the environment (or any lookup in tests).
/// Every value is checked when it is read, whether or not persistence is on,
/// so a bad value stops the start with its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPersistSettings {
    pub store_dir: Option<std::path::PathBuf>,
    /// Sharing namespace (ADR 1.2 field 13), at most 64 bytes. Default `default`.
    pub namespace: String,
    /// Chunk size in pool blocks. Default 16.
    pub chunk_blocks: u32,
    /// Restore holder blocks in flight at once. `None`: a quarter of the pool.
    pub restore_max_blocks: Option<usize>,
    /// Default on when the store dir is set.
    pub park: bool,
    pub restore: bool,
    /// `AIEN_KV_RESTORE_RETRY_WINDOW_MS`, default 60000.
    pub restore_retry_window_ms: u64,
    /// `AIEN_KV_STORE_QUOTA_BYTES`; `None`: the host store default.
    pub quota_bytes: Option<u64>,
    /// `AIEN_KV_STORE_MAX_OBJECTS`; `None`: the host store default.
    pub max_objects: Option<u64>,
    /// `AIEN_KV_STORE_RECOVERY_SLICE_MS`; `None`: the host store default.
    pub recovery_slice_ms: Option<u64>,
    /// The storage worker's settings (`WorkerConfig::from_lookup`: the
    /// `AIEN_KV_STORE_MAX_*` bounds and `AIEN_KV_STORE_DRAIN_MS`).
    pub worker: aien_kv_store::worker::WorkerConfig,
    /// The store key from `AIEN_KV_STORE_KEY` (ADR 0003 section 20), read only
    /// when the store dir is set (`None` otherwise). A refusal (missing or
    /// malformed) does not stop the start: persistence is refused (fail
    /// closed) and the status says why. The key is redacted in `Debug`.
    pub store_key:
        Option<Result<Arc<aien_kv_store::seal::SealKey>, aien_kv_store::seal::KeyRefusal>>,
}

fn parse_switch(name: &str, v: Option<String>) -> Result<bool, String> {
    match v.as_deref() {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(format!("{name}={other:?}: must be 1 or 0")),
    }
}

fn parse_positive(name: &str, v: &str) -> Result<u64, String> {
    if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("{name}={v:?} is not a positive decimal integer"));
    }
    match v.parse::<u64>() {
        Ok(0) => Err(format!("{name}=0: must be greater than zero")),
        Ok(n) => Ok(n),
        Err(e) => Err(format!("{name}={v:?}: {e}")),
    }
}

fn parse_optional_positive(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
) -> Result<Option<u64>, String> {
    get(name).map(|v| parse_positive(name, &v)).transpose()
}

impl KvPersistSettings {
    /// Every value is checked; a bad value is refused, never defaulted.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let store_dir = get(KV_STORE_DIR_ENV)
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from);
        let namespace = get(KV_STORE_NAMESPACE_ENV).unwrap_or_else(|| "default".to_string());
        if namespace.is_empty()
            || namespace.len() > aien_kv_store::identity::MAX_SHARING_NAMESPACE_BYTES
        {
            return Err(format!(
                "{KV_STORE_NAMESPACE_ENV}: must be 1 to {} bytes",
                aien_kv_store::identity::MAX_SHARING_NAMESPACE_BYTES
            ));
        }
        let chunk_blocks = match get(KV_CHUNK_BLOCKS_ENV) {
            None => 16,
            Some(v) => u32::try_from(parse_positive(KV_CHUNK_BLOCKS_ENV, &v)?)
                .map_err(|_| format!("{KV_CHUNK_BLOCKS_ENV}={v:?} is too large"))?,
        };
        let restore_max_blocks = match get(KV_RESTORE_MAX_BLOCKS_ENV) {
            None => None,
            Some(v) => Some(
                usize::try_from(parse_positive(KV_RESTORE_MAX_BLOCKS_ENV, &v)?)
                    .map_err(|_| format!("{KV_RESTORE_MAX_BLOCKS_ENV}={v:?} is too large"))?,
            ),
        };
        let restore_retry_window_ms =
            parse_optional_positive(&get, KV_RESTORE_RETRY_WINDOW_MS_ENV)?
                .unwrap_or(DEFAULT_RESTORE_RETRY_WINDOW_MS);
        let worker = aien_kv_store::worker::WorkerConfig::from_lookup(&get)
            .map_err(|e| format!("persistent KV worker setting {e}"))?;
        let store_key = store_dir
            .is_some()
            .then(|| aien_kv_store::seal::SealKey::from_lookup(&get).map(Arc::new));
        Ok(Self {
            store_dir,
            namespace,
            chunk_blocks,
            restore_max_blocks,
            park: parse_switch(KV_PARK_ENV, get(KV_PARK_ENV))?,
            restore: parse_switch(KV_RESTORE_ENV, get(KV_RESTORE_ENV))?,
            restore_retry_window_ms,
            quota_bytes: parse_optional_positive(&get, KV_STORE_QUOTA_BYTES_ENV)?,
            max_objects: parse_optional_positive(&get, KV_STORE_MAX_OBJECTS_ENV)?,
            recovery_slice_ms: parse_optional_positive(&get, KV_STORE_RECOVERY_SLICE_MS_ENV)?,
            worker,
            store_key,
        })
    }

    /// From the process environment. `AIEN_KV_STORE_KEY` is removed from the
    /// environment once read (whatever its value), so no child process
    /// inherits it; the key lives on only in the settings (ADR 0003 section 20).
    pub fn from_env() -> Result<Self, String> {
        let got = Self::from_lookup(|n| std::env::var(n).ok());
        std::env::remove_var(aien_kv_store::seal::KEY_ENV);
        got
    }

    /// Persistence is on only when a store directory is configured.
    pub fn enabled(&self) -> bool {
        self.store_dir.is_some()
    }

    /// The residency settings for a pool of `total_blocks`, capped at the
    /// operator context `context_cap` (`AIEN_KV_CONTEXT_TOKENS`).
    pub fn residency_config(
        &self,
        total_blocks: usize,
        context_cap: Option<usize>,
    ) -> crate::kv_residency::KvResidencyConfig {
        crate::kv_residency::KvResidencyConfig {
            chunk_blocks: self.chunk_blocks,
            context_cap,
            restore_max_blocks: self
                .restore_max_blocks
                .unwrap_or_else(|| (total_blocks / 4).max(1)),
            park: self.park,
            restore: self.restore,
            restore_retry_window_ms: self.restore_retry_window_ms,
            drain_ms: self.worker.drain_ms,
        }
    }
}

/// Opens the host object store under `settings.store_dir` for `fields` (whose
/// `chunk_tokens` must be `chunk_blocks * block_size`), puts the storage worker
/// in front of it, and builds the residency layer over `kv`. `Ok(None)` when
/// persistence is off. The worker's drive runs the resumable recovery in
/// `AIEN_KV_STORE_RECOVERY_SLICE_MS` slices on the worker thread until it is
/// done; until then the store is `Recovering` and every lookup misses (ADR
/// 4.1), so the step loop is never blocked and a cold start only costs plain
/// prefills.
pub fn open_host_kv_residency(
    settings: &KvPersistSettings,
    fields: &aien_kv_store::NamespaceFields<'_>,
    kv: SharedKvManager,
    context_cap: Option<usize>,
) -> Result<Option<crate::kv_residency::KvResidency>, String> {
    use aien_kv_store::host_fs::{HostFsConfig, HostFsStore};
    use aien_kv_store::worker::StorageWorker;
    use aien_kv_store::KvStore;

    let Some(root) = settings.store_dir.clone() else {
        return Ok(None);
    };
    let seal_key = match &settings.store_key {
        Some(Ok(k)) => k.clone(),
        Some(Err(r)) => return Err(format!("persistent KV refused: {r}")),
        None => return Err("persistent KV: the store key was not read".to_string()),
    };
    let identity = crate::kv_residency::KvIdentity::new(fields)
        .map_err(|e| format!("persistent KV identity: {e}"))?;
    let mut cfg = HostFsConfig::from_fields(root.clone(), fields)
        .map_err(|e| format!("persistent KV store config: {e:?}"))?;
    if let Some(v) = settings.quota_bytes {
        cfg.quota_bytes = v;
    }
    if let Some(v) = settings.max_objects {
        cfg.max_objects = v;
    }
    if let Some(v) = settings.recovery_slice_ms {
        cfg.recovery_slice_ms = v;
    }
    cfg.seal_key = Some(seal_key);
    let slice_ms = cfg.recovery_slice_ms;
    let store = Arc::new(HostFsStore::open(cfg).map_err(|e| {
        format!(
            "persistent KV store open at {} ({KV_STORE_DIR_ENV}): {e:?}",
            root.display()
        )
    })?);
    let drive = {
        let store = store.clone();
        let mut recovered = false;
        move || {
            if !recovered {
                let p = store.recover_step(slice_ms);
                recovered = p.done || p.failed;
            }
        }
    };
    let worker = Arc::new(
        StorageWorker::spawn(store, settings.worker.clone(), drive)
            .map_err(|e| format!("persistent KV worker: {e:?}"))?,
    );
    let total_blocks = kv.read().total_block_count();
    let residency = crate::kv_residency::KvResidency::new(
        settings.residency_config(total_blocks, context_cap),
        identity,
        worker,
        kv,
    )?;
    Ok(Some(residency))
}

/// What the daemon knows about the loaded model that the namespace binds
/// (ADR 1.2): the resolved config, the backend that computes the K/V, the
/// weights digest, and the engine identity (ADR 1.2.2, sc#504).
/// `engine` has no default: the host path passes
/// `compute_config::HostEngineIdentity::reference(arith_id).identity()`, an
/// engine with its own build, weight format or run switches passes its own.
#[derive(Debug, Clone, Copy)]
pub struct ModelKvBinding<'a> {
    pub model: &'a aien_abi_core::ModelConfig,
    /// `TensorBackend::name()` of the backend that writes the K/V.
    pub arith_id: &'a str,
    pub model_sha256: [u8; 32],
    pub model_digest_kind: &'a str,
    /// Weight format as loaded, engine build, rounding class and engine config
    /// digest of the engine that writes the K/V.
    pub engine: aien_kv_store::EngineIdentity<'a>,
}

/// Parses a 64-hex-digit sha256 (the daemon's `ModelIdentity::model_sha256`).
pub fn parse_sha256_hex(hex: &str) -> Result<[u8; 32], String> {
    let hex = hex.trim();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("model digest {hex:?} is not 64 hex digits"));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|e| format!("model digest: {e}"))?;
    }
    Ok(out)
}

fn kv_dtype_of(d: KvDType) -> aien_kv_store::KvDtype {
    match d {
        KvDType::Fp4 => aien_kv_store::KvDtype::Fp4,
        KvDType::Fp8 => aien_kv_store::KvDtype::Fp8,
        KvDType::Bf16 => aien_kv_store::KvDtype::Bf16,
        KvDType::Fp16 => aien_kv_store::KvDtype::Fp16,
        KvDType::Fp32 => aien_kv_store::KvDtype::Fp32,
    }
}

/// Daemon wiring (cut C7): with `AIEN_KV_STORE_DIR` set, binds the namespace
/// to `model` and the spine's pool, opens the host store, starts the storage
/// worker (recovery runs on it in slices), and installs the residency layer
/// and its finish hook in the spine. Returns the startup log line, or
/// `Ok(None)` with persistence off, in which case nothing at all is created or
/// changed.
pub fn enable_persistent_kv(
    spine: &mut AienRuntimeSpine,
    settings: &KvPersistSettings,
    model: &ModelKvBinding<'_>,
    context_cap: Option<usize>,
) -> Result<Option<String>, String> {
    use crate::compute_config::{ModelBinding, StoreInputs};
    if !settings.enabled() {
        return Ok(None);
    }
    // Fail closed without a valid store key (ADR 0003 section 20): no store is
    // opened and nothing is written; the daemon serves with plain prefill.
    if let Some(Err(refusal)) = &settings.store_key {
        aien_kv_store::seal::note_key_refusal();
        spine.refuse_kv_persistence(refusal.reason());
        return Ok(Some(format!(
            "persistent KV: REFUSED ({refusal}); running without persistence: no park, no restore, every prompt is prefilled"
        )));
    }
    let kv = spine.kv_manager.clone();
    let (layout, dtype, block_size) = {
        let guard = kv.read();
        let pool = guard
            .pool()
            .ok_or("persistent KV: the runtime KV manager has no tensor pool")?;
        (
            crate::kv_residency::layout_strides(pool.layout()),
            kv_dtype_of(pool.config().dtype),
            guard.block_size(),
        )
    };
    let chunk_tokens = (settings.chunk_blocks as usize)
        .checked_mul(block_size)
        .and_then(|t| u32::try_from(t).ok())
        .ok_or_else(|| format!("{KV_CHUNK_BLOCKS_ENV}: chunk size overflows"))?;
    let binding = ModelBinding::from_model(model.model, model.arith_id)
        .map_err(|e| format!("persistent KV model binding: {e}"))?;
    let fields = binding
        .namespace_fields(
            &StoreInputs {
                model_sha256: model.model_sha256,
                model_digest_kind: model.model_digest_kind,
                dtype,
                layout,
                backend_format: aien_kv_store::BackendFormat::HostPagedPoolV1,
                chunk_tokens,
                sharing_namespace: &settings.namespace,
            },
            &model.engine,
        )
        .map_err(|e| format!("persistent KV namespace: {e}"))?;
    let Some(residency) = open_host_kv_residency(settings, &fields, kv, context_cap)? else {
        return Ok(None);
    };
    let line = format!(
        "persistent KV: store {} namespace {:?} chunk {} tokens, weights {}, engine build {:?}, rounding {:?}, park {}, restore {}, retry window {} ms, drain {} ms (recovering in the background)",
        settings
            .store_dir
            .as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        settings.namespace,
        chunk_tokens,
        model.engine.weight_format.as_str(),
        model.engine.engine_build_id,
        model.engine.rounding_class,
        if settings.park { "on" } else { "off" },
        if settings.restore { "on" } else { "off" },
        settings.restore_retry_window_ms,
        settings.worker.drain_ms,
    );
    spine.enable_kv_residency(residency);
    Ok(Some(line))
}

/// [`enable_persistent_kv`] with the settings read from the environment.
pub fn enable_persistent_kv_from_env(
    spine: &mut AienRuntimeSpine,
    model: &ModelKvBinding<'_>,
    context_cap: Option<usize>,
) -> Result<Option<String>, String> {
    let settings = KvPersistSettings::from_env()?;
    enable_persistent_kv(spine, &settings, model, context_cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_inference_abi::model_config_from_hf_json;

    /// Blocks the daemon asked for before #239 (aien-cli/src/commands.rs,
    /// `SharedKvSizing { total_blocks: 8192 }`, sovereign-core 328a7e9).
    const OLD_FIXED_DAEMON_BLOCKS: usize = 8192;

    /// unsloth/Llama-3.2-1B-Instruct config.json, snapshot 5a8abab (the KV-relevant
    /// fields as on disk; same text as the model_dir.rs fixture).
    const LLAMA32_1B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 128000, "eos_token_id": 128009, "head_dim": 64, "hidden_act": "silu",
  "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 8192,
  "max_position_embeddings": 131072, "mlp_bias": false, "model_type": "llama",
  "num_attention_heads": 32, "num_hidden_layers": 16, "num_key_value_heads": 8,
  "pad_token_id": 128004, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
  "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192, "rope_type": "llama3"},
  "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "vocab_size": 128256
}"#;

    /// HuggingFaceTB/SmolLM2-1.7B-Instruct config.json, revision 31b70e2e (the
    /// fields this engine reads, as on disk; transformers.js block dropped).
    const SMOLLM2_17B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 2048,
  "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 8192,
  "mlp_bias": false, "model_type": "llama", "num_attention_heads": 32,
  "num_hidden_layers": 24, "num_key_value_heads": 32, "pad_token_id": 2,
  "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 130000,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "use_cache": true,
  "vocab_size": 49152
}"#;

    fn cfg(id: &str, json: &str) -> aien_inference_abi::ModelConfig {
        model_config_from_hf_json(id, json, None).expect("config.json parses")
    }

    fn old_fixed_bytes(c: &aien_inference_abi::ModelConfig) -> usize {
        KvPoolConfig::for_model(
            OLD_FIXED_DAEMON_BLOCKS,
            c.block_size,
            c.num_layers,
            c.num_kv_heads,
            c.head_dim,
            SHARED_KV_DTYPE,
        )
        .total_bytes()
    }

    #[test]
    fn llama32_1b_plan_is_byte_identical_to_the_old_fixed_pool() {
        let c = cfg("unsloth/Llama-3.2-1B-Instruct", LLAMA32_1B_CONFIG);
        let plan = KvPoolPlan::for_model(&c).unwrap();
        assert_eq!(plan.context_tokens, 131_072);
        assert_eq!(plan.block_size, 16);
        assert_eq!(
            (plan.num_layers, plan.num_kv_heads, plan.head_dim),
            (16, 8, 64)
        );
        assert_eq!(plan.total_blocks, OLD_FIXED_DAEMON_BLOCKS);
        // 2 * 16 * 8 * 64 * 2 (bf16 pool, SHARED_KV_DTYPE)
        assert_eq!(plan.bytes_per_token, 32_768);
        assert_eq!(plan.total_bytes, 4_294_967_296); // 4.00 GiB
        assert_eq!(plan.total_bytes, old_fixed_bytes(&c));
        assert_eq!(plan.total_bytes, plan.pool_config().total_bytes());
    }

    #[test]
    fn smollm2_plan_is_sized_from_its_own_context() {
        let c = cfg("HuggingFaceTB/SmolLM2-1.7B-Instruct", SMOLLM2_17B_CONFIG);
        let plan = KvPoolPlan::for_model(&c).unwrap();
        assert_eq!(plan.context_tokens, 8192);
        assert_eq!(
            (plan.num_layers, plan.num_kv_heads, plan.head_dim),
            (24, 32, 64)
        );
        assert_eq!(plan.total_blocks, 512);
        // 2 * 24 * 32 * 64 * 2 (bf16)
        assert_eq!(plan.bytes_per_token, 196_608);
        assert_eq!(plan.total_bytes, 1_610_612_736); // 1.50 GiB
        assert_eq!(plan.total_bytes, plan.pool_config().total_bytes());
        // The old fixed pool was 16x larger: 24.00 GiB (25.8 GB) at the same bf16 element.
        assert_eq!(old_fixed_bytes(&c), 25_769_803_776);
        assert_eq!(old_fixed_bytes(&c), 16 * plan.total_bytes);
    }

    #[test]
    fn plan_rounds_a_partial_block_up_and_refuses_zero_or_overflow() {
        let mut c = cfg("unsloth/Llama-3.2-1B-Instruct", LLAMA32_1B_CONFIG);
        let plan = KvPoolPlan::for_context(&c, 17, "test").unwrap();
        assert_eq!(plan.total_blocks, 2);
        assert_eq!(plan.total_bytes, 2 * 16 * 32_768);
        assert!(KvPoolPlan::for_context(&c, 0, "test")
            .unwrap_err()
            .contains("context_tokens is 0"));
        c.num_layers = usize::MAX / 2;
        assert!(KvPoolPlan::for_model(&c).unwrap_err().contains("overflow"));
    }

    #[test]
    fn log_line_names_bytes_gib_and_every_parameter() {
        let c = cfg("unsloth/Llama-3.2-1B-Instruct", LLAMA32_1B_CONFIG);
        let line = KvPoolPlan::for_model(&c).unwrap().log_line();
        for want in [
            "4294967296 bytes",
            "4.00 GiB",
            "8192 blocks",
            "16 tokens",
            "16 layers",
            "8 kv heads",
            "64 head_dim",
            "2 bytes",
            "context 131072 tokens",
            "max_position_embeddings",
        ] {
            assert!(line.contains(want), "{want:?} missing from {line:?}");
        }
    }

    #[test]
    fn mem_available_parses_kb_and_rejects_garbage() {
        let meminfo =
            "MemTotal:       130000000 kB\nMemFree:         1000 kB\nMemAvailable:   62000000 kB\n";
        assert_eq!(parse_mem_available(meminfo), Some(62_000_000 * 1024));
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: x kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 5\n"), None);
    }

    #[test]
    fn memory_check_refuses_with_requested_and_available_bytes() {
        let err = check_kv_memory(3_221_225_472, Ok(1_073_741_824), false).unwrap_err();
        assert!(err.contains("3221225472 bytes (3.00 GiB)"), "{err}");
        assert!(err.contains("1073741824 bytes (1.00 GiB)"), "{err}");
        assert!(err.contains("refusing"), "{err}");
        assert!(check_kv_memory(10, Ok(10), false).is_ok());
    }

    fn tiny_config() -> aien_inference_abi::ModelConfig {
        aien_inference_abi::ModelConfig {
            model_id: "kv239-tiny".to_string(),
            max_sequence_length: 40,
            block_size: 16,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            num_kv_heads: 2,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            eos_token_ids: Vec::new(),
            qk_norm: false,
        }
    }

    // ---- #254: fail closed when memory cannot be established ----

    #[test]
    fn unreadable_meminfo_refuses_the_pool() {
        let cfg = tiny_config();
        let err = plan_checked_model_kv(
            &cfg,
            None,
            &|| Err("/proc/meminfo cannot be read: No such file".to_string()),
            false,
        )
        .unwrap_err();
        assert!(err.contains("cannot be checked"), "{err}");
        assert!(err.contains("refusing to allocate"), "{err}");
        assert!(err.contains(ALLOW_UNCHECKED_MEMORY_ENV), "{err}");
    }

    #[test]
    fn unreadable_meminfo_with_override_proceeds() {
        // The override path also prints a WARNING line on stderr (see check_kv_memory).
        let cfg = tiny_config();
        let plan = plan_checked_model_kv(&cfg, None, &|| Err("unreadable".to_string()), true)
            .expect("override set: proceeds");
        assert_eq!(plan.total_blocks, 3);
        // The override never bypasses a known shortage.
        assert!(check_kv_memory(10, Ok(1), true).is_err());
    }

    #[test]
    fn unparsable_meminfo_refuses_the_pool() {
        for text in [
            "MemTotal: 1 kB\n",
            "MemAvailable: x kB\n",
            "MemAvailable: 5\n",
            "",
        ] {
            let cfg = tiny_config();
            let err =
                plan_checked_model_kv(&cfg, None, &|| parse_mem_available_checked(text), false)
                    .unwrap_err();
            assert!(err.contains("no parsable"), "{text:?}: {err}");
        }
    }
    #[test]
    fn daemon_plan_refuses_before_allocating_and_sizes_the_manager() {
        // Exercises the planning, memory check and pooled manager without building a
        // spine: a spine creates a RuntimeController, which reads the process-wide
        // AIEN_RUNTIME_STATE_DIR that other tests in this crate point at temp dirs.
        let cfg = tiny_config();
        let err = plan_checked_model_kv(&cfg, None, &|| Ok(1), false).unwrap_err();
        assert!(err.contains("refusing"), "{err}");

        let plan =
            plan_checked_model_kv(&cfg, None, &|| Ok(1 << 30), false).expect("enough memory");
        // ceil(40 / 16) = 3 blocks of 16 tokens * (2 * 2 * 2 * 16 * 2) bytes per token (bf16).
        assert_eq!(plan.total_blocks, 3);
        assert_eq!(plan.total_bytes, 3 * 16 * 256);

        let weights = TransformerWeights::reference_test_weights(&cfg);
        let mgr = build_model_kv_manager(&weights, plan.total_blocks).expect("pool built");
        assert_eq!(mgr.read().total_block_count(), 3);
        assert_eq!(plan.pool_config().total_bytes(), plan.total_bytes);
    }

    #[test]
    fn kv_context_cap_parse_is_strict() {
        assert_eq!(parse_kv_context_cap("8192"), Ok(8192));
        assert_eq!(parse_kv_context_cap("1"), Ok(1));
        for bad in [
            "",
            "0",
            "00",
            "-1",
            "+5",
            " 8",
            "8 ",
            "8k",
            "1e3",
            "0x10",
            "99999999999999999999999",
        ] {
            let err = parse_kv_context_cap(bad).unwrap_err();
            assert!(err.contains(KV_CONTEXT_CAP_ENV), "{bad:?}: {err}");
        }
    }

    #[test]
    fn kv_context_cap_only_lowers_the_model_context() {
        let cfg = tiny_config(); // max_sequence_length 40, block 16
        assert_eq!(
            KvPoolPlan::for_model_capped(&cfg, None).unwrap(),
            KvPoolPlan::for_model(&cfg).unwrap()
        );
        let capped = KvPoolPlan::for_model_capped(&cfg, Some(17)).unwrap();
        assert_eq!((capped.context_tokens, capped.total_blocks), (17, 2));
        assert!(capped.context_source.contains(KV_CONTEXT_CAP_ENV));
        assert_eq!(
            KvPoolPlan::for_model_capped(&cfg, Some(40))
                .unwrap()
                .total_blocks,
            3
        );
        let err = KvPoolPlan::for_model_capped(&cfg, Some(41)).unwrap_err();
        assert!(err.contains("can only lower"), "{err}");
    }

    #[test]
    fn kv_context_cap_lets_a_long_context_model_fit_and_sizes_the_real_pool() {
        let cfg = tiny_config();
        let uncapped = KvPoolPlan::for_model(&cfg).unwrap().total_bytes as u64;
        let capped = KvPoolPlan::for_model_capped(&cfg, Some(16))
            .unwrap()
            .total_bytes as u64;
        assert!(capped < uncapped);
        // Memory between the two: the model's own context is refused, the cap fits.
        let avail = capped;
        assert!(plan_checked_model_kv(&cfg, None, &|| Ok(avail), false).is_err());
        let plan = plan_checked_model_kv(&cfg, Some(16), &|| Ok(avail), false).unwrap();
        assert_eq!(plan.total_blocks, 1);
        // The capped plan sizes the real pool. No spine here: a spine creates a
        // RuntimeController, which reads the process-wide AIEN_RUNTIME_STATE_DIR that
        // control::tests points at a dir holding deliberately damaged state files.
        let plan = plan_checked_model_kv(&cfg, Some(17), &|| Ok(1 << 30), false).unwrap();
        assert_eq!(plan.total_blocks, 2);
        let weights = TransformerWeights::reference_test_weights(&cfg);
        let mgr = build_model_kv_manager(&weights, plan.total_blocks).expect("pool built");
        assert_eq!(mgr.read().total_block_count(), 2);
    }
}
