//! Reading guide: the Omega engine behind the HTTP door's `Engine` trait (OWN-30B).
//!
//! NOT COMPILED. The live code is `crates/aien-serve-http/src/omega_engine.rs`; when this
//! sketch and that file disagree, that file wins. Refreshed for sc#522 against main bdd58a2b
//! (the first version of this file was the cut 3 x cut 4 seam, before the GPU lock, sampling,
//! whole-prompt prefill and persistent KV existed). Bodies are trimmed to the steps, in order.
//!
//! Wiring in `main.rs` (`--engine omega`):
//!     let mut door = OmegaDoor::load(&model_dir, max_ctx, weights, spin_us, prefill)?;
//!     if AIEN_KV_STORE_DIR is set { door.set_persistence(kv_persist::open_with_test_fault(..).0) }
//!     Arc::new(door)  // the server's Arc<dyn Engine>
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use aien_inference_abi::qwen3_moe_omega::{self as omega, OmegaEngine, WeightFormat};

use crate::engine::{Engine, EngineError, GenerateOutcome, GenerateParams};
use crate::kv_persist::PersistState;

/// The hive's GPU lock: held only around a generation (and the load), never while idle.
pub const GPU_LOCK_PATH: &str = "/tmp/aien-gb10.lock";
/// How long `generate` waits for the GPU lock before answering `EngineError::Busy` (HTTP 503).
pub const LOCK_WAIT: Duration = Duration::from_secs(30);
/// Host memory the load needs free.
pub const LOAD_MIN_FREE_GB: f64 = 45.0;

/// Prefill path; both ON by default in the door (the engine and `own30b_check` default OFF:
/// see the doc on `PrefillMode` in the live file).
pub struct PrefillMode {
    pub whole_prompt: bool,
    pub device_host: bool,
}

pub struct OmegaDoor {
    /// Declared first: a running store drains its parks before the engine goes.
    persist: PersistState,
    engine: OmegaEngine,
    lock_wait: Duration,
    lock_path: PathBuf,
    /// Prompt tokens the last generation reused (RAM or restored prefix).
    reused_last: AtomicUsize,
}

impl OmegaDoor {
    /// 1. refuse below LOAD_MIN_FREE_GB available, or when the binary carries the CPU stub of
    ///    omega, or when FP8 experts are asked for without the FP8 GEMV linked;
    /// 2. apply `spin_us`; take the GPU lock (LOCK_WAIT);
    /// 3. `OmegaEngine::load_with(model_dir, max_context, format, progress)`;
    /// 4. still under the lock: `set_whole_prompt(..)`, then `set_whole_prompt_device_host(..)`
    ///    when whole-prompt is on (a refusal is fatal and names the `off` switch);
    /// 5. log the load report and the paths in use; `from_parts(engine, GPU_LOCK_PATH, LOCK_WAIT)`.
    pub fn load(
        model_dir: &Path,
        max_context: usize,
        format: WeightFormat,
        spin_us: Option<u32>,
        prefill: PrefillMode,
    ) -> Result<Self, String> {
        unimplemented!("see omega_engine.rs")
    }

    /// A door over a built engine (tests: a private lock file and the host test engine).
    /// Persistence starts `PersistState::Off`.
    pub fn from_parts(engine: OmegaEngine, lock_path: PathBuf, lock_wait: Duration) -> Self {
        unimplemented!("see omega_engine.rs")
    }

    /// Installs the persistence state, before the door is shared.
    pub fn set_persistence(&mut self, state: PersistState) {
        self.persist = state;
    }
}

impl Engine for OmegaDoor {
    fn kind(&self) -> &str {
        self.engine.kind()
    }

    fn max_context_tokens(&self) -> usize {
        self.engine.max_context_tokens()
    }

    /// 1. empty prompt -> `Failed`; prompt >= context -> `PromptTooLong`;
    /// 2. map the params to `omega::GenerateParams`;
    /// 3. persistent KV on with restore: fetch the longest stored prefix that beats the RAM
    ///    prefix, WITHOUT the GPU lock;
    /// 4. take the GPU lock at `lock_path` (wait `lock_wait`), else `EngineError::Busy`;
    /// 5. restore the fetched K/V (all or nothing; on error the engine resets and prefills);
    /// 6. `engine.generate_with(prompt, &q, &mut sampler, on_token)` with the server's own
    ///    `SamplingObserver`, so CPU and Omega pick tokens the same way;
    /// 7. persistent KV on: snapshot (park) under the lock; release the lock; then submit the
    ///    park for writing;
    /// 8. map the finish (`Constraint` reads as `Stop`) and return the sampling record.
    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        unimplemented!("see omega_engine.rs")
    }

    /// `/health` `persistent_kv`: absent when off, `refused` with the reason, or the store's
    /// status when on.
    fn persistence_status(&self) -> Option<serde_json::Value> {
        unimplemented!("see omega_engine.rs")
    }
}
