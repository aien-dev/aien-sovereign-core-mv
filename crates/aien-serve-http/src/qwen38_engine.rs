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

//! Qwen3.8-27B behind the door (sc #642 serving): the whole model resident on AIEN's own Omega
//! kernels (64 layers, fused FP8 projections, bf16 lm_head), prompts through omega's prefill,
//! replies through the chip-proven decode graph, next turn resumed from the live state or the
//! snapshot taken before the prompt's last turn header (`aien_inference_abi::qwen38_omega::
//! Qwen38Session`).
//!
//! Chip etiquette as the 30B door ([`crate::omega_engine`]): `/tmp/aien-gb10.lock` (or the file
//! `AIEN_SERVE_GPU_LOCK` names, [`serve_lock_path`]) is taken for the upload and around each
//! generation, never while idle. Persistent KV parks
//! (`AIEN_KV_STORE_DIR`) are not offered: they hold attention KV only, not the DeltaNet state.
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Session, Reuse};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{serve_lock_path, GpuLock};
use crate::sampling::SamplingObserver;

/// Model id the door answers to unless `--model-id` says otherwise.
pub const DEFAULT_MODEL_ID: &str = "qwen3.8-27b-fp8";
/// Prefill chunk length (the 2026-10-18 TTFT receipts ran 4096).
pub const PREFILL_CHUNK: usize = 4096;
/// How long a request waits in the chip lock's queue before answering busy (the hive's chip
/// jobs queue with `flock(1)` and run for seconds to minutes each).
pub const REQUEST_LOCK_WAIT: Duration = Duration::from_secs(300);
/// How long the upload waits for the chip lock (other jobs' chip runs take minutes; a request
/// waits [`REQUEST_LOCK_WAIT`]).
pub const LOAD_LOCK_WAIT: Duration = Duration::from_secs(1800);
/// Memory kept free beyond [`need_bytes`] at load (other jobs share the box).
pub const KEEP_FREE: u64 = 8 << 30;

/// `config.json` says `model_type: qwen3_5` (the Qwen3.8 family this door serves).
pub fn is_qwen38_dir(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .is_some_and(|v| v["model_type"].as_str() == Some("qwen3_5"))
}

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// This process's resident set in bytes (0 when unreadable).
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("VmRSS:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

/// Takes the lock with `acquire`, and only then reads free memory with `mem_available` and checks
/// it against `need` plus [`KEEP_FREE`]. On a refusal the lock is released (dropped) before the
/// error returns. The order is the point: a check made before a long lock wait is stale by the time
/// the upload starts.
fn admit_load<L>(
    acquire: impl FnOnce() -> Result<L, String>,
    mem_available: impl FnOnce() -> u64,
    need: u64,
) -> Result<(L, u64), String> {
    let lock = acquire()?;
    let avail = mem_available();
    if avail < need + KEEP_FREE {
        return Err(format!(
            "needs {} MiB plus {} MiB kept free ({} MiB in all), MemAvailable {} MiB after taking the GPU lock (another model may be loaded); refusing",
            need >> 20,
            KEEP_FREE >> 20,
            (need + KEEP_FREE) >> 20,
            avail >> 20
        ));
    }
    Ok((lock, avail))
}

pub struct Qwen38Door {
    session: Mutex<Qwen38Session>,
    max_context: usize,
    lock_path: PathBuf,
    lock_wait: Duration,
}

impl Qwen38Door {
    /// Uploads the checkpoint at `dir` with a KV cache for `max_context` positions (rounded up
    /// to whole 256-position passes), under the GPU lock.
    pub fn load(dir: &Path, max_context: usize) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        if !aien_omega_q38::has_prefill() {
            return Err("this binary has no qwen38 prefill entries linked (build against omega qwen38/prefill-graph)".into());
        }
        let c = Qwen38Config::from_checkpoint(dir)?;
        let max_pos = max_context.next_multiple_of(256);
        let need = need_bytes(&c, max_pos, PREFILL_CHUNK);
        // Memory is judged only once the lock is ours: the wait can last many minutes and free
        // memory on a shared box swings by tens of GiB meanwhile (alpha test: 6 to 59 GiB).
        let t_lock = Instant::now();
        let lock_path = serve_lock_path();
        let (_lock, avail) = admit_load(
            || GpuLock::acquire_queued_at(&lock_path, LOAD_LOCK_WAIT),
            mem_available,
            need,
        )?;
        eprintln!(
            "aien-serve-http: qwen38: lock taken after {:.1} s, memory re-checked after the lock: MemAvailable {} MiB, needs {} MiB plus {} MiB kept free",
            t_lock.elapsed().as_secs_f64(),
            avail >> 20,
            need >> 20,
            KEEP_FREE >> 20
        );
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen38: uploading {} layers, KV {max_pos} positions (need {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let model = Qwen38Omega::open(dir, None, max_pos)?;
        eprintln!(
            "aien-serve-http: qwen38 engine ready in {:.1} s: context {max_context} tokens, weights {:.2} GB resident, KV {:.2} GB, rss {:.2} GB, prefill chunk {PREFILL_CHUNK}, spin_us {:?}",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            model.kv_bytes() as f64 / 1e9,
            rss_bytes() as f64 / 1e9,
            aien_omega_gpu::spin_us(),
        );
        Ok(Self {
            session: Mutex::new(Qwen38Session::new(model, PREFILL_CHUNK)),
            max_context,
            lock_path,
            lock_wait: REQUEST_LOCK_WAIT,
        })
    }
}

impl Engine for Qwen38Door {
    fn kind(&self) -> &str {
        "omega-qwen38"
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context
    }

    fn generate(
        &self,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        if prompt.is_empty() {
            return Err(EngineError::Failed("empty prompt".into()));
        }
        if prompt.len() >= self.max_context {
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max: self.max_context,
            });
        }
        let t_lock = Instant::now();
        let lock = GpuLock::acquire_queued_at(&self.lock_path, self.lock_wait)
            .map_err(EngineError::Busy)?;
        let lock_wait_ms = t_lock.elapsed().as_millis();
        let mut s = self
            .session
            .lock()
            .map_err(|_| EngineError::Failed("qwen38 session poisoned".into()))?;
        let t0 = Instant::now();
        let (mut logits, fs) = s.start(prompt).map_err(EngineError::Failed)?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut observer = SamplingObserver::new(p);
        let (mut emitted, mut chip_ns) = (0usize, 0u64);
        let finish = loop {
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let tok = observer.sample(&logits, emitted);
            if p.stop_token_ids.contains(&tok) {
                break FinishReason::Stop;
            }
            emitted += 1;
            if !on_token(tok) {
                break FinishReason::Cancelled;
            }
            // The last allowed token is not fed: nothing would read its logits.
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let (next, ns) = s.advance(tok).map_err(EngineError::Failed)?;
            logits = next;
            chip_ns += ns;
        };
        let decode_ms = t1.elapsed().as_millis() as u64;
        let steps = emitted.saturating_sub(1).max(1) as f64;
        let reused = fs.reuse.map_or(0, Reuse::tokens);
        drop(s);
        drop(lock);
        let pf = &fs.prefill;
        eprintln!(
            "aien-serve-http: qwen38: lock wait {lock_wait_ms} ms, prompt {} tok, reuse {:?} ({reused} tok), fed {} ({} stepped), prefill {prefill_ms} ms (gemm chip {} ms, gdn chip {} ms, attn chip {} ms, host {} ms, restore {} ms, snapshot {} ms), decode {emitted} tok in {decode_ms} ms, graph chip {:.1} ms/token, wall {:.1} ms/token",
            prompt.len(),
            fs.reuse.unwrap_or(Reuse::Cold),
            fs.fed,
            fs.stepped,
            pf.gemm_chip_ns / 1_000_000,
            pf.gdn_chip_ns / 1_000_000,
            pf.attn_chip_ns / 1_000_000,
            pf.host_ns / 1_000_000,
            fs.restore_ns / 1_000_000,
            fs.snapshot_ns / 1_000_000,
            chip_ns as f64 / 1e6 / steps,
            decode_ms as f64 / steps,
        );
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms,
            sampling: Some(observer.record()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Guard<'a>(&'a RefCell<Vec<&'static str>>);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("release");
        }
    }

    const NEED: u64 = 41_305 << 20;

    #[test]
    fn memory_is_read_after_the_lock_is_taken() {
        let log = RefCell::new(Vec::new());
        let r = admit_load(
            || {
                log.borrow_mut().push("lock");
                Ok(Guard(&log))
            },
            || {
                log.borrow_mut().push("memory");
                NEED + KEEP_FREE
            },
            NEED,
        );
        let (g, avail) = r.expect("exactly need plus keep-free is enough");
        assert_eq!(avail, NEED + KEEP_FREE);
        assert_eq!(*log.borrow(), ["lock", "memory"]);
        drop(g);
    }

    #[test]
    fn low_memory_after_the_wait_refuses_and_releases_the_lock() {
        // 59 GiB was free before the wait, 6 GiB after it: only the reading after the lock counts.
        let log = RefCell::new(Vec::new());
        let e = admit_load(
            || Ok(Guard(&log)),
            || {
                log.borrow_mut().push("memory");
                6 << 30
            },
            NEED,
        )
        .err()
        .expect("6 GiB must be refused");
        assert!(
            e.contains("41305 MiB plus 8192 MiB kept free (49497 MiB in all)"),
            "{e}"
        );
        assert!(
            e.contains("MemAvailable 6144 MiB after taking the GPU lock"),
            "{e}"
        );
        assert_eq!(*log.borrow(), ["memory", "release"]);
    }

    #[test]
    fn one_byte_short_is_refused() {
        let r = admit_load(|| Ok(()), || NEED + KEEP_FREE - 1, NEED);
        assert!(r.is_err());
    }

    #[test]
    fn no_lock_means_memory_is_never_read() {
        let r: Result<((), u64), String> = admit_load(
            || Err("lock wait timed out".into()),
            || panic!("memory must not be read without the lock"),
            NEED,
        );
        assert_eq!(r.unwrap_err(), "lock wait timed out");
    }
}
