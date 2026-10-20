//! Qwen3.8-27B behind the door (sc #642 serving): the whole model resident on AIEN's own Omega
//! kernels (64 layers, fused FP8 projections, bf16 lm_head), prompts through omega's prefill,
//! replies through the chip-proven decode graph, next turn resumed from the live state or the
//! snapshot taken before the prompt's last turn header (`aien_inference_abi::qwen38_omega::
//! Qwen38Session`).
//!
//! Chip etiquette as the 30B door ([`crate::omega_engine`]): `/tmp/aien-gb10.lock` is taken for
//! the upload and around each generation, never while idle. Persistent KV parks
//! (`AIEN_KV_STORE_DIR`) are not offered: they hold attention KV only, not the DeltaNet state.
//!
//! Speculative decode (speed2/specdecode, opt-in): with `AIEN_Q38_SPEC_K=k` (0 ..= 8) greedy
//! requests reply through the MTP head's chain of `k` drafts and one verify pass per round on the
//! decode graph's rows mode (`aien_inference_abi::qwen38_omega::spec::decode_rounds`, verify
//! kernel `Graph`). The reply is the one-row tv path's greedy reply bit for bit (`k = 0` is that
//! reference). Sampled requests keep the plain decode loop.
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::spec::{
    decode_rounds, Mtp, MtpOptions, SpecOptions, SpecRun, VerifyKernel, MAX_CHAIN,
};
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Session, Reuse};

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
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
/// Turns speculative decode on for greedy requests: the MTP chain length `k` (0 ..= 8).
pub const SPEC_K_ENV: &str = "AIEN_Q38_SPEC_K";
/// The MTP head (fc bf16, one attention layer at FP8, its KV), counted generously as the
/// `qwen38_spec` tool does.
pub const MTP_BYTES: u64 = 1 << 30;

/// Parses [`SPEC_K_ENV`]: unset or empty is off.
pub fn parse_spec_k(v: Option<&str>) -> Result<Option<usize>, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(None),
        Some(s) => match s.parse::<usize>() {
            Ok(k) if k <= MAX_CHAIN => Ok(Some(k)),
            _ => Err(format!("{SPEC_K_ENV}={s}: want a chain length 0 ..= {MAX_CHAIN}")),
        },
    }
}

/// The speculative path's state: the MTP head (none at `k = 0`) and the chain length.
struct Spec {
    mtp: Option<Mtp>,
    k: usize,
}

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

pub struct Qwen38Door {
    session: Mutex<Qwen38Session>,
    /// Locked only while `session` is held.
    spec: Option<Mutex<Spec>>,
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
        let spec_k = parse_spec_k(std::env::var(SPEC_K_ENV).ok().as_deref())?;
        let max_pos = max_context.next_multiple_of(256);
        let need = need_bytes(&c, max_pos, PREFILL_CHUNK)
            + if spec_k.is_some_and(|k| k > 0) { MTP_BYTES } else { 0 };
        let avail = mem_available();
        if avail < need + KEEP_FREE {
            return Err(format!(
                "needs {} MiB plus {} MiB kept free, MemAvailable {} MiB (another model may be loaded); refusing",
                need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let _lock = GpuLock::acquire_queued_at(Path::new(GPU_LOCK_PATH), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        eprintln!(
            "aien-serve-http: qwen38: uploading {} layers, KV {max_pos} positions (need {} MiB, MemAvailable {} MiB)",
            c.layers,
            need >> 20,
            avail >> 20
        );
        let mut model = Qwen38Omega::open(dir, None, max_pos)?;
        let spec = match spec_k {
            None => None,
            Some(k) => {
                let mtp = if k > 0 {
                    Some(Mtp::open(dir, &model, MtpOptions::default())?)
                } else {
                    None
                };
                // Sizes the rows mode and the DeltaNet checkpoint now, under the upload's lock.
                model
                    .graph_mut()
                    .rows_reserve(MAX_CHAIN + 1)
                    .map_err(|e| format!("q38 rows reserve: {e}"))?;
                eprintln!(
                    "aien-serve-http: qwen38: speculative decode for greedy requests: MTP chain k {k}{}",
                    mtp.as_ref()
                        .map_or(String::new(), |m| format!(", head {} MiB", m.weight_bytes() >> 20))
                );
                Some(Mutex::new(Spec { mtp, k }))
            }
        };
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
            spec,
            max_context,
            lock_path: PathBuf::from(GPU_LOCK_PATH),
            lock_wait: REQUEST_LOCK_WAIT,
        })
    }

    /// A greedy reply on the speculative path ([`SPEC_K_ENV`]): the prompt as [`Engine::generate`]
    /// feeds it (same reuse and snapshot), the first token from the head over the last row, then
    /// [`decode_rounds`] on the graph's rows mode. Same limits and finish reasons as the plain loop.
    #[allow(clippy::too_many_arguments)]
    fn generate_spec(
        &self,
        s: &mut Qwen38Session,
        sp: &mut Spec,
        prompt: &[u32],
        p: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
        lock: GpuLock,
        lock_wait_ms: u128,
    ) -> Result<GenerateOutcome, EngineError> {
        let t0 = Instant::now();
        let (logits, rows, first, fs) = s.start_rows(prompt).map_err(EngineError::Failed)?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut observer = SamplingObserver::new(p);
        // The plain loop's two limits: the request's cap and the context window.
        let n = p.max_new_tokens.min(self.max_context - prompt.len());
        let mut run = SpecRun {
            accepted_hist: vec![0; sp.k + 1],
            ..SpecRun::default()
        };
        let mut emitted = 0usize;
        let finish = 'reply: {
            if n == 0 {
                break 'reply FinishReason::Length;
            }
            let t = observer.sample(&logits, 0);
            run.tokens.push(t);
            if p.stop_token_ids.contains(&t) {
                break 'reply FinishReason::Stop;
            }
            emitted = 1;
            if !on_token(t) {
                break 'reply FinishReason::Cancelled;
            }
            // The last allowed token is not fed: nothing would read its logits.
            if n == 1 {
                break 'reply FinishReason::Length;
            }
            let mut history = prompt.to_vec();
            history.push(t);
            let r = (|| {
                let model = s.model_mut();
                let mut draft = None;
                if let Some(m) = sp.mtp.as_mut() {
                    // The MTP head pairs each fed row with the token after it.
                    let mut next = prompt[first + 1..].to_vec();
                    next.push(t);
                    draft = Some(m.feed(model, &rows, &next, first, &mut run.mtp)?);
                }
                let opts = SpecOptions {
                    kernel: VerifyKernel::Graph,
                    ..SpecOptions::default()
                };
                let mut count = |tok| {
                    emitted += 1;
                    on_token(tok)
                };
                decode_rounds(
                    model,
                    sp.mtp.as_mut(),
                    n,
                    sp.k,
                    &p.stop_token_ids,
                    opts,
                    &mut run,
                    &mut history,
                    t,
                    draft,
                    &mut count,
                )
            })();
            // The state holds the history without its last token (not fed).
            let fed = history.len() - 1;
            if let Err(e) = r.and_then(|()| match s.model().pos {
                pos if pos == fed => Ok(()),
                pos => Err(format!("spec reply: model at position {pos}, history {fed}")),
            }) {
                s.forget_live();
                return Err(EngineError::Failed(e));
            }
            s.extend_history(&history[prompt.len()..fed]);
            observer.note_greedy(run.tokens.len() - 1);
            match run.tokens.last() {
                _ if run.cancelled => FinishReason::Cancelled,
                Some(x) if p.stop_token_ids.contains(x) => FinishReason::Stop,
                _ => FinishReason::Length,
            }
        };
        let decode_ms = t1.elapsed().as_millis() as u64;
        drop(lock);
        let (m, pf) = (&run.main, &fs.prefill);
        eprintln!(
            "aien-serve-http: qwen38 spec k {}: lock wait {lock_wait_ms} ms, prompt {} tok, reuse {:?} ({} tok), fed {} ({} stepped), prefill {prefill_ms} ms (gemm chip {} ms, gdn chip {} ms, attn chip {} ms), decode {emitted} tok in {decode_ms} ms over {} passes ({:.2} words per pass, {} of {} drafts accepted), graph chip {} ms, commit chip {} ms, drafts {} ms, wall {:.1} ms/token",
            sp.k,
            prompt.len(),
            fs.reuse.unwrap_or(Reuse::Cold),
            fs.reuse.map_or(0, Reuse::tokens),
            fs.fed,
            fs.stepped,
            pf.gemm_chip_ns / 1_000_000,
            pf.gdn_chip_ns / 1_000_000,
            pf.attn_chip_ns / 1_000_000,
            run.rounds,
            run.tokens.len().saturating_sub(1) as f64 / run.rounds.max(1) as f64,
            run.accepted,
            run.drafted,
            m.graph_chip_ns / 1_000_000,
            m.commit_chip_ns / 1_000_000,
            run.draft_ns / 1_000_000,
            decode_ms as f64 / emitted.saturating_sub(1).max(1) as f64,
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
        if let Some(spec) = self.spec.as_ref().filter(|_| p.temperature <= 0.001) {
            let mut sp = spec
                .lock()
                .map_err(|_| EngineError::Failed("qwen38 spec state poisoned".into()))?;
            return self.generate_spec(&mut s, &mut sp, prompt, p, on_token, lock, lock_wait_ms);
        }
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

    #[test]
    fn spec_k_is_off_unless_set_and_bounded() {
        assert_eq!(parse_spec_k(None), Ok(None));
        assert_eq!(parse_spec_k(Some(" ")), Ok(None));
        assert_eq!(parse_spec_k(Some("0")), Ok(Some(0)));
        assert_eq!(parse_spec_k(Some("8")), Ok(Some(8)));
        assert!(parse_spec_k(Some("9")).is_err());
        assert!(parse_spec_k(Some("-1")).is_err());
        assert!(parse_spec_k(Some("k8")).is_err());
    }
}
