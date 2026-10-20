//! `--engine wormhole` (L190): the 27B + 30B pair behind the same OpenAI surface as the 27B door.
//!
//! The 27B (Qwen3.8, the same session and decode graph as the 27B door on 18091) writes every token: its own
//! choice through the same sampler, so the output is the 27B alone, token for token, at any temperature.
//! Beside it the Qwen3-30B-A3B drafts (CPU reference, everything but the routed experts resident, the 8
//! routed experts of every layer read through the Heartbeat ring from the SSD, BLAKE3-checked) and is
//! scored against each token the 27B writes, through the text bridge between the two vocabularies.
//!
//! Speed: this door checks one guess per 27B step, so it is not faster than the 27B alone; it is the
//! pairing and its live acceptance, measured. Checking many guesses in one 27B pass is the exact-rows
//! path (docs/WORMHOLE.md, evidence/WORMHOLE-30B/00-prediction.md), not this door.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aien_heartbeat::manifest::Manifest;
use aien_inference_abi::qwen38::Qwen38Config;
use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Session};
use aien_wormhole::align::{ByteVocab, DraftQueue, Vocab};
use aien_wormhole::moe30::Moe30Ring;

use crate::engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams};
use crate::omega_engine::{GpuLock, GPU_LOCK_PATH};
use crate::qwen38_engine::{rss_bytes, KEEP_FREE, LOAD_LOCK_WAIT, PREFILL_CHUNK, REQUEST_LOCK_WAIT};
use crate::sampling::SamplingObserver;

pub const DEFAULT_MODEL_ID: &str = "aien-wormhole-27b-30b";
/// Host bytes of the 30B without its routed experts (attention f32, embed + lm_head f32, routers), about.
const DENSE_30B: u64 = 7 << 30;

pub struct Drafter {
    pub ring: Moe30Ring,
    pub v30: ByteVocab,
    pub k: usize,
}

pub struct WormholeDoor {
    session: Mutex<Qwen38Session>,
    drafter: Option<Mutex<Drafter>>,
    v27: ByteVocab,
    max_context: usize,
    lock_path: PathBuf,
    lock_wait: Duration,
}

fn mem_available() -> u64 {
    (crate::omega_engine::mem_available_gb() * (1u64 << 30) as f64) as u64
}

/// The 30B expert manifest, built once (reads every expert to digest it) and kept next to the cache.
fn manifest(dir30: &Path) -> Result<Manifest, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let path = Path::new(&home).join(".cache/aien/wormhole-30b-manifest.json");
    if let Ok(m) = Manifest::load(&path) {
        if m.model_dir == dir30 {
            return Ok(m);
        }
    }
    let m = Manifest::build(dir30).map_err(|e| e.to_string())?;
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    m.save(&path).map_err(|e| e.to_string())?;
    Ok(m)
}

impl WormholeDoor {
    /// The 27B at `dir27` on the chip; the 30B at `dir30` as the drafter with `slots` ring slots, `k` drafter
    /// tokens per draft. `dir30 = None`: the 27B alone in this door (the yardstick for the identity receipt).
    pub fn load(dir27: &Path, dir30: Option<&Path>, max_context: usize, k: usize, slots: usize) -> Result<Self, String> {
        if !aien_omega_gpu::is_native() {
            return Err("this binary carries the CPU stub of the Omega engine".into());
        }
        let c = Qwen38Config::from_checkpoint(dir27)?;
        let max_pos = max_context.next_multiple_of(256);
        let slot_bytes = 4_800_000u64; // one Qwen3-30B-A3B expert is 4.72 MB (evidence/WORMHOLE-30B)
        let draft_need = dir30.map_or(0, |_| DENSE_30B + slots as u64 * slot_bytes);
        let need = need_bytes(&c, max_pos, PREFILL_CHUNK) + draft_need;
        let avail = mem_available();
        if avail < need + KEEP_FREE {
            return Err(format!(
                "needs {} MiB (27B {} + 30B drafter {}) plus {} MiB kept free, MemAvailable {} MiB; refusing",
                (need) >> 20,
                (need - draft_need) >> 20,
                draft_need >> 20,
                KEEP_FREE >> 20,
                avail >> 20
            ));
        }
        let v27 = ByteVocab::from_file(&dir27.join("tokenizer.json"))?;
        let drafter = match dir30 {
            Some(d) => {
                let t = Instant::now();
                let ring = Moe30Ring::open(d, manifest(d)?, slots, 8, true)?;
                let v30 = ByteVocab::from_file(&d.join("tokenizer.json"))?;
                eprintln!(
                    "aien-serve-http: wormhole: 30B drafter ready in {:.1} s: dense {:.2} GB resident, ring {slots} slots, k {k}",
                    t.elapsed().as_secs_f64(),
                    ring.dense_bytes() as f64 / 1e9
                );
                Some(Mutex::new(Drafter { ring, v30, k }))
            }
            None => None,
        };
        let _lock = GpuLock::acquire_queued_at(Path::new(GPU_LOCK_PATH), LOAD_LOCK_WAIT)?;
        let t0 = Instant::now();
        let model = Qwen38Omega::open(dir27, None, max_pos)?;
        eprintln!(
            "aien-serve-http: wormhole: 27B ready in {:.1} s: weights {:.2} GB, KV {:.2} GB, rss {:.2} GB",
            t0.elapsed().as_secs_f64(),
            model.weight_bytes() as f64 / 1e9,
            model.kv_bytes() as f64 / 1e9,
            rss_bytes() as f64 / 1e9
        );
        Ok(Self {
            session: Mutex::new(Qwen38Session::new(model, PREFILL_CHUNK)),
            drafter,
            v27,
            max_context,
            lock_path: PathBuf::from(GPU_LOCK_PATH),
            lock_wait: REQUEST_LOCK_WAIT,
        })
    }
}

impl Engine for WormholeDoor {
    fn kind(&self) -> &str {
        if self.drafter.is_some() {
            "wormhole-qwen38+qwen3-30b"
        } else {
            "wormhole-qwen38-alone"
        }
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
            return Err(EngineError::PromptTooLong { len: prompt.len(), max: self.max_context });
        }
        let lock = GpuLock::acquire_queued_at(&self.lock_path, self.lock_wait).map_err(EngineError::Busy)?;
        let mut s = self.session.lock().map_err(|_| EngineError::Failed("wormhole session poisoned".into()))?;
        let mut dg = match &self.drafter {
            Some(m) => Some(m.lock().map_err(|_| EngineError::Failed("wormhole drafter poisoned".into()))?),
            None => None,
        };
        let mut queue = match dg.as_deref_mut() {
            Some(Drafter { ring, v30, k }) => {
                let text = String::from_utf8_lossy(&self.v27.decode(prompt)).to_string();
                let d_prompt = v30.encode_str(&text).map_err(EngineError::Failed)?;
                Some(DraftQueue::new(ring, &*v30, &self.v27, d_prompt, *k))
            }
            None => None,
        };
        let t0 = Instant::now();
        let (mut logits, _fs) = s.start(prompt).map_err(EngineError::Failed)?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut observer = SamplingObserver::new(p);
        let mut emitted = 0usize;
        let mut draft_err = None;
        // the 27B door's loop, unchanged; the drafter only watches
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
            if let Some(q) = queue.as_mut() {
                if let Err(e) = q.observe(tok) {
                    draft_err = Some(e);
                    queue = None;
                }
            }
            if emitted >= p.max_new_tokens || prompt.len() + emitted >= self.max_context {
                break FinishReason::Length;
            }
            let (next, _ns) = s.advance(tok).map_err(EngineError::Failed)?;
            logits = next;
        };
        let decode_ms = t1.elapsed().as_millis() as u64;
        drop(s);
        drop(lock);
        let scored = queue.map(|q| (q.stats.clone(), q.draft_ns));
        match (&scored, &draft_err) {
            (Some((st, draft_ns)), _) => eprintln!(
                "aien-serve-http: wormhole: {emitted} tok in {decode_ms} ms; drafts {} rounds, 27B ids drafted {} accepted {} ({:.3}), drafter {:.0} ms, ring hits {} misses {}",
                st.rounds,
                st.drafted_v,
                st.accepted,
                st.accepted as f64 / emitted.max(1) as f64,
                *draft_ns as f64 / 1e6,
                dg.as_ref().map_or(0, |d| d.ring.cache.stats.hits),
                dg.as_ref().map_or(0, |d| d.ring.cache.stats.misses),
            ),
            (None, Some(e)) => eprintln!("aien-serve-http: wormhole: drafting stopped for this request: {e}"),
            (None, None) => eprintln!("aien-serve-http: wormhole (27B alone): {emitted} tok in {decode_ms} ms"),
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms,
            sampling: Some(observer.record()),
        })
    }
}
