//! Independent reference and per-op native accounting for the GB10 strict gate.
//!
//! The independent reference is the Hugging Face FP32 oracle fixture checked into
//! `aien-inference-abi/fixtures` (tinyllama_oracle_manifest.json + tinyllama_oracle.safetensors):
//! TinyLlama-1.1B-Chat-v1.0 logits for a 46-token prompt and its 16-step greedy decode,
//! produced outside this codebase. The CPU reference backend is our own code, so comparing
//! Omega only against it cannot catch a bug both share.
//!
//! [`CountingBackend`] wraps the backend the production path selected and counts calls per
//! op, so a receipt can show every op that ran and that none of them fell back.
#![allow(dead_code)]
use aien_inference_abi::backend::TensorBackend;
use aien_inference_abi::native_ops::{NativeOpMask, OpReport, TensorOp};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../aien-inference-abi/fixtures")
}

pub struct Oracle {
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub safetensors_sha256: String,
    pub prompt_text: String,
    pub prompt_tokens: Vec<u32>,
    pub greedy_tokens: Vec<u32>,
    pub greedy_text: String,
    /// Oracle logits for the token after the prompt (32000 values).
    pub last_token_logits: Vec<f32>,
}

fn str_field(v: &serde_json::Value, k: &str) -> String {
    v[k].as_str()
        .unwrap_or_else(|| panic!("oracle manifest: missing {k}"))
        .to_string()
}

fn u32_list(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("oracle manifest: expected an array")
        .iter()
        .map(|t| t.as_u64().expect("oracle manifest: token id") as u32)
        .collect()
}

/// One f32 tensor from a safetensors file (header JSON, then raw little-endian data).
fn read_f32_tensor(path: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + hlen]).expect("oracle safetensors header");
    let t = &header[name];
    assert_eq!(t["dtype"].as_str(), Some("F32"), "{name} dtype: {t}");
    let off = t["data_offsets"].as_array().expect("data_offsets");
    let (a, b) = (
        off[0].as_u64().unwrap() as usize,
        off[1].as_u64().unwrap() as usize,
    );
    let (words, rest) = bytes[8 + hlen + a..8 + hlen + b].as_chunks::<4>();
    assert!(rest.is_empty(), "{name}: byte length not a multiple of 4");
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
}

pub fn load() -> Oracle {
    let dir = fixtures_dir();
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("tinyllama_oracle_manifest.json")).expect("oracle manifest"),
    )
    .expect("oracle manifest JSON");
    let st = dir.join(str_field(&manifest, "safetensors_file"));
    let last_token_logits = read_f32_tensor(&st, "last_token_logits");
    assert_eq!(last_token_logits.len(), 32000, "oracle last_token_logits");
    Oracle {
        model_sha256: str_field(&manifest, "model_sha256"),
        tokenizer_sha256: str_field(&manifest, "tokenizer_sha256"),
        safetensors_sha256: str_field(&manifest, "safetensors_sha256"),
        prompt_text: str_field(&manifest, "prompt_text"),
        prompt_tokens: u32_list(&manifest["prompt_tokens"]),
        greedy_tokens: u32_list(&manifest["decoded_16_steps"]["token_ids"]),
        greedy_text: str_field(&manifest["decoded_16_steps"], "text"),
        last_token_logits,
    }
}

/// Delegates every op to the backend the production path chose and counts calls per op.
pub struct CountingBackend {
    inner: Arc<dyn TensorBackend>,
    calls: [AtomicU64; TensorOp::COUNT],
}

impl CountingBackend {
    pub fn new(inner: Arc<dyn TensorBackend>) -> Self {
        Self {
            inner,
            calls: Default::default(),
        }
    }

    fn hit(&self, op: TensorOp) {
        self.calls[TensorOp::ALL.iter().position(|o| *o == op).unwrap()]
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn calls(&self, op: TensorOp) -> u64 {
        self.calls[TensorOp::ALL.iter().position(|o| *o == op).unwrap()].load(Ordering::Relaxed)
    }

    /// Calls per op minus reference runs and fallbacks of that op: the native executions.
    pub fn native_calls(&self) -> Vec<(TensorOp, u64, u64)> {
        let rep = self.inner.op_report();
        let count = |v: &[(String, u64)], op: TensorOp| {
            v.iter()
                .find(|(n, _)| n == op.name())
                .map_or(0, |(_, c)| *c)
        };
        TensorOp::ALL
            .into_iter()
            .map(|op| {
                let off = count(&rep.native_fallbacks, op) + count(&rep.reference_runs, op);
                (op, self.calls(op), self.calls(op).saturating_sub(off))
            })
            .collect()
    }

    /// `NATIVE_CALLS op:calls/native,...` for the run log.
    pub fn line(&self) -> String {
        let parts: Vec<String> = self
            .native_calls()
            .into_iter()
            .map(|(op, c, n)| format!("{}:{c}/{n}", op.name()))
            .collect();
        format!("NATIVE_CALLS {}", parts.join(","))
    }
}

impl TensorBackend for CountingBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn fallback_count(&self) -> u64 {
        self.inner.fallback_count()
    }
    fn native_ops(&self) -> NativeOpMask {
        self.inner.native_ops()
    }
    fn op_report(&self) -> OpReport {
        self.inner.op_report()
    }
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.hit(TensorOp::Rmsnorm);
        self.inner.rmsnorm(out, x, weight, eps)
    }
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        theta: f32,
    ) {
        self.hit(TensorOp::ApplyRope);
        self.inner
            .apply_rope(q, k, pos, head_dim, num_q_heads, num_kv_heads, theta)
    }
    fn matmul_vec(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) {
        self.hit(TensorOp::MatmulVec);
        self.inner.matmul_vec(out, x, weight, out_dim, in_dim)
    }
    fn matmul_batch(
        &self,
        out: &mut [f32],
        x: &[f32],
        weight: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        self.hit(TensorOp::MatmulBatch);
        self.inner
            .matmul_batch(out, x, weight, batch_size, in_dim, out_dim)
    }
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        self.hit(TensorOp::Swiglu);
        self.inner.swiglu(out, gate, up)
    }
    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::GqaAttention);
        self.inner.gqa_attention(
            out,
            q,
            k_cache,
            v_cache,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn paged_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_ids: &[usize],
        context_len: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::PagedAttention);
        self.inner.paged_attention(
            out,
            q,
            pool,
            block_ids,
            context_len,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn paged_attention_batch(
        &self,
        out: &mut [f32],
        q: &[f32],
        pool: &aien_kv_cache::UnifiedKvTensorPool,
        block_tables: &[i32],
        context_lens: &[i32],
        max_blocks_per_seq: usize,
        num_seqs: usize,
        layer_idx: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.hit(TensorOp::PagedAttentionBatch);
        self.inner.paged_attention_batch(
            out,
            q,
            pool,
            block_tables,
            context_lens,
            max_blocks_per_seq,
            num_seqs,
            layer_idx,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
    }
    fn compute_logits(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        embed_weight: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) {
        self.hit(TensorOp::ComputeLogits);
        self.inner
            .compute_logits(logits, hidden, embed_weight, vocab_size, hidden_dim)
    }
}
