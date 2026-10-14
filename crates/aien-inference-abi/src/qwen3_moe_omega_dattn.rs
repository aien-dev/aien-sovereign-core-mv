//! The fused attention path of the Omega engine (OWN-30B cut 9): omega's one-launch decode
//! attention (`aien-omega-dattn`) as a [`KvAttention`]. One launch per layer per token does the
//! q/k per-head norms, RoPE, the bf16 K/V append and the attention over the whole context on
//! the chip, from the raw fused q|k|v projection row, so the host work between the q/k/v GEMV
//! and the o GEMV disappears.
//!
//! The cache owns the KV of every layer by position, so prefix reuse works as before:
//! `truncate(len)` forgets positions at or past `len` and the next step rewrites them.
//! `append` / `attend` (the split path) are refused: the engine must take [`KvAttention::fused_attend`].
use crate::qwen3_moe_omega::{KvAttention, OmegaLayer, Qwen3MoeConfig};

/// Omega's one-launch decode attention with its GPU-resident KV, one sequence.
pub struct OmegaFusedKv {
    cache: aien_omega_dattn::DecodeAttnCache,
    capacity: usize,
    filled: Vec<usize>,
}

impl OmegaFusedKv {
    /// Whether the kernel's fixed shape matches `config` (32 query heads, 4 KV heads, 128 wide).
    pub fn fits(config: &Qwen3MoeConfig) -> bool {
        config.q_heads == aien_omega_dattn::Q_HEADS && config.kv_heads == aien_omega_dattn::KV_HEADS && config.head_dim == aien_omega_dattn::HEAD_DIM
    }

    /// Allocates the cache for `max_context` tokens (rounded up to 256) and uploads every
    /// layer's q_norm / k_norm weights.
    pub fn new(config: &Qwen3MoeConfig, max_context: usize, layers: &[OmegaLayer]) -> Result<Self, String> {
        if !aien_omega_dattn::is_native() {
            return Err("this binary has no decode attention linked (omega before cut 8)".into());
        }
        if !Self::fits(config) {
            return Err(format!(
                "fused attention needs 32 q heads x 4 kv heads x 128, config has {} x {} x {}",
                config.q_heads, config.kv_heads, config.head_dim
            ));
        }
        if layers.len() != config.layers {
            return Err(format!("fused attention: {} layers of weights, config says {}", layers.len(), config.layers));
        }
        let max_pos = aien_omega_dattn::round_max_pos(max_context);
        let mut cache = aien_omega_dattn::DecodeAttnCache::alloc(config.layers, max_pos, config.rope_theta as f64)
            .map_err(|e| format!("decode attn cache alloc ({} layers x {max_pos}): {e}", config.layers))?;
        for (l, layer) in layers.iter().enumerate() {
            cache.set_norms(l, &layer.q_norm, &layer.k_norm).map_err(|e| format!("decode attn norms layer {l}: {e}"))?;
        }
        Ok(Self { cache, capacity: max_pos, filled: vec![0; config.layers] })
    }

    /// Bytes the chip holds for the K and V planes.
    pub fn kv_bytes(&self) -> usize {
        aien_omega_dattn::kv_bytes(self.cache.layers(), self.cache.max_pos())
    }

    /// omega's prefill attention over this cache (whole-prompt prefill appends K/V for every layer).
    pub fn prefill_attention(&mut self) -> aien_omega_pf::OmegaPrefillAttention<'_> {
        aien_omega_pf::OmegaPrefillAttention::new(&mut self.cache)
    }

    /// Marks positions `0..upto` stored in every layer (after a whole-prompt prefill).
    pub fn mark_filled(&mut self, upto: usize) {
        self.filled.iter_mut().for_each(|f| *f = upto);
    }
}

impl KvAttention for OmegaFusedKv {
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn reset(&mut self) {
        self.filled.iter_mut().for_each(|f| *f = 0);
    }
    fn truncate(&mut self, len: usize) {
        // The cache keeps its bytes; positions past `len` are rewritten by later steps.
        for f in self.filled.iter_mut() {
            *f = (*f).min(len);
        }
    }
    fn append(&mut self, _layer: usize, pos: usize, _k: &[f32], _v: &[f32]) -> Result<(), String> {
        Err(format!("fused kv: split append at {pos} is not a path here (use fused_attend)"))
    }
    fn attend(&mut self, _layer: usize, _q: &[f32], context_len: usize, _out: &mut [f32]) -> Result<u64, String> {
        Err(format!("fused kv: split attend over {context_len} is not a path here (use fused_attend)"))
    }
    fn fused_attend(&mut self, layer: usize, pos: usize, qkv: &[f32], eps: f32, out: &mut [f32]) -> Option<Result<u64, String>> {
        if pos != self.filled[layer] || pos >= self.capacity {
            return Some(Err(format!("fused kv: step at {pos}, have {} (capacity {})", self.filled[layer], self.capacity)));
        }
        // out_scale 1.0: the linked FP8 GEMV takes a plain f32 row (no x_prescaled entry in
        // omega_gpu_fp8_api.h at own30b/integration 831001cc).
        let r = self
            .cache
            .attend(layer, pos, qkv, eps, 1.0, out)
            .map(|info| info.elapsed_ns)
            .map_err(|e| format!("decode attn layer {layer} pos {pos}: {e}"));
        if r.is_ok() {
            self.filled[layer] = pos + 1;
        }
        Some(r)
    }
    fn resident(&self) -> bool {
        true
    }
    fn path_name(&self) -> &'static str {
        "fused"
    }
    fn fused_kv(&mut self) -> Option<&mut OmegaFusedKv> {
        Some(self)
    }
    fn dgraph_cache(&mut self, pos: usize) -> Option<Result<*mut std::ffi::c_void, String>> {
        if pos >= self.capacity {
            return Some(Err(format!("fused kv (decode graph): step at {pos}, capacity {}", self.capacity)));
        }
        if let Some((l, &f)) = self.filled.iter().enumerate().find(|(_, &f)| f != pos) {
            return Some(Err(format!("fused kv (decode graph): step at {pos}, layer {l} holds {f} positions")));
        }
        Some(Ok(self.cache.as_raw()))
    }
    fn dgraph_commit(&mut self, pos: usize) {
        self.filled.iter_mut().for_each(|f| *f = pos + 1);
    }
}
