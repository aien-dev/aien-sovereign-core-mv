//! Many conversations on one weight read (cascade/continuous-batching).
//!
//! The 27B reads 26.9 GB of weights per decode pass whoever the pass is for. Here every
//! conversation keeps its own chip state (a [`Stream`]: its own DeltaNet tails and states and
//! its own KV cache) and one [`Qwen38Omega::step_rows`] call advances any number of them by one
//! token each: the dense projections run as one multi-row FP8 GEMM over all the rows (the
//! weights stream once), while the mixers (`attn256 decode`, `gdn_step`) run per row on that
//! row's own state, exactly as a single conversation's step would.
//!
//! The step boundary shared with the MTP chain (cascade/mtp-chain): `step_rows(rows) ->
//! outputs`, one call per token for every active conversation. Rows join (after their prompt
//! is fed by [`Qwen38Omega::stream_prefill`]) and leave (stop, length, cancel) between calls;
//! nothing inside a call knows or cares which conversation a row belongs to.
//!
//! Exactness contract (the f7 rule, [`super::spec`] module doc): every row of an `m`-row step
//! is bit for bit the single-row step of that conversation alone, because the GEMM is
//! row-for-row identical to the per-row GEMV (omega's contract, chip-checked at every shape),
//! the mixers are the same calls on the same state, and the host glue is row by row. The
//! word-scoring head (bf16 lm_head) runs as one GEMV per row: that read is NOT shared yet.
use std::time::Instant;

use aien_omega_q38 as q;
use aien_omega_q38::{Attn256Cache, Gdn};

use std::sync::atomic::Ordering;

use super::spec::{layer_rows, Mixer, RowStats, RowTargets, TREE_GEMM};
use super::{gpu, PrefillStats, Qwen38Omega};
use crate::qwen3_moe::bf16_to_f32;
use crate::tensor::rmsnorm;

/// How one [`Qwen38Omega::step_rows`] call reads the weights (speed5/batching).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowKernel {
    /// Dense projections on the row-pair FP8 GEMM (row for row the per-row GEMV, but each
    /// extra row costs close to another weight read: 2 rows per pass), bf16 lm_head GEMV per row.
    #[default]
    RowPair,
    /// Dense projections on the tensor-core tree-verify GEMM (omega cascade/tree-verify-gemm):
    /// the weights are read once for up to [`aien_omega_fp8::TV_MAX_ROWS`] rows, and row `r` is
    /// the same bits whatever the row count or the other rows (activations go in as bf16, so
    /// these bits are not the row-pair GEMM's). With the FP8 lm_head loaded
    /// (`AIEN_Q38_FP8_HEAD=1`) the head is one tree-verify GEMM over every row as well;
    /// without it the head stays bf16 per row.
    Tree,
}

/// Clears [`TREE_GEMM`] when a step ends, error or not.
struct TreeGemmOn;

impl TreeGemmOn {
    fn set(on: bool) -> Self {
        TREE_GEMM.store(on, Ordering::Relaxed);
        TreeGemmOn
    }
}

impl Drop for TreeGemmOn {
    fn drop(&mut self) {
        TREE_GEMM.store(false, Ordering::Relaxed);
    }
}

/// One conversation's chip state: DeltaNet tails and states for every DeltaNet layer, a KV
/// cache of `max_pos` positions for every attention layer, and the next token's position.
pub struct Stream {
    attn: Option<Attn256Cache>,
    gdn: Option<Gdn>,
    /// Tokens the state holds (the next token's position).
    pub pos: usize,
    max_pos: usize,
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stream {{ pos: {}, max_pos: {} }}",
            self.pos, self.max_pos
        )
    }
}

impl Stream {
    /// Positions this stream's KV cache holds.
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Bytes one stream takes on the box for a cache of `max_pos` positions (ON PAPER from the
    /// kernel constants: the DeltaNet tails and states in f32, the bf16 KV of every attention
    /// layer). Allocator rounding is not counted.
    pub fn bytes(n_attn: usize, n_gdn: usize, max_pos: usize) -> usize {
        n_gdn * 4 * (q::GDN_TAIL_FLOATS + q::GDN_STATE_FLOATS)
            + n_attn * Attn256Cache::layer_bytes(max_pos)
    }
}

/// One conversation's row of a step: its stream and the token to feed at `stream.pos`.
pub struct RowInput<'a> {
    pub stream: &'a mut Stream,
    pub token: u32,
}

/// What a row got back: the logits after its token (`[vocab]`).
#[derive(Debug, Clone)]
pub struct RowOutput {
    pub logits: Vec<f32>,
}

/// Rows that each bring their own state.
pub(super) struct Streams<'a, 'b>(pub &'a mut [RowInput<'b>]);

impl RowTargets for Streams<'_, '_> {
    fn attn(&mut self, r: usize) -> (&mut Attn256Cache, usize) {
        let s = &mut *self.0[r].stream;
        (
            s.attn.as_mut().expect("an attention layer has a cache"),
            s.pos,
        )
    }
    fn gdn(&mut self, r: usize) -> &mut Gdn {
        self.0[r]
            .stream
            .gdn
            .as_mut()
            .expect("a DeltaNet layer has state")
    }
}

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

impl Qwen38Omega {
    /// A new conversation's state with a KV cache of `max_pos` positions, set up with this
    /// model's per-layer norms and DeltaNet parameters. Position 0, DeltaNet state zero.
    pub fn new_stream(&self, max_pos: usize) -> Result<Stream, String> {
        let c = &self.config;
        if max_pos == 0 || max_pos > q::MAX_POS {
            return Err(format!(
                "new_stream: {max_pos} positions (1..={})",
                q::MAX_POS
            ));
        }
        let gdn = if self.gdn_params.is_empty() {
            None
        } else {
            let mut g = Gdn::alloc(self.gdn_params.len()).map_err(gpu("stream gdn alloc"))?;
            for (gi, p) in self.gdn_params.iter().enumerate() {
                g.set_layer(gi, &p.conv, &p.a_log, &p.dt_bias, &p.norm)
                    .map_err(gpu("stream gdn set_layer"))?;
            }
            g.reset().map_err(gpu("stream gdn reset"))?;
            Some(g)
        };
        let attn = if self.attn_norms.is_empty() {
            None
        } else {
            let mut a =
                Attn256Cache::alloc(self.attn_norms.len(), max_pos, c.rope_theta, c.rotary_dim)
                    .map_err(gpu("stream attn256 alloc"))?;
            for (ai, (qn, kn)) in self.attn_norms.iter().enumerate() {
                a.set_norms(ai, qn, kn)
                    .map_err(gpu("stream attn256 set_norms"))?;
            }
            Some(a)
        };
        Ok(Stream {
            attn,
            gdn,
            pos: 0,
            max_pos,
        })
    }

    /// Bytes a stream of `max_pos` positions takes for this model ([`Stream::bytes`]).
    pub fn stream_bytes(&self, max_pos: usize) -> usize {
        Stream::bytes(self.attn_norms.len(), self.gdn_params.len(), max_pos)
    }

    /// Makes `s` the model's live state (and the model's own state the stream's) until the
    /// next swap. The decode graph keeps pointing at the model's own objects, so it must not run
    /// while a stream is swapped in; the prefill path and the rows path go through the fields.
    fn swap_state(&mut self, s: &mut Stream) {
        std::mem::swap(&mut self.attn, &mut s.attn);
        std::mem::swap(&mut self.gdn, &mut s.gdn);
        std::mem::swap(&mut self.pos, &mut s.pos);
        std::mem::swap(&mut self.max_pos, &mut s.max_pos);
    }

    /// Feeds `tokens` cold into `s` (state reset, position 0): the logits after the last token.
    /// Prompts of at least [`q::GDN_PF_MIN_T`] tokens go through the chip prefill in chunks of
    /// `chunk`; shorter ones through the exact rows path. Other streams are untouched.
    pub fn stream_prefill(
        &mut self,
        s: &mut Stream,
        tokens: &[u32],
        chunk: usize,
        st: &mut RowStats,
    ) -> Result<(Vec<f32>, PrefillStats), String> {
        if tokens.is_empty() {
            return Err("stream_prefill: empty prompt".into());
        }
        self.swap_state(s);
        let r = self.stream_prefill_inner(tokens, chunk, st);
        self.swap_state(s);
        r
    }

    fn stream_prefill_inner(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        st: &mut RowStats,
    ) -> Result<(Vec<f32>, PrefillStats), String> {
        self.reset()?;
        if tokens.len() >= q::GDN_PF_MIN_T {
            let (_, logits, pf) = self.prefill(tokens, chunk)?;
            Ok((logits, pf))
        } else {
            let h = self.config.hidden;
            let x = self.exact_rows(tokens, None, st)?;
            let last = &x[(tokens.len() - 1) * h..];
            let logits = self.head_logits(last)?;
            Ok((logits, PrefillStats::default()))
        }
    }

    /// One token for every row: feeds `rows[r].token` at `rows[r].stream.pos` through every
    /// layer, the weights streaming once for all rows, and returns each row's logits. Advances
    /// every row's stream by one position. The cascade's step boundary.
    pub fn step_rows(
        &mut self,
        rows: &mut [RowInput<'_>],
        kernel: RowKernel,
        st: &mut RowStats,
    ) -> Result<Vec<RowOutput>, String> {
        let m = rows.len();
        let (h, v) = (self.config.hidden, self.config.vocab);
        let max_rows = match kernel {
            RowKernel::RowPair => aien_omega_fp8::GEMM_MAX_ROWS,
            RowKernel::Tree => aien_omega_fp8::TV_MAX_ROWS,
        };
        if m == 0 || m > max_rows {
            return Err(format!("step_rows: {m} rows (1..={max_rows}, {kernel:?})"));
        }
        for (r, row) in rows.iter().enumerate() {
            if row.token as usize >= v {
                return Err(format!(
                    "step_rows: row {r} token {} outside the vocab {v}",
                    row.token
                ));
            }
            if row.stream.pos >= row.stream.max_pos {
                return Err(format!(
                    "step_rows: row {r} at position {}, its KV cache holds {}",
                    row.stream.pos, row.stream.max_pos
                ));
            }
        }
        let t = Instant::now();
        let mut x: Vec<f32> = rows
            .iter()
            .flat_map(|row| {
                let r = row.token as usize * h;
                self.embed[r..r + h].iter().map(|&b| bf16_to_f32(b))
            })
            .collect();
        let gather = q::gather_table(self.config.rotary_dim);
        st.host_ns += ns(t);
        let eps = self.config.eps;
        let tree = TreeGemmOn::set(kernel == RowKernel::Tree);
        let mut targets = Streams(rows);
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..self.config.layers {
            let lt = &self.layers[l];
            let (in_norm, post_norm) = &self.norms[l];
            let mixer = if self.config.full_attention[l] {
                ai += 1;
                Mixer::Attn {
                    idx: ai - 1,
                    gather: &gather,
                }
            } else {
                gi += 1;
                Mixer::Gdn {
                    idx: gi - 1,
                    ab: lt.ab.as_ref().expect("a DeltaNet layer has a|b"),
                    keep: None,
                }
            };
            layer_rows(
                lt,
                in_norm,
                post_norm,
                mixer,
                &mut targets,
                m,
                eps,
                &mut x,
                st,
            )
            .map_err(|e| format!("layer {l}: {e}"))?;
        }
        drop(tree);
        for row in targets.0.iter_mut() {
            row.stream.pos += 1;
        }
        st.rows += m as u64;
        st.passes += 1;
        if let (RowKernel::Tree, Some(h8)) = (kernel, self.lm_head8.as_ref()) {
            // One read of the FP8 head for every row (decode-graph head rule: final norm, then W . x).
            let t = Instant::now();
            let mut hidden = vec![0f32; m * h];
            for (o, r) in hidden.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
                rmsnorm(r, &self.final_norm, eps, o);
            }
            let mut logits = vec![0f32; m * v];
            h8.tv_f32(m, &hidden, &mut logits)
                .map_err(|e| format!("lm_head fp8 tree-verify gemm x{m}: {e}"))?;
            st.head_ns += ns(t);
            st.head_calls += 1;
            return Ok(logits
                .chunks_exact(v)
                .map(|l| RowOutput { logits: l.to_vec() })
                .collect());
        }
        let mut out = Vec::with_capacity(m);
        for r in 0..m {
            let t = Instant::now();
            let logits = self.head_logits(&x[r * h..(r + 1) * h])?;
            st.head_ns += ns(t);
            st.head_calls += 1;
            out.push(RowOutput { logits });
        }
        Ok(out)
    }
}
