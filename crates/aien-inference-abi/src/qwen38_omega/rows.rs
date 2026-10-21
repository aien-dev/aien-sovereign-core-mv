//! Many conversations on one 35B upload (sc L354, server batching): each row is one conversation's
//! own state (DeltaNet tails and states, the attention KV cache, its position) with its own
//! Qwen3.6-35B-A3B decode graph pointed at the SAME resident weights, router and lm_head.
//!
//! [`Qwen38Omega::step_rows`] feeds one token into each listed row and returns each row's greedy
//! token and logits. With L312's strokes linked it feeds up to `MAX_HEADS` rows per command
//! list (one stroke graph whose head slots are re-pointed at each chunk's rows, nothing copied);
//! `AIEN_Q36_STROKE=0` runs each row's own one-row graph in turn (the first cut). Either way every
//! row's bits must be exactly what that conversation gets alone (the gate checks it).
//!
//! Negative control (TEST ONLY): `AIEN_Q36_TEST_ROW_SWAP=<k>` swaps the states of the first two
//! fed rows once, at the first multi-row step from the `k`th on (counted from 1) that feeds two or
//! more rows, so two conversations continue
//! from each other's state: the per-row identity gate must go red.
use aien_omega_q38::q36::{self as q, Q36Config, Q36Graph};
use aien_omega_q38::{Attn256Cache, Gdn, GraphStats};

use super::{gpu, Qwen38Omega, Qwen38Snapshot};
use crate::qwen3_moe::bf16_to_f32;

/// One DeltaNet layer's parameters as the upload set them (host copy, for each new row's state).
pub(super) struct GdnParams {
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
}

/// What a new row needs besides the shared device tensors (the 35B only).
pub(super) struct RowSource {
    pub layers: usize,
    pub attn_mask: u64,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub gdn_params: Vec<GdnParams>,
    /// Every attention layer's (q_norm, k_norm) as the checkpoint stores them.
    pub attn_norms: Vec<(Vec<f32>, Vec<f32>)>,
    pub mutant: Option<i32>,
}

/// One conversation's state and graph. Field order is drop order: the graph before the state it
/// points at.
pub(super) struct Q36Row {
    graph: Q36Graph,
    gdn: Option<Gdn>,
    #[allow(dead_code)] // owned only: the graph points at it
    attn: Option<Attn256Cache>,
    pos: usize,
    max_pos: usize,
}

/// One row's share of a [`Qwen38Omega::step_rows`] call.
#[derive(Debug, Clone, Copy)]
pub struct RowFeed {
    /// The row ([`Qwen38Omega::add_rows`]).
    pub row: usize,
    /// The token it takes at its next position.
    pub token: u32,
    /// Copy the row's logits back (a row still feeding its prompt needs only the last ones).
    pub logits: bool,
}

/// What one row got from a step.
#[derive(Debug, Clone)]
pub struct RowOut {
    /// The graph's greedy token.
    pub token: u32,
    /// `[vocab]` when asked for.
    pub logits: Option<Vec<f32>>,
    pub stats: GraphStats,
}

impl Qwen38Omega {
    /// Makes `n` more conversation rows of `max_pos` positions each (rounded up to whole
    /// 256-position passes) over this upload's weights; returns the first new row's index.
    /// The 35B only. Rows start empty (position 0).
    pub fn add_rows(&mut self, n: usize, max_pos: usize) -> Result<usize, String> {
        let first = self.rows.len();
        let src = self
            .row_src
            .as_ref()
            .ok_or("rows: the 35B (qwen3_5_moe) only")?;
        let max_pos = max_pos.next_multiple_of(256);
        let router = self.router.as_ref().ok_or("rows: no router")?.as_raw();
        let lm_head = self.lm_head.as_raw();
        for _ in 0..n {
            let mut gdn = if src.gdn_params.is_empty() {
                None
            } else {
                Some(
                    Gdn::alloc_ratio(src.gdn_params.len(), q::GDN_RATIO)
                        .map_err(gpu("row gdn alloc"))?,
                )
            };
            if let Some(g) = gdn.as_mut() {
                for (i, p) in src.gdn_params.iter().enumerate() {
                    g.set_layer(i, &p.conv, &p.a_log, &p.dt_bias, &p.norm)
                        .map_err(gpu("row gdn set_layer"))?;
                }
            }
            let mut attn = if src.attn_norms.is_empty() {
                None
            } else {
                Some(
                    Attn256Cache::alloc_shape(
                        src.attn_norms.len(),
                        max_pos,
                        src.rope_theta,
                        src.rotary_dim,
                        q::Q_HEADS,
                        q::KV_HEADS,
                        q::Q_PER_CTA,
                    )
                    .map_err(gpu("row attn256 alloc"))?,
                )
            };
            if let Some(a) = attn.as_mut() {
                for (i, (qn, kn)) in src.attn_norms.iter().enumerate() {
                    a.set_norms(i, qn, kn)
                        .map_err(gpu("row attn256 set_norms"))?;
                }
            }
            let cfg = Q36Config {
                layers: src.layers as u32,
                attn_mask: src.attn_mask,
                vocab: src.vocab as u32,
                eps: src.eps,
                eps_l2: super::EPS_L2,
                gdn: gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                attn: attn
                    .as_ref()
                    .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                router,
                lm_head,
            };
            // SAFETY: the state handles live in the row (dropped after its graph); the router and
            // lm_head live in `self`, which drops `rows` first (field order).
            let mut graph =
                unsafe { Q36Graph::create(&cfg) }.map_err(gpu("row q36 graph create"))?;
            self.wire(&mut graph)?;
            self.rows.push(Q36Row {
                graph,
                gdn,
                attn,
                pos: 0,
                max_pos,
            });
        }
        // The strokes' graph (L312): MAX_HEADS head slots, pointed at each chunk's rows per step.
        if self.row_stroke.is_none() && stroke_on() && !self.rows.is_empty() {
            let r0 = &self.rows[0];
            let cfg = Q36Config {
                layers: src.layers as u32,
                attn_mask: src.attn_mask,
                vocab: src.vocab as u32,
                eps: src.eps,
                eps_l2: super::EPS_L2,
                gdn: r0.gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                attn: r0
                    .attn
                    .as_ref()
                    .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                router,
                lm_head,
            };
            // SAFETY: as the rows' graphs; `row_stroke` drops before `rows` (field order).
            let mut g = unsafe { Q36Graph::create_heads(&cfg, q::MAX_HEADS) }
                .map_err(gpu("rows stroke graph create"))?;
            self.wire(&mut g)?;
            self.row_stroke = Some(g);
        }
        Ok(first)
    }

    /// Points a new graph at this upload's layers and final norm (and the test mutant).
    fn wire(&self, graph: &mut Q36Graph) -> Result<(), String> {
        for (l, (t, (in_norm, _))) in self.layers.iter().zip(&self.norms).enumerate() {
            let ab = t.ab.as_ref().map_or(std::ptr::null_mut(), |b| b.as_raw());
            // SAFETY: the tensors live in `self.layers`, which outlive the rows and the stroke
            // graph (field order).
            unsafe {
                graph.set_layer(
                    l,
                    t.mix.as_raw(),
                    ab,
                    t.o.as_raw(),
                    t.gate_up.as_raw(),
                    t.down.as_raw(),
                    in_norm,
                )
            }
            .map_err(gpu("row q36 graph set_layer"))?;
        }
        graph
            .set_final_norm(&self.final_norm)
            .map_err(gpu("row q36 graph set_final_norm"))?;
        if let Some(m) = self.row_src.as_ref().and_then(|s| s.mutant) {
            graph.test_set_mutant(m);
        }
        Ok(())
    }

    /// Rows made so far.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Device bytes one row's state takes on paper at `max_pos` (DeltaNet state and tails, KV).
    pub fn row_bytes(&self, max_pos: usize) -> usize {
        let Some(src) = self.row_src.as_ref() else {
            return 0;
        };
        let max_pos = max_pos.next_multiple_of(256);
        let gdn = self.gdn.as_ref().map_or(0, |g| {
            src.gdn_params.len() * (g.tail_floats() + g.state_floats()) * 4
        });
        gdn + src.attn_norms.len() * Attn256Cache::layer_bytes(max_pos)
    }

    /// Empties row `r` for a new conversation (DeltaNet state to zero, position 0; KV rows are
    /// overwritten as the new conversation reaches them).
    pub fn row_reset(&mut self, r: usize) -> Result<(), String> {
        let n = self.rows.len();
        let row = self
            .rows
            .get_mut(r)
            .ok_or_else(|| format!("row {r} of {n}"))?;
        if let Some(g) = row.gdn.as_mut() {
            g.reset().map_err(gpu("row gdn reset"))?;
        }
        row.pos = 0;
        Ok(())
    }

    /// Copies row `r`'s DeltaNet tails and states and its position into `snap` (host memory,
    /// reused between calls), as [`Self::snapshot`] does for the main state.
    pub fn row_snapshot(&mut self, r: usize, snap: &mut Qwen38Snapshot) -> Result<(), String> {
        let n = self.rows.len();
        let row = self
            .rows
            .get_mut(r)
            .ok_or_else(|| format!("row {r} of {n}"))?;
        snap.pos = row.pos;
        let Some(g) = row.gdn.as_mut() else {
            return Ok(());
        };
        let (tf, sf) = (g.tail_floats(), g.state_floats());
        snap.tails.resize(g.layers() * tf, 0.0);
        snap.states.resize(g.layers() * sf, 0.0);
        for (l, (t, st)) in snap
            .tails
            .chunks_exact_mut(tf)
            .zip(snap.states.chunks_exact_mut(sf))
            .enumerate()
        {
            g.get_state(l, t, st).map_err(gpu("row gdn get_state"))?;
        }
        Ok(())
    }

    /// Puts a [`Self::row_snapshot`] back into row `r` (any row: the copy is plain host memory).
    pub fn row_restore(&mut self, r: usize, snap: &Qwen38Snapshot) -> Result<(), String> {
        let n = self.rows.len();
        let row = self
            .rows
            .get_mut(r)
            .ok_or_else(|| format!("row {r} of {n}"))?;
        if snap.pos > row.max_pos {
            return Err(format!(
                "row_restore: position {} past the row's {}",
                snap.pos, row.max_pos
            ));
        }
        if let Some(g) = row.gdn.as_mut() {
            let (tf, sf) = (g.tail_floats(), g.state_floats());
            if snap.tails.len() != g.layers() * tf || snap.states.len() != g.layers() * sf {
                return Err("row_restore: the snapshot is not of this model".into());
            }
            for (l, (t, st)) in snap
                .tails
                .chunks_exact(tf)
                .zip(snap.states.chunks_exact(sf))
                .enumerate()
            {
                g.set_state(l, t, st).map_err(gpu("row gdn set_state"))?;
            }
        }
        row.pos = snap.pos;
        Ok(())
    }

    /// Tokens row `r` holds (its next position) and the most it holds.
    pub fn row_pos(&self, r: usize) -> Option<(usize, usize)> {
        self.rows.get(r).map(|row| (row.pos, row.max_pos))
    }

    /// One token into each listed row (each row at most once): each row's greedy token, and its
    /// logits when asked. On an error no position has moved past the failing row.
    pub fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
        for (i, f) in feed.iter().enumerate() {
            if f.row >= self.rows.len() {
                return Err(format!("step_rows: row {} of {}", f.row, self.rows.len()));
            }
            if feed[..i].iter().any(|g| g.row == f.row) {
                return Err(format!("step_rows: row {} fed twice", f.row));
            }
            if f.token as usize >= v {
                return Err(format!(
                    "step_rows: token {} outside the vocab {v}",
                    f.token
                ));
            }
        }
        self.row_steps += 1;
        if feed.len() >= 2
            && row_swap_step().is_some_and(|k| self.row_steps >= k)
            && !ROW_SWAPPED.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            eprintln!(
                "qwen36 omega: TEST ROW SWAP at step {}: rows {} and {} trade state (negative control; never serve this)",
                self.row_steps, feed[0].row, feed[1].row
            );
            let (a, b) = (feed[0].row.min(feed[1].row), feed[0].row.max(feed[1].row));
            let (lo, hi) = self.rows.split_at_mut(b);
            std::mem::swap(&mut lo[a], &mut hi[0]);
        }
        for f in feed {
            let row = &self.rows[f.row];
            if row.pos >= row.max_pos {
                return Err(format!(
                    "step_rows: row {} holds {} positions",
                    f.row, row.max_pos
                ));
            }
        }
        let mut outs = Vec::with_capacity(feed.len());
        // Strokes (L312): up to MAX_HEADS rows per command list, the head slots pointed at them.
        if let Some(sg) = self.row_stroke.as_mut() {
            for chunk in feed.chunks(q::MAX_HEADS) {
                let mut x = Vec::with_capacity(chunk.len() * h);
                let mut pos = Vec::with_capacity(chunk.len());
                for (hd, f) in chunk.iter().enumerate() {
                    let t = f.token as usize;
                    x.extend(
                        self.embed[t * h..(t + 1) * h]
                            .iter()
                            .map(|&b| bf16_to_f32(b)),
                    );
                    let row = &self.rows[f.row];
                    let (g, a) = (
                        row.gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                        row.attn
                            .as_ref()
                            .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                    );
                    // SAFETY: the row's state lives in `self.rows`, which outlives the stroke
                    // graph (field order); a re-point copies nothing.
                    unsafe { sg.set_head_state(hd, g, a) }
                        .map_err(gpu("rows stroke set_head_state"))?;
                    pos.push(row.pos);
                }
                let (tokens, stats) = sg.stroke(&x, &pos).map_err(gpu("rows stroke"))?;
                for (hd, (f, token)) in chunk.iter().zip(tokens).enumerate() {
                    self.rows[f.row].pos += 1;
                    let logits = if f.logits {
                        let mut l = vec![0f32; v];
                        sg.read_head(hd, None, None, Some(&mut l))
                            .map_err(gpu("rows stroke read logits"))?;
                        Some(l)
                    } else {
                        None
                    };
                    outs.push(RowOut {
                        token,
                        logits,
                        // The chip time is the whole stroke's: counted once, on its first head.
                        stats: if hd == 0 {
                            stats
                        } else {
                            GraphStats::default()
                        },
                    });
                }
            }
            return Ok(outs);
        }
        for f in feed {
            let t = f.token as usize;
            let x: Vec<f32> = self.embed[t * h..(t + 1) * h]
                .iter()
                .map(|&b| bf16_to_f32(b))
                .collect();
            let row = &mut self.rows[f.row];
            if row.pos >= row.max_pos {
                return Err(format!(
                    "step_rows: row {} holds {} positions",
                    f.row, row.max_pos
                ));
            }
            let (token, stats) = row.graph.token(&x, row.pos).map_err(gpu("row q36 token"))?;
            row.pos += 1;
            let logits = if f.logits {
                let mut l = vec![0f32; v];
                row.graph
                    .read(None, None, Some(&mut l))
                    .map_err(gpu("row q36 read logits"))?;
                Some(l)
            } else {
                None
            };
            outs.push(RowOut {
                token,
                logits,
                stats,
            });
        }
        Ok(outs)
    }
}

/// The row-swap negative control fires once per process.
static ROW_SWAPPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn row_swap_step() -> Option<u64> {
    std::env::var("AIEN_Q36_TEST_ROW_SWAP")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
}

/// Strokes for the rows (`AIEN_Q36_STROKE=0`: each row's own one-row graph, in turn).
fn stroke_on() -> bool {
    std::env::var("AIEN_Q36_STROKE").as_deref() != Ok("0")
}
