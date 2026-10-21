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
    /// The exact two-pass head's Row coefficients (HUM L314), when the screen was built: every row
    /// graph and the stroke graph screen with the door's `head_screen`.
    pub head_coef: Option<Vec<f32>>,
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

/// The merge train's switches for one row of a [`Qwen38Omega::step_rows_test`] call (L384):
/// which path this row's token takes this step. The default is the plain served path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowTest {
    /// Ride a stroke with the other plain rows (`false`: this row's own one-row graph).
    pub stroke: bool,
    /// The exact two-pass head for this row's greedy token (`false`: the plain head).
    pub exact_head: bool,
    /// `AIEN_Q36_TEST_MUTANT` number for this row's step only (`0`: none). Set on the row's
    /// own graph before its token is recorded and cleared after: the command list is recorded
    /// afresh every token, so the mutant kernels are chosen for this step and no other.
    pub mutant: i32,
}

impl Default for RowTest {
    fn default() -> Self {
        Self {
            stroke: true,
            exact_head: true,
            mutant: 0,
        }
    }
}

impl RowTest {
    /// The served path: nothing to apply.
    pub fn is_plain(&self) -> bool {
        *self == Self::default()
    }
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
            // L312 v2 (the batch): every layer recorded for all heads at once, one weight read per layer;
            // its final reads the lm_head once per group of heads (the exact head is left out there).
            let batch = batch_on();
            if batch {
                g.set_batch(true)
                    .map_err(|rc| format!("rows stroke set_batch: rc {rc}"))?;
            }
            eprintln!(
                "qwen36 omega: rows ride strokes of up to {} ({}; AIEN_Q36_STROKE=0 rows in turn)",
                q::MAX_HEADS,
                if batch {
                    "batch v2, AIEN_Q36_STROKE_V2=0 for v1"
                } else {
                    "v1 (AIEN_Q36_STROKE_V2=0)"
                }
            );
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
        if let (Some(screen), Some(coef)) = (
            self.head_screen.as_ref(),
            self.row_src.as_ref().and_then(|s| s.head_coef.as_ref()),
        ) {
            // SAFETY: the screen lives in `self.head_screen`, which outlives the rows and the stroke
            // graph (field order); the coefficients are copied.
            unsafe { graph.set_exact_head(screen.as_raw(), coef) }
                .map_err(gpu("row q36 graph set_exact_head"))?;
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

    /// [`Self::step_rows`] with the merge train's switches per row (`tests[i]` goes with
    /// `feed[i]`; shorter or empty = plain). Plain rows ride strokes together as in `step_rows`;
    /// a row that turns strokes off, turns the exact head off or asks for a mutant takes its own
    /// one-row graph for this step with those settings, so no other row's bits can change.
    /// Outputs are in feed order.
    pub fn step_rows_test(
        &mut self,
        feed: &[RowFeed],
        tests: &[RowTest],
    ) -> Result<Vec<RowOut>, String> {
        if tests.iter().all(RowTest::is_plain) {
            return self.step_rows(feed);
        }
        self.check_feed(feed)?;
        self.row_steps += 1;
        let test = |i: usize| tests.get(i).copied().unwrap_or_default();
        let plain: Vec<usize> = (0..feed.len()).filter(|&i| test(i).is_plain()).collect();
        let mut outs: Vec<Option<RowOut>> = vec![None; feed.len()];
        if !plain.is_empty() {
            let sub: Vec<RowFeed> = plain.iter().map(|&i| feed[i]).collect();
            let mut got = Vec::with_capacity(sub.len());
            if sub.len() >= 2 && self.row_stroke.is_some() {
                let mut sg = self.row_stroke.take().expect("checked above");
                let r = self.stroke_chunks(&mut sg, &sub, &mut got);
                self.row_stroke = Some(sg);
                r?;
            } else {
                for f in &sub {
                    got.push(self.row_token(f)?);
                }
            }
            for (i, o) in plain.into_iter().zip(got) {
                outs[i] = Some(o);
            }
        }
        for (i, f) in feed.iter().enumerate() {
            let t = test(i);
            if t.is_plain() {
                continue;
            }
            outs[i] = Some(self.row_token_test(f, t)?);
        }
        Ok(outs
            .into_iter()
            .map(|o| o.expect("every row answered"))
            .collect())
    }

    /// One row's token through its own graph with the row's switches: the mutant set for this
    /// record only (the graph's standing mutant, if the process has one, is put back after).
    fn row_token_test(&mut self, f: &RowFeed, t: RowTest) -> Result<RowOut, String> {
        let standing = self.row_src.as_ref().and_then(|s| s.mutant).unwrap_or(0);
        if t.mutant != 0 {
            self.rows[f.row].graph.test_set_mutant(t.mutant);
        }
        let r = self.row_token_with_head(f, t.exact_head);
        if t.mutant != 0 {
            self.rows[f.row].graph.test_set_mutant(standing);
        }
        r
    }

    /// Argument checks shared by [`Self::step_rows`] and [`Self::step_rows_test`].
    fn check_feed(&self, feed: &[RowFeed]) -> Result<(), String> {
        let v = self.config.vocab;
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
        Ok(())
    }

    /// One token into each listed row (each row at most once): each row's greedy token, and its
    /// logits when asked. On an error no position has moved past the failing row.
    pub fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        let v = self.config.vocab;
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
        // Strokes (L312) for two rows or more; a lone row takes its own one-row graph (with the exact
        // head): one head costs more in a stroke (L312: N=1 v2 27.3 ms vs 24.4 ms v1).
        if feed.len() >= 2 && self.row_stroke.is_some() {
            let mut sg = self.row_stroke.take().expect("checked above");
            let r = self.stroke_chunks(&mut sg, feed, &mut outs);
            self.row_stroke = Some(sg);
            r?;
            return Ok(outs);
        }
        for f in feed {
            outs.push(self.row_token(f)?);
        }
        Ok(outs)
    }

    /// Up to MAX_HEADS rows per command list, the head slots pointed at them; a chunk of one row
    /// takes the row's own graph.
    fn stroke_chunks(
        &mut self,
        sg: &mut Q36Graph,
        feed: &[RowFeed],
        outs: &mut Vec<RowOut>,
    ) -> Result<(), String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
        {
            for chunk in feed.chunks(q::MAX_HEADS) {
                if let [f] = chunk {
                    outs.push(self.row_token(f)?);
                    continue;
                }
                // The exact head (when built) makes off-candidate logits -inf: on for the whole stroke only
                // when no row in it reads logits (sampled rows need every one).
                sg.set_exact_head_on(!chunk.iter().any(|f| f.logits));
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
        }
        Ok(())
    }

    /// One row's token through its own one-row graph.
    fn row_token(&mut self, f: &RowFeed) -> Result<RowOut, String> {
        self.row_token_with_head(f, true)
    }

    /// [`Self::row_token`]; `exact_head` false keeps the plain head for this token (the merge
    /// train's `exact_head: false` arm; the words must not change).
    fn row_token_with_head(&mut self, f: &RowFeed, exact_head: bool) -> Result<RowOut, String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
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
        row.graph.set_exact_head_on(exact_head && !f.logits);
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
        Ok(RowOut {
            token,
            logits,
            stats,
        })
    }
}

impl Qwen38Omega {
    /// L369: `tokens` (a prompt chunk, at most [`q::PREFILL_MAX_ROWS`]) into row `r` at its next
    /// positions through the chip's prefill: the state ends bit for bit as `tokens.len()` steps of
    /// [`Self::step_rows`] would leave it. The last token's greedy token, and its logits when asked
    /// (then the exact head is off for that token, as in a stroke with a row reading logits).
    pub fn prefill_row(&mut self, r: usize, tokens: &[u32], logits: bool) -> Result<RowOut, String> {
        let (v, h) = (self.config.vocab, self.config.hidden);
        let n = tokens.len();
        if r >= self.rows.len() {
            return Err(format!("prefill_row: row {r} of {}", self.rows.len()));
        }
        if n == 0 || n > q::PREFILL_MAX_ROWS {
            return Err(format!("prefill_row: {n} tokens (1..={})", q::PREFILL_MAX_ROWS));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= v) {
            return Err(format!("prefill_row: token {t} outside the vocab {v}"));
        }
        let (pos, max_pos) = (self.rows[r].pos, self.rows[r].max_pos);
        if pos + n > max_pos {
            return Err(format!("prefill_row: row {r} holds {max_pos} positions, {pos} + {n} asked"));
        }
        let mut x = Vec::with_capacity(n * h);
        for &t in tokens {
            let t = t as usize;
            x.extend(self.embed[t * h..(t + 1) * h].iter().map(|&b| bf16_to_f32(b)));
        }
        // The stroke graph's head slot 0 pointed at the row (its prefill buffers serve every row), or
        // the row's own graph when the strokes are off.
        let row = &mut self.rows[r];
        let g: &mut Q36Graph = match self.row_stroke.as_mut() {
            Some(sg) => {
                let (gd, a) = (
                    row.gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                    row.attn.as_ref().map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                );
                // SAFETY: the row's state lives in `self.rows`, which outlives the stroke graph.
                unsafe { sg.set_head_state(0, gd, a) }.map_err(gpu("prefill_row set_head_state"))?;
                sg
            }
            None => &mut row.graph,
        };
        if g.prefill_cap() < n {
            g.prefill_alloc(q::PREFILL_MAX_ROWS).map_err(gpu("prefill_row alloc"))?;
        }
        g.set_exact_head_on(!logits);
        let (token, stats) = g.prefill(0, &x, pos, true).map_err(gpu("prefill_row"))?;
        let out_logits = if logits {
            let mut l = vec![0f32; v];
            g.read_head(0, None, None, Some(&mut l)).map_err(gpu("prefill_row read logits"))?;
            Some(l)
        } else {
            None
        };
        self.rows[r].pos += n;
        Ok(RowOut {
            token: token.ok_or("prefill_row: no token")?,
            logits: out_logits,
            stats,
        })
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

fn batch_on() -> bool {
    std::env::var("AIEN_Q36_STROKE_V2").as_deref() != Ok("0")
}
