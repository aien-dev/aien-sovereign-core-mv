//! The DFlash 2 drafter as a second draft source on the exact verify path (cascade/dflash-drafter).
//!
//! The drafter maths is [`crate::qwen38_dflash`] (host reference, `docs/qwen38/DFLASH2.md`). What
//! this module adds is the 27B side:
//! - [`Qwen38Omega::exact_rows_taps`]: the exact rows pass that also returns the residual stream
//!   after the drafter's tap layers (5, 19, 33, 47, 61) for every row. Same chip calls in the same
//!   order as [`Qwen38Omega::exact_rows`], so the tokens and the residual rows are the same bits.
//! - [`ChipHead`]: the 27B's bf16 lm_head on the chip as the drafter's word-scoring head.
//! - [`capture`]: greedy generation one token at a time with every position's taps recorded, and
//!   [`replay_rounds`]: the DFlash round loop run over that record. Greedy verify accepts the
//!   longest prefix of drafts equal to the model's own tokens, and the exact path gives every row
//!   the same bits whether it rides in a pass of 1 or of 8, so the rounds over the record are the
//!   rounds a live run makes (checked against [`generate`] in the tool).
//! - [`generate`]: the live loop, the drafter in the place of the MTP chain: one verify pass over
//!   the anchor and 7 drafts, acceptance, the DeltaNet state put back and the accepted rows
//!   replayed when a draft is refused, the accepted rows' taps into the drafter's cache.
use std::time::Instant;

use aien_omega_q38 as q;

use super::spec::{GdnRows, Mixer, RowStats};
use super::{gpu, Qwen38Omega, Qwen38Snapshot};
use crate::qwen38_dflash::{accepted, DraftCache, DraftMutant, Drafter, Head};

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

impl Qwen38Omega {
    /// [`Self::exact_rows`] plus, for every row, the residual stream after each layer in `taps`
    /// (`[m][taps.len()][hidden]`).
    pub fn exact_rows_taps(
        &mut self,
        tokens: &[u32],
        mut keep: Option<&mut Vec<GdnRows>>,
        taps: &[usize],
        st: &mut RowStats,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let m = tokens.len();
        let (h, v) = (self.config.hidden, self.config.vocab);
        if m == 0 || m > aien_omega_fp8::GEMM_MAX_ROWS {
            return Err(format!("exact_rows_taps: {m} rows"));
        }
        if self.pos + m > self.max_pos {
            return Err(format!(
                "exact_rows_taps: positions {} + {m} > KV cache of {}",
                self.pos, self.max_pos
            ));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= v) {
            return Err(format!("exact_rows_taps: token {t} outside the vocab {v}"));
        }
        if let Some(&l) = taps.iter().find(|&&l| l >= self.config.layers) {
            return Err(format!(
                "exact_rows_taps: tap layer {l} of {}",
                self.config.layers
            ));
        }
        let t = Instant::now();
        let mut x: Vec<f32> = Vec::with_capacity(m * h);
        for &tk in tokens {
            x.extend(self.embed_row(tk));
        }
        let gather = q::gather_table(self.config.rotary_dim);
        st.host_ns += ns(t);
        let n_gdn = self.ab_mm.len();
        if let Some(k) = keep.as_deref_mut() {
            k.resize_with(n_gdn, GdnRows::default);
        }
        let tw = taps.len() * h;
        let mut out_taps = vec![0f32; m * tw];
        let (pos0, eps) = (self.pos, self.config.eps);
        let (mut gi, mut ai) = (0usize, 0usize);
        for l in 0..self.config.layers {
            let lt = &self.layers[l];
            let (in_norm, post_norm) = &self.norms[l];
            let mixer = if self.config.full_attention[l] {
                ai += 1;
                Mixer::Attn {
                    cache: self.attn.as_mut().expect("an attention layer has a cache"),
                    idx: ai - 1,
                    gather: &gather,
                }
            } else {
                gi += 1;
                Mixer::Gdn {
                    gdn: self.gdn.as_mut().expect("a DeltaNet layer has state"),
                    idx: gi - 1,
                    ab: lt.ab.as_ref().expect("a DeltaNet layer has a|b"),
                    keep: keep.as_deref_mut().map(|k| &mut k[gi - 1]),
                }
            };
            super::spec::layer_rows(lt, in_norm, post_norm, mixer, pos0, m, eps, &mut x, st)
                .map_err(|e| format!("layer {l}: {e}"))?;
            for (ti, _) in taps.iter().enumerate().filter(|(_, &tl)| tl == l) {
                for r in 0..m {
                    out_taps[r * tw + ti * h..r * tw + (ti + 1) * h]
                        .copy_from_slice(&x[r * h..(r + 1) * h]);
                }
            }
        }
        self.pos += m;
        st.rows += m as u64;
        st.passes += 1;
        Ok((x, out_taps))
    }

    /// The 27B's embedding row of `token` (the drafter has no table of its own).
    pub fn embedding(&self, token: u32) -> Vec<f32> {
        self.embed_row(token)
    }
}

/// The 27B's lm_head on the chip (bf16 GEMV) scoring the drafter's normed rows: no final norm
/// of the 27B, the drafter's own `norm` has run.
pub struct ChipHead<'a> {
    pub model: &'a mut Qwen38Omega,
    pub calls: u64,
    pub ns: u64,
}

impl Head for ChipHead<'_> {
    fn logits(&mut self, h: &[f32]) -> Result<Vec<f32>, String> {
        let t = Instant::now();
        let mut logits = vec![0f32; self.model.config.vocab];
        self.model
            .lm_head
            .gemv_f32(h, &mut logits)
            .map_err(gpu("dflash lm_head"))?;
        self.calls += 1;
        self.ns += ns(t);
        Ok(logits)
    }
}

/// A greedy generation with every position's taps.
#[derive(Debug, Clone, Default)]
pub struct Capture {
    /// Prompt then generated tokens.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    /// `[position][taps][hidden]` for positions `0 .. tokens.len() - 1` (the last token was
    /// never run through the model).
    pub taps: Vec<f32>,
    pub tap_width: usize,
    pub main: RowStats,
}

impl Capture {
    pub fn generated(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }
}

/// Greedy generation on the exact path one token at a time (`k = 0`), taps recorded.
pub fn capture(
    model: &mut Qwen38Omega,
    taps: &[usize],
    prompt: &[u32],
    n: usize,
    stop: &[u32],
    chunk: usize,
) -> Result<Capture, String> {
    if prompt.is_empty() || n == 0 {
        return Err("capture: empty prompt or n = 0".into());
    }
    let h = model.config.hidden;
    let mut cap = Capture {
        tap_width: taps.len() * h,
        prompt_len: prompt.len(),
        tokens: prompt.to_vec(),
        ..Capture::default()
    };
    model.reset()?;
    let mut last = Vec::new();
    for c in prompt.chunks(chunk.max(1)) {
        let (x, tp) = model.exact_rows_taps(c, None, taps, &mut cap.main)?;
        cap.taps.extend_from_slice(&tp);
        last = x[(c.len() - 1) * h..].to_vec();
    }
    let mut t = model.head_token(&last, &mut cap.main)?;
    cap.tokens.push(t);
    while cap.tokens.len() - prompt.len() < n && !stop.contains(&t) {
        let (x, tp) = model.exact_rows_taps(&[t], None, taps, &mut cap.main)?;
        cap.taps.extend_from_slice(&tp);
        t = model.head_token(&x, &mut cap.main)?;
        cap.tokens.push(t);
    }
    Ok(cap)
}

/// What a DFlash round loop did.
#[derive(Debug, Clone, Default)]
pub struct DflashRun {
    pub tokens: Vec<u32>,
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Accepted drafts per round, `[0 ..= block - 1]`.
    pub accepted_hist: Vec<u64>,
    /// `(start position, accepted)` of every round.
    pub per_round: Vec<(usize, usize)>,
    pub rollbacks: u64,
    pub decode_ns: u64,
    pub prompt_ns: u64,
    pub draft_ns: u64,
    pub context_ns: u64,
    pub verify_ns: u64,
    pub snapshot_ns: u64,
    pub rollback_ns: u64,
    pub head_calls: u64,
    pub head_ns: u64,
    pub main: RowStats,
}

impl DflashRun {
    /// Words per verify pass: the accepted drafts plus the model's own word, per round.
    pub fn words_per_pass(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.accepted + self.rounds) as f64 / self.rounds as f64
        }
    }
}

/// The DFlash round loop over a [`Capture`] (no 27B pass; the 27B's lm_head and embedding are
/// still used for the drafts). Round at `s`: the anchor is `tokens[s]`, the context is the taps of
/// positions `0 .. s`, the drafts are compared with `tokens[s + 1 ..]`.
pub fn replay_rounds(
    drafter: &Drafter,
    model: &mut Qwen38Omega,
    cap: &Capture,
    mutant: DraftMutant,
) -> Result<DflashRun, String> {
    let c = &drafter.cfg;
    let tw = c.tap_width();
    if cap.tap_width != tw {
        return Err(format!(
            "replay: capture rows of {}, drafter wants {tw}",
            cap.tap_width
        ));
    }
    let total = cap.tokens.len();
    let mut run = DflashRun {
        accepted_hist: vec![0; c.block],
        ..DflashRun::default()
    };
    let mask = model.embedding(c.mask_token);
    let mut cache = DraftCache::new(c.layers);
    let mut s = cap.prompt_len;
    let t1 = Instant::now();
    while s + 1 < total {
        let tc = Instant::now();
        let from = cache.len;
        drafter.add_context(&mut cache, &cap.taps[from * tw..s * tw], mutant)?;
        run.context_ns += ns(tc);
        let anchor = cap.tokens[s];
        let emb = model.embedding(anchor);
        let td = Instant::now();
        let mut head = ChipHead {
            model: &mut *model,
            calls: 0,
            ns: 0,
        };
        let (drafts, _) = drafter.draft(&cache, anchor, &emb, &mask, &mut head, mutant)?;
        run.head_calls += head.calls;
        run.head_ns += head.ns;
        run.draft_ns += ns(td);
        let truth = &cap.tokens[s + 1..];
        let room = truth.len().min(drafts.len());
        let a = accepted(&drafts[..room], truth);
        run.rounds += 1;
        run.drafted += room as u64;
        run.accepted += a as u64;
        run.accepted_hist[a] += 1;
        run.per_round.push((s, a));
        s += a + 1;
    }
    run.decode_ns = ns(t1);
    run.tokens = cap.generated().to_vec();
    Ok(run)
}

/// Live greedy generation with the drafter: one exact verify pass per round over the anchor and
/// its drafts. The tokens equal the one-at-a-time path's whatever the drafts were.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut Qwen38Omega,
    drafter: &Drafter,
    prompt: &[u32],
    n: usize,
    stop: &[u32],
    chunk: usize,
    mutant: DraftMutant,
    no_rollback: bool,
) -> Result<DflashRun, String> {
    if prompt.is_empty() || n == 0 {
        return Err("dflash generate: empty prompt or n = 0".into());
    }
    let c = &drafter.cfg;
    let (h, tw) = (model.config.hidden, c.tap_width());
    let mut run = DflashRun {
        accepted_hist: vec![0; c.block],
        ..DflashRun::default()
    };
    let mask = model.embedding(c.mask_token);
    let mut cache = DraftCache::new(c.layers);
    model.reset()?;
    let t0 = Instant::now();
    let mut last = Vec::new();
    for ch in prompt.chunks(chunk.max(1)) {
        let (x, tp) = model.exact_rows_taps(ch, None, &c.taps, &mut run.main)?;
        let tc = Instant::now();
        drafter.add_context(&mut cache, &tp, mutant)?;
        run.context_ns += ns(tc);
        last = x[(ch.len() - 1) * h..].to_vec();
    }
    let mut t = model.head_token(&last, &mut run.main)?;
    run.tokens.push(t);
    run.prompt_ns = ns(t0);

    let t1 = Instant::now();
    let mut snap = Qwen38Snapshot::default();
    let mut keep: Vec<GdnRows> = Vec::new();
    while run.tokens.len() < n && !stop.contains(&t) {
        let p = model.pos;
        let room = (model.max_pos() - p)
            .saturating_sub(1)
            .min(c.block - 1)
            .min(n - run.tokens.len());
        let td = Instant::now();
        let emb = model.embedding(t);
        let drafts = if room > 0 {
            let mut head = ChipHead {
                model: &mut *model,
                calls: 0,
                ns: 0,
            };
            let (d, _) = drafter.draft(&cache, t, &emb, &mask, &mut head, mutant)?;
            run.head_calls += head.calls;
            run.head_ns += head.ns;
            d
        } else {
            Vec::new()
        };
        run.draft_ns += ns(td);
        let mut feed = vec![t];
        feed.extend(drafts.iter().copied().take(room));
        let nd = feed.len() - 1;
        run.rounds += 1;
        run.drafted += nd as u64;

        let ts = Instant::now();
        if nd > 0 {
            model.snapshot(&mut snap)?;
        }
        run.snapshot_ns += ns(ts);
        let tv = Instant::now();
        let (out, tp) =
            model.exact_rows_taps(&feed, (nd > 0).then_some(&mut keep), &c.taps, &mut run.main)?;
        let mut j = 0;
        let next = loop {
            let pj = model.head_token(&out[j * h..(j + 1) * h], &mut run.main)?;
            if j < nd && pj == feed[j + 1] {
                j += 1;
                continue;
            }
            break pj;
        };
        run.verify_ns += ns(tv);
        run.accepted += j as u64;
        run.accepted_hist[j] += 1;
        run.per_round.push((p, j));
        let left = n - run.tokens.len();
        run.tokens.extend(feed[1..=j].iter().take(left));
        if run.tokens.len() < n {
            run.tokens.push(next);
        }
        if let Some(s) = run.tokens.iter().position(|x| stop.contains(x)) {
            run.tokens.truncate(s + 1);
            break;
        }
        if j < nd {
            run.rollbacks += 1;
            if !no_rollback {
                let tr = Instant::now();
                model.restore(&snap)?;
                model.replay_gdn(&keep, j + 1, &mut run.main)?;
                if model.pos != p + j + 1 {
                    return Err(format!(
                        "dflash rollback: position {} after replay, want {}",
                        model.pos,
                        p + j + 1
                    ));
                }
                run.rollback_ns += ns(tr);
            } else {
                model.pos = p + j + 1;
            }
        }
        // The verified rows the model kept are the drafter's new context.
        let tc = Instant::now();
        drafter.add_context(&mut cache, &tp[..(j + 1) * tw], mutant)?;
        run.context_ns += ns(tc);
        t = next;
    }
    run.decode_ns = ns(t1);
    Ok(run)
}
