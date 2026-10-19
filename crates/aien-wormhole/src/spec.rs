//! The Wormhole round: a drafter proposes up to k words, the verifier checks all of them in ONE pass,
//! keeps the longest matching prefix and adds its own next word. Greedy only. The rule: the output is
//! exactly what the verifier alone would write, word for word. `generate_alone` is that reference.
//!
//! States are committed only on what the verifier accepted: after a round both models are re-fed the
//! accepted words from their last committed state (simple and exact; the chip path replaces this with
//! snapshot + rollback, which the 27B MTP work already proved exact).

pub trait GreedyModel {
    type State: Clone;
    fn new_state(&self) -> Self::State;
    /// Feed `tokens` after `st`; the model's greedy choice after each of them.
    fn argmax_rows(&mut self, st: &mut Self::State, tokens: &[u32]) -> Result<Vec<u32>, String>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct SpecStats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub verifier_passes: usize,
    pub drafter_steps: usize,
}

/// Faults for the negative controls; production uses `Fault::None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    None,
    /// also accept the first refused draft
    AcceptOneExtra,
    /// compare draft i against the verifier row i+1 (off by one)
    RowShift,
}

pub fn generate_alone<M: GreedyModel>(m: &mut M, prompt: &[u32], n: usize, eos: &[u32]) -> Result<Vec<u32>, String> {
    let mut st = m.new_state();
    let mut next = *m.argmax_rows(&mut st, prompt)?.last().ok_or("empty prompt")?;
    let mut out = Vec::new();
    while out.len() < n {
        out.push(next);
        if eos.contains(&next) || out.len() == n {
            break;
        }
        next = m.argmax_rows(&mut st, &[next])?[0];
    }
    Ok(out)
}

pub fn generate_wormhole<D: GreedyModel, V: GreedyModel>(
    d: &mut D,
    v: &mut V,
    prompt: &[u32],
    n: usize,
    k: usize,
    eos: &[u32],
    fault: Fault,
) -> Result<(Vec<u32>, SpecStats), String> {
    assert!(k >= 1);
    let mut s = SpecStats::default();
    let mut vst = v.new_state();
    let mut pending = *v.argmax_rows(&mut vst, prompt)?.last().ok_or("empty prompt")?;
    s.verifier_passes += 1;
    let mut dst = d.new_state();
    d.argmax_rows(&mut dst, prompt)?;
    let mut out: Vec<u32> = Vec::new();
    loop {
        out.push(pending);
        if eos.contains(&pending) || out.len() >= n {
            break;
        }
        // draft k words from the drafter's committed state
        let kk = k.min(n - out.len());
        let mut dtmp = dst.clone();
        let mut drafts = Vec::with_capacity(kk);
        let mut x = pending;
        for _ in 0..kk {
            let a = d.argmax_rows(&mut dtmp, &[x])?[0];
            s.drafter_steps += 1;
            drafts.push(a);
            if eos.contains(&a) {
                break;
            }
            x = a;
        }
        // verify every draft in one pass: rows[i] = the verifier's word after [pending, drafts[..i]]
        let mut feed = vec![pending];
        feed.extend_from_slice(&drafts);
        let mut vtmp = vst.clone();
        let rows = v.argmax_rows(&mut vtmp, &feed)?;
        s.verifier_passes += 1;
        s.rounds += 1;
        s.drafted += drafts.len();
        let mut m = 0;
        while m < drafts.len() {
            let r = match fault {
                Fault::RowShift => rows.get(m + 1).copied().unwrap_or(u32::MAX),
                _ => rows[m],
            };
            if r != drafts[m] {
                break;
            }
            m += 1;
        }
        if fault == Fault::AcceptOneExtra && m < drafts.len() {
            m += 1;
        }
        s.accepted += m;
        // commit: pending + drafts[..m]; the next pending is the verifier's own word after them
        let mut commit = vec![pending];
        commit.extend_from_slice(&drafts[..m]);
        let mut stop = false;
        for &w in &drafts[..m] {
            out.push(w);
            if eos.contains(&w) || out.len() >= n {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
        pending = if m < rows.len() { rows[m] } else { *rows.last().unwrap() };
        if m == drafts.len() && m + 1 == feed.len() && fault == Fault::None {
            vst = vtmp; // the pass fed exactly the committed words
        } else {
            v.argmax_rows(&mut vst, &commit)?;
            s.verifier_passes += 1;
        }
        d.argmax_rows(&mut dst, &commit)?;
        s.drafter_steps += commit.len();
    }
    out.truncate(n);
    Ok((out, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy causal model: the next word is a hash of the whole context; `wrong_every` makes a drafter
    /// that disagrees on some contexts.
    struct Toy {
        vocab: u32,
        wrong_every: u64,
    }
    fn h(ctx: &[u32]) -> u64 {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for &t in ctx {
            x ^= t as u64;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(17);
        }
        x
    }
    impl GreedyModel for Toy {
        type State = Vec<u32>;
        fn new_state(&self) -> Vec<u32> {
            Vec::new()
        }
        fn argmax_rows(&mut self, st: &mut Vec<u32>, tokens: &[u32]) -> Result<Vec<u32>, String> {
            let mut out = Vec::new();
            for &t in tokens {
                st.push(t);
                let x = h(st);
                // the true next word depends only on the last 3 words so drafts can be right
                let w = h(&st[st.len().saturating_sub(3)..]) % self.vocab as u64;
                let w = if self.wrong_every > 0 && x % self.wrong_every == 0 { (w + 1) % self.vocab as u64 } else { w };
                out.push(w as u32);
            }
            Ok(out)
        }
    }

    #[test]
    fn output_equals_verifier_alone_for_every_block_size() {
        for wrong in [0u64, 2, 3, 5] {
            for k in 1..=16 {
                let mut v = Toy { vocab: 50, wrong_every: 0 };
                let mut d = Toy { vocab: 50, wrong_every: wrong };
                let alone = generate_alone(&mut Toy { vocab: 50, wrong_every: 0 }, &[1, 2, 3], 200, &[49]).unwrap();
                let (out, s) = generate_wormhole(&mut d, &mut v, &[1, 2, 3], 200, k, &[49], Fault::None).unwrap();
                assert_eq!(out, alone, "wrong_every {wrong} k {k}");
                if wrong == 0 {
                    assert_eq!(s.accepted, s.drafted, "a perfect drafter is always accepted");
                }
            }
        }
    }

    #[test]
    fn negative_controls_go_red() {
        let alone = generate_alone(&mut Toy { vocab: 50, wrong_every: 0 }, &[1, 2, 3], 200, &[]).unwrap();
        for fault in [Fault::AcceptOneExtra, Fault::RowShift] {
            let mut v = Toy { vocab: 50, wrong_every: 0 };
            let mut d = Toy { vocab: 50, wrong_every: 3 };
            let (out, _) = generate_wormhole(&mut d, &mut v, &[1, 2, 3], 200, 4, &[], fault).unwrap();
            assert_ne!(out, alone, "{fault:?} must change the output");
        }
    }
}
