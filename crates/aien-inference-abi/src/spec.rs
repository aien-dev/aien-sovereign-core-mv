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

//! Speculative decoding bookkeeping shared by the MTP drafter (DRA-70, sc#866) and the CPU
//! suffix drafter (DRA-77, sc#867): the truncated draft vocabulary, the greedy and the sampled
//! acceptance rules, and the expected-length model.
//!
//! One verify step feeds the last accepted token plus k drafts (N = 1 + k rows) through the
//! target and gets k + 1 target tokens back.
//!
//! - Greedy: the longest prefix where `draft[j] == target[j]` is kept, then `target[a]` (the
//!   target's own next token after the accepted prefix). Output is token-identical to plain greedy
//!   decoding by construction: every emitted token is a target argmax.
//! - Sampled (Leviathan et al. 2023, Chen et al. 2023): draft x with draft probability q(x) is
//!   kept with probability `min(1, p(x) / q(x))`; on rejection the token is drawn from
//!   `max(0, p - q)` renormalised. The emitted token is distributed exactly as p. A truncated draft
//!   head only makes q zero outside the draft vocabulary; the rule and its exactness are unchanged.

use std::path::Path;

use sha2::{Digest, Sha256};

/// The draft head's vocabulary: ascending unique target token ids.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftVocab {
    ids: Vec<u32>,
    /// `slot[id]` = position of `id` in `ids`, or `u32::MAX` when absent.
    slot: Vec<u32>,
}

impl DraftVocab {
    /// `ids` in any order, each below `vocab`, no repeats.
    pub fn new(mut ids: Vec<u32>, vocab: usize) -> Result<Self, String> {
        ids.sort_unstable();
        if ids.windows(2).any(|p| p[0] == p[1]) {
            return Err("draft vocab: repeated id".into());
        }
        if ids.last().is_some_and(|&l| l as usize >= vocab) {
            return Err(format!("draft vocab: id outside the vocab {vocab}"));
        }
        let mut slot = vec![u32::MAX; vocab];
        for (j, &id) in ids.iter().enumerate() {
            slot[id as usize] = j as u32;
        }
        Ok(Self { ids, slot })
    }

    /// Reads `{"vocab": V, "ids": [...]}` (the `q36_draft_vocab` output) and checks its
    /// `ids_sha256` when present.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let vocab = v["vocab"].as_u64().ok_or("draft vocab: no \"vocab\"")? as usize;
        let ids = v["ids"]
            .as_array()
            .ok_or("draft vocab: no \"ids\"")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as u32).ok_or("draft vocab: bad id"))
            .collect::<Result<Vec<u32>, _>>()?;
        let dv = Self::new(ids, vocab)?;
        if let Some(want) = v["ids_sha256"].as_str() {
            if want != dv.sha256_hex() {
                return Err(format!(
                    "draft vocab: ids_sha256 {want} but the ids hash to {}",
                    dv.sha256_hex()
                ));
            }
        }
        Ok(dv)
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: u32) -> bool {
        self.slot(id).is_some()
    }

    /// Row of `id` in the truncated head.
    pub fn slot(&self, id: u32) -> Option<usize> {
        match self.slot.get(id as usize) {
            Some(&s) if s != u32::MAX => Some(s as usize),
            _ => None,
        }
    }

    /// sha256 of the ids as little-endian u32, ascending (the identity of the head's row set).
    pub fn sha256_hex(&self) -> String {
        let mut h = Sha256::new();
        for id in &self.ids {
            h.update(id.to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Argmax over the truncated head's logits (`[len]`) as a target token id (first on ties).
    pub fn argmax(&self, logits: &[f32]) -> u32 {
        assert_eq!(logits.len(), self.ids.len());
        self.ids[crate::qwen38::argmax(logits)]
    }
}

/// Greedy verify: `drafts` (k) against `target` (k + 1 target argmaxes, row j being the target's
/// token after the accepted prefix plus `drafts[..j]`). Returns the accepted draft count a; the
/// step emits `drafts[..a]` and then `target[a]`.
pub fn greedy_accept(drafts: &[u32], target: &[u32]) -> usize {
    assert_eq!(target.len(), drafts.len() + 1, "greedy_accept: want k + 1 target rows");
    drafts
        .iter()
        .zip(target)
        .take_while(|(d, t)| d == t)
        .count()
}

/// The sampled rule for one draft token: `p` and `q` are the target and draft distributions over
/// the full vocabulary (q zero outside the draft vocab), `u` a uniform draw in [0, 1). `Ok(())`
/// keeps the draft; `Err(residual)` rejects it and gives `max(0, p - q)` renormalised to draw the
/// replacement from.
pub fn sampled_accept(p: &[f64], q: &[f64], draft: usize, u: f64) -> Result<(), Vec<f64>> {
    assert_eq!(p.len(), q.len());
    let (pd, qd) = (p[draft], q[draft]);
    if qd > 0.0 && u < (pd / qd).min(1.0) {
        return Ok(());
    }
    let mut r: Vec<f64> = p.iter().zip(q).map(|(a, b)| (a - b).max(0.0)).collect();
    let s: f64 = r.iter().sum();
    if s > 0.0 {
        r.iter_mut().for_each(|x| *x /= s);
    } else {
        // p == q: the rejection branch has probability zero; fall back to p itself.
        r.copy_from_slice(p);
    }
    Err(r)
}

/// Probability that a draft drawn from q survives against p: `sum_x min(p(x), q(x))`.
pub fn acceptance_rate(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(a, b)| a.min(*b)).sum()
}

/// Expected tokens emitted per verify step with conditional acceptance `alpha[j]` at depth j + 1
/// (draft j + 1 kept given drafts 1..=j kept): `1 + a1 + a1 a2 + ... + a1..ak`.
pub fn expected_tokens(alpha: &[f64]) -> f64 {
    let mut e = 1.0;
    let mut run = 1.0;
    for a in alpha {
        run *= a;
        e += run;
    }
    e
}

/// Softmax of `logits` in f64.
pub fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_keeps_the_matching_prefix() {
        assert_eq!(greedy_accept(&[5, 6, 7], &[5, 6, 7, 8]), 3);
        assert_eq!(greedy_accept(&[5, 9, 7], &[5, 6, 7, 8]), 1);
        assert_eq!(greedy_accept(&[1, 6, 7], &[5, 6, 7, 8]), 0);
        assert_eq!(greedy_accept(&[], &[4]), 0);
    }

    #[test]
    fn expected_tokens_matches_the_series() {
        assert_eq!(expected_tokens(&[]), 1.0);
        // The DRA-70 issue's NVFP4 per-position rates 0.86 / 0.67 / 0.55 read as conditional.
        let e = expected_tokens(&[0.86, 0.67, 0.55]);
        assert!((e - (1.0 + 0.86 + 0.86 * 0.67 + 0.86 * 0.67 * 0.55)).abs() < 1e-12);
    }

    /// Exactness of the sampled rule with a truncated draft (q zero off its vocab): the law of the
    /// emitted token is p, computed in closed form from the rule's own branches.
    #[test]
    fn sampled_rule_emits_the_target_law_with_a_truncated_draft() {
        let p = [0.30f64, 0.25, 0.20, 0.15, 0.10];
        // Draft vocab {0, 1, 3}: q is the draft's softmax renormalised over those ids.
        let q = [0.50, 0.10, 0.0, 0.40, 0.0];
        let mut out = [0f64; 5];
        for (x, &qx) in q.iter().enumerate() {
            if qx == 0.0 {
                continue;
            }
            let keep = (p[x] / qx).min(1.0);
            out[x] += qx * keep;
            let r = sampled_accept(&p, &q, x, 1.0).unwrap_err();
            for (o, rv) in out.iter_mut().zip(&r) {
                *o += qx * (1.0 - keep) * rv;
            }
        }
        for (o, pv) in out.iter().zip(&p) {
            assert!((o - pv).abs() < 1e-12, "{out:?} vs {p:?}");
        }
        let a = acceptance_rate(&p, &q);
        assert!((a - (0.30 + 0.10 + 0.15)).abs() < 1e-12);
    }

    /// Same law by sampling (deterministic generator), so the branch logic is exercised and not
    /// only the arithmetic.
    #[test]
    fn sampled_rule_by_draws() {
        let p = [0.30f64, 0.25, 0.20, 0.15, 0.10];
        let q = [0.50, 0.10, 0.0, 0.40, 0.0];
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut uni = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let draw = |d: &[f64], u: f64| {
            let mut c = 0.0;
            for (i, v) in d.iter().enumerate() {
                c += v;
                if u < c {
                    return i;
                }
            }
            d.len() - 1
        };
        let n = 400_000;
        let mut hist = [0usize; 5];
        for _ in 0..n {
            let x = draw(&q, uni());
            let tok = match sampled_accept(&p, &q, x, uni()) {
                Ok(()) => x,
                Err(r) => draw(&r, uni()),
            };
            hist[tok] += 1;
        }
        for (h, pv) in hist.iter().zip(&p) {
            let f = *h as f64 / n as f64;
            // 5 sigma of a binomial proportion at n = 400k is below 0.0037.
            assert!((f - pv).abs() < 0.004, "{hist:?}");
        }
    }

    /// Negative control: dropping the residual (re-drawing from p on rejection) is biased.
    #[test]
    fn redraw_from_p_on_rejection_is_biased() {
        let p = [0.30f64, 0.25, 0.20, 0.15, 0.10];
        let q = [0.50, 0.10, 0.0, 0.40, 0.0];
        let mut out = [0f64; 5];
        for (x, &qx) in q.iter().enumerate() {
            if qx == 0.0 {
                continue;
            }
            let keep = (p[x] / qx).min(1.0);
            out[x] += qx * keep;
            for (o, pv) in out.iter_mut().zip(&p) {
                *o += qx * (1.0 - keep) * pv;
            }
        }
        assert!(out.iter().zip(&p).any(|(o, pv)| (o - pv).abs() > 0.01));
    }

    #[test]
    fn draft_vocab_slots_and_hash() {
        let v = DraftVocab::new(vec![9, 2, 5], 10).unwrap();
        assert_eq!(v.ids(), &[2, 5, 9]);
        assert_eq!(v.slot(5), Some(1));
        assert_eq!(v.slot(3), None);
        assert_eq!(v.slot(99), None);
        assert_eq!(v.argmax(&[0.0, 3.0, 1.0]), 5);
        assert!(DraftVocab::new(vec![1, 1], 10).is_err());
        assert!(DraftVocab::new(vec![10], 10).is_err());
        let w = DraftVocab::new(vec![2, 5, 9], 10).unwrap();
        assert_eq!(v.sha256_hex(), w.sha256_hex());
        assert_ne!(
            v.sha256_hex(),
            DraftVocab::new(vec![2, 5, 8], 10).unwrap().sha256_hex()
        );
    }
}
