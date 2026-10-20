//! Text-level bridge for two models with DIFFERENT vocabularies (L190: Qwen3.8-27B, 248,044 tokens, and
//! Qwen3-30B-A3B, 151,643 tokens). The drafter's guesses are turned into raw bytes, the verifier cuts those
//! bytes into its own tokens and checks them in one pass, exactly as in `spec.rs`. The rule is unchanged:
//! every token written is the verifier's own greedy choice on its own committed prefix, so the output is
//! the verifier alone, token for token, whatever the drafter or the bridge does.
//!
//! Round (`generate_bridged`):
//! 1. The verifier holds committed ids and one pending id (its own choice). Text C = its bytes so far.
//! 2. The drafter's context is its own prompt + `encode_d(C)`; its state keeps the longest common id
//!    prefix with what it already fed (KV truncate) and is fed the rest.
//! 3. The drafter drafts k ids greedily; their bytes B (raw, byte-level decode, never lossy UTF-8).
//! 4. The verifier cuts the longest valid UTF-8 prefix of B into its ids b1..bm (standalone: its next
//!    token starts right after the pending id), feeds [pending, b1..bm] in one pass, accepts b_j while it
//!    equals its own choice, and its choice after the last accepted id is the next pending id.
//! 5. The verifier truncates to what it accepted; the drafter re-syncs at the next round (step 2).

use std::collections::HashMap;
use std::time::Instant;

use crate::spec::Fault;

/// A vocabulary as raw bytes per id.
pub trait Vocab {
    fn token_bytes(&self, id: u32) -> &[u8];
    /// Cut `text` (valid UTF-8) into ids, no special tokens added.
    fn encode_str(&self, text: &str) -> Result<Vec<u32>, String>;
    fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &i in ids {
            out.extend_from_slice(self.token_bytes(i));
        }
        out
    }
    /// Cut the longest valid UTF-8 prefix of `text` into ids; the cut-off tail (an unfinished
    /// character) is dropped. Returns the ids and how many bytes they cover.
    fn encode_bytes(&self, text: &[u8]) -> Result<(Vec<u32>, usize), String> {
        let valid = match std::str::from_utf8(text) {
            Ok(s) => s,
            Err(e) => std::str::from_utf8(&text[..e.valid_up_to()]).map_err(|e| e.to_string())?,
        };
        Ok((self.encode_str(valid)?, valid.len()))
    }
}

/// GPT-2 byte-level map (byte -> char) as used by both Qwen tokenizers, inverted (char -> byte).
fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter().zip(&cs).map(|(&b, &c)| (char::from_u32(c).expect("byte map char"), b as u8)).collect()
}

/// A real tokenizer.json (byte-level BPE) with raw bytes per id. Added tokens (`<|im_end|>`, ...) map
/// to their literal content and are flagged.
pub struct ByteVocab {
    pub tok: tokenizers::Tokenizer,
    bytes: Vec<Vec<u8>>,
    pub added: Vec<bool>,
    pub special: Vec<bool>,
}

impl ByteVocab {
    pub fn from_file(path: &std::path::Path) -> Result<Self, String> {
        let tok = tokenizers::Tokenizer::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let n = tok.get_vocab_size(true);
        let dec = byte_decoder();
        let mut bytes = vec![Vec::new(); n];
        let mut added = vec![false; n];
        let mut special = vec![false; n];
        for (s, id) in tok.get_vocab(false) {
            let mut b = Vec::with_capacity(s.len());
            for c in s.chars() {
                b.push(*dec.get(&c).ok_or_else(|| format!("token {id} {s:?}: char {c:?} is not byte-level"))?);
            }
            bytes[id as usize] = b;
        }
        for (id, t) in tok.get_added_tokens_decoder() {
            bytes[id as usize] = t.content.as_bytes().to_vec();
            added[id as usize] = true;
            special[id as usize] = t.special;
        }
        Ok(Self { tok, bytes, added, special })
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Vocab for ByteVocab {
    fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], |b| b.as_slice())
    }
    fn encode_str(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tok.encode(text, false).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string())
    }
}

/// A model with a KV-style state that can be cut back (the 30B's KV cache; for the 27B's DeltaNet a
/// snapshot per committed length). `feed` returns the greedy choice after every fed token.
pub trait Lm {
    fn feed(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String>;
    fn truncate(&mut self, len: usize) -> Result<(), String>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The drafter side: what it has fed and its choice after each fed token, so a re-sync feeds only
/// what changed.
struct Fed<'a, M: Lm> {
    m: &'a mut M,
    ids: Vec<u32>,
    rows: Vec<u32>,
}

impl<M: Lm> Fed<'_, M> {
    /// Make the state equal `ctx`; returns the choice after its last id and how many ids were fed.
    fn sync(&mut self, ctx: &[u32]) -> Result<(u32, usize), String> {
        let lcp = self.ids.iter().zip(ctx).take_while(|(a, b)| a == b).count();
        // keep at least one id to re-feed so the last row is fresh (rows after a cut are kept)
        let keep = if lcp == ctx.len() { lcp - 1 } else { lcp };
        if keep < self.ids.len() {
            self.m.truncate(keep)?;
            self.ids.truncate(keep);
            self.rows.truncate(keep);
        }
        let rows = self.m.feed(&ctx[keep..])?;
        self.ids.extend_from_slice(&ctx[keep..]);
        self.rows.extend_from_slice(&rows);
        Ok((*self.rows.last().ok_or("empty drafter context")?, ctx.len() - keep))
    }
    fn step(&mut self, id: u32) -> Result<u32, String> {
        let r = self.m.feed(&[id])?[0];
        self.ids.push(id);
        self.rows.push(r);
        Ok(r)
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct BridgeStats {
    pub rounds: usize,
    /// drafter ids drafted
    pub drafted_d: usize,
    /// verifier ids the drafts became (block size the verifier checked, without the pending id)
    pub drafted_v: usize,
    /// verifier ids accepted from drafts
    pub accepted: usize,
    pub verifier_passes: usize,
    pub drafter_steps: usize,
    /// drafter ids re-fed to re-sync after a round (beyond the drafted ones)
    pub drafter_refeed: usize,
    /// rounds where the draft bytes ended inside a UTF-8 character (tail dropped)
    pub cut_tails: usize,
    /// host time in the bridge itself (decode + encode both sides), nanoseconds
    pub bridge_ns: u128,
}

/// Verifier alone (reference for the exactness rule), same Lm interface.
pub fn generate_alone_lm<V: Lm>(v: &mut V, prompt: &[u32], n: usize, eos: &[u32]) -> Result<Vec<u32>, String> {
    let mut next = *v.feed(prompt)?.last().ok_or("empty prompt")?;
    let mut out = Vec::new();
    while out.len() < n {
        out.push(next);
        if eos.contains(&next) || out.len() == n {
            break;
        }
        next = v.feed(&[next])?[0];
    }
    Ok(out)
}

/// One wormhole generation across two vocabularies. Returns the verifier ids written (never more than
/// `n`) and the counters. `d_prompt` / `v_prompt` are each model's own rendering of the same prompt.
#[allow(clippy::too_many_arguments)]
pub fn generate_bridged<D: Lm, V: Lm, DV: Vocab, VV: Vocab>(
    d: &mut D,
    dv: &DV,
    v: &mut V,
    vv: &VV,
    d_prompt: &[u32],
    v_prompt: &[u32],
    n: usize,
    k: usize,
    eos: &[u32],
    fault: Fault,
) -> Result<(Vec<u32>, BridgeStats), String> {
    assert!(k >= 1);
    let mut s = BridgeStats::default();
    let mut pending = *v.feed(v_prompt)?.last().ok_or("empty prompt")?;
    s.verifier_passes += 1;
    let mut out: Vec<u32> = Vec::new();
    let mut d = Fed { m: d, ids: Vec::new(), rows: Vec::new() };
    loop {
        if out.len() + 1 >= n || eos.contains(&pending) {
            out.push(pending);
            break;
        }
        s.rounds += 1;
        // 1-2: drafter context = its prompt + its cut of the committed text (pending included)
        let t0 = Instant::now();
        let mut committed = out.clone();
        committed.push(pending);
        let text = vv.decode(&committed);
        let (dtail, _) = dv.encode_bytes(&text)?;
        let mut ctx = d_prompt.to_vec();
        ctx.extend_from_slice(&dtail);
        s.bridge_ns += t0.elapsed().as_nanos();
        let (mut next, fed) = d.sync(&ctx)?;
        s.drafter_steps += fed;
        s.drafter_refeed += fed.saturating_sub(1);
        // 3: k drafter ids
        let mut drafts = Vec::with_capacity(k);
        for i in 0..k {
            drafts.push(next);
            if i + 1 < k {
                next = d.step(next)?;
                s.drafter_steps += 1;
            }
        }
        s.drafted_d += drafts.len();
        // 4: verifier's cut of the draft bytes, at most k ids and never past n
        let t1 = Instant::now();
        let bytes = dv.decode(&drafts);
        let (mut block, covered) = vv.encode_bytes(&bytes)?;
        if covered < bytes.len() {
            s.cut_tails += 1;
        }
        s.bridge_ns += t1.elapsed().as_nanos();
        block.truncate(k.min(n - out.len() - 1));
        s.drafted_v += block.len();
        let base = v.len();
        let mut feed = vec![pending];
        feed.extend_from_slice(&block);
        let rows = v.feed(&feed)?;
        s.verifier_passes += 1;
        let mut a = 0;
        while a < block.len() && block[a] == rows[a] {
            a += 1;
        }
        if fault == Fault::AcceptOneExtra && a < block.len() {
            a += 1;
        }
        let row = if fault == Fault::RowShift { (a + 1).min(block.len()) } else { a };
        // 5: commit pending + accepted, the next pending is the verifier's own choice
        out.push(pending);
        out.extend_from_slice(&block[..a]);
        s.accepted += a;
        pending = rows[row];
        v.truncate(base + 1 + a)?;
        if let Some(p) = out.iter().position(|t| eos.contains(t)) {
            out.truncate(p + 1);
            break;
        }
    }
    out.truncate(n);
    Ok((out, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toy vocabulary: greedy longest match over a fixed list of byte strings (covers every single byte).
    struct ToyVocab(Vec<Vec<u8>>);
    impl ToyVocab {
        fn new(extra: &[&str]) -> Self {
            let mut v: Vec<Vec<u8>> = (b'a'..=b'h').map(|b| vec![b]).collect();
            v.extend(extra.iter().map(|s| s.as_bytes().to_vec()));
            Self(v)
        }
    }
    impl Vocab for ToyVocab {
        fn token_bytes(&self, id: u32) -> &[u8] {
            &self.0[id as usize]
        }
        fn encode_str(&self, text: &str) -> Result<Vec<u32>, String> {
            let t = text.as_bytes();
            let mut i = 0;
            let mut out = Vec::new();
            while i < t.len() {
                let (id, len) = self
                    .0
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| t[i..].starts_with(b))
                    .map(|(id, b)| (id as u32, b.len()))
                    .max_by_key(|&(_, l)| l)
                    .ok_or("byte outside the toy vocab")?;
                out.push(id);
                i += len;
            }
            Ok(out)
        }
    }

    fn h(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, &x| (a ^ x as u64).wrapping_mul(0x100_0000_01b3))
    }

    /// Toy verifier: its next id is a function of the TEXT of its context (so its own output is well
    /// defined), drawn from its own vocabulary.
    struct ToyV<'a> {
        voc: &'a ToyVocab,
        ids: Vec<u32>,
    }
    fn v_next(voc: &ToyVocab, ids: &[u32]) -> u32 {
        (h(&voc.decode(ids)) % voc.0.len() as u64) as u32
    }
    impl Lm for ToyV<'_> {
        fn feed(&mut self, t: &[u32]) -> Result<Vec<u32>, String> {
            let mut r = Vec::new();
            for &x in t {
                self.ids.push(x);
                r.push(v_next(self.voc, &self.ids));
            }
            Ok(r)
        }
        fn truncate(&mut self, len: usize) -> Result<(), String> {
            self.ids.truncate(len);
            Ok(())
        }
        fn len(&self) -> usize {
            self.ids.len()
        }
    }

    /// Toy drafter: knows the verifier's rule, so it predicts the verifier's next bytes and emits them in
    /// its OWN vocabulary (first drafter id of the verifier's next bytes), wrong on every `bad`-th context.
    struct ToyD<'a> {
        dvoc: &'a ToyVocab,
        vvoc: &'a ToyVocab,
        vprompt: Vec<u32>,
        dprompt_len: usize,
        bad: u64,
        ids: Vec<u32>,
    }
    impl ToyD<'_> {
        fn next(&self) -> u32 {
            let text = self.dvoc.decode(&self.ids[self.dprompt_len..]);
            // the verifier's continuation of this text, a few of its tokens ahead
            let mut vids = self.vprompt.clone();
            vids.extend(self.vvoc.encode_bytes(&text).unwrap().0);
            let mut ahead = Vec::new();
            for _ in 0..3 {
                let t = v_next(self.vvoc, &vids);
                vids.push(t);
                ahead.extend_from_slice(self.vvoc.token_bytes(t));
            }
            let first = self.dvoc.encode_bytes(&ahead).unwrap().0[0];
            if self.bad > 0 && h(&text) % self.bad == 0 {
                (first + 1) % self.dvoc.0.len() as u32
            } else {
                first
            }
        }
    }
    impl Lm for ToyD<'_> {
        fn feed(&mut self, t: &[u32]) -> Result<Vec<u32>, String> {
            let mut r = Vec::new();
            for &x in t {
                self.ids.push(x);
                r.push(self.next());
            }
            Ok(r)
        }
        fn truncate(&mut self, len: usize) -> Result<(), String> {
            self.ids.truncate(len);
            Ok(())
        }
        fn len(&self) -> usize {
            self.ids.len()
        }
    }

    fn run(k: usize, bad: u64, fault: Fault) -> (Vec<u32>, Vec<u32>, BridgeStats) {
        let vv = ToyVocab::new(&["ab", "cd", "abc", "efg", "ha"]);
        let dv = ToyVocab::new(&["bc", "de", "fgh", "ca", "hab"]);
        let vp = vec![0u32, 1, 2];
        let dp = vec![7u32];
        let alone = generate_alone_lm(&mut ToyV { voc: &vv, ids: vec![] }, &vp, 60, &[]).unwrap();
        let mut d = ToyD { dvoc: &dv, vvoc: &vv, vprompt: vp.clone(), dprompt_len: dp.len(), bad, ids: vec![] };
        let mut v = ToyV { voc: &vv, ids: vec![] };
        let (w, s) = generate_bridged(&mut d, &dv, &mut v, &vv, &dp, &vp, 60, k, &[], fault).unwrap();
        (alone, w, s)
    }

    #[test]
    fn bridged_output_equals_verifier_alone() {
        for k in 1..=12 {
            for bad in [0, 2, 3, 5, 1] {
                let (alone, w, s) = run(k, bad, Fault::None);
                assert_eq!(w, alone, "k {k} bad {bad} {s:?}");
            }
        }
        // the bridge must actually carry guesses across the two vocabularies: with the right-text toy drafter
        // the verifier needs under half the passes it needs alone (60). Not every right byte is accepted: the
        // verifier cuts the draft bytes on their own, which can differ from how it would have cut its own text.
        let (_, _, s) = run(4, 0, Fault::None);
        assert!(s.verifier_passes * 2 < 60 && s.accepted > 0, "the bridge saved too few verifier passes: {s:?}");
    }

    #[test]
    fn bridged_negative_controls_go_red() {
        for f in [Fault::AcceptOneExtra, Fault::RowShift] {
            let (alone, w, _) = run(4, 3, f);
            assert_ne!(w, alone, "control {f:?} did not change the output");
        }
    }

    #[test]
    fn byte_decoder_covers_every_byte() {
        let d = byte_decoder();
        assert_eq!(d.len(), 256);
        let mut seen: Vec<u8> = d.values().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
        assert_eq!(d[&'Ġ'], b' ');
        assert_eq!(d[&'Ċ'], b'\n');
    }
}
