//! Suffix-index drafter for speculative decoding (W-L4c cut 2, SuffixDecoding style, no weights).
//!
//! A [`SuffixTree`] holds every substring of up to `depth` tokens of what it was fed, with how
//! often each continuation followed. [`SuffixDrafter`] keeps two: `local` (the current request:
//! prompt plus everything generated so far, fed one token at a time) and `global` (finished
//! requests: past conversations, kept across requests until a node cap resets it). A draft is the
//! most frequent continuation of the longest suffix of the context that occurred before with a
//! continuation, followed greedily while the estimated probability stays high.
//!
//! The drafter only decides WHICH tokens to propose and how many. Verification is unchanged
//! (`generate_spec_with`), so the emitted tokens never depend on it.
//!
//! Draft length is chosen by expected throughput, not a fixed k: with per-position estimates
//! `c_i` (product of the branch probabilities so far), a prefix of `k` drafts is worth
//! `(1 + sum c_i) / (verify_ms(k + 1) + lm_ms * (1 + sum c_i))` tokens per ms; the best prefix is
//! drafted only when that beats plain decode `1 / plain_ms` by `gain`. Short or rare matches thus
//! draft nothing (auto-off on ordinary text) and long repeated runs draft long blocks.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplicative hasher for the (node, token) edge keys (SipHash is ~5x slower here).
#[derive(Default, Clone, Copy)]
pub struct EdgeHasher(u64);

impl Hasher for EdgeHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7cc1_b727_220a_95);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7cc1_b727_220a_95);
    }
}

type Edges = HashMap<u64, u32, BuildHasherDefault<EdgeHasher>>;

fn key(node: u32, token: u32) -> u64 {
    ((node as u64) << 32) | token as u64
}

#[derive(Clone, Copy, Debug, Default)]
struct Node {
    /// Times a continuation followed this path (sum of the children's counts).
    cont: u32,
    /// Times this path occurred as a prefix of a later token (its own count as a child).
    count: u32,
    /// Most frequent child (`u32::MAX` = none); ties go to the most recent.
    best: u32,
    best_token: u32,
    depth: u16,
}

/// Every substring of up to `depth` tokens of the fed sequences, as a trie with counts.
#[derive(Clone, Debug)]
pub struct SuffixTree {
    nodes: Vec<Node>,
    edges: Edges,
    depth: usize,
    /// Paths ending at the last fed token (one per start position still shorter than `depth`).
    active: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl SuffixTree {
    pub fn new(depth: usize) -> Self {
        let mut t = Self { nodes: Vec::new(), edges: Edges::default(), depth: depth.max(2), active: Vec::new() };
        t.clear();
        t
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.active.clear();
        self.nodes.push(Node { best: NONE, ..Node::default() });
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Starts a new sequence: the next token does not continue the previous one.
    pub fn end_sequence(&mut self) {
        self.active.clear();
    }

    /// Appends one token to the current sequence: every active path (and a new one from the
    /// root) gains it as a continuation.
    pub fn push(&mut self, token: u32) {
        let mut next = Vec::with_capacity(self.active.len() + 1);
        let starts: Vec<u32> = std::iter::once(0).chain(self.active.iter().copied()).collect();
        for parent in starts {
            let pd = self.nodes[parent as usize].depth as usize;
            if pd >= self.depth {
                continue;
            }
            let k = key(parent, token);
            let child = match self.edges.get(&k) {
                Some(&c) => c,
                None => {
                    let c = self.nodes.len() as u32;
                    self.nodes.push(Node { best: NONE, depth: (pd + 1) as u16, ..Node::default() });
                    self.edges.insert(k, c);
                    c
                }
            };
            self.nodes[child as usize].count += 1;
            let cc = self.nodes[child as usize].count;
            let best_count = match self.nodes[parent as usize].best {
                NONE => 0,
                b => self.nodes[b as usize].count,
            };
            let p = &mut self.nodes[parent as usize];
            p.cont += 1;
            if p.best == NONE || p.best == child || cc >= best_count {
                p.best = child;
                p.best_token = token;
            }
            if pd + 1 < self.depth {
                next.push(child);
            }
        }
        self.active = next;
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push(t);
        }
    }

    fn walk(&self, path: &[u32]) -> Option<u32> {
        let mut n = 0u32;
        for &t in path {
            n = *self.edges.get(&key(n, t))?;
        }
        Some(n)
    }

    /// The node of the longest suffix of `ctx` (at most `max_len` tokens) that has a
    /// continuation, and its length. Having one is monotone in the length (an earlier occurrence
    /// of a suffix is also one of every shorter suffix), so a binary search finds it.
    pub fn longest_match(&self, ctx: &[u32], max_len: usize) -> Option<(u32, usize)> {
        let hi_cap = max_len.min(ctx.len()).min(self.depth - 1);
        let ok = |p: usize| self.walk(&ctx[ctx.len() - p..]).filter(|&n| self.nodes[n as usize].cont > 0);
        let (mut lo, mut hi, mut best) = (1usize, hi_cap, None);
        while lo <= hi {
            let mid = (lo + hi) / 2;
            match ok(mid) {
                Some(n) => {
                    best = Some((n, mid));
                    lo = mid + 1;
                }
                None => hi = mid - 1,
            }
        }
        best
    }

    /// Greedy continuation from `node`: tokens and the cumulative probability estimate after each
    /// (`best.count / (cont + prior / depth)` multiplied along the path), at most `max` tokens, stopping
    /// when the cumulative estimate drops below `min_cum`.
    pub fn follow(&self, mut node: u32, max: usize, prior: f32, min_cum: f32) -> (Vec<u32>, Vec<f32>) {
        let (mut toks, mut cums) = (Vec::new(), Vec::new());
        let mut cum = 1.0f32;
        while toks.len() < max {
            let n = &self.nodes[node as usize];
            if n.best == NONE || n.cont == 0 {
                break;
            }
            // The pseudo-count shrinks with the matched context length (the node depth): one earlier
            // occurrence of a 12-token context predicts better than one of a 2-token context.
            let p = self.nodes[n.best as usize].count as f32 / (n.cont as f32 + prior / n.depth.max(1) as f32);
            cum *= p;
            if cum < min_cum {
                break;
            }
            toks.push(n.best_token);
            cums.push(cum);
            node = n.best;
        }
        (toks, cums)
    }
}

/// Measured costs the length choice weighs (ms). Defaults: chip probe own30b-l4c-cut1 (GB10,
/// p4-long at 657 positions): verify ~24.3 ms at 1 row + ~4.5 ms per extra row, lm_head 2.9 ms,
/// plain step + lm_head 28.3 ms.
#[derive(Clone, Debug)]
pub struct CostModel {
    pub verify_base_ms: f32,
    pub verify_row_ms: f32,
    pub lm_ms: f32,
    pub plain_ms: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        Self { verify_base_ms: 24.3, verify_row_ms: 4.5, lm_ms: 2.9, plain_ms: 28.3 }
    }
}

impl CostModel {
    pub fn verify_ms(&self, rows: usize) -> f32 {
        self.verify_base_ms + self.verify_row_ms * rows.saturating_sub(1) as f32
    }
}

/// Drafter settings.
#[derive(Clone, Debug)]
pub struct SuffixConfig {
    /// Deepest path kept (match plus draft).
    pub depth: usize,
    /// Longest suffix matched.
    pub max_match: usize,
    /// Shortest match that may draft.
    pub min_match: usize,
    /// Draft at most `alpha * match length` tokens (SuffixDecoding's adaptive length).
    pub alpha: f32,
    /// Pseudo-count in the probability estimate (count / (cont + prior / matched length)).
    pub prior: f32,
    /// Stop following when the cumulative estimate drops below this.
    pub min_cum: f32,
    /// Draft only when the expected rate beats plain decode by this factor.
    pub gain: f32,
    /// Global tree node cap (reset when exceeded).
    pub global_max_nodes: usize,
    /// Feed finished requests into the global tree.
    pub use_global: bool,
    pub cost: CostModel,
}

impl Default for SuffixConfig {
    fn default() -> Self {
        Self {
            depth: 40,
            max_match: 24,
            min_match: 1,
            alpha: 4.0,
            prior: 4.0,
            min_cum: 0.05,
            gain: 1.1,
            global_max_nodes: 8_000_000,
            use_global: true,
            cost: CostModel::default(),
        }
    }
}

impl SuffixConfig {
    /// Defaults overridden by `AIEN_OWN30B_SPEC_SUFFIX_{DEPTH,MAX_MATCH,MIN_MATCH,ALPHA,PRIOR,MIN_CUM,GAIN,GLOBAL}`
    /// and the cost model by `AIEN_OWN30B_SPEC_COST_{VERIFY_BASE,VERIFY_ROW,LM,PLAIN}_MS`.
    pub fn from_env() -> Self {
        let get = |k: &str| std::env::var(format!("AIEN_OWN30B_SPEC_{k}")).ok();
        let u = |k: &str, d: usize| get(k).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(d);
        let f = |k: &str, d: f32| get(k).and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(d);
        let d = Self::default();
        let c = d.cost.clone();
        Self {
            depth: u("SUFFIX_DEPTH", d.depth),
            max_match: u("SUFFIX_MAX_MATCH", d.max_match),
            min_match: u("SUFFIX_MIN_MATCH", d.min_match),
            alpha: f("SUFFIX_ALPHA", d.alpha),
            prior: f("SUFFIX_PRIOR", d.prior),
            min_cum: f("SUFFIX_MIN_CUM", d.min_cum),
            gain: f("SUFFIX_GAIN", d.gain),
            global_max_nodes: d.global_max_nodes,
            use_global: !matches!(get("SUFFIX_GLOBAL").as_deref().map(str::trim), Some("0") | Some("off")),
            cost: CostModel {
                verify_base_ms: f("COST_VERIFY_BASE_MS", c.verify_base_ms),
                verify_row_ms: f("COST_VERIFY_ROW_MS", c.verify_row_ms),
                lm_ms: f("COST_LM_MS", c.lm_ms),
                plain_ms: f("COST_PLAIN_MS", c.plain_ms),
            },
        }
    }
}

/// One draft decision, for logging.
#[derive(Clone, Debug, Default)]
pub struct Draft {
    pub tokens: Vec<u32>,
    pub match_len: usize,
    /// Expected tokens per round (1 + sum of cumulative estimates) for the chosen length.
    pub expected: f32,
    pub from_global: bool,
}

/// Local (current request) plus global (finished requests) suffix trees.
#[derive(Clone, Debug)]
pub struct SuffixDrafter {
    pub cfg: SuffixConfig,
    local: SuffixTree,
    global: SuffixTree,
}

impl SuffixDrafter {
    pub fn new(cfg: SuffixConfig) -> Self {
        let d = cfg.depth;
        Self { cfg, local: SuffixTree::new(d), global: SuffixTree::new(d) }
    }

    /// New request: the local tree holds only `prompt` (which already contains this
    /// conversation's earlier turns).
    pub fn begin(&mut self, prompt: &[u32]) {
        self.local.clear();
        self.local.extend(prompt);
    }

    /// A token the model emitted (or the request's next accepted token).
    pub fn push(&mut self, token: u32) {
        self.local.push(token);
    }

    /// Request finished: its whole sequence (prompt plus output) goes into the global tree.
    pub fn finish(&mut self, sequence: &[u32]) {
        if !self.cfg.use_global {
            return;
        }
        if self.global.nodes() > self.cfg.global_max_nodes {
            self.global.clear();
        }
        self.global.end_sequence();
        self.global.extend(sequence);
        self.global.end_sequence();
    }

    pub fn clear_global(&mut self) {
        self.global.clear();
    }

    fn candidate(&self, tree: &SuffixTree, ctx: &[u32], k_max: usize) -> Option<(Vec<u32>, Vec<f32>, usize)> {
        let (node, p) = tree.longest_match(ctx, self.cfg.max_match)?;
        if p < self.cfg.min_match {
            return None;
        }
        let cap = ((self.cfg.alpha * p as f32).floor() as usize).min(k_max);
        let (t, c) = tree.follow(node, cap, self.cfg.prior, self.cfg.min_cum);
        (!t.is_empty()).then_some((t, c, p))
    }

    /// The draft for context `ctx` (prompt plus everything emitted, the round's first token
    /// included), at most `k_max` tokens; empty when drafting is not expected to pay.
    pub fn draft(&self, ctx: &[u32], k_max: usize) -> Draft {
        self.draft_with_cost(ctx, k_max, &self.cfg.cost)
    }

    /// [`Self::draft`] weighing `cost` instead of the configured cost model (W-L4e: the measured costs).
    pub fn draft_with_cost(&self, ctx: &[u32], k_max: usize, cost: &CostModel) -> Draft {
        let mut best = Draft::default();
        let mut best_rate = 1.0 / cost.plain_ms * self.cfg.gain;
        let cands = [(self.candidate(&self.local, ctx, k_max), false), (if self.cfg.use_global { self.candidate(&self.global, ctx, k_max) } else { None }, true)];
        for (c, from_global) in cands {
            let Some((toks, cums, p)) = c else { continue };
            let mut exp = 1.0f32;
            for k in 1..=toks.len() {
                exp += cums[k - 1];
                let rate = exp / (cost.verify_ms(k + 1) + cost.lm_ms * exp);
                if rate > best_rate {
                    best_rate = rate;
                    best = Draft { tokens: toks[..k].to_vec(), match_len: p, expected: exp, from_global };
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_counts_continuations_and_finds_the_longest_earlier_match() {
        let mut t = SuffixTree::new(8);
        // 1 2 3 4 | 1 2 3 5 | 1 2 3
        t.extend(&[1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3]);
        let ctx = [1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3];
        let (n, p) = t.longest_match(&ctx, 6).unwrap();
        // "5 1 2 3" never continued; the longest suffix seen before WITH a continuation is "1 2 3"
        // (followed by 4 once and 5 once): length 3 ("3" alone and "2 3" too, but shorter).
        assert_eq!(p, 3);
        let (toks, cums) = t.follow(n, 4, 0.0, 0.0);
        assert_eq!(toks[0], 5, "tie goes to the most recent continuation");
        assert!((cums[0] - 0.5).abs() < 1e-6);
        // No earlier occurrence at all: nothing.
        assert!(t.longest_match(&[9, 9], 4).is_none());
    }

    #[test]
    fn depth_caps_paths_and_end_sequence_separates_requests() {
        let mut t = SuffixTree::new(3);
        t.extend(&[1, 2, 3, 4]);
        assert!(t.walk(&[1, 2, 3]).is_some());
        assert!(t.walk(&[1, 2, 3, 4]).is_none(), "deeper than depth is not stored");
        t.end_sequence();
        t.extend(&[7, 8]);
        assert!(t.walk(&[4, 7]).is_none(), "no path across sequences");
    }

    #[test]
    fn drafter_drafts_long_repeats_and_nothing_on_fresh_text() {
        let mut d = SuffixDrafter::new(SuffixConfig::default());
        let block: Vec<u32> = (100..130).collect();
        let mut prompt = block.clone();
        prompt.extend_from_slice(&[7, 8]);
        prompt.extend_from_slice(&block[..6]);
        d.begin(&prompt);
        let dr = d.draft(&prompt, 16);
        assert_eq!(dr.match_len, 6);
        assert_eq!(dr.tokens, block[6..6 + dr.tokens.len()].to_vec());
        assert!(dr.tokens.len() >= 2, "a long exact repeat drafts: {:?}", dr.tokens);
        // Seen three times: the estimate rises and the cost model drafts a longer block.
        let mut p3 = block.clone();
        p3.extend_from_slice(&block);
        p3.extend_from_slice(&block);
        p3.extend_from_slice(&block[..6]);
        d.begin(&p3);
        assert!(d.draft(&p3, 16).tokens.len() > dr.tokens.len());
        // Fresh tokens: no match, no draft.
        let fresh: Vec<u32> = (1000..1040).collect();
        d.begin(&fresh);
        assert!(d.draft(&fresh, 16).tokens.is_empty());
    }

    #[test]
    fn global_tree_remembers_finished_requests() {
        let mut d = SuffixDrafter::new(SuffixConfig::default());
        let past: Vec<u32> = (200..240).collect();
        d.finish(&past);
        let now: Vec<u32> = vec![5, 6, 200, 201, 202, 203, 204];
        d.begin(&now);
        let dr = d.draft(&now, 16);
        assert!(dr.from_global);
        assert_eq!(dr.tokens[0], 205);
        d.clear_global();
        assert!(d.draft(&now, 16).tokens.is_empty());
    }

    #[test]
    fn cost_model_keeps_short_uncertain_drafts_off() {
        let mut d = SuffixDrafter::new(SuffixConfig::default());
        // "1 2" was followed by 3, 4, 5 and 6 once each: estimate 1/5 per branch -> not worth a verify.
        let ctx = [1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2];
        d.begin(&ctx);
        assert!(d.draft(&ctx, 16).tokens.is_empty());
    }
}
