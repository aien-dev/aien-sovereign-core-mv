//! Shared next-token sampling for engines that produce logits on the host.
//!
//! The rule (OWN-8B stage S, unchanged from before it, written down here):
//! - `temperature <= 0.001` is argmax, whatever the seed.
//! - Otherwise: top-k mask (here, because the workspace `SamplingParams` has no
//!   top-k field), then the workspace `sample_with_params` math (softmax at
//!   1/temperature in f64, nucleus cut at `top_p`, one uniform draw) seeded per
//!   step by `decode_sampling_seed(seed, step)`.
//!
//! So the same seed and prompt give the same token stream on the same build and
//! weights. Every draw is observed ([`SamplingObserver`]): the request's
//! [`SamplingRecord`] carries the kept-set size of each sampled step, so a model
//! that is confident (kept set of 1) can be told apart from a sampler that
//! ignores its seed (kept set above 1, replies still identical).
//!
//! Negative controls (`AIEN_SAMPLING_MUTANT`, builds with debug assertions only,
//! read once at start): `ignore-seed` draws every request with seed 0,
//! `force-greedy` takes the argmax at any temperature. A build without debug
//! assertions refuses to start when the variable is set.

use std::sync::OnceLock;

use aien_inference_abi::transformer_backend::{decode_sampling_seed, sample_with_params_traced};
use aien_inference_abi::{DecodeObservation, SamplingParams};
use serde::Serialize;

use crate::engine::GenerateParams;

/// The environment variable that selects a sampling mutant.
pub const MUTANT_ENV: &str = "AIEN_SAMPLING_MUTANT";

/// A deliberately broken sampler, for the gate's negative controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingMutant {
    /// Every request is drawn with seed 0.
    IgnoreSeed,
    /// Argmax at any temperature.
    ForceGreedy,
}

impl SamplingMutant {
    pub fn name(self) -> &'static str {
        match self {
            Self::IgnoreSeed => "ignore-seed",
            Self::ForceGreedy => "force-greedy",
        }
    }
}

/// Parses the value of [`MUTANT_ENV`]. `debug_build` says the binary was built
/// with debug assertions; without them any non-empty value is refused.
pub fn parse_mutant(
    value: Option<&str>,
    debug_build: bool,
) -> Result<Option<SamplingMutant>, String> {
    let v = match value.map(str::trim) {
        None | Some("") => return Ok(None),
        Some(v) => v,
    };
    if !debug_build {
        return Err(format!(
            "{MUTANT_ENV}={v} is refused: sampling mutants exist only in builds with debug assertions"
        ));
    }
    match v {
        "ignore-seed" => Ok(Some(SamplingMutant::IgnoreSeed)),
        "force-greedy" => Ok(Some(SamplingMutant::ForceGreedy)),
        _ => Err(format!(
            "{MUTANT_ENV}={v} is not a mutant (ignore-seed | force-greedy)"
        )),
    }
}

static MUTANT: OnceLock<Option<SamplingMutant>> = OnceLock::new();

/// Reads [`MUTANT_ENV`] once for the whole process (the server calls this at start).
/// An error means the server must not start.
pub fn install_mutant_from_env() -> Result<Option<SamplingMutant>, String> {
    let parsed = parse_mutant(
        std::env::var(MUTANT_ENV).ok().as_deref(),
        cfg!(debug_assertions),
    )?;
    Ok(*MUTANT.get_or_init(|| parsed))
}

/// The process's mutant; `None` until [`install_mutant_from_env`] ran, and always
/// `None` in a build without debug assertions.
pub fn active_mutant() -> Option<SamplingMutant> {
    if cfg!(debug_assertions) {
        MUTANT.get().copied().flatten()
    } else {
        None
    }
}

/// How one request was sampled, as observed at each draw. Goes into the server
/// log line and the reply (`usage.aien_sampling`, an extra field OpenAI clients ignore).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SamplingRecord {
    /// `greedy` (argmax at every step) or `sampled`.
    pub mode: &'static str,
    /// The seed the draws used (the request's, or the server's choice when the
    /// request sent none). Send it back to replay the request.
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Whether the top-k mask ran (`0 < top_k < vocab` and not greedy).
    pub top_k_applied: bool,
    /// Draws decided by argmax. Every draw counts, including the one that picked a
    /// stop token, so `greedy_tokens + sampled_tokens` can be one more than the reply's
    /// `completion_tokens`.
    pub greedy_tokens: usize,
    /// Draws made at random (same counting as `greedy_tokens`).
    pub sampled_tokens: usize,
    /// Kept-set size of every sampled draw, in step order (empty when greedy).
    pub kept: Vec<usize>,
    /// Smallest, median (lower middle), largest kept-set size and the mean;
    /// all 0 when nothing was sampled.
    pub kept_min: usize,
    pub kept_median: usize,
    pub kept_max: usize,
    pub kept_mean: f64,
    /// Sampled draws whose kept set was a single token (the seed had no choice).
    pub kept_single: usize,
    /// Smallest and mean probability of the most likely token at a sampled draw (after
    /// temperature, before the nucleus cut); 1.0 when nothing was sampled. Shows confidence
    /// where the kept set cannot (at top_p 1.0 every token is kept).
    pub top_prob_min: f64,
    pub top_prob_mean: f64,
    /// Natural-log probability of this exact sequence of sampled draws under the kept
    /// distributions (the sum of each chosen token's log-probability); 0 when nothing was sampled.
    pub reply_logprob: f64,
    /// `exp(reply_logprob)`: the chance that an honest sampler with another seed repeats these
    /// draws exactly. Identical replies across seeds are expected when this is large and are a
    /// sign of an ignored seed when it is small.
    pub p_repeat: f64,
    /// The sampling mutant in force, if any (debug builds only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutant: Option<&'static str>,
}

impl SamplingRecord {
    /// One-line form for the server log.
    pub fn log_fields(&self) -> String {
        let mut s = format!(
            "sampling mode={} seed={} temp={} top_p={} top_k={} greedy={} sampled={} kept(min/med/mean/max)={}/{}/{:.2}/{} kept1={}/{} top_prob(min/mean)={:.4}/{:.4} p_repeat={:.3e}",
            self.mode,
            self.seed,
            self.temperature,
            self.top_p,
            self.top_k,
            self.greedy_tokens,
            self.sampled_tokens,
            self.kept_min,
            self.kept_median,
            self.kept_mean,
            self.kept_max,
            self.kept_single,
            self.sampled_tokens,
            self.top_prob_min,
            self.top_prob_mean,
            self.p_repeat
        );
        if let Some(m) = self.mutant {
            s.push_str(&format!(" MUTANT={m}"));
        }
        s
    }
}

/// The sampler of one request: picks tokens and records how.
#[derive(Debug, Clone)]
pub struct SamplingObserver {
    params: GenerateParams,
    mutant: Option<SamplingMutant>,
    top_k_applied: bool,
    greedy: usize,
    kept: Vec<usize>,
    top_prob_min: f64,
    top_prob_sum: f64,
    logprob_sum: f64,
    obs: DecodeObservation,
}

impl SamplingObserver {
    /// An observer under the process's mutant (none in a normal server).
    pub fn new(params: &GenerateParams) -> Self {
        Self::with_mutant(params, active_mutant())
    }

    /// An observer under an explicit mutant (tests).
    pub fn with_mutant(params: &GenerateParams, mutant: Option<SamplingMutant>) -> Self {
        Self {
            params: params.clone(),
            mutant,
            top_k_applied: false,
            greedy: 0,
            kept: Vec::new(),
            top_prob_min: 1.0,
            top_prob_sum: 0.0,
            logprob_sum: 0.0,
            obs: DecodeObservation::default(),
        }
    }

    fn seed(&self) -> u64 {
        match self.mutant {
            Some(SamplingMutant::IgnoreSeed) => 0,
            _ => self.params.seed,
        }
    }

    fn temperature(&self) -> f32 {
        match self.mutant {
            Some(SamplingMutant::ForceGreedy) => 0.0,
            _ => self.params.temperature,
        }
    }

    /// Picks the token for decode step `step` (0-based count of tokens already
    /// generated in this request).
    pub fn sample(&mut self, logits: &[f32], step: usize) -> u32 {
        let temperature = self.temperature();
        let k = self.params.top_k;
        let masked;
        let logits = if k > 0 && k < logits.len() && temperature > 0.001 {
            self.top_k_applied = true;
            masked = top_k_mask(logits, k);
            masked.as_slice()
        } else {
            logits
        };
        let sp = SamplingParams {
            temperature,
            top_p: self.params.top_p,
            max_tokens: 1,
            stop_token_ids: Vec::new(),
        };
        let seed = decode_sampling_seed(self.seed(), step);
        let (token, logprob, trace) = sample_with_params_traced(logits, &sp, seed, &mut self.obs);
        if temperature > 0.001 {
            self.kept.push(trace.kept);
            self.top_prob_min = self.top_prob_min.min(trace.top_prob);
            self.top_prob_sum += trace.top_prob;
            self.logprob_sum += f64::from(logprob);
        } else {
            self.greedy += 1;
        }
        token
    }

    /// The record of every draw so far.
    pub fn record(&self) -> SamplingRecord {
        let mut sorted = self.kept.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let (min, median, max, mean) = if n == 0 {
            (0, 0, 0, 0.0)
        } else {
            (
                sorted[0],
                sorted[(n - 1) / 2],
                sorted[n - 1],
                sorted.iter().sum::<usize>() as f64 / n as f64,
            )
        };
        SamplingRecord {
            mode: if self.temperature() > 0.001 {
                "sampled"
            } else {
                "greedy"
            },
            seed: self.seed(),
            temperature: self.params.temperature,
            top_p: self.params.top_p,
            top_k: self.params.top_k,
            top_k_applied: self.top_k_applied,
            greedy_tokens: self.greedy,
            sampled_tokens: n,
            kept_min: min,
            kept_median: median,
            kept_max: max,
            kept_mean: mean,
            kept_single: sorted.iter().take_while(|&&k| k == 1).count(),
            top_prob_min: self.top_prob_min,
            top_prob_mean: if n == 0 {
                1.0
            } else {
                self.top_prob_sum / n as f64
            },
            reply_logprob: self.logprob_sum,
            p_repeat: self.logprob_sum.exp(),
            kept: self.kept.clone(),
            mutant: self.mutant.map(SamplingMutant::name),
        }
    }
}

/// Picks the next token from `logits` for decode step `step`, unobserved (the
/// same choice as [`SamplingObserver::sample`]).
pub fn sample_next(logits: &[f32], params: &GenerateParams, step: usize) -> u32 {
    SamplingObserver::new(params).sample(logits, step)
}

/// Rank order: higher value first, lower token id first on exact ties.
fn rank_cmp(logits: &[f32], a: usize, b: usize) -> std::cmp::Ordering {
    logits[b]
        .partial_cmp(&logits[a])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.cmp(&b))
}

/// Copies `logits` with everything outside the `k` largest values set to -inf.
/// Ties at the boundary keep the lower token id, matching the nucleus tie rule.
///
/// Uses an O(V) partial selection. The comparator is a strict total order only
/// when no logit is NaN (NaN compares Equal to everything, which is not
/// transitive), so any NaN input takes the original full-sort path to keep its
/// exact historical output.
fn top_k_mask(logits: &[f32], k: usize) -> Vec<f32> {
    if logits.iter().any(|v| v.is_nan()) {
        return top_k_mask_full_sort(logits, k);
    }
    let mut out = vec![f32::NEG_INFINITY; logits.len()];
    if k == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    let winners = if k < order.len() {
        order.select_nth_unstable_by(k - 1, |&a, &b| rank_cmp(logits, a, b));
        &order[..k]
    } else {
        &order[..]
    };
    for &i in winners {
        out[i] = logits[i];
    }
    out
}

/// The original O(V log V) implementation: reference for tests and the path for
/// vectors containing NaN.
fn top_k_mask_full_sort(logits: &[f32], k: usize) -> Vec<f32> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| rank_cmp(logits, a, b));
    let mut out = vec![f32::NEG_INFINITY; logits.len()];
    for &i in &order[..k] {
        out[i] = logits[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(temperature: f32, top_k: usize) -> GenerateParams {
        GenerateParams {
            max_new_tokens: 1,
            temperature,
            top_p: 1.0,
            top_k,
            seed: 42,
            stop_token_ids: vec![],
        }
    }

    #[test]
    fn greedy_is_argmax() {
        assert_eq!(sample_next(&[0.1, 5.0, 2.0], &p(0.0, 0), 0), 1);
    }

    #[test]
    fn top_k_never_picks_outside_the_k_best() {
        let logits = [1.0, 9.0, 8.0, 0.5, 7.9];
        for step in 0..200 {
            let t = sample_next(&logits, &p(2.0, 2), step);
            assert!(t == 1 || t == 2, "step {step} picked {t}");
        }
    }

    #[test]
    fn same_seed_same_token() {
        let logits = [1.0, 1.1, 0.9, 1.05];
        for step in 0..20 {
            assert_eq!(
                sample_next(&logits, &p(1.0, 0), step),
                sample_next(&logits, &p(1.0, 0), step)
            );
        }
    }

    /// Tokens of 16 steps over a flat 8-way distribution for `seed` under `mutant`.
    fn stream(seed: u64, temperature: f32, mutant: Option<SamplingMutant>) -> Vec<u32> {
        let params = GenerateParams {
            seed,
            temperature,
            ..p(temperature, 0)
        };
        let mut o = SamplingObserver::with_mutant(&params, mutant);
        (0..16).map(|s| o.sample(&[0.0; 8], s)).collect()
    }

    #[test]
    fn seeds_spread_and_mutants_break_the_spread() {
        let honest: Vec<_> = (1..=3).map(|s| stream(s, 1.0, None)).collect();
        assert!(honest[0] != honest[1] && honest[1] != honest[2] && honest[0] != honest[2]);
        let ign: Vec<_> = (1..=3)
            .map(|s| stream(s, 1.0, Some(SamplingMutant::IgnoreSeed)))
            .collect();
        assert!(
            ign[0] == ign[1] && ign[1] == ign[2],
            "ignore-seed must not spread"
        );
        let fg = stream(1, 1.0, Some(SamplingMutant::ForceGreedy));
        assert_eq!(
            fg,
            stream(2, 0.0, None),
            "force-greedy is the argmax stream"
        );
    }

    #[test]
    fn record_counts_kept_sets_and_modes() {
        let params = GenerateParams {
            top_p: 0.8,
            ..p(0.7, 0)
        };
        let mut o = SamplingObserver::with_mutant(&params, None);
        o.sample(&[10.0, 0.0, 0.0, 0.0], 0); // confident: kept 1
        o.sample(&[0.0; 4], 1); // flat: nucleus 0.8 of 4 equal keeps 4
        o.sample(&[0.0; 4], 2);
        let r = o.record();
        assert_eq!(r.mode, "sampled");
        assert_eq!(r.seed, 42);
        assert_eq!(r.kept, vec![1, 4, 4]);
        assert_eq!(
            (r.kept_min, r.kept_median, r.kept_max, r.kept_single),
            (1, 4, 4, 1)
        );
        assert!((r.kept_mean - 3.0).abs() < 1e-12);
        assert_eq!((r.greedy_tokens, r.sampled_tokens), (0, 3));
        // Two flat draws of 4: the chance of repeating them is (1/4)^2.
        assert!((r.p_repeat - 1.0 / 16.0).abs() < 1e-6, "{r:?}");
        assert!((r.reply_logprob - 2.0 * 0.25f64.ln()).abs() < 1e-5);
        assert!((r.top_prob_min - 0.25).abs() < 1e-12);
        assert!((r.top_prob_mean - 0.5).abs() < 1e-3, "{r:?}");
        assert!(
            r.log_fields().contains(
                "kept(min/med/mean/max)=1/4/3.00/4 kept1=1/3 top_prob(min/mean)=0.2500/0.5000 p_repeat=6.250e-2"
            ),
            "{}",
            r.log_fields()
        );

        let mut g = SamplingObserver::with_mutant(&p(0.0, 5), None);
        g.sample(&[0.0, 1.0], 0);
        let r = g.record();
        assert_eq!(
            (r.mode, r.greedy_tokens, r.sampled_tokens),
            ("greedy", 1, 0)
        );
        assert!(!r.top_k_applied && r.kept.is_empty());
        assert_eq!(
            (r.p_repeat, r.top_prob_min, r.top_prob_mean),
            (1.0, 1.0, 1.0)
        );

        let mut m = SamplingObserver::with_mutant(&params, Some(SamplingMutant::IgnoreSeed));
        m.sample(&[0.0; 4], 0);
        let r = m.record();
        assert_eq!((r.seed, r.mutant), (0, Some("ignore-seed")));
        assert!(r.log_fields().ends_with("MUTANT=ignore-seed"));
    }

    /// Deterministic xorshift for reproducible vectors.
    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn check_all_k(logits: &[f32], ks: &[usize]) {
        for &k in ks.iter().filter(|&&k| k <= logits.len()) {
            assert!(
                same_bits(&top_k_mask(logits, k), &top_k_mask_full_sort(logits, k)),
                "k={k} len={}",
                logits.len()
            );
        }
    }

    #[test]
    fn top_k_selection_matches_full_sort_random_and_tied() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        for len in [1usize, 2, 5, 64, 1000, 4097] {
            let ks = [0, 1, 20, 100, len.saturating_sub(1), len];
            // Random finite, with negatives.
            let v: Vec<f32> = (0..len)
                .map(|_| (rng(&mut st) % 20001) as f32 / 1000.0 - 10.0)
                .collect();
            check_all_k(&v, &ks);
            // Tie-heavy: only 4 distinct values, includes -0.0 and 0.0.
            let t: Vec<f32> = (0..len)
                .map(|_| [-1.0f32, 0.0, -0.0, 2.5][(rng(&mut st) % 4) as usize])
                .collect();
            check_all_k(&t, &ks);
            // All equal.
            check_all_k(&vec![1.5; len], &ks);
            // Infinities mixed in.
            let i: Vec<f32> = (0..len)
                .map(|_| match rng(&mut st) % 6 {
                    0 => f32::INFINITY,
                    1 => f32::NEG_INFINITY,
                    _ => (rng(&mut st) % 7) as f32,
                })
                .collect();
            check_all_k(&i, &ks);
            // NaN mixed in: falls back to the original path, bit-identical.
            let n: Vec<f32> = (0..len)
                .map(|_| match rng(&mut st) % 5 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => (rng(&mut st) % 5) as f32,
                })
                .collect();
            // The original comparator is not a total order with NaN, so the
            // full sort may itself panic; the new path must match either way.
            for k in [0, 1, 20.min(len), len] {
                let a = std::panic::catch_unwind(|| top_k_mask(&n, k));
                let b = std::panic::catch_unwind(|| top_k_mask_full_sort(&n, k));
                match (a, b) {
                    (Ok(a), Ok(b)) => assert!(same_bits(&a, &b)),
                    (Err(_), Err(_)) => {}
                    _ => panic!("NaN behavior diverged at k={k}"),
                }
            }
        }
    }

    #[test]
    fn top_k_ties_at_cutoff_keep_lower_token_id() {
        let m = top_k_mask(&[1.0, 3.0, 1.0, 1.0, 3.0], 3);
        assert_eq!(m, vec![1.0, 3.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0]);
    }

    #[test]
    fn sampled_tokens_and_record_match_with_reference_mask() {
        let mut st = 7u64;
        let logits: Vec<f32> = (0..5000)
            .map(|_| (rng(&mut st) % 400) as f32 / 40.0 - 5.0)
            .collect();
        for k in [1usize, 20, 100] {
            let masked = top_k_mask(&logits, k);
            let reference = top_k_mask_full_sort(&logits, k);
            assert!(same_bits(&masked, &reference));
            let params = GenerateParams {
                top_p: 0.9,
                ..p(0.8, k)
            };
            let mut o = SamplingObserver::with_mutant(&params, None);
            let mut r = SamplingObserver::with_mutant(
                &GenerateParams {
                    top_k: 0,
                    ..params.clone()
                },
                None,
            );
            for step in 0..50 {
                assert_eq!(o.sample(&logits, step), r.sample(&reference, step));
            }
            assert_eq!(o.record().kept, r.record().kept);
        }
    }

    /// Microbenchmark: `cargo test --release -p aien-serve-http --lib
    /// sampling::tests::bench_top_k -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_top_k() {
        let mut st = 1u64;
        let logits: Vec<f32> = (0..152_064)
            .map(|_| (rng(&mut st) % 100_000) as f32 / 5000.0 - 10.0)
            .collect();
        for k in [1usize, 20, 100] {
            for (name, f) in [
                (
                    "full_sort",
                    top_k_mask_full_sort as fn(&[f32], usize) -> Vec<f32>,
                ),
                ("select", top_k_mask),
            ] {
                let mut t: Vec<f64> = (0..21)
                    .map(|_| {
                        let s = std::time::Instant::now();
                        std::hint::black_box(f(std::hint::black_box(&logits), k));
                        s.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "k={k:<3} {name:<9} median {:.3} ms min {:.3} max {:.3}",
                    t[10], t[0], t[20]
                );
            }
        }
    }

    #[test]
    fn mutant_variable_is_refused_without_debug_assertions() {
        assert_eq!(parse_mutant(None, false), Ok(None));
        assert_eq!(parse_mutant(Some(""), false), Ok(None));
        assert!(parse_mutant(Some("ignore-seed"), false).is_err());
        assert_eq!(
            parse_mutant(Some("ignore-seed"), true),
            Ok(Some(SamplingMutant::IgnoreSeed))
        );
        assert_eq!(
            parse_mutant(Some("force-greedy"), true),
            Ok(Some(SamplingMutant::ForceGreedy))
        );
        assert!(parse_mutant(Some("bogus"), true).is_err());
    }
}
