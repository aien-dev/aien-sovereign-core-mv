# SC-PREFILL-LOGITS tolerance (P-DRA-4 Lane P G1, #877)

Status: PROPOSED. It is enforced by `prefill-logits.om` but has not been measured against chip A/A or A/B
noise. Chip runs are NOT_RUN.

## What is compared
- Arm A is the legacy GEMV prefill (f32 activations, chunk 128).
- Arm B is the path under test, for example DRA-82 `AIEN_OMEGA_Q36_PF_QMMA=1`.
- Both arms read the same prompt ids. Each writes the last prompt token's logits as 248,320 LE f32
  (`qwen36_prefill_bench`, `NAME.logits.f32`).

## Rule
PASS requires all three of:
1. Every value is finite in both files.
2. Top-1 agrees. B's top-1 equals A's top-1 (a tie goes to the lowest id), or A's own logit gap between
   its top-1 and B's top-1 is below BAND = 0.1. BAND is the house `MARGIN_MIN`, written 2026-10-21 in
   `qwen38_omega_full.rs` / `qwen36_stroke_gate.rs`, and is never derived from a run.
3. max |A − B| ≤ TAU = 1.0 logit, inclusive.

The judge always reports these alongside the verdict: A's top-two margin, max |Δ| and its id, rms Δ,
top-16 overlap, nonfinite counts, TAU, and the flip gap.

## Why these numbers (Wolfram, quoted)
- B quantizes GEMM input activations to e4m3 with one power-of-two scale per row per 128-column block.
  The e4m3 unit roundoff is 2^-4: `N[2^-4, 6]` → 0.0625. The rms relative error of uniform rounding is
  2^-4/√6: `N[2^-4/Sqrt[6], 6]` → 0.0255155.
- That predicted 0.0255 agrees with Prefill's host measurement on a synthetic single GEMM (k 384):
  rms|Δy|/rms|y| = 0.0261, and worst |y_B − y_A| = 0.053 of Σ|terms|.
- The maximum of 248,320 Gaussian deviations sits near √(2 ln V) σ: `N[Sqrt[2 Log[248320]], 6]` →
  4.98447. So TAU = 1.0 allows a per-logit noise σ up to about `N[1.0/Sqrt[2 Log[248320]], 6]` → 0.200623
  logit before the max alone trips it.
- A 1.0 logit shift changes a token's probability ratio by at most `N[Exp[2*1.0], 6]` → 7.38906 (both
  ends move). TAU is therefore a numerical-fault screen, not a sampling-equivalence claim.
- File size check: `248320*4` → 993280.

## Path B numerics (from Prefill, DRA-82, CUDApp 721826bd)
- e4m3 quantization uses RNE and saturates. The scale is `2^(ceil(log2 amax)-8)` per row per 128-column
  block.
- QMMA accumulates in fp32 per 128-k block, then multiplies by act_scale·w_scale.
- This applies to the GDN/attention mix, o (multi-token only), and expert gate_up/down. A keeps f32
  activations.

## Risks / open
- Logit-level B-vs-A error through 40 layers is unmeasured (NOT_RUN). The single-GEMM 2.6 % may grow
  with depth. If the G1 chip slice shows max |Δ| > 1.0 together with a stable top-1, revisit TAU with
  the A/A and A/B distributions. Do not adjust TAU after looking at the results.
- The A/A noise floor (two A runs) is a required arm of the G1 slice. A is expected to be bit-identical
  run to run (`reps_bit_identical` in bench.json). If it is not, record that as a finding.
- One prompt checks only the last token. Mid-prompt errors show up only through state carried into
  later chunks (GDN state, KV).
