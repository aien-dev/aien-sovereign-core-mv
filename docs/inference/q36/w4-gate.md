# Qwen3.6-35B-A3B 4-bit weights: accuracy gate (L355), frozen before any lossy run

Written 2026-10-21 ~02:15Z, before any W4 drift number exists. Adopted from plan review (Fable, ~/handoffs/l355/review-v1.txt).
A run that changes these numbers after seeing results is void.

## Arms and reference
- Reference: FP8 checkpoint weights, f32 DeltaNet state, host reference (sc qwen38.rs). Its greedy ids and top-1 margins for
  512 steps on the five goldens (gen-p1..p4, gen-chat-max-24; prompts from evidence/q36-host-20261020) come from L313's
  drift-*.json (f32 arm) where present, else from the FP8 arm of q36_w4_drift itself.
- Exactness control: the FP8 arm of q36_w4_drift, teacher-forced on L313's p1 ids, must give top-1 = those ids at every step
  and margins within 1e-3 of L313's f32_margin. If it fails, every number of the run is void.
- Arm under test: the same, with the 4-bit tensors read from the store (exactly the bytes the chip uploads). Teacher-forced
  on the reference ids: one flip does not cascade, every step is scored.
- Mutant (must REJECT, calibrated, not absurd): the same 4-bit experts with every e2m1 mantissa bit cleared (a 3-bit grid
  0, 1, 2, 4 x scale). If it passes, the rule is too weak and the run is void.

## Rule (every line must hold for ACCEPT)
- R1 (primary) mean NLL of the reference tokens over the 2560 steps: ratio arm / reference <= 1.07.
  Anchor: the 27B all-weights MXFP4 receipt, NLL 1.062x on LICENSE-256 (sc qwen38/mtp-spec-decode 891d33c6), accepted.
- R1b (secondary) AGPL passage, 400 tokens (L313's probe): mean NLL rise <= 0.069 nats (L313's adopted bound).
- R2 top-1 agreement: pooled >= 0.98 over steps with reference margin >= 0.1 (near-ties excluded, L313 convention), and
  >= 0.998 over steps with reference margin >= 1.0 logit.
- Reported, not gated: first divergence per golden in a free run (one flip changes every later word), KL, near-tie flips.

## Levers and stacking
- W4 experts (routed + shared, the graph's expert slabs) is judged alone first; W4 trunk (mix + o) is judged as
  experts + trunk. Before any swap request, the exact stack that would be served (with L313's state format if it is in the
  same build) is judged under this same rule.
- Chip implementation check (separate from accuracy): the chip with the 4-bit store must give the host 4-bit arm's words,
  by the motor identity gate (sc qwen38_omega_full, error band written in advance, L312).

## Scale rule (chosen by the survey, before any drift run)
evidence/l355-w4/survey.jsonl (16 routed experts per layer + the shared expert, and the whole trunk, every layer): relative
RMS error experts OCP 0.116313, FIT 0.118971, BEST 0.112581; trunk OCP 0.118405, FIT 0.121358, BEST 0.114692. The store uses
BEST (the smaller squared error per 32-weight group of OCP and FIT; the chip decodes every rule the same way).

## GPTQ arm (written 2026-10-21 before its calibration ran; stricter than the rule above, never looser)
Plain rounding (BEST) REJECTed: R1 1.0945, R2 confident 0.9970 (evidence/l355-w4/verdict-w4e.json); the layers 0-9 FP8 mix
REJECTed too (verdict-w4em0.json). One more arm, then no more: error-feedback rounding (sc q36_w4_gptq.rs), same store
format and kernel. Everything below is fixed now; nothing is retuned after a result.
- Arm w4g: experts (routed + shared) from a store with rule `gptq`; trunk unchanged (FP8). Calibration: FP8 host reference
  over ~/handoffs/l355/calib/calib-v1.txt (mind-map notes and ideas, repo docs, Rust and C sources; files holding any gate
  prompt or GNU licence text skipped; sha256 prefix 2bcdb4691789d2c6; file list in evidence/l355-w4/calib-v1-files.txt),
  first 32768 tokens, fresh context every 2048. H for gate|up = sum x x^T of every token's MoE input per layer; H for each
  down = sum g^2 a a^T over the rows routed to that expert (g = router weight, the shared expert's gate). Damp 0.01 x mean
  diagonal; BEST-style scale per group on the error-updated values (one up when the pass clips, lower loss wins).
- ACCEPT needs all three:
  A. The frozen rule above (control, R1, R1b, R2; mutant still REJECTs) on the five goldens.
  B. R1 and R2 (same thresholds) on five held-out prompts no arm has run before this line: h1..h5 below, 512 steps each,
     reference = the FP8 arm's own greedy ids, margins and NLL on them, the arm teacher-forced on those ids.
  C. Paired against plain rounding: per-step NLL(w4g) - NLL(w4e) over the 5120 steps of A and B (w4e also run on h1..h5),
     block bootstrap (blocks of 64 consecutive steps, 10000 resamples, seed 355): the 95 % upper bound of the mean
     difference is below 0. The verdict prints the bootstrap 95 % interval of the R1 ratio for A and for B.
- Held-out prompts (raw text, `\n` = newline):
  h1 `The main reasons the Roman Empire declined were`
  h2 `<|im_start|>user\nExplain how a refrigerator keeps food cold, in simple terms.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  h3 `fn main() {\n    let numbers = vec![3, 1, 4, 1, 5, 9];\n`
  h4 `Photosynthesis is the process by which`
  h5 `<|im_start|>user\nWrite a short email asking a colleague to move our Thursday meeting to Friday.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

## NV16 arm (written 2026-10-21 ~12:00Z before any nv16 or route run; plan ~/handoffs/l355/plan-nv16.md, Fable review adopted)
A finer 4-bit grid, NVFP4-style (sc q36_w4_nv.rs): e2m1 x e4m3 scale per 16 weights x one f32 global per linear (each
expert's gate, up and down), global = amax / (448 x 6), scale = nearest e4m3 code (one up when a value clips and that lowers
the squared error), e2m1 ties to the even code (cvt.rn). Host widen applies the global to each output once (the chip's
epilogue). An ACCEPT here says nothing about speed: that needs the chip kernel and the 8-head stroke receipt.
Order and go / no-go, decided now:
1. Diagnostics first (no verdict arm before both):
   a. Router flips: forced runs of the FP8 arm and w4e (MXFP4 BEST store) on gen-p4 and chat-max-24 with AIEN_L355_ROUTES=1
      (each step's top-8 per layer). Per step: layers whose top-8 set differs from FP8; excess = NLL(w4e) - NLL(FP8).
      NO-GO for every 4-bit grid if the mean count of differing layers at the 20 steps with the largest excess is at least
      2x the mean over the other steps AND those 20 steps carry at least half of the excess: then the error is router flips and
      W4 experts close for this model.
   b. Weight error: the nv16 store's squared relative error (rel_rms^2 over all its tensors) must be at most half of the BEST
      store's: rel_rms^2 <= 0.0063333 (= 0.5 x 0.112546^2, calc:466b6d16). Otherwise NO-GO (under the second-order
      reading, the held-out gap needs about that cut).
2. Then arm w4n: experts from the nv16 store, trunk FP8. ACCEPT = A (frozen rule, five goldens, control rerun with this build:
   forced fp8 p1 must equal forced-fp8-p1.json) AND B (held-out h1..h5) AND C (paired block bootstrap vs w4e, upper bound < 0).
   Controls: the mantissa mutant still REJECTs; the scale control w4e-smut (e4m3 scale mantissa cleared, powers of two at
   group 16) runs on the goldens, reported, not gated: it says whether a gain comes from the scale mantissa or the group size.
3. Fallback named now: if (1) says the error is smooth but w4n misses B, the next arm is mixed inside the expert (gate|up
   nv16, down FP8, or the reverse by per-matrix sensitivity), same A + B + C. If (1a) says router flips: no more 4-bit expert
   arms for this model; bytes go to the draft head and the accuracy-free levers.
4. Mixed go / no-go (written 2026-10-21 ~13:00Z after (1b) said NO-GO for nv16, before any split was measured; Fable
   final review point 2): `q36_w4_store split <dir> [n]` converts the survey's expert sample (ids 0, n/16 apart, plus the
   shared expert, every layer) with BEST and nv16 and prints sq_err and sq_ref for gate|up and down apart. Mixed arm
   m-gu (nv16 gate|up, FP8 down): sq = nv16 sq_err(gate|up) / sq_ref(all experts); m-dn (FP8 gate|up, nv16 down):
   sq = nv16 sq_err(down) / sq_ref(all experts). Each passes (b) only if its sq <= 0.5 x BEST's sq on the same sample.
   A pass only opens the mixed arm (needs per-tensor mixing in the loader and host reference, then A + B + C, then an nv16
   chip kernel); a fail for both closes 4-bit experts at any grid without a rotation.
   Order (written after the split, before any mixed run): m-gu first (more bytes off: 0.708 of FP8 expert bytes, calc:449379b7), m-dn only
   if m-gu REJECTs. Host arms w4e-mgu / w4e-mdn on the nv16 store, every 35B load under q36-extra (L366), one at a time.
   Sanity (reported, not gated): the mixed arm's per-prompt mean NLL beside the FP8 arm's and w4e's.
