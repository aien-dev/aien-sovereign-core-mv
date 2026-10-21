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
