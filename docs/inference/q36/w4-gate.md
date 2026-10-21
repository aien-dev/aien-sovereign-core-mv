# Qwen3.6-35B-A3B 4-bit weights: accuracy gate (L355), frozen before any lossy run

Written 2026-10-21 ~02:55Z, before any W4 drift number exists. Adopted from plan review (Fable, ~/handoffs/l355/review-v1.txt).
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
