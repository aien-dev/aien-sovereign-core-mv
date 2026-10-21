# Qwen3.6-35B-A3B 4-bit weights: results (L355, 2026-10-21)

Rule: docs/inference/q36/w4-gate.md (frozen before every run). Evidence: evidence/l355-w4/.

## Speed and chip correctness (the chip side works)
- q36_levers on the served decode graph, same build, chip-exclusive (band 64-288; indicative, not a spark-window
  receipt): FP8 39.90 tok/s (0.631 of the 237 GB/s ceiling, calc:7ab2013f); W4 experts 47.94 (1.201x, calc:e336c63f;
  0.651 of ceiling, calc:29e8dede); W4 experts + trunk 54.07 (1.355x, calc:790b5f54; 0.597, calc:b86699b6).
- Chip vs host 4-bit identity (qwen38_omega_full, band 0.1): W4 experts and W4 experts + trunk p1..p4 32/32 PASS; the
  routing mutant and the chip-only nibble mutant FAIL. q36_levers served path: ALL IDENTICAL on five goldens.

## Accuracy verdicts (all REJECT)
| arm | goldens R1 (<= 1.07) | R2 confident (>= 0.998) | held-out R1 | verdict |
|---|---|---|---|---|
| w4e: experts, BEST rounding | 1.0945 | 0.9970 | 1.1377 | REJECT |
| w4em0: experts, layers 0-9 kept FP8 | 1.0978 | 0.9970 | not run | REJECT |
| w4g: experts, error-feedback (GPTQ-style) rounding | 1.1645 | 0.9957 | 1.1473 | REJECT |
| w4et: experts + trunk, BEST | 1.3678 | 0.9902 | not run | REJECT |

- Control (FP8 arm == L313 f32 reference on p1, 512/512, margins equal) PASS in every run; the 3-bit mutant REJECTs.
- Bootstrap 95 % intervals of R1 (blocks of 64 steps): w4g goldens 1.086-1.287, held-out 1.088-1.208; w4e held-out
  1.081-1.198. Paired w4g - w4e: +0.0037 nats per step, interval -0.0003 to +0.0080: error feedback did not help.
- The excess sits on a few positions per text (Fable review of the GPTQ plan: 20 positions carry 74 % of it), the
  pattern of a few fragile positions rather than smooth rounding error (whether they are flipped expert choices was not
  measured). That fits better rounding of the same grid not moving it.

## What this means
- The 4-bit expert format with one power-of-two scale per 32 weights (MXFP4) is not accurate enough to serve on this
  model, with either rounding. W4 stays off in the served build. The code (kernel, loader, store, gates) is kept: it is
  correct on chip and is reused below.
- Use that does not need the accuracy: the MTP draft head in 4 bits (docs/inference/q36/mtp-draft.md: 1.547x fewer
  bytes per emitted token on p1, 1.365x on chat at k=1, with the bf16 head 1.335x / 1.177x; the 4-bit figures assume
  FP8 acceptance, the 4-bit head is not measured: the store holds no lm_head or mtp.* tensors yet). The verify pass keeps the
  words exact, so a 4-bit draft only lowers acceptance, measured before it ships.
- Paths that could make 4-bit experts pass, both kernel changes, not tried here: finer scales (16 weights per scale with
  an FP8 scale, NVFP4-style) or a block Hadamard rotation of x before the expert matmul (MR-GPTQ). Each needs its own
  frozen arm.
- Limit, said plainly: the error-feedback arm was checked on synthetic data (output error drops; H = I gives omega's
  exact bytes; a rounding mutant goes red), not by a per-layer output-error measurement on the real calibration
  activations. A slip in that path cannot be ruled out; the held-out result (w4g about equal to w4e) and the method's
  known weak spot on MoE experts (one input H per layer for all 257 experts' gate|up) fit the result without one.

## Settled: the FP8 gen-p4 near-tie flip is the chip FP8 graph, not this branch
evidence/l355-w4/levers-fp8.json (03:52Z, eight minutes after the 03:44Z RM_ALLOC fault) has `all_identical: false`:
gen-p4 differs at generated token 4 (chip 8432, golden 7592). The reference margin there is 0.010 logit (L313
drift-p4 f32_margin[4]), a near-tie inside the identity gate's 0.1 band; p1, p2, p3 and chat are identical and both W4
arms are identical 5/5 to their host goldens. Not isolated: the flip came either with L356's a|b overlap lever
(0915000b, `AIEN_OMEGA_Q36_OVERLAP_AB`, merged into this branch) or with this branch's FP8-path edits (loader Option
fields, need_bytes, moe.rs). Two one-variable runs settle it (this build with the overlap switch off on gen-p4; the
pinned bin-0915000b on gen-p4). Done 2026-10-21 12:10Z (the full tool's memory margin cannot be met beside the 8-row 18091,
so through qwen38_omega_full, WL=5, FP8, gen-p4): base (sc 0915000b + lib 678df626) and this branch (pre-merge + lib 979f928a)
give the same JSON in every field except timings, both 31/32 with token 4 = 8432 (one flip inside the 0.1 band);
evidence/l355-w4/p4check/. The overlap lever was off in the original run (omega default, no env), so it is excluded too.
The flip is the chip FP8 arithmetic at a 0.010 margin; this branch's FP8 path changes no number.

Receipts are the JSON files under evidence/l355-w4/ (the per-run .log.txt captures came out empty and were removed).
