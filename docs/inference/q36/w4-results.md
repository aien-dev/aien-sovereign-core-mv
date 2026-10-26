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
| nv16: experts, e2m1 x e4m3 scale per 16 x f32 global | not run | not run | not run | NO-GO before the arm (weight error) |

- Control (FP8 arm == L313 f32 reference on p1, 512/512, margins equal) PASS in every run; the 3-bit mutant REJECTs.
- Bootstrap 95 % intervals of R1 (blocks of 64 steps): w4g goldens 1.086-1.287, held-out 1.088-1.208; w4e held-out
  1.081-1.198. Paired w4g - w4e: +0.0037 nats per step, interval -0.0003 to +0.0080: error feedback did not help.
- The excess sits on a few positions per text (Fable review of the GPTQ plan: 20 positions carry 74 % of it), the
  pattern of a few fragile positions rather than smooth rounding error (whether they are flipped expert choices: measured
  below, they are not). That fits better rounding of the same grid not moving it.
- Router-flip check (NV16 go/no-go (a), frozen in w4-gate.md, evidence/l355-w4/routes/verdict.json): on the w4e forced
  runs (1024 steps) the 20 steps with the most excess differ from FP8 in the top-8 expert picks on 18.8 layers on average,
  the other steps on 23.2 (ratio 0.81; NO-GO needs 2 or more). The excess is not driven by flipped expert choices: GO.
- Weight-error check (NV16 go/no-go (b), evidence/l355-w4/nv16-gonogo-b.json): the held-out gap needs the squared weight
  error at about half of BEST's, limit 0.5 x 0.112546^2 = 0.0063333 (calc:466b6d16). The full nv16 store (160 tensors,
  33.6 G weights, 263 s) gives rel_rms 0.092202, squared 0.0085: NO-GO. Finer scales cut the rms error by 18 %
  (calc:7f0abae2), short of the halved square the gap needs, so the rule stopped the nv16 arm before a model load.
  Fable's final review: the squared-error ratio nv16 / BEST is 0.671 (calc:4993a394); under the second-order reading the
  projected held-out R1 is 1.0924 (over 1.07) while the goldens would project 1.0634 (under), so the stop is set by the
  held-out line B. The second-order reading is the generous one (at a position where the nudge crosses a margin, loss
  grows about linearly, projecting 1.1128), and nv16 clips 1.74x more values than BEST. Limit: the held-out interval
  1.081 to 1.198 projects to about 1.054 to 1.133, so this is a point-estimate stop whose interval includes a pass; a
  follow-up re-opens nv16 with that evidence, not by argument.
- Mixed inside the expert (w4-gate.md NV16 item 4, frozen before the measurement; evidence/l355-w4/split/split-16.json):
  on the survey's expert sample (16 experts + the shared one per layer, 2.14 G weights; its nv16 squared error 0.0085029
  matches the full store's 0.0085012), BEST 0.0126745, limit 0.0063373. nv16 gate|up with FP8 down (m-gu) 0.0057170: GO;
  FP8 gate|up with nv16 down (m-dn) 0.0027859: GO. Per matrix the two formats err alike (gate|up 0.0922, down 0.0922 rms),
  so the split follows the bytes. m-gu (0.708 of FP8 expert bytes, calc:449379b7) runs first as host arm w4mgu, A + B + C,
  every load under q36-extra; m-dn only if m-gu REJECTs. Result: see the verdict files verdict-w4mgu.json / verdict-w4mdn.json.

## What this means
- The 4-bit expert format with one power-of-two scale per 32 weights (MXFP4) is not accurate enough to serve on this
  model, with either rounding. W4 stays off in the served build. The code (kernel, loader, store, gates) is kept: it is
  correct on chip and is reused below.
- Use that does not need the accuracy: the MTP draft head in 4 bits (docs/inference/q36/mtp-draft.md: 1.547x fewer
  bytes per emitted token on p1, 1.365x on chat at k=1, with the bf16 head 1.335x / 1.177x; the 4-bit figures assume
  FP8 acceptance, the 4-bit head is not measured: the store holds no lm_head or mtp.* tensors yet). The verify pass keeps the
  words exact, so a 4-bit draft only lowers acceptance, measured before it ships.
- Finer scales (nv16, NVFP4-style) were tried host-side and fail the weight-error go/no-go above. The one path left
  for 4-bit experts is a block Hadamard rotation of x before the expert matmul (MR-GPTQ), a kernel change that needs its own
  frozen arm, or mixed precision inside the expert (gate|up 4-bit, down FP8, by measured sensitivity). Neither is tried here.
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

## Low-rank down correction on m-dn (2026-10-26, lane W4Correct; rule: w4-gate.md "Low-rank arm", frozen f0c53629 + 33a6c59c)
- Plain screen (b-P), MEASURED, weights only (`q36_w4_store lowrank <M> q36-nv16-v1.store 16 <sidecar> - - all 6`,
  all 10,280 down matrices, bf16 U and V): sum |E|^2 = 10095.38, after rank 16 = 9407.98, ratio 0.93191 > 0.5 -> **NO-GO**
  (per layer 0.905 .. 0.933). nv16's down error is near full rank: rank 16 removes 6.8 % of it (WolframLanguageEvaluator:
  `1 - 9407.980470180006/10095.378622277292 -> 0.06809`); its relative squared error is 0.008513 (`10095.378622277292 /
  1.1859018473596044*^6 -> 0.0085128`). Receipt ~/handoffs/W4Correct/screen-plain.json (store binary sha256 in
  store-bin.sha256). The plain sidecar was deleted (that fit cannot run).
- Calibrated screen (b-C) and the arm: NOT_RUN. Calibration (hdump) and every forced run load the full 35B on the host,
  so they wait for an exclusive window Drake authorises; evidence/l355-w4/run-lowrank-window.sh runs them in order
  (resumable, also as a q27-window.sh hook) and stops at NO-GO on (b-C).
- Bytes (integer bookkeeping): 81,920 per down matrix, sidecar 842,137,664 B. MODELED (Wolfram, linear byte model fitted
  to the MEASURED FP8 39.90 and w4e 47.94 tok/s): m-dn 42.10 tok/s (1.055x FP8), m-dn + rank 16 41.69 tok/s (1.045x); an
  ACCEPT would buy about 4.5 % at N=1 decode before any chip kernel exists.
- Gate hygiene: the verdict is on fresh held-out h6..h10, not h1..h5 (every arm has seen those); h1 and the chat golden
  are diagnostics only. m-dn without h1 (0.99836) is not a pass.
