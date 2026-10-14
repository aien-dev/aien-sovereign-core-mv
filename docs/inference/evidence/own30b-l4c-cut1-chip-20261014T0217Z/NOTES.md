# W-L4c cut 1: verify-pass cost on the chip (HARDWARE, GB10, 2026-10-14)

Build: sc own30b/spec-wp 1667896 (cut 1) linked against omega own30b/rc d0889a94 (libomega_gpu.a 6a6cb4df..., copied from
W-RC's /tmp/wrc/omega; development override, not a qualified build). Checkpoint ~/models/qwen3-30b-a3b-instruct-2507-fp8,
FP8 everywhere, fused KV, whole-prompt driver on, AIEN_OMEGA_SPIN_US=2000 applied (cut 1 makes the spec binaries apply it).
Exact commands, binary hashes and archive hash in *.start; raw output in *.raw.

## Verify cost (probe.raw, p4-long at 657 positions, median of 6 after 1 warm-up, rewind after each call)
Before = receipt own30b-l4c-verify-probe-20261014T0207Z/p0 (sc f7ac6a4 = same code as L4b + RC merge, no spin window).
| rows m | before host path ms | before device-router path ms | after (cut 1) ms |
| 1 | 39.6 | 51.1 | 24.7 |
| 2 | 63.6 | 90.5 | 29.5 |
| 4 | 71.1 | 82.3 | 39.8 |
| 5 | 71.5 | 88.3 | 44.3 |
| 6 | 89.7 | 109.1 | 47.4 |
| 8 | 99.2 | 114.0 | 55.0 |
After, m=5: device calls 38.4 ms (GEMMs/quant 34.5 incl. launch, attention 3.6), host 5.9 ms (router 4.0, combine 0.9, norms 0.7, plan 0.3).
The GEMM part grows ~4 ms per extra row = the expert union (INFERRED from the per-row slope; experts read once per pass).
Plain decode in the same process: step 25.5 ms + lm_head 2.9 ms = 28.4 ms/token (35.2 tok/s).
lm_head is NOT in the verify ms above: the spec loop runs one 2.9 ms lm_head per produced token (no batched bf16 GEMV in omega).
Where the "before" time went: rayon pool wake-ups on 2-9 row batches (norms/residual/combine/plan: 7-25 ms), a serial
one-expert-at-a-time host router (9.5-17 ms), the device router kernel at small m (~0.46 ms per layer), and omega's 50 us
default spin wait (the spec binaries never applied AIEN_OMEGA_SPIN_US; attention alone 10 ms -> 3.4 ms with it).
Decode-rounding verify (the exact path, p0 vs p1 probes): m=4 126 -> 102 ms, m=8 196 -> 152 ms (spin window only; not reworked).

## Gate (gate.raw: own30b_spec_check --runs 2 --new 64 --verify both --wp-prefill, prompt-lookup drafter, unchanged)
RESULT PASS, negative-control mutants caught on both paths.
Decode-rounding verify: token-identical to spec-off 8/8 (as required). Still slower than off: 11-25 tok/s vs 36-37.
Whole-prompt verify: tok/s off vs on: p1 37.4 vs 59.9 (+60%), p2 36.6 vs 27.3, p3 36.0 vs 32.0, p4 35.6 vs 31.0.
Whole-prompt identity vs spec-off: 2 of 8 runs identical; p1 diverges at token 14, p3 at 7, p4 at 11 (deterministic).
Cause check (diagA.raw): the PRE-cut-1 code with the same archive diverges at exactly the same tokens with the same
acceptance counts, so cut 1 does not change the whole-prompt output (VERIFIED); the extra divergence vs L4b's 18/20
comes with the RC archive / RC merge (L4b used archive e059453f) (INFERRED: W8A8 activation rounding of the newer prefill
GEMM at near-ties; not traced). Whole-prompt verify output is therefore NOT plain-decode-identical; it stays opt-in.
