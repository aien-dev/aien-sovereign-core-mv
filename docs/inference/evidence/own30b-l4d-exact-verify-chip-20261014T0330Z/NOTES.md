# W-L4d: fast exact speculative verify (HARDWARE, GB10, 2026-10-14)
Builds: omega own30b/spec-exact b9ae0096 (cut A 2c41bd98 + rows attention), archive sha256 8d5a42e8 (dev override);
sc own30b/spec-wp ff919fc + this commit. Live server on :18091 untouched; only small synthetic or 6-real-layer loads.

VERIFIED (chip):
- r3-rows-attn: omega_gpu_decode_attn_rows vs per-row omega_gpu_decode_attn, 66 cases (pos0 0..2032, rows 1..16,
  split boundaries crossed): output and appended K/V bit-identical in all 66; mutant 2 caught 20/20.
- mode0-bytes-*: the existing decode attention kernel's encoded bytes unchanged (20 specs, host, no chip).
- r4-rows-attn-probe (6 real layers, p4-long KV): step_verify_exact rows bit-identical to plain step for m = 1..8;
  whole-prompt verify differs in every word (instrument can say no). Wall per 6 layers, m=5: old exact verify
  14.6 ms (r0) -> 9.54 ms; m=8 16.45 (r2) -> 12.72; m=1 2.63 ms (0.82x a plain step).
- r5-mutant-identity: the numerics mutant (bf16-truncated attention in verify) differs in every row.
- r6-gate-rehearsal-6layers: own30b_spec_check --layers 6, 4 golden prompts + code/JSON, suffix drafter, 2 runs,
  --verify decode,exact: every ON run IDENTICAL to OFF on both paths; extra-accept mutant caught on every prompt;
  numerics mutant caught on 4 of 5 prompts; RESULT PASS. Verify ms/round Exact 3.5-5.2 vs Decode 5.6-7.8.
INFERRED: on the full model the exact verify should cost about 0.65x the old decode-rounding verify per round.
UNKNOWN: full-model tok/s and identity. Prepared, not run: scripts/own30b-l4d-gate.sh (needs a pause window).
