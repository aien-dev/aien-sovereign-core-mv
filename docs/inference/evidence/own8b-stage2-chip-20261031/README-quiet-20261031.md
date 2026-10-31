# own8b stage 2, quiet-window additions (2026-10-31, CoreChip)

Added to this directory after the shared-window runs listed in `README.md` (add-only). Flint stopped, driver
`/tmp/corechip/quiet/driver.sh`. Binary `own8b_check` sha256
`3e882deb82d174f755d689f5766b68cee0b81d3cf25a32f721b7f92122f674db`, rebuilt natively from PR head
`bf4512fb4102f9ce017012f5923dbe7132608e75` (identical sha to the earlier build), omega a647b6a9.
Chip runs: `--gb10 shared --quiet none` spark-window plus `flock -s /tmp/aien-gb10.lock`, telemetry `<step>.dmon`.
`SHA256SUMS-quiet-20261031` lists these files.

| Step | rc | Result |
|---|---|---|
| `8b-red-attn` (`--attention-mutant`, window 20261031T071603Z-911af4) | 1 | negative control FAILS as required: tokens 1/32, 0/32, 0/32, 0/32, 3/32 |
| `8b-bf16-host-golden` (host only, `own8b_check golden ~/models/qwen3-8b ... --new 32`) | 0 | bf16 greedy golden in this binary's format, `8b-bf16-host-golden.json` |
| `8b-bf16-green` (chip, bf16 checkpoint against that golden, window 20261031T071906Z-dd6a0f) | 1 | tokens 32/32 identical on all 5 prompts, prefill bit-equal; oracle A p95_rel 2.27e-3 / 4.19e-2 / 6.08e-2 at layers 0/17/35 over the 2e-3 bound with cosine >= 0.999984 (the same pattern as the FP8 runs) |

## Oracle B (design 2026-10-15 section 2.4 B)

- Free-running leg, MEASURED (`oracle-b-freerun.py`, output `oracle-b-freerun.txt`): first greedy divergence
  of the FP8 host golden (`SMOKE-8B-FP8-v1`) from the bf16 host golden: p1-one-token 0, p2-short 21,
  p3-medium, p4-code, p5-512 no divergence in 32. The prior guess ">= 8 on every prompt" is missed on
  p1-one-token only: a single-token prompt whose top-1 margin at position 0 is 0.134 (FP8) and 0.108 (bf16).
  Per the design this is a finding about the checkpoint for Drake, not an engine verdict.
- Teacher-forced leg (FP8 top-1 against bf16 tokens, >= 0.95 overall, no prompt below 0.90): NOT_RUN. No
  mode of `own8b_check` computes it; it needs host code that feeds the bf16 golden to the FP8 host engine.
