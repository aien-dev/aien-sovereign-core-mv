# Qwen3.8-27B decode ceiling on the DGX Spark

How fast can the 27B model write tokens on this machine at most? Writing one token reads every
weight the model uses once, so memory speed sets the limit.

- Weight bytes read per generated token, FP8 weights with the bf16 final layer (lm_head):
  26,893,352,960 bytes (calc:ca9900d8).
- Nominal memory speed of the DGX Spark (GB10, LPDDR5x): 273 GB/s.
- Ceiling: 273,000,000,000 / 26,893,352,960 = 10.15 tokens per second (calc:9822c2ba).
- At the best weight-streaming speed measured with our engine, about 219 GB/s (218.57 GB/s,
  calc:e726ff77), the same arithmetic gives 219,000,000,000 / 26,893,352,960 = 8.14 tokens per
  second (calc:c3bbcc06). That is the practical ceiling until the engine streams weights faster.

The live server's 7.63 tokens per second (median chip time of 4 runs of 256 greedy tokens,
evidence/qwen38-serve-fast-20261019) is 0.752 of the 10.15 ceiling (calc:64f74f90).

Source: the byte count and the 10.15 figure were first written down in the operator handoff
2026-10-18-qwen38-27b-on-omega (line 26), which lives only on the project's Spark; this file is
the public copy. The calc receipts are in calc-receipts/ next to this file. Check one with
`aien-calc verify calc-receipts/<file>.json` (aien-calc recomputes it from its own inputs).

What this does not cover: time to the first token (reading the prompt), which is limited by
compute, not by memory speed; and power.
