# Oracle B teacher-forced leg, 2026-10-31 (CoreChip)

Supersedes the "Teacher-forced leg: NOT_RUN" line of `README-quiet-20261031.md` (that file is left
unchanged; evidence is add-only).

Tool: `own8b_check oracle-b`, added in this branch (sovereign-core commit
8efd27832838b67b4ba2c809341fa92bbf548c76). Host only, no GPU. It loads the FP8 checkpoint's host
engine, prefills each golden prompt, and for every golden position compares the FP8 host top-1 with
the golden token, then feeds the golden token (teacher forcing). Binary sha256 and command are in
each `*.start`.

| Run | golden | rc | agree | per prompt |
|---|---|---|---|---|
| oracle-b-selfcontrol | `docs/campaigns/open-model-qwen3/SMOKE-8B-FP8-v1.golden.json` (FP8's own golden) | 0 | 160 of 160 | 32/32 on all 5 |
| oracle-b-teacher | `8b-bf16-host-golden.json` (bf16 checkpoint, this directory) | 0 | 157 of 160 = 0.98125 | p1 30/32 = 0.9375 (first disagreement at 0), p2 31/32 = 0.96875 (at 21), p3 p4 p5 32/32 |

The self-control shows the counting reproduces a checkpoint's own golden exactly. The teacher run
shows the tool does report disagreements (3 positions).

Verdict against the design's prior guesses (>= 0.95 overall, no prompt below 0.90): PASS
(overall 0.98125, worst prompt 0.9375). Golden length is 32 tokens per prompt, not the design's 48.
Rates checked with Wolfram: `N[{157/160, 30/32, 31/32}, 6] -> {0.98125, 0.9375, 0.96875}`.

`SHA256SUMS-oracle-b-20261031` covers the `oracle-b-selfcontrol.*`, `oracle-b-teacher.*` files and
this README.
