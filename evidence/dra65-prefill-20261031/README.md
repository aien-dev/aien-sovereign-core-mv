# DRA-65: chunked prefill through the door (2026-10-31)

Refs aien-dev/aien-sovereign-core-mv#856, Linear DRA-65.

## Which door serves, and what still read prompts one token at a time (source, origin/main d3dec0eb)

- Production 35B (last start 2026-10-30T18:49:06Z, `start-35b-batch.sh`, bin-19173f9e, `--batch-slots 8`) is the
  **batch door**: `main.rs` sends `--batch-slots > 0` with a MoE dir to `Qwen36BatchDoor::load`. It already reads
  prompts in chip chunks of 128 (L369; `qwen36_batch.rs` chunk block, BUILD.txt "L369 prefill (chunks of 128)").
  The Linear premise "the door reads prompts one token per beat" is **stale for the batch door**. What it still
  steps token by token: a stretch shorter than `AIEN_Q36_PREFILL_MIN` (16) before the snapshot point, and the
  answer header after the snapshot (~7 tokens), fed per token in the same beat.
- The **single door** (`Qwen38Door`, `--batch-slots 0`) read nearly the whole prompt one token per decode step:
  `session.rs` feed_rows -> `prefill_rows` MoE branch -> `self.step(t)` per token (chunk ignored). Only the last
  piece (`feed_last` -> `prefill`) was chunked.

## Change (IMPLEMENTED)

- `prefill.rs`: `Qwen38Omega::prefill_state(tokens, chunk)` feeds tokens for their state only; on the 35B as chip
  prefill chunks of `AIEN_Q36_PREFILL_CHUNK` (default 128, 0 = the old token-by-token path, same binary) at any
  length from 1. `prefill_rows` on the 35B now chunks all but the last token and steps the last one for its
  residual (the chip prefill keeps its rows in `pf.x`, not the decode graph's x, so `residual()` is only valid
  after a step); `all=true` still steps every token (its rows are not exported by the chip prefill).
  `q36_chunk_lengths` is the one chunk plan `prefill_moe_head` iterates.
- `session.rs`: `feed_rows` uses `prefill_state` (35B: never stepped); `snapshot_point` and `rows_by_steps` are
  the pure split rules start_inner uses; off-mode tokens are counted in `FeedStats::stepped`.
- Signatures of `Qwen38Session::{start, advance, advance_greedy}` and `RowStep::{step_rows, prefill_row}`
  unchanged. Batch door code unchanged (see "Next lever").

## 32-prompt set (pinned)

`set32.json` = the merge train's concurrent-32 prompts (`evidence/q36-batch-20261021/prompts.txt`, max_tokens
`i==0 ? 24 : 8 + (i*7) % 41`), goldens = reference build **bin-5b726818** ("plain correct path: prompt fed token
by token, no speed levers"), each request alone, recorded 2026-10-21 (before this change). Prompt 0's ids equal
the host-oracle golden `evidence/q36-host-20261020/gen-chat-max-24.json`. `long-goldens.json` adds the 365 /
2486 / 8210-token prompts with goldens from bin-16b0e6f8 (batch door, prompts token by token). 35 cases in all.
Judge: exact id identity (the door returns no logit margins, so the 0.1-logit band cannot be judged through it;
any flip is FAIL). `SHA256SUMS` pins every input; the script refuses to run if one changed.

## Host evidence (EXERCISED on host, not chip)

`nice -n 19 taskset -c 0-9 cargo test -p aien-inference-abi --lib qwen38_omega`: 9 passed (chunk boundaries at
1, 127, 128, 129, 2048, 8192 and 8210; env switch 0/default/cap/129/garbage; snapshot point and rows/last split at
the same lengths and on turn 2; 35B rows never stepped). Negative control: `q36_chunk_lengths` mutated to drop a
remainder of 1 -> `q36_chunks_at_the_boundaries` FAILED, then restored. Judge exercised against a mock door
(exact goldens / one id changed): green PASS + red FAIL, and green FAIL (exit 1) + red PASS.

Mutant 4 wiring (source, omega 474197e4f): `AIEN_Q36_TEST_MUTANT=4` -> `moe.rs` `graph_mutant()` ->
`test_set_mutant` on the main graph and (`rows.rs` wire) every row and stroke graph; in C mutant 4 swaps K_CN for
K_CN_M3 in the decode path and in `omega_gpu_q36_graph_prefill` (the chunk path), K_CN_R for K_CN_R_M3 in rows.
So the env mutant reaches the chunked prefill. The per-request `aien_test` mutant does not (the batch door steps
such rows token by token), so the window uses the env mutant.

## Window (Main; chip gates NOT_RUN until then)

Quiet exclusive window per the usual procedure (marker, stop 18091 and Flint, `spark-window open ... --gb10
exclusive --quiet exclusive`). NEW = the build of this branch. Each line: start the server, run, stop.

1. Single door, NEW, `AIEN_Q36_TEST_MUTANT=4`: `window-dra65.sh golden PORT OUT/w1-mutant red` -> must PASS (red).
2. Single door, NEW: `window-dra65.sh golden PORT OUT/w2-single green` -> PASS = 35/35 identical.
3. Batch door (`--batch-slots 8`), NEW: `window-dra65.sh golden PORT OUT/w3-batch green`.
4. `window-dra65.sh time PORT OUT/w4-<arm>` on: single NEW; single NEW with `AIEN_Q36_PREFILL_CHUNK=0` (before
   arm); batch NEW; batch NEW with `AIEN_Q36_PREFILL_MIN=1` (chunks the answer header too).
   Receipts: `time.jsonl` (t0, first content delta, end, usage per run). TTFT = t_first - t0; e2e tok/s =
   completion_tokens / (t_end - t_first). Medians of the 3 runs through Wolfram (`Median[{...}]`).
   Target 1 compares e2e against chip-only decode from DRA-66's `chip_ns` in the same runs.

## Predictions (MODELED, Wolfram)

- Single door before: 1 token per decode step, 20.1 ms -> `N[1/0.0201]` = 49.75 tok/s; 512 tokens
  `N[512*0.0201]` = 10.29 s; 8210 tokens `N[8210*0.0201]` = 165.0 s.
- Single door after, at the measured prefill-gate rates (250 tok/s at 2k, 196.8 at 8k): 512 tokens
  `N[512/250.]` = 2.05 s; 8210 tokens `N[8210/196.8]` = 41.7 s.
- Target 2 (2,000 tok/s at 2k) needs `N[2000/250.]` = 8x the current chunked rate: not reachable by
  chunking alone.

## Next lever (batch and single door alike; omega kernel work, not host-testable)

Expert bytes per 128-token chunk if every expert is read once per layer: `256*(3145728+384)*40` = 32,216,186,880
B; at 273 GB/s `N[perChunk/273*^9]` = 0.118 s, i.e. a byte floor of `N[128/0.118...]` = 1,084.7 tok/s; the
measured 250 tok/s is `N[(128/250.)/0.118...]` = 4.34x above that floor, so the chunk is not bytes-bound. The
FLOP floor for 2,048 tokens at ~3e9 active parameters and 237.5 TFLOPS is `N[2*3*^9*2048/237.5*^12]` = 0.052 s.
The lever is compute efficiency of the prefill GEMMs: a tensor-core grouped expert GEMM (OMEGA.md bar C3) in
`omega_gpu_q36_graph_prefill`, then chunks above 128 (the 1024-row fp8 GEMM cap). The batch door's remaining
per-token stretches (< 16 before the snapshot, the ~7-token header) are an env A/B (`AIEN_Q36_PREFILL_MIN=1`,
step 4) because small-chunk chip cost is unmeasured.
