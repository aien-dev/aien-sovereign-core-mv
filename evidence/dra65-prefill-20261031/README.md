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
  prefill chunks of `AIEN_Q36_PREFILL_CHUNK` (default 128) at any length from 1. `AIEN_Q36_PREFILL_CHUNK=0` makes
  the new binary feed EVERY prompt token through the decode graph, the answer-header piece included; the base
  binary's default already chunked that last piece when it was >= 3 tokens, so the CHUNK=0 arm is slower than the
  old production single door by about one header (`N[7*0.0201]` = 0.14 s), not equal to it. `prefill_rows` on
  the 35B now chunks all but the last token and steps the last one for its
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
2486 / 8210-token prompts and **turn 2** of long-2048 (2528 tokens, three messages, `turn2.messages.json`) with
goldens from bin-16b0e6f8 (prompts token by token). The script sends turn 2 right after long-2048, so the door
restores the snapshot it took at long-2048's answer header after a chunked prefill (`plan_reuse` ->
`Reuse::Snapshot`). 36 cases in all.

Provenance: both files name `source_dir` (the merge-train store); every case carries `ref_sha256` of its source
golden (`alone/<i>.json`, `l369-long/<name>.json`), and `long-goldens.json` carries the store's `done` receipt
text and sha256 (it names bin-16b0e6f8). `SHA256SUMS` pins every committed input; the script refuses to run if
one changed.

Judge: exact id identity (the door returns no logit margins, so the 0.1-logit band cannot be judged through it).
Per case SAME / DIFF (a flip inside the common length, or more ids than the golden) / SHORT (a strict prefix: no
flip shown) / NO_IDS (HTTP error, missing or empty ids). Red PASS needs a DIFF and no SHORT/NO_IDS (else
INCONCLUSIVE); green PASS needs 36 SAME (NO_IDS -> INCONCLUSIVE, else FAIL). Green refuses to run without two red
PASS receipts with distinct labels for the same server binary sha256.

## Host evidence (EXERCISED on host, not chip)

`nice -n 19 taskset -c 0-9 cargo test -p aien-inference-abi --lib qwen38_omega`: 10 passed (chunk boundaries at
1, 127, 128, 129, 2048, 8192 and 8210; env switch 0/default/cap/129/garbage; snapshot point and rows/last split at
the same lengths and on turn 2; 35B rows never stepped; the feed path below). Negative control: `q36_chunk_lengths`
mutated to drop a remainder of 1 -> `q36_chunks_at_the_boundaries` FAILED, then restored.

Feed path: `start_inner`'s feed is `session::feed_prompt` -> `feed_rows` / `feed_last` over the `PromptFeed`
trait, and the 35B's state feed is `prefill::q36_state` + `q36_chunk_walk` (the loop `prefill_moe_head` runs on
the chip) over `Q36Feed`; the model implements both. `every_prompt_token_is_fed_once_in_order` drives that same
code with a fake that records (position, token id, step/row chunk/head chunk) and refuses a chunk at a position
other than its own: lengths 1, 127, 128, 129, 259 (2*128+3), cached prefix 0, 61 (ends mid-chunk) and len-3
(past the header), chunk 128 and 0. It asserts each token of `prompt[from..]` fed exactly once, in order, at its
position; the snapshot taken at the header right after the rows; chunks within 1..=128; no row stepped with
chunking, every token stepped with 0. Review Rev862's mutations, each applied alone and reverted: M4
(`q36_state` drops the last row token) -> FAILED; M5 (`feed_rows` skips the first row token) -> FAILED; restored
-> 10 passed. What stays chip-only: the wrappers (`prefill_state`'s env read, the embed + `g.prefill` per chunk),
hence the window's prefill-only red arm (step 2).

Script probes against a mock door (`/tmp/dra65mock/mock2.py`, `SPARK_WINDOW=w-test-2`): red with one turn-2 id
changed PASS rc 0; red on exact goldens FAIL rc 1; red on empty ids, short ids or HTTP 500 INCONCLUSIVE rc 2; green
with 0 or 1 red receipt, or with receipts of another binary sha, refused rc 2; green with two receipts: exact PASS
rc 0, one id changed FAIL rc 1, empty ids INCONCLUSIVE rc 2; time: exact RECORDED rc 0 (6/6 runs ok), HTTP 500 and
missing usage INCONCLUSIVE rc 2; no `SPARK_WINDOW` NOT_RUN rc 3. Every `golden.jsonl` / `time.jsonl` line and
receipt carried `window: w-test-2` and the binary sha. `run` guard, with `DRA65_TEST_LOCK` pointing at a scratch
lock (it never reaches the chip): no `SPARK_WINDOW` rc 3; lock held by another process rc 3 ("held by another
process"); `DRA65_LOCKED` spoofed without holding the lock rc 3 ("held by pid(s) ..., none an ancestor"); lock
free: taken, its flock(1) holder found among the script's ancestors, then rc 3 as the test lock demands. The
part past the guard (lease, listener and chip checks, server start, step loop) is NOT_RUN on host.

Mutant 4 wiring (source, omega 474197e4f): `AIEN_Q36_TEST_MUTANT=4` -> `moe.rs` `graph_mutant()` ->
`test_set_mutant` on the main graph and (`rows.rs` wire) every row and stroke graph; in C mutant 4 swaps K_CN for
K_CN_M3 in the decode path and in `omega_gpu_q36_graph_prefill` (the chunk path), K_CN_R for K_CN_R_M3 in rows.
So the env mutant reaches the chunked prefill. It also breaks decode, so it cannot show the chunk path runs. The
prefill-only mutant does: `AIEN_OMEGA_Q36_PF_MUTANT` is read by `omega_gpu_q36_graph.c` L925 when the prefill
buffers are set up in the server process (header L146: 2 = every expert segment one row short); the decode graph
never reads it, and with `AIEN_Q36_PREFILL_CHUNK=0` the single door runs no chip prefill at all
(`prefill.rs` L298, `q36_state`). So PF_MUTANT=2 must be red with chunking and green with CHUNK=0.

## Window (Main; chip gates NOT_RUN until then)

Main opens a quiet exclusive window without the chip lock (marker, `spark-window open ... --gb10 none --quiet
exclusive --lease 20`), stops 18091 and Flint, and runs one lease's steps:
`spark-window run bash evidence/dra65-prefill-20261031/window-dra65.sh run BIN OUT STEP ...`, BIN = the server
binary built from this branch, OUT = one receipt root for every lease (step N writes `OUT/N/`). The 35B server's
chip reservation (`crates/aien-serve-http/src/chip_reservation.rs` L88-95) accepts `/tmp/aien-gb10.lock` only
when one of its ancestors holds it, and spark-window's holder never is; so the script takes `flock -n` on the
lock itself (flock(1) holds it and runs the script as its child), checks in `/proc/locks` that the exclusive
holder is its ancestor, and starts each server as its own child with `AIEN_GB10_LOCK_HELD=1`. Exit 3 (NOT_RUN,
chip untouched): no `SPARK_WINDOW`; the lock held by anyone else; a listener on 18091, 18300, 18310 or the test
port 18192; a compute app on the chip; BIN or the model missing; green steps without the red receipts in
`OUT/1`, `OUT/2`; or the steps' estimates longer than the lease has left (`$SPARK_WINDOW_DIR/held` + `lease`).
Servers: the live `start-35b.sh` flags (`--max-context 131072`) for the single door, `start-35b-batch.sh`
(`--max-context 49152 --batch-slots 8`) for the batch door, `--bind 127.0.0.1 --max-tokens-cap 16384 --thinking
off AIEN_Q36_STROKE=0`, plus the step's env. Each step writes `OUT/N/step-receipt.json` (window id, binary path
and sha256, door, env, est_s, server_ready_s, elapsed_s, rc, lock holder pid, the probe's receipt) beside the
probe's own receipts; `run` stops at the first step that does not exit 0, and posts GPU-START/GPU-END lines on
the board.

Estimated durations (MODELED, upper bounds at max_tokens; `window-dra65.sh plan` prints them), with decode
0.0201 s/token, chunked prompts 250 tok/s (8210 tokens at 196.8), stepped prompts 0.0201 s/token, +33 s start:
golden chunked `dec*(756 + 49 + 876) + 365/250. + 2486/250. + 8210/196.8` = 86.9 s -> 120 s with start;
golden all-stepped (step 3) `dec*(756 + 365 + 2486 + 8210 + 49 + 876)` = 256.1 s -> 289 s; time chunked
`3*(512/250. + 8210/196.8) + 6*256*dec` = 162.2 s -> 195 s; time all-stepped `dec*(3*(512 + 8210) + 6*256)` =
556.8 s -> 590 s (756 = the set's prompt tokens, 876 = its decode tokens 798 + 3*24 + 6; 49 = turn 2 past the
snapshot; 512 = planning size of long-512). 20-min leases: L1 = `run BIN OUT 1 2 3 4 5`
(`4*119.91 + 289.11` = 769 s), L2 = `6a 6b` (785 s), L3 = `6c 6d` (390 s).

| step | door | server env | probe | expect | est_s |
|---|---|---|---|---|---|
| 1 | single | `AIEN_Q36_TEST_MUTANT=4` | golden red `mutant4` | PASS (red) | 120 |
| 2 | single | `AIEN_OMEGA_Q36_PF_MUTANT=2` | golden red `pf2` | PASS (red: the chunk path runs) | 120 |
| 3 | single | `AIEN_OMEGA_Q36_PF_MUTANT=2 AIEN_Q36_PREFILL_CHUNK=0` | golden green `pf2-chunk0` | PASS (the mutant lives on the chunk path only) | 289 |
| 4 | single | - | golden green `single` | PASS = 36/36 | 120 |
| 5 | batch | - | golden green `batch` | PASS = 36/36 | 120 |
| 6a | single | - | time `single` | RECORDED | 195 |
| 6b | single | `AIEN_Q36_PREFILL_CHUNK=0` (all token steps) | time `single-chunk0` | RECORDED | 590 |
| 6c | batch | - | time `batch` | RECORDED | 195 |
| 6d | batch | `AIEN_Q36_PREFILL_MIN=1` (chunks the answer header too) | time `batch-min1` | RECORDED | 195 |

Green steps get `OUT/1/golden-receipt.json` and `OUT/2/golden-receipt.json` and refuse unless both are red PASS
of the same binary sha. Time steps: exit 0 only if all 6 runs have HTTP 200, a first content delta and usage;
receipts `time.jsonl`. TTFT = t_first - t0; e2e tok/s = completion_tokens / (t_end - t_first). Medians of the 3
runs through Wolfram (`Median[{...}]`). Record `usage.prompt_tokens` of long-512 (the host has no tokenizer).
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
