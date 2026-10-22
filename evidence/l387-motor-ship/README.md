# L387 motor ship build (2026-10-21)

Drake: "hurry, motor -> Allen -> alpha". Every READY speed lever with a green gate in ONE merge-train build, one train
load, timing receipts, and the swap of the live door 18091.

## What is in the ship build

| lever | branch @ sha | state in the ship build | gate evidence |
|---|---|---|---|
| L354 batch door + L312 strokes v2 (lever 2) + tools L374 | sc serve/q36-batch-v2 9c62690a (bin-16b0e6f8 source) | ON (`AIEN_Q36_STROKE=1`, as live) | live gate2 32/32 on bin-16b0e6f8 |
| L384 per-request test switches | sc serve/merge-train 59b896b1 | test build only (cargo feature `aien-test-switches`); the live build answers 400 to `aien_test` | the train itself |
| L369 chip prefill (prompt chunks of 128: GEMMs, multi-token GDN, attention, gate rows) | omega serve/q36-prefill-v2 60ba3051, sc serve/q36-prefill-v2 6127db50 | ON (`AIEN_Q36_PREFILL_CHUNK=0` = token by token) | full 35B bit-identical 256/2k/8k (targets/l369/ev/full-time-v3.log); train case `l369-long-prompts` |
| L314 exact head per stroke slot | sc hum/exact-head-batch 8dd86a36 (omega 41269bd8 via prefill-v2) | ON (`AIEN_Q36_EXACT_HEAD=0` off) | batch gate PASS real8 + mutant5 red (sc-hum-exact-batch evidence/hum-exact-head/batch/gate-4d377169.result) |
| L356 COMBINE+NORM rows | omega 4b079116 via prefill-v2 | ON (`AIEN_OMEGA_Q36_CN_ROWS=0` off) | the train (strokes at N >= 2) |
| L313 bf16 DeltaNet state | omega hum/state-precision 19c3b5d2 | OFF (`AIEN_OMEGA_GDN_STATE_BF16` unset): not bit-exact by design (door drift 30/33 identical vs f32), not validated with the prefill's multi-token GDN | sc-l313-door evidence/hum-state-door |
| L312 levers 3-7 | omega motor/serve 881f4397 | OUT: stroke gate3 unfinished (l6 real PASS 21:57Z; ladder, swap controls, lever 7 pending) | next train |
| L375 shared prompt cache | sc serve/q36-prefix a7dab040 | OUT: gate never finished | next train |
| L376 team board | sc serve/q36-board be32649f | OUT: chip run never served | next train |

Merges: omega ship/motor-l387 6de999da; sc ship/motor-l387 fe5a0d6e (code). Two conflicts resolved by keeping both sides:
the scheduler runs the L384 row swaps then the L369 prefill chunks, and a row carrying `aien_test` switches skips the
chunked prefill (its prompt goes token by token through `step_rows_test`, so every switch applies to every step); the
lone row keeps its own graph (16b0e6f8) and strokes set the exact head per head slot (L314).

Library: lib sha256 110295ef; q36 dec sim 27/27 PASS; q36 kernel hashes identical to L369's lib-v3 (the merge of
L313 changes no 35B program while the switch is off).

## Train
Run `20261022T0244Z-bfc41659` on bin-bfc41659 (the same commit as the live build bin-19173f9e, built with
`--features aien-test-switches`): **1 load**, load 91 s + suite 437 s = 528 s wall, against reference build 5b726818.
Verdict: [`../merge-train/runs/20261022T0244Z-bfc41659/verdict.md`](../merge-train/runs/20261022T0244Z-bfc41659/verdict.md).

- goldens 5/5, concurrent-32 32/32 identical to alone, turn 2 identical, tool call parses (`get_weather`, `{"city":"Paris"}`),
  on each of the arms plain, stroke-off and exact-head-off.
- red controls 3 RED-OK of 3: mutant-4-red, mutant-row-isolated (row 5 differs, 7 others identical),
  row-swap-red (rows 0 and 3 differ, 6 others identical).
- l369-long-prompts (365 / 2486 / 8210 prompt tokens, one sampled request and the 32 short prompts at once, then turn 2):
  37 identical to the references.
- Lever verdicts: L354 PASS, L312 PASS, L314 PASS, L374 PASS, L369 PASS, L313 PASS (OFF; plain arm only, its own arm disabled);
  L375, L371, L355, L376 NOT-ON-TRAIN.

Run 1 (`20261022T0119Z-faac363f`, stopped after the fault) found two harness faults, both fixed red then green before run 2:
59b05a71 (a row that carried a mutant or a row trade left its turn-header snapshot for the next request: row-swap-red
RED-FAIL, 3 rows red; host test `a_negative_control_leaves_no_state_for_a_later_request` fails without the fix) and
f0230a59 (`enabled // true` in jq loaded the disabled state-bf16 arm).

## Timing receipts (18091, same flags, shared machine)
Both builds were timed on the live door 18091 with `AIEN_Q36_STROKE=1`. The chip was reserved for the lever owners, so
spark-window refused L387 a window: **every figure below is a shared-machine receipt.** Renda and the courier were also using
the door (each line in `before-16b0e6f8/time.jsonl` and `after-19173f9e/time.jsonl` logs `in_flight_before` and
`"window":"shared machine"`).

Prompt reading (one request, prompt tokens / wall):

| prompt | bin-16b0e6f8 (before) | bin-19173f9e (after) | after / before |
|---|---|---|---|
| ~2.5k | 2486 tok in 109.536 s = **22.696 tok/s** (calc:d6eb7524) [1] | 2532 tok in 11.987 s = **211.23 tok/s** (calc:b4d69bb4) | 9.31x (calc:3bbd50da) |
| ~8k | 8210 tok in 429.259 s = **19.126 tok/s** (calc:67b6d9e6) [1] | 8256 tok in 52.894 s = **156.09 tok/s** (calc:6edf5912) | 8.16x (calc:bd8f42ad) |
| ~21k | 21421 tok in 789.738 s = **27.124 tok/s** (calc:16e80123) | 21421 tok in 196.787 s = **108.85 tok/s** (calc:6d32757e) | 4.01x less wall (calc:c16a81e4) |

[1] The time-motor before-run lost its 2k and 8k prompt files (they were not yet copied here; its two 58-token lines are
void). These two before figures are the L387 reference capture on the same build and port from 2026-10-21T22:02:15Z to 22:11:14Z, one request
at a time: wall = the gap between consecutive reply files (`refs/.../l369-long/long-{256,2048,8192}.json.raw` mtimes), so it
includes 24 generated tokens per request.

The 21k "before" figure is faster per token than the 8k one: the 8k reference capture ran beside other load, and both were
taken on the shared machine. Treat the before column as a rough floor, not a fine figure.

Short replies (N requests at once from short.txt, greedy, max 128 tokens, completion tokens / wall of the slowest;
the wall includes reading the short prompt; median of 3):

| N | before | after |
|---|---|---|
| 1 | 1.711 tok/s (calc:83ad1ec4) | 14.66 tok/s (calc:985c5509) |
| 8 | 20.77 tok/s (calc:61850490) | 46.29 tok/s (calc:1bf7f597) |

On the old build, N=8 got 6 replies of 8 each time. The door's other two slots were busy, so two requests were refused with
`server_busy`. After the swap, N=8 got 7, 8 and 7. These short-reply figures are end-to-end and noisy: they are not
decode-only numbers, and they do not set a ceiling fraction. The decode ceiling for one conversation is 63.25 tok/s
(motor-chip-20261021/README.md). Prompt-reading ceiling: not measured.

## Swap
`swap.sh bin-19173f9e` (copy here). Timeline from `~/workspace/targets/l387/swap.log`:

- 03:11Z: a first attempt posted SWAP NOW and then stopped itself before any restart. The window had been refused, so the
  2k/8k/decode before-timing would have been lost. The BOARD has a SWAP POSTPONED line.
- 03:16Z: BOARD SWAP NOW. The script waits until in_flight is 0 and never cuts a live request: if the door stays busy
  for 30 min, there is no swap.
- 03:16:38Z: old server (bin-16b0e6f8) stopped by SIGTERM in 3 s. The flag `/tmp/aien-q36-live-restart` was set,
  so q36-extra admitted no load meanwhile.
- 03:17:50Z: 18091 serves bin-19173f9e with `AIEN_Q36_STROKE=1`. **Downtime 74 s** (limit 3 min).
- 03:18:11Z: live gate2 `gate-batch.sh 18091 18091 live-19173f9e-1022T0317Z`: **PASS**, 32/32 replies identical to alone,
  and turn 2 identical (`../q36-batch-20261021/live-19173f9e-1022T0317Z/`).
- 03:18:17Z: live tool call **parsed**: `get_weather`, `{"city":"Paris"}` (`live-tool-call.json`).
- 03:25:19Z: after-timing done, SWAP DONE.

Back-out line (in ledger L387 too):
`Q36_BATCH_BIN=bin-16b0e6f8 AIEN_Q36_STROKE=1 ~/.local/state/aien-q36/start-35b-batch.sh`, after a SIGTERM of serve.pid.
