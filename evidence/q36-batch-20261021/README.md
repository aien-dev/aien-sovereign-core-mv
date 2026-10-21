# L354 stage 1: many conversations at once through the 35B batch door (2026-10-21)

Binary `~/.local/state/aien-q36/bin-99230310` (sc serve/q36-batch 99230310, omega lib 678df626 = the live lib),
`--batch-slots 8`, test server on 127.0.0.1:18094, context 2048 per row. Correctness only (shared chip, no timing).

## Gate (rule written before the runs, `gate-batch.sh`)
PASS = every reply sent all at once to the batch door has exactly the greedy token ids the same request gets alone
from the single door, the turn-2 request (next-turn reuse) too, and prompt 0 alone equals the host golden
`../q36-host-20261020/gen-chat-max-24.json`. 32 mixed chat prompts (`prompts.txt`), started in 8 waves 250 ms apart,
max_tokens 8 to 48 (prompt 0: 24), so rows join and leave mid-spell and 32 requests share 8 slots (503s retried as a
client would). Negative control: `AIEN_Q36_TEST_ROW_SWAP=40` swaps two rows' state once; the gate must go red.

## Results (bin-99230310)
| run | identical | differ | turn 2 | golden | verdict |
|---|---|---|---|---|---|
| real8-99230310 | 32/32 | none | identical | identical | PASS (want PASS) |
| swap40-8-99230310 (row swap at step 54, rows 7 and 6) | 30/32 | prompts 0 and 3 (the two swapped rows) | identical | identical | FAIL (want FAIL) |

References: `alone/` from bin-5b726818 (02:42Z) and `alone-bin-0915000b/` from the live L356 build: 33/33 replies
identical between the two.

## Kept, not results
- `real8-8b5e95bb-busy503/`: first run; all 8 answered replies identical, 24 were 503 server_busy (gate client did
  not retry yet).
- `real8-70e25b8a/` PASS 32/32 and `swap40-8-70e25b8a-noswap/`: the control never fired (it waited for exactly step 40,
  which fed one row); fixed in 99230310 (first step from k on with two rows).

## Limits
- Stage 1 runs the rows' one-row graphs in turn: the same chip work as one request at a time, so total tokens per
  second do not rise; what it adds is no server_busy until 8 conversations, per-row next-turn reuse, streaming.
  Speed comes with L312's multi-row stroke behind the same `step_rows` call.
- 32 rows need omega's 4096 handle slots (serve/q36-handles 5d3e28d0); not used in this build.
- Context 2048 in the test; the swap config is 131072 per row (8 rows ~20.8 GiB on paper, calc:252e20b5).

## Stage 2: rows on L312's batch strokes (bin-16b0e6f8, 2026-10-21)
Build: sc serve/q36-batch-v2 16b0e6f8 (batch door + tools + L314 exact head + L356 levers + L312 stroke v2 through lever 2,
8 rows per stroke, a lone row on its own graph), omega serve/q36-batch-v2 ab81d8f3 (509 q36 kernel hashes identical to
L312's 0bef2e59; bf16 MASK and XBATCH forms both pass the simulator, both mutant 4 caught).

Gate (same rule, alone = bin-5b726818 single door replies kept in `alone/`):
| build | real8 | swap40-8 (must fail) | real32 (32 slots) |
|---|---|---|---|
| 16b27643 (v1 strokes) | PASS 32/32, turn2, golden | not run (superseded) | not run |
| 8144655d (v2) | PASS 32/32, turn2, golden | FAIL as required: 30/32, prompts 3 and 28 | PASS 32/32 |
| 16b0e6f8 (v2 + lone row) | PASS 32/32, turn2, golden | FAIL as required: 30/32, prompts 10 and 26 | PASS 32/32 |

Timing (`time-run.sh 16b0e6f8`, spark-window 20261021T141700Z-f7b5e0, chip exclusive + quiet, 32 slots, ctx 2048,
N requests at once, greedy, max 128 tokens, wall = first request out to last reply in, prompt reading included, 2 reps):
| N | v2 tok/s | v1 tok/s | rows in turn tok/s |
|---|---|---|---|
| 1 | 21.95 / 22.46 | 22.67 / 22.63 | 22.65 / 22.75 |
| 2 | 20.75 / 20.83 | 17.25 / 17.29 | 17.14 / 17.17 |
| 4 | 42.27 / 44.59 | 31.63 / 31.49 | 30.84 / 30.89 |
| 8 | 57.10 / 57.09 | 33.65 / 33.80 | 32.01 / 31.50 |
| 16 | 62.42 / 62.40 | 33.63 / 33.38 | 31.12 / 32.01 |
| 32 | 65.66 / 66.38 | 34.30 / 35.69 | 35.39 / 34.83 |
v2 over rows in turn: N=8 1.799x (calc:3658d8e7), N=16 1.978x (calc:3adea8f3), N=32 1.881x (calc:01a7d873), N=1 0.977x
(calc:9035675a, inside the run-to-run spread). N=8 total 57.11 tok/s (calc:f67ae6c1) vs 31.75 (calc:b261e933).
Limits: short replies (many end before 128 tokens), so prompt reading (one token per beat today, L369) is a large share of
the wall; the per-token decode speed alone is higher. 18091 was stopped for tests during the window (no live traffic).
The one-conversation ceiling (63.25 tok/s) bounds N=1 only; batch strokes share the weight reads, so N >= 16 passes it.
Raw: time-v2-16b0e6f8/, time-v1-16b0e6f8/, time-inturn-16b0e6f8/ (time.jsonl + every reply).
