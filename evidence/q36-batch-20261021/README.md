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
