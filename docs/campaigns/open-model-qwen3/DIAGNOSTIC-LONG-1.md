# DIAGNOSTIC-LONG-1: does truncation hide a good answer or delay a bad one? (plan; run 2026-10-07 by session 031756, results in DIAGNOSTIC-LONG-1-RESULTS.md)

**This is a measurement, not a campaign. It has no verdict and scores nothing.** It is not frozen, is
not an acceptance spec and does not replace or amend ACCEPTANCE-v2.md or VERDICT-v2.md. Its goals are
diagnostic only and will NOT be reused in the next campaign, which needs fresh tasks. Draft written by
session 031756, 2026-10-07. Nothing here has been run.

## 0. Question

In v2 (VERDICT-v2.md), T4 and T6 were cut at 256 tokens and refused. I do not know what Qwen3-4B would
have written. Two readings: (a) it was writing a good document that needed more room (truncation hides
a good answer), or (b) it would have kept going without finishing, or finished with something that fails
the rows (truncation delays a bad one). The answer decides what the product needs: a larger budget for
long documents, concise output, or splitting a long task into parts.

## 1. Tasks (5 launches, one each, greedy)

`diagnostic-long-1-tasks.json` in this folder:

- D1 = the v8 T4 goal and D2 = the v8 T6 goal, unchanged text (so the v2 cut can be compared).
- D3 (architecture doc, 20 lines), D4 (contributing guide, 15 lines), D5 (FAQ, 10 questions) are new
  representative long-document goals written for this diagnostic. They are diagnostic only and will not
  be reused in the next campaign.

All five use `max_tokens` 1024.

## 2. Bounded budget

- `AIEN_COMPOSE_MAX_TOKENS` = 1024 per launch; `AIEN_COMPOSE_BUDGET_MS` = 120000 (omega waits 121 000 ms).
  At about 88 ms per token (VERDICT-v2 Section 3), 1024 tokens is about 90 s of decode plus prefill, so
  120 s is enough for the cap and still bounded: no launch can run past about 121 s.
- One launch each, greedy, no retry, no repeat. Same model, digests, `AIEN_KV_CONTEXT_TOKENS=4096`,
  `AIEN_REQUIRE_BLACKWELL=1` and declared-attempt setting as ACCEPTANCE-v2 Section 2.
- Everything else is the NEXT-PHASE-1 v8 driver path (`run-campaign.sh`), unchanged.

## 3. What is measured per task

1. Tokens generated until the model stopped.
2. Finish reason: `eos` (the model chose to stop) or cut at 1024.
3. Elapsed ms, split into prefill and decode where the daemon logs them.
4. Whether the reply parses as a file proposal (`parse_file_proposal`), and the refusal reason if not.
5. The existing v8 rows for the launch (`rows-v8.jq`, the T4 and T6 rows for D1 and D2; the same row
   shapes applied to D3 to D5 with their own phrases and min_lines) as a **correctness reading only**:
   the row results are reported, never summed into a verdict.

Reading the result (so the numbers are interpreted before they are seen):

- eos before 1024 and the rows pass: truncation was hiding a good answer.
- eos before 1024 and rows fail: the answer was bad, a larger budget would not have helped.
- cut at 1024: the model does not stop on this kind of task; a bigger budget only delays, so the product
  needs concise output or task splitting.

## 4. Run plan

- Build: local, unmerged, so the whole run is labelled measurement. omega branch `031756/compose-wait-ms`
  (adds `rx_compose_set_wait_ms`) plus sovereign-core branch `031756/compose-budget` (adds
  `AIEN_COMPOSE_BUDGET_MS`). Known gap: sovereign-core's call needs omega to expose a host-facade wrapper
  `rxc_host_set_wait_ms` over `rx_compose_set_wait_ms` (the host handle hides the composition); if the omega
  branch lacks it, the budget above 29 s is refused at open and this run cannot start until that is
  added. The build commits and sha256 are recorded with the results.
- One quietlock hold of 20 minutes or less. Estimate: 5 launches x 2 daemon starts x about 64 s = 10 min 40
  s of starts, plus decode of at most 90 s per launch (7 min 30 s worst case) = about 18 min 10 s worst
  case. That is too close to 20 minutes, so the run is two holds: D1 D2 D3, then D4 D5, each under 12
  minutes by the same arithmetic. (If a start is faster than 64 s in a dry check, one hold may do.)
- Fresh run base, for example `/home/drakestapleton/workspace/oq-long1-runs`; never under `oq3-v2-*`
  (sealed evidence). The wrapper is not written yet: if `run-campaign.sh` runs a custom tasks file
  without change, no wrapper is needed; otherwise a small bash wrapper is added to this PR before the
  run. No Python.
- Nothing is run on the GPU by the author of this draft; the run is a separate approved step.

## 5. Prediction (stated before any run; UNVERIFIED, from v2 numbers and judgment)

- D1 and D2: Qwen3 began a long markdown guide and was still writing at 256 tokens. My guess is it stops
  by eos between 400 and 800 tokens, about 60 percent that both stop before 1024. If they stop, I expect
  D1 to pass its rows more likely than D2 (D1 about 55 percent, D2 about 45 percent) with the risk on the
  12-line count and the four phrases.
- D3 to D5: likely longer (more lines requested). I expect D3 to be the one most likely cut at 1024
  (about 45 percent), D5 (a list) the most likely to stop by eos.
- Overall: more likely than not (about 60 percent) that at least one of the five is cut at 1024, which
  would point at concise output or splitting, not just a bigger budget.

## 6. After

The numbers go to Drake for a product decision: a separate budget for short edits versus long documents,
concise-output instructions, or splitting long tasks. Whatever is chosen, the next campaign gets its own
frozen spec with fresh tasks (none of D1 to D5, and not T4 or T6, which Qwen3 has now seen).

## 7. Limits

- Five launches, one observation each, greedy decoding: not a rate.
- D1 and D2 were seen by Qwen3 in v2 (cut); D3 to D5 were never run. Nothing here is held-out evidence.
- Unmerged build; GPU memory failure omega#327 stays open.
- No row is changed, no score is declared, nothing here is a pass or a fail.
