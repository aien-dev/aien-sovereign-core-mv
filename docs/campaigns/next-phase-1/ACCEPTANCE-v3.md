# NEXT-PHASE-1 campaign v3: acceptance criteria (frozen before any v3 code or run)

```text
campaign_id  = "next-phase-1"
spec_version = 3
status       = FROZEN at the commit that adds this file. The v3 code (measured
               attempt budget, driver S8 fix) lands in LATER commits. The one
               commit before this file (3ec13bd) adds only observation: an
               AIEN_STEP_LOG line and chat template tests, no behaviour change.
base         = sovereign-core main 91751b6 (v2, #217 merged).
```

v1 and v2 stand as recorded; their receipts (`92acb566...` and the three before it, `2c12a5ba...`)
are never edited. v2 verdict FAIL on task completion: attempt 1 echoed the prompt in 48 tokens
(16 056 ms) and the 25 s budget left no room for attempt 2.

## 1. Task, steps, receipts, pass criteria: unchanged

ACCEPTANCE.md Sections 1, 2 and 4 and ACCEPTANCE-v2.md Section 2 apply unchanged: the bounded
task, steps S1..S8, the environment class, both receipt formats, every threshold of the v2 table
and the rescue definition. One driver launch, one goal, one task, two daemon starts.

## 2. Measured input (control run 2026-10-06 03:11Z, GB10, quietlock hold)

Raw records (outputs verbatim, client timestamps, `step:` lines) are in the campaign log
directory `np1-logs/v3-control/` of the run host; the numbers below are copied from them.

**Prompt format (control).** The exact v2 request (`proposal_prompt` with the v2 goal, the v2
workspace path and `README.md, docs`) was sent through plain `StreamTurn`, outside the
composition, greedy, 48 tokens.
- The production path already applies the model's chat template: `submit_turn`
  (crates/aien-runtime/src/server.rs) formats every `StreamTurn` and every compose proposal with
  `format_tinyllama_chat`, then encodes with special tokens. Source of the template:
  `TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json`, sha256
  `7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a`, field `chat_template`.
  Rendered with the Hugging Face settings (trim_blocks, lstrip_blocks) it gives
  `<|user|>\n{prompt}</s>\n<|assistant|>\n`, byte-identical to AIEN's (PROVEN BY TEST:
  `chat_template_test::streamturn_formatting_matches_the_model_template`; the hand rendering
  of the jinja is INFERRED, no jinja engine was run). With the real tokenizer the prompt starts
  with BOS (id 1) and holds one EOS (id 2): 135 tokens (PROVEN BY TEST).
- So "raw as v2 did" and "model template" are the same request. Sent twice (cold, warm), the
  output was identical both times (sha256 `0fd0c04e5198bd53...`, equal to v2 S3 attempt 1),
  first line `Goal: Create the file README.md ...`: no `filename:` first line (MEASURED).
  **The echo is not a prompt-format bug.** v3 therefore declares no change to the request text.
- Exploratory, outside the brief: the format instruction as a system turn and the goal as the
  user turn gave `To create the file NOTES.md ... you can follow these steps:` (no filename line).

**Fixed cost (one daemon, 135-token prompt, wall from send to `TurnFinished`).**

| max_tokens | wall, two calls | backend steps |
|-----------:|-----------------|---------------|
| 1  | 3.131 s, 3.144 s | prefill 128-token chunk + 7-token chunk |
| 8  | 4.258 s, 4.174 s | prefill + 7 decode steps |
| 48 | 11.056 s, 11.028 s | prefill + 47 decode steps |
| 48 (first call after start) | 16.122 s | first 128-token chunk 8 010 ms |

- Prefill, warm: 128-token chunk 2 769 to 2 872 ms (mean 2 839 ms over 8 calls, about 45 tokens/s);
  7-token chunk mean 255 ms; 15-token chunk 389 ms (MEASURED, `step:` lines).
- Decode: 162 to 169 ms per token, mean 166.6 ms, i.e. **6.0 tokens/s** (MEASURED).
- Wall minus backend step time: under 60 ms per call (11.5 ms and 58 ms on the 1- and 48-token calls checked) (socket, tokenizer, block_on, polling).
- **Where the fixed cost sits:** the prompt prefill, about 3.09 s warm for 135 tokens. Not a model
  load per call (the model loads once at daemon start, 15.5 s), not the socket or `block_on`
  (under 60 ms). KV allocation is inside the prefill step and not separable from these logs
  (UNKNOWN). The first call after a daemon start pays about 5.2 s more inside its first prefill
  step (MEASURED; cause UNKNOWN from these logs). v2 attempt 1 was such a first call.
- Model: T(n) = 3.14 s + (n - 1) x 0.1666 s for this prompt; predicts 3.14 / 4.31 / 10.97 s
  for 1 / 8 / 48 tokens against 3.13 / 4.17 to 4.26 / 11.03 to 11.06 s measured.

## 3. Declared changes (v3)

(a) **Chat template: no change, pinned.** The Skill keeps formatting the proposal request with
the model's own chat template through `submit_turn` (production path, source cited in Section 2).
v3 adds the test that pins it to the tokenizer_config.json template.

(b) **Attempt budget from the measurement.**
- Ceiling: `rx_compose_run` waits at most 30 000 ms for quiescence
  (omega 62b6a28 `src/runtime/rx_compose.c:1192`), NOT raised. Non-Skill time inside S3 in v2:
  16 107 - 16 056 = 51 ms (MEASURED).
- Total Skill budget B = 30 000 - 1 000 margin = **29 000 ms** (v2: 25 000 ms).
- Full warm attempt with the retry prompt (173 tokens = chunks of 128 and 45; 48 new tokens):
  2 839 (chunk 128) + 896 (chunk 45: 136 + 45 x 16.9, the line through the 7- and 15-token
  chunks, INFERRED extrapolation) + 47 x 166.6 = 7 830 (decode) + 60 (wall overhead)
  = 11 625 ms. Per-attempt budget A = **12 000 ms** (11 625 rounded up, about 3 % margin).
- Rule: attempt 1 always starts; attempt k > 1 starts only if the remaining budget is at least
  A (v2: at least the measured duration of attempt k-1, which a cold first attempt inflated);
  each attempt's limit is the remaining budget.
- Expected in this run: attempt 1 cold, about 16.1 s, leaves about 12.9 s >= 12.0 s, so attempt 2
  runs (about 11.6 s) and leaves about 1.3 s < 12.0 s: no attempt 3 unless an attempt ends early
  at end of sequence. What attempt 2 answers is UNKNOWN.

(c) **Unchanged:** `max_tokens` = 48, at most 3 attempts per task, greedy decoding, the parser
and the correction line of ACCEPTANCE-v2 Section 3.

(d) **Driver: S8 is scored on its own.** When S6 cites nothing, the driver still runs S8 (and
the pre-restart recall), without `--ids` / `--prefix`; the recall then returns the host
records, so "Correctness: recall" compares the S1 constraint text with what S8 returns, no
longer a side effect of S3. S6 tolerates an empty citation list.

## 4. Binding and run conditions

As ACCEPTANCE-v2 Section 5: omega pinned by `omega.lock` = 62b6a28 for both the GPU engine and
librx_compose; GB10 hold through `quietlock hold --minutes <= 20` with start and release
whispers; one driver launch; the verdict is whatever the rows give. Performance is an
observation: the summary reports tokens/s.
