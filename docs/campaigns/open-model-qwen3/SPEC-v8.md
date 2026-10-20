# OPEN-MODEL-QWEN3 v8: corrective retry, qualification specification for Qwen3-8B

**Status: DRAFT NOT FROZEN. NOT RUN. `frozen-v8.json` says DRAFT NOT FROZEN and every pin is the placeholder, so `run-qwen3-v8.sh`
refuses a real run.** Owning issue: aien-spark/aien-sovereign-core#384. Ruling: Drake 2026-10-12, aien-architecture#190 comment
6116159138 ("AIEN D1, Qwen3-8B Qualification v8": Option 3, validation-aware corrective feedback for rejected model replies).
Preceding result: RESULT-v7.md (FAIL, 259 of 334, kept as is with all its evidence).

## 0. What v8 is

v7 (`SPEC-v7-DRAFT.md`, frozen as `frozen-v7.json`) with one product change and one harness repair. Nothing else changes.

```text
model        = Qwen/Qwen3-8B rev b968826d9c46dd6066d109eabc6255188de91218, thinking off; the same weights and hashes as v7
rows         = the 334 declared rows of oq3-v5.decl.json, unchanged
tasks, caps, budgets, seeds = the v5 files, unchanged (documents 1024 tokens / 120 s, D6 1536 / 170 s, edits 96 / 29 s, N2 16 tokens)
attempt rule = unchanged: at most 3 attempts; attempt k > 1 starts only if the per-attempt budget is left (30 s documents, 12 s edits)
scoring      = scoring-v5 score-rows.sh, unchanged; PASS only if every declared row is present once and PASS
new required = the five feature checks of Section 3 (v8-checks.sh); a v8 PASS needs the scorer PASS AND V8 CHECKS PASS
```

No row, threshold, task, cap, time budget, retry limit or effect authorization changes. Qwen3-8B is promoted only if the whole frozen v8
campaign passes; a FAIL blocks the public bundle (ruling 2). If v8 fails, the remaining failures are classified before any other model or
acceleration is proposed. Aienux CPU int8 and Omega GB10 inference acceleration are separate workstreams and not part of v8.

## 1. Product change: corrective retry (sc#460)

Before (v7): a refused attempt k-1 led to attempt k with the task prompt plus one sentence, "Your previous answer was refused (<reason>).
Start your answer with the line "filename: <relative path>"." The model never saw what it had written.

After (v8): `spine::corrective_retry_prompt` adds, after the exact refusal reason, the refused reply of attempt k-1 between the markers
`<<<REFUSED ANSWER` and `REFUSED ANSWER>>>`, and asks for a new complete answer that meets every requirement of the goal. Rules:

- Only attempt k-1's reply is shown, never an older one. The end marker inside a reply is replaced, so a reply cannot close the quote.
  A reply longer than 8192 bytes is cut on a character boundary and marked (v7's longest refused reply was about 3 KB).
- After a timeout or an error there is no reply: the prompt is the v7 retry prompt byte for byte.
- The quoted reply is prompt text only. It never reaches the write path: only a reply that passes every check becomes a proposal,
  and a committed write still needs the same authorization as before.
- Each attempt records `shown_refused_text_sha256`: the `text_sha256` of the reply its prompt showed (none on attempt 1).

Unit tests (`crates/aien-runtime/tests/corrective_retry_test.rs`, 11, with `librx_compose` linked): see sc#460. aien-runtime 519 of 519.

## 2. Harness repair: R1-P replays the exact GB10 prompt (sc#459)

v7 R1-P failed because the CPU reference replay `np1_reference` was given the bare task prompt while the product puts the ALLEN style block
in front of it (164 against 99 tokens; the replies were identical). `np1_reference` now takes `NP1_REFERENCE_ALLEN=default`, which applies the
product's own `persona::prefix_prompt` with `PersonaContext::defaults()`, and records `"allen": "default"`. The v8 wrapper calls it so; it
differs from the v7 wrapper only in names and that one line (`test-v8.sh` proves it). Proof (`evidence-v8/r1p-replay-cpu.json`): the full
CPU run now gives prompt_ids_sha256 `eee4e094...adfe6`, 164 tokens, the v7 GB10 ids exactly, and the identical 26-token reply. The row R1-P
already compares `prompt_ids_sha256` and `prompt_tokens`, so it requires exact token identity, not a count.

## 3. Feature checks and negative controls (`v8-checks.sh`, required for PASS)

| Check | Requires | Negative control (proved red in `test-v8.sh`, 29 of 29) |
|---|---|---|
| C1 shown | attempt k>1 shows exactly attempt k-1's refused reply; nothing on attempt 1 or after a timeout or error | a retry that shows nothing after a refusal (the v7 records go red on D2, D3, D5, D6, N2) |
| C2 never kept | at most one parsed attempt, the last, and its reply equals no refused reply of the launch | accepted reply equal to a refused one |
| C3 limits | at most 3 attempts, numbered 1..n | 4 attempts |
| C4 replay | R1 CPU reference says `allen: default` and has the accepted attempt's prompt_tokens and prompt_ids_sha256 | the v7 replay (no block, 99 tokens) |
| C5 unchanged | attempt 1's prompt ids equal the v7 record of the same launch | a different attempt-1 prompt |

Product negative controls inside the 334 rows stay as they are: N1 refused for its destination before any model call, N2 cut at 16
tokens and refused on every attempt (in v8 its attempts 2 and 3 see the cut reply, C1, and must still be refused), containment rows,
nothing written outside the workspace, no write without authorization.

## 4. Deterministic replay checks (before any chip time)

`np1_reference` example test `v8_replay` (tokenizer only, `AIEN_QWEN3_MODEL_DIR`): every prompt of the v7 GB10 run, rebuilt from the frozen
task file and seeds with the product's own functions and the default ALLEN block, equals the v7 record token for token: 14 prompts
(attempt 1 of the ten model launches and the four recorded retries). Each v7 refused reply put through `corrective_retry_prompt` fits the
4096-token context with its task cap: largest D6 1013 + 1536 = 2549 tokens (D1 969 + 1024, D2 and D3 750 + 1024, D5 456 + 1024, N2 299 + 16).
Recorded at the freeze build in `evidence-v8/` (Section 6 item 5).

## 5. Predictions (stated before any run; UNVERIFIED)

Attempt 1 is unchanged (C5, proven by Section 4), and greedy decoding on this engine is repeatable: the v7 dry run and the v7 real run
gave the same reply sha256 and token counts on every attempt of D1, D2 and D3 (D1 d1786e61, D2 5581d0a6 then a41fb8e0, D3 9785e182 then
the same timeout). So the attempt-1 replies are predicted to equal v7's (90 percent). Then, launch by launch:

- D4, E1, E2, R1: pass on attempt 1 as in v7 (85 percent). R1-P now PASS (95 percent): the prompt ids are proven equal in Section 2.
- N1, N2: refused as required, all their rows PASS (90 percent).
- D1: FAIL as in v7 (85 percent). Its first reply took 95.8 s of 120 s; 24.2 s were left, under the 30 s per-attempt minimum, so no second
  attempt runs and the corrective retry never acts. The budget rule is not changed.
- D6: FAIL (80 percent). The first reply took 101.5 s of 170 s; a corrected 700-token essay needs about 100 s at the observed 7 tokens per
  second, more than the 68.5 s left.
- D3: FAIL more likely than not (65 percent). 57.5 s are left after the first reply; an 18-line document of about 450 tokens needs about 65 s.
- D2, D5: these passed on attempt 2 in v7 with the old prompt; the corrective prompt changes attempt 2, so each may pass or fail
  (pass 70 percent each). A regression here would be a result, recorded as such.
- Verdict: FAIL predicted (85 percent), because three of six document tasks are bound by the unchanged budgets, where feedback cannot act.
  This prediction is a reason to run, not to skip: v8 measures what feedback does where a retry fits, and confirms or refutes the budget
  reading of v7.
- V8 CHECKS PASS predicted (90 percent).

## 6. Freeze checklist (all must be true; each is a yes or no with evidence)

1. [ ] sc#459 (replay repair) and sc#460 (corrective retry) merged on main; this spec and its launchers merged.
2. [ ] Model files re-hashed on the chip host on the freeze day and equal to the `model` block of `frozen-v8.json` (= `frozen-v7.json`).
3. [ ] Clean build at the freeze commit (`evidence-v8/build.sh`), every pin in `frozen-v8.json` filled, checkouts clean.
4. [ ] `test-v8.sh` passes at the freeze commit; the v5 campaign file hashes still match (`evidence-v5/campaign-files.sha256`).
5. [ ] At the freeze build: aien-runtime tests pass with `librx_compose` linked (including `corrective_retry_test`), and the
   deterministic replay checks of Section 4 and the R1 exact-ids test pass; outputs recorded in `evidence-v8/`.
6. [ ] `frozen-v8.json` status FROZEN in the freeze PR, merged before any launch; `quietlock check` clear.
7. [ ] Gate G6 dry run of part 1 on GB10 in its own 20-minute hold AFTER the freeze (`evidence-v8/run-dry.sh`): `g6-gate.sh` exit 0 over
   D1 D2 D3 and `v8-checks.sh` PASS on the dry launches; `model_sha256` from the daemon's `CHECKPOINT_SHARDS` line equals `frozen-v8.json`.
8. [ ] Real run (`evidence-v8/run.sh`): four parts, each in its own 20-minute hold; it refuses without G6 PASS and V8 CHECKS PASS on the dry
   launches. Every outcome is recorded; no part is rerun to pass. Authorization: Drake's ruling of 2026-10-12 ("followed by the authorized
   GB10 qualification").

## 7. What v8 does not prove

A v8 PASS would qualify Qwen3-8B with the corrective retry on these eleven launches, greedy decoding, thinking off, one run. It would not show
the model is better outside them, and the ALLEN subject is still a host-built fixture. A v8 FAIL says nothing about int8 or other
acceleration, which are separate workstreams.

## 8. Launchers (`evidence-v8/`)

As `evidence-v7/` with the v8 names: `build.sh <sc commit>`, `run-part.sh`, `run-dry.sh <freeze commit> <build folder>` (G6 plus
`v8-checks.sh` over the dry D1 D2 D3, written to `records/v8-checks-dry.txt`), `run.sh <freeze commit> <build folder>` (four real parts;
refuses without G6 PASS and V8 CHECKS PASS on the dry launches; runs `v8-checks.sh` over all eleven launches after part 4, written to
`records/v8-checks.txt`). Run base `~/workspace/oq3-v8-runs`, records `~/workspace/oq3-v8-real`.
