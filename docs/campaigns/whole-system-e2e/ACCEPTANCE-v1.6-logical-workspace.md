# WHOLE-SYSTEM-E2E, harness v1.6: the logical workspace prompt baseline (RUN-7-CPU and RUN-7-GB10)

Tracking: aien-spark/aien-sovereign-core#402 (R6, owning issue), #429 (product issue), #431 (product fix),
aien-spark/aien-architecture#190 (Drake ruling D2, 2026-10-11). Contract: `ACCEPTANCE_E2E.md` FROZEN at
aien-architecture b2400fe, sha256 `9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8`. This file is
committed before any run under the changed product. `ACCEPTANCE-v1.md` to `-v1.5-fixed-root.md` stay as written; RUN-1
to RUN-6 stay as recorded. No row, rule, score, threshold, objective text or input changes. The harness script is not
changed by this version.

## Disclosure: written after RUN-6 and after the product fix

RUN-6-GB10 (`RESULT-RUN-6-GB10.md`) failed E3 (D4) and E4 under the fixed run root of v1.5. The ruling recorded there
was: no further R6 attempt without a pre-registered input change. Drake's decision D2 (arch#190, 2026-10-11) approved
the product change #429: the model-visible proposal prompt names the workspace by the constant label `workspace`
(`spine::WORKSPACE_LABEL`, PR #431) instead of the machine-specific absolute path. The real path stays in the
authorization, confinement, execution and evidence layers. This is a product change, not a harness control: it changes
every model input on the compose path, so every prompt id and every greedy output recorded in RUN-1 to RUN-6 is a
different baseline from the one measured here. Those runs are historical evidence and are not re-judged.

The label `workspace` was chosen as a plain constant before any model input under it was built or run. It was NOT
chosen by comparing model outputs under candidate labels; choosing a label because its output passes is selecting on
outcome and is forbidden. The E3, E3m and E4 prompt token counts and `prompt_ids_sha256` under the new prompt are
UNKNOWN at declaration.

## The control

1. **Product under test:** the merged main commit that contains #431. The PRE receipt already records the product
   commit and the pins; the result names the commit.
2. **Prompt independence, proven before the run:** at that commit, `cargo test -p aien-runtime --test
   logical_workspace_test` is run with `AIEN_TOKENIZER_PATH` set to the CAND-4 tokenizer (Llama 3.2 1B Instruct) and
   its output is kept in the run folder as `logical_workspace_test.txt`. It must show 2 passed (equal prompts and equal
   token ids for two workspaces at different absolute paths). A failure stops the attempt before any daemon starts.
3. **Run root:** `WSE2E_RUN_ROOT=/home/drakestapleton/.aien-e2e/run` as v1.5. It no longer enters the prompt; it is
   kept so that the evidence paths of RUN-6 and RUN-7 compare directly.
4. **Recorded prompt ids:** rows E3, E3m and E4 record `prompt_attempts` (`prompt_tokens`, `prompt_ids_sha256`) as
   v1.5. Recording only; no row is judged on them. The result lists them as the new baseline values.
5. **Negative control (recorded, not judged):** after the run, the CPU reconstruction of the E3 model input with the
   product's own functions (persona block, `spine::task_prompt`, chat template, compose assistant prefix, model
   tokenizer), performed as in `RESULT-RUN-5-GB10.md`, is built twice: once under the run root and once under a
   throwaway folder with a different name and depth. The two `prompt_ids_sha256` must be equal and must equal the id
   the run recorded for E3 attempt 1. Inequality is reported as a harness or product defect, never hidden.
6. A copy of this file is written by hand into each run folder as `ACCEPTANCE-v1.6-logical-workspace.as-run.md`
   (the harness copies v1.5; this version adds no harness code).

## The two declared runs

| run | mode | purpose |
|---|---|---|
| RUN-7-CPU | `WSE2E_BACKEND=cpu` | the new CPU whole-system baseline for the logical-workspace prompt (replaces RUN-1 as the reference input set for later R7 cross-machine comparison; RUN-1 stays as recorded) |
| RUN-7-GB10 | `WSE2E_BACKEND=gb10` | the R6 GB10 qualification under the changed product, attempt 5 (RUN-2, RUN-3, RUN-5 and RUN-6 named as earlier attempts) |

Order: RUN-7-CPU first, then RUN-7-GB10. One declared attempt each. An outcome stands; neither run is repeated to
find a passing input.

## Predictions (written before either run)

| row | prediction | why |
|---|---|---|
| PRE, CTRL-E3a, E2, CTRL-E2, E6, CTRL-E3b, NET, CTRL-E6, CTRL-E4 | PASS | PASS in RUN-5 and RUN-6 with fallback 0; nothing here reads the prompt text |
| E3 | PASS or FAIL as measured | UNVERIFIED: the 1B model's E3 text under the new prompt has never been generated; no outcome is predicted |
| E3m | PASS or FAIL as measured | same; PASS in RUN-3, RUN-5 and RUN-6 under the old prompt |
| E4 | PASS or FAIL as measured | FAIL in RUN-6 (edit reply unchanged), PASS in RUN-5; depends on the prompt text |
| CTRL-E5 | PASS if E3 PASS, else FAIL as a consequence | as RUN-3, RUN-5 and RUN-6 |
| CTRL-E4b | RECORDED | not judged (v1.2) |
| E1, E5, CTRL-E1 | NOT_RUN | source-build modes; R8 (install) and R7 (second machine) |
| G4, G5 (gb10 only) | `fallback_detected: false`; every model-calling row `gpu_executions >= 1` | as RUN-3, RUN-5 and RUN-6 |

## Rules of these runs

As `ACCEPTANCE-v1.5-fixed-root.md`: from the merged main commit that contains #431 and this file, folders
`runs/RUN-7-CPU-<utc>/` and `runs/RUN-7-GB10-<utc>/`, `WSE2E_RUN_ROOT=/home/drakestapleton/.aien-e2e/run` (the RUN-6
state moved aside by the harness), started detached, never killed, `quietlock check` first, the chip free of other
runs for the gb10 run, nothing under `runs/` edited afterwards. If a row fails, that outcome stands and the next step
is a pre-registered input change (model decision D1 or a product issue), never a rerun under this control. No program
PASS is claimed by any outcome of these runs: R6 PASS requires every required row of the frozen contract; R8 stays
blocked on the real signing key, the qualified public model and the CAND-5 pins.

## Not part of this control

The public model (Drake D1: Qwen3-8B evaluation and a frozen v7 campaign), the release signing ceremony (D3), the
install run (R8) and the native Aienux work (D4) are separate lanes with their own pre-registered controls.
