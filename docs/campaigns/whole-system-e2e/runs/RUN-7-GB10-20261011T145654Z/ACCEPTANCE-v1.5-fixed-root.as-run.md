# WHOLE-SYSTEM-E2E, harness v1.5: a fixed run root (RUN-6-GB10 onward)

Tracking: aien-dev/aien-sovereign-core#402 (R6, owning issue), aien-dev/aien-architecture#190. Contract:
`ACCEPTANCE_E2E.md` FROZEN at aien-architecture b2400fe, sha256
`9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8`. This file is committed before the harness change
that implements it. `ACCEPTANCE-v1.md` to `-v1.4-e1.md` stay as written; RUN-1 to RUN-5 stay as recorded. No row, rule,
score, threshold, objective text or input changes.

## Disclosure: written after RUN-5 showed the path sensitivity

This control was written after RUN-5-GB10 (`RESULT-RUN-5-GB10.md`). There, E3 failed D4, and a CPU reconstruction of
the E3 model input with the product's own functions showed that the only difference from RUN-3-GB10 (E3 PASS) was the
absolute workspace path the proposal prompt carries once (`spine::proposal_prompt`: `Authorized workspace: {workspace}`):
345 tokens, `prompt_ids_sha256` `ede9f1e8...` under RUN-3's job folder, 347 tokens, `6847127a...` under RUN-5's. Until
now the run state lived under `$CLAUDE_JOB_DIR/tmp/e2e/run`, so the folder name of whichever session started the run
entered the model input. That is an uncontrolled harness variable: attempts under different job folders are not
comparable.

## The control

New option `WSE2E_RUN_ROOT=<absolute path>`: the run state (workspaces, home, daemon state, socket, logs) lives there
instead of `$CLAUDE_JOB_DIR/tmp/e2e/run`. Build targets, the verifier binary and the pin checkouts stay under
`$CLAUDE_JOB_DIR/tmp/e2e`; they do not enter the prompt. Without the variable the harness behaves exactly as v1.4 (cpu,
gb10 and artifact modes unchanged, `selftest` still passes).

**Declared run root for every declared attempt from RUN-6 on: `/home/drakestapleton/.aien-e2e/run`.** It is a neutral
constant, chosen without regard to any prior outcome. It was NOT chosen because of RUN-2/3's path (which was
`.claude/jobs/ade3c837/tmp/e2e/run`); reusing a path because it once passed would be selecting on outcome and is
forbidden. Its E3 prompt token count and `prompt_ids_sha256` are UNKNOWN at declaration: no model input under it has
been built or run before this file is merged. Every later declared attempt uses the same root, so its E3 model input is
the same across attempts and sessions (on this host; the home directory differs on another machine, which is stated as
a limit, not hidden).

Rules for the option:

1. The path must be absolute and must be its own real path once created (no symlink, no `..`, no trailing `/`), so the
   text in the prompt is exactly the declared text. Otherwise the run stops before any build.
2. An empty directory at the root is removed and recreated; a non-empty one is moved aside to `<root>.old-<epoch>`, as
   v1.4 does with its default run folder. The prompt path is the same either way.
3. The PRE receipt records `run_root` (the path) and `run_root_rule` (this file).
4. Rows E3, E3m and E4 record `prompt_attempts`: for each proposal attempt the daemon reports in its propose output,
   `prompt_tokens` and `prompt_ids_sha256` (`report.proposal_attempts[]`, fields the daemon already exposes;
   the same values Lane A2 matched in `RESULT-RUN-5-GB10.md`). Recording only; no row is judged on them. A missing
   propose file or missing fields give an empty list or nulls, never a failure.
5. A copy of this file is written into the run folder as `ACCEPTANCE-v1.5-fixed-root.as-run.md`.

## Predictions for RUN-6-GB10 (written before the run)

| row | prediction | why |
|---|---|---|
| PRE, CTRL-E3a, E2, CTRL-E2, E6, CTRL-E3b, NET, CTRL-E6, CTRL-E4 | PASS | PASS in RUN-5-GB10 with fallback 0; nothing here depends on the path text |
| E3 | PASS or FAIL as measured | UNVERIFIED: the 1B model's E3 text is sensitive to the workspace path (RUN-3 vs RUN-5); the new path has never been run, so no outcome is predicted |
| E3m | PASS or FAIL as measured | same path sensitivity applies to its prompt; PASS in RUN-3 and RUN-5 |
| E4 | PASS or FAIL as measured | PASS on the chip in RUN-5; depends on E3m and on the path text |
| CTRL-E5 | PASS if E3 PASS, else FAIL as a consequence | as RUN-3 and RUN-5: the control cannot change the verdict id when E3 already failed |
| CTRL-E4b | RECORDED | not judged (v1.2) |
| E1, E5, CTRL-E1 | NOT_RUN | gb10 source-build mode; R8 (install) and R7 (second machine) |
| G4, G5 | `fallback_detected: false`; every model-calling row `gpu_executions >= 1` | as RUN-3 and RUN-5 |

## Rules of this run

As `ACCEPTANCE-v1.3-gb10.md`: one declared attempt, gb10 mode, from the merged main commit that contains v1.5, folder
`runs/RUN-6-GB10-<utc>/`, `WSE2E_RUN_ROOT=/home/drakestapleton/.aien-e2e/run`, started detached, never killed,
`quietlock check` first, the chip free of other runs, nothing under `runs/` edited afterwards. The result names RUN-2,
RUN-3 and RUN-5 as earlier attempts. If E3 fails again under the fixed root, that outcome stands; the root is not
changed to find a passing path. No program PASS is claimed by any outcome of this run.

## Not part of this control

The product fix (a stable logical workspace name in the prompt instead of the machine path) is a separate product
issue for Drake's decision: it changes every model input and needs a new baseline. v1.5 only holds the path constant.
