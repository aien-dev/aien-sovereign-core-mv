# WHOLE-SYSTEM-E2E, harness v1.2: rows for RUN-1 proper (milestone R5, CPU)

Tracking: aien-dev/aien-architecture#190, lane L1, Drake's directive of 2026-10-10. Contract: `ACCEPTANCE_E2E.md`
FROZEN at aien-architecture b2400fe, sha256 `9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8`
(the harness writes this digest into `VERDICT_ROWS.txt`; the verifier takes it on the command line).
`ACCEPTANCE-v1.md` and `ACCEPTANCE-v1.1.md` stay as written; their runs stay recorded. This file is committed before
the harness change that implements it. Repairs it depends on, all merged with pre-registered regression files:
sc#388 (objective record, CTRL-E2), sc#391 (memory recall, ALLEN engagement, E4 budget), sc#392 (input reading,
destination), sc#389 (verifier), sc#390 (evidence logs restored).

## Rows changed or added for RUN-1 proper

| row | v1.2 wording and rule | prediction |
|---|---|---|
| PRE | as v1.1, plus: every daemon start from PRE onward engages the host-built fixture ALLEN subject (sc#391); the PRE receipt carries `allen_identity_fingerprint`; daemon logs and the console transcript are written as `.txt` so the repository keeps them; `chain/SHA256SUMS` is written after the console transcript closes | PASS |
| E2 | the objective is recorded before any work by the installation: `aien-cli objective record --text "$OBJ"` (receipt fields `objective_id`, `objective_text_sha256`, `recorded_before_work`, `operator_surface`, `utc`, record path and sha256); every later propose for that objective passes `--objective-id`; PASS when the record exists with the same `objective_id` the harness computes and no effect receipt predates it | PASS |
| CTRL-E2 | `aien-cli objective record --text "$OBJ" --effect-class EXTERNAL_IRREVERSIBLE` exits non-zero with `OBJECTIVE_REFUSED ForbiddenEffectClass` and writes no record; the declared class is what is tested (the CLI does not classify prose), as sc#388 states | PASS |
| E3 | contract objective (frozen text); the proposer now receives the three inbox files (sc#392); the propose report's `inputs` list (name, sha256, bytes) is copied into the receipt; rule unchanged (DONE, digest match, `# ` first line, at most 200 words, all three file names, second execute `AlreadySpent`) | PASS or FAIL as measured: the 1B model on CPU exceeded 200 words and the time budget in dry run 2; a FAIL here is a model or budget finding, not a harness one |
| E3m | unchanged from v1.1 | PASS |
| E4 | ALLEN engaged; `compose remember` must return ok (it refuses when not engaged, sc#391); SIGKILL; restart with the same subject; the E4 daemon start sets `AIEN_COMPOSE_MAX_TOKENS=400` (recorded in the receipt as `compose_max_tokens`); second objective `Append one line to report.md naming the topic you reported on first.`; PASS when the propose report shows `memory.items_included >= 1`, the appended line contains `harbour`, `recall_after_restart` true, `memory_store` `aien-allen-memory` (ADR 0036) | PASS or FAIL as measured |
| E5 | `tools/aien-verify` (sc#389) is built with `rustc` alone at setup; its binary and source sha256 go into the E5 receipt with the frozen `contract_sha256`; after the chain closes and the manifest is written the harness runs it over the run folder with `--contract-sha256 9e3d5f4e...`, `--objective-id` and `--write`, so `VERIFIER_VERDICT.txt` and `VERIFIER_RECEIPT.json` sit in the folder outside the manifest (the verifier excludes exactly those two and the manifest); the verifier derives every row from receipts and files, not from the harness status words; E5 itself is NOT_RUN in RUN-1 because cross-machine agreement is R7 (the verifier prints `base_verdict_id` for the second machine) | NOT_RUN by rule; chain_unbroken true and no row disagreement expected |
| E6 | unchanged (no-hold SIGKILL between authorize and execute; `kill_point` recorded as `after_authorize_before_execute`); the held variants (before_intent, after_intent, after_write, reconcile holds) are lane L2 evidence: `runs/FAULTS-20261010T144049Z` (sc#393, `RESULT-FAULTS-1.md`), referenced by path and manifest digest in the E6 receipt | PASS |
| CTRL-E6 | new, no hold needed, as the fault-harness control (sc#393 CTRL-E6): propose and authorize a fourth file objective (`state.md`); SIGKILL the daemon; remove the compose home, copy the desk key back, keep the record mark (`compose.cortex-mark`); restart (ALLEN engagement cannot be required on a fresh home and is recorded as such); execute the same grant: PASS when the state is not DONE, a refusal is named (`E_MARK_TRUNCATED` expected, as sc#393 observed; `NotAuthorized`, `Stale` or `ReconciliationRequired` also count), `state.md` is absent and `effects_after_restart` is 0; never a silent success | PASS (refusal) |
| CTRL-E3b | unchanged; with the destination fix (sc#392) the propose should now be accepted, so the revoke step is reached: `revoke` then `execute` refused `Revoked` | PASS |
| CTRL-E4 | mechanics unchanged except the stop: it runs on the daemon CTRL-E6 left (one removal serves both controls), so the stop before the removal is the CTRL-E6 SIGKILL instead of a graceful shutdown, recorded as `stop_kind`; the desk key is copied back and the record mark kept; the second objective must be refused by name and the report unchanged. CTRL-E4b (recorded, not judged, runs before CTRL-E6 while the home is intact): only the ALLEN memory files (`compose.allen-memory`, store.rs `memory_dir`) are moved away after a graceful stop, the second objective is proposed and whatever the installation does is recorded, then the files are restored and the daemon restarted (a failed restart is a `RESTORE FAIL` receipt and aborts the remaining controls) | PASS; E4b recorded |

Every other row, rule, prediction and limit of v1 and v1.1 applies unchanged. E1, CTRL-E1 (lane L5), cross-machine
E5 (R7) and CTRL-E5 (L4 on the second machine) stay NOT_RUN in RUN-1.

## Rules of this run

One declared attempt (contract rule 7). If the run aborts for a harness defect, the folder stays and the next attempt
is a new folder with a note. Nothing under `runs/` is edited after the run. `git status --ignored` on the run folder
and `sha256sum -c chain/SHA256SUMS` from the committed tree are checked before the evidence PR (Cortex lesson
8fe1ebe8).
