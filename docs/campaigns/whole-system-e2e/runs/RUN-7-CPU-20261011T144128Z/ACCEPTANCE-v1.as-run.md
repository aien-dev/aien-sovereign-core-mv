# WHOLE-SYSTEM-E2E, harness v1 (RUN-1 dry run on the Linux host): pre-registered rows

Tracking: aien-dev/aien-architecture#190, lane L1 of `docs/plans/release-readiness/INTEGRATION_PLAN.md`.
Contract: aien-architecture `docs/plans/release-readiness/ACCEPTANCE_E2E.md` at commit `0631cf39`, sha256
`5fd4b3d4a671f70d3799c5efaca11a71093852148425e006465ce9d3b320927d` (status PROPOSED; this file names the
version it was written against). Harness: `scripts/whole_system_e2e.sh` (shell plus `jq`, no Python).

These rows are committed before the harness code. The harness measures them and never edits them. A row is
PASS only when its rule holds and, where a control exists, the control fails as predicted. NOT_RUN is never
PASS. Every status word below is NOT_RUN at the time of writing.

## 1. What this harness is and is not

It chains the steps E2 to E6 of the contract into one receipt chain on the Linux host, one daemon, the
CPU-reference backend, the internal CAND-4 model (Llama 3.2 1B, Drake ruling 2 of 2026-10-10: internal-test
only). It reuses what exists: `aien-cli compose propose / authorize / execute / remember / recall / effects /
shutdown`, the approval desk with its key (sc#342 required by default), SIGKILL and restart as the ALLEN demo
S5 and the NEXT-PHASE-2 fault harness C2a do. It is the first measurement, not a PASS claim for the program.

Known limits of the engine on 2026-10-10, pre-registered so the rows below are read correctly:

- **The model sees only the goal text, the workspace path and, in edit mode, the prior content of the one
  file it edits** (`crates/aien-runtime/src/spine.rs` `task_decision`, `proposal_prompt`). It cannot read
  other workspace files. The contract's fixed objective asks it to read three inbox files, so row E3 is
  predicted to FAIL on the "mentions each file by name" check unless the model guesses the names. The
  harness runs that objective anyway (the prediction is on record, red before green) and then a mechanics
  objective that names the files in the goal, so the chain can continue through E4 and E6.
- **No installation-side objective record exists before work starts** (lane L6). `compose propose` records
  the goal inside the proposal report, after the model ran. Row E2 is therefore NOT_RUN; the harness still
  writes the objective receipt as the head of the chain so every later receipt names the objective identity.
- **No independent verifier exists yet** (lane L4). Row E5 is NOT_RUN; the harness writes the verdict rows
  and the chain manifest the verifier will read.
- **The mid-effect hold points need a test-only build feature** (`fault-hold`). Harness v1 uses the
  production binary and kills the daemon between the approval and the start of execution (no hold); the
  held variants (kill while the executor is between intent and write) belong to lane L2.

## 2. Identities (from the contract, section 1)

- `objective_id` = sha256 of the bytes `AIEN_E2E_OBJECTIVE_V1`, newline, objective text (no trailing newline).
- `run_id` = sha256 of `objective_id`, newline, daemon binary sha256, newline, model weights sha256, newline,
  host name (no trailing newline).
- Each receipt is `chain/NNN-<step>.json` with `objective_id`, `run_id`, `step`, `status`, `utc`,
  `prev_receipt_sha256` (64 zeros for the first), `evidence` and the step's fields. `chain/SHA256SUMS` lists
  every file of the run folder. `VERDICT_ROWS.txt` holds the lines the verifier hashes (contract section 1,
  `verdict_id`); the harness does not compute `verdict_id` (that is the verifier's job).

## 3. Rows

| row | the harness does | PASS rule | prediction |
|---|---|---|---|
| PRE | records host, `ip route`, quiet flag state, daemon and CLI sha256 (same binary), model weights and tokenizer sha256 against `release/candidate.toml` CAND-4, the daemon's `Backend:` line | backend line contains `CPU-reference`; weights and tokenizer digests equal the candidate record; quiet flag not held; no non-loopback established socket of the daemon at the end of the run | PASS |
| E1 | nothing installed; the daemon runs from a release build of this tree | NOT_RUN by design in RUN-1 (`signature_check` `none`); RUN-3 measures it | NOT_RUN |
| E2 | writes the objective receipt (chain head) with `objective_text_sha256` and `recorded_before_work` true, then starts work | installation-side record before work: NOT_RUN until lane L6; the harness-side receipt is `001-E2` and every later receipt chains to it | NOT_RUN |
| E3 | the contract objective, exactly: `Read the three text files in the inbox folder and write a Markdown report under 200 words named report.md in the outbox folder; begin it with a heading line; mention each file by name.` through propose, authorize (desk on), execute in workspace `ws-contract/` whose `inbox/` holds three UTF-8 files named in the PRE receipt | execute state DONE; the written file exists where the engine put it (recorded) and its sha256 equals the execute `content_sha256`; first line starts with `# `; at most 200 words; contains all three file names; a second execute with the same grant is refused `AlreadySpent` | FAIL on the file-name check (engine limit above) |
| E3m | the mechanics objective in workspace `ws/`: `Write a Markdown report named report.md about the three inbox files named <a>, <b> and <c>. Start with a Markdown heading line that begins with "# ". Use at most 200 words. Mention each of the three file names.` (names filled from the fixture) | same rule as E3 | PASS |
| E4 | `compose remember --text "E4 memory item: the inbox file reported on first is <a>"` (operator surface; automatic recording is lane L3), SIGKILL the daemon, restart, second objective exactly: `Append one line to report.md naming the file you reported on first.` with `--context work` | daemon log shows the restart (`Replay reconcile:` line of daemon 2); the proposal's memory report has context `work` and `items_included` at least 1; `report.md` has exactly one more line than before and that line contains `<a>`; execute DONE; the recall receipt's `utc` is not earlier than the restart | PASS or FAIL as measured (edit mode on a file named without its folder is untested) |
| E5 | writes `VERDICT_ROWS.txt` and `chain/SHA256SUMS` | NOT_RUN until lane L4 | NOT_RUN |
| E6 | objective `Write a file named done.md in the outbox folder containing one line: finished.`: propose, authorize, then SIGKILL the daemon before any execute (`kill_point` `after_authorize_before_execute`), execute once while the daemon is dead (recorded refusal), restart, execute with the same grant, execute a third time | first execute refused (no daemon); after restart the same grant executes to DONE exactly once (`done.md` exists, written once: inode and mtime unchanged by the third call); third execute refused `AlreadySpent`; `duplicates` 0; `restart_count` 1 | PASS (C2a of NEXT-PHASE-2 showed this shape) |
| CTRL-E3a | before the desk key exists: start a daemon with the desk required | the daemon exits without creating the socket and its log names the missing desk key (sc#328, sc#342) | PASS |
| CTRL-E3b | objective `Write a file named ctrl.md in the outbox folder containing one line: control.`: propose, authorize, `compose revoke`, execute | execute refused `Revoked`; `ctrl.md` absent | PASS |
| CTRL-E4 | last step: graceful stop, remove the compose home (the records the recall read from), restart, run the E4 second objective again | a named refusal (not a silently written line) | FAIL predicted (a fresh home is opened silently); recorded either way |
| CTRL-E1, CTRL-E2, CTRL-E5, CTRL-E6 | not measurable in harness v1 (no artifact, no installation-side record, no verifier, no durable objective state) | NOT_RUN with the lane that supplies them (L5, L6, L4, L2) | NOT_RUN |

## 4. Rules of the run (contract section 5, applied)

No network use by the run (recorded, not enforced by a sandbox); no Python, no systemd, no CUDA, no GPU;
real weights (CAND-4 digests in every receipt); harness shell plus `jq`; the run folder is never edited after
the manifest is written; one attempt per declared run, a failed run stays recorded.

## 5. What a PASS of any row does not show

Nothing about an installed artifact, a public model, the GB10 path, native AIENOS, a hostile operator, or the
correctness of the prose beyond the stated checks. Nothing here is evidence for the program's milestone R5;
this is milestone R4 (harness merged, first dry run recorded).
