# REGRESSION-sc380: operator objective record and the CTRL-E2 refusal

Tracking: aien-sovereign-core#380 (lane L6), aien-architecture#190. Contract: `docs/plans/release-readiness/ACCEPTANCE_E2E.md` step E2 and CTRL-E2.

## Failing behaviour as observed

In the harness runs under `docs/campaigns/whole-system-e2e/runs/` step E2 is recorded NOT_RUN: the installation has no command that accepts an objective and records it. `aien compose propose --goal` runs the model first and only afterwards stores the goal inside the proposal report. The harness therefore wrote the objective receipt itself, and CTRL-E2 (a forbidden effect class refused before execution) had nothing to test.

## What is added

- `aien objective record --text TEXT [--effect-class CLASS]` writes `objectives/<objective_id>.json` under `AIEN_PROVENANCE_DIR` (same fallback as effect receipts) and prints `ok`, `objective_id`, `path`, `objective_text_sha256`, `recorded_before_work` (true), `operator_surface` (`aien-cli`), `utc`.
- `objective_id` = lowercase hex sha256 of `AIEN_E2E_OBJECTIVE_V1`, LF, then the text bytes. Text is at most 2000 bytes UTF-8.
- The operator declares the effect class. Default is `WORKSPACE_WRITE`. The CLI does not classify natural language. CTRL-E2 therefore tests the declared class: `EXTERNAL_IRREVERSIBLE` is refused with `OBJECTIVE_REFUSED ForbiddenEffectClass: ...` on stderr, `ok:false` on stdout, non-zero exit, and no record is written.
- `aien objective show ID` prints the record.
- `aien compose propose --objective-id ID` refuses (`ObjectiveMissing`, `ObjectiveMismatch`) before any model call if the record is missing or sha256 of `--goal` differs from the record; on success the report carries `objective_id`. Without the flag nothing changes.

## Tests (crates/aien-cli/tests/objective_record_test.rs and unit tests in src/objective.rs)

| test | on current main | after fix |
|---|---|---|
| record_writes_file_and_id_is_reproducible | FAIL (no command) | PASS |
| record_refuses_over_length_text | FAIL | PASS |
| record_refuses_forbidden_effect_class_before_any_work | FAIL | PASS |
| show_prints_the_record | FAIL | PASS |
| propose_with_mismatched_objective_id_is_refused | FAIL | PASS |
| propose_with_missing_objective_id_is_refused | FAIL | PASS |
| unit: matching_objective_id_is_accepted_and_carried | FAIL (does not compile) | PASS |

The matching case is a unit test of the check and the report attachment, because a live propose needs a daemon and the linked composition library.

## Must not change

Every existing command and flag, `compose propose` without `--objective-id`, the effect receipt format and location, `scripts/whole_system_e2e.sh`, and anything under `runs/`.
