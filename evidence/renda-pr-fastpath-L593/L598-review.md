# L598 independent review — L593 frozen source

Reviewed `aaedbfea..9ccfc410d1d78a00d389b097112daa1a1e991a6f` in the Spark worktree. Read `DELIVERY-L593.md`, `delivery.py`, harness and packaging/startup changes, relevant inherited queue/receipt/PR-policy paths, and `test_delivery.py`. Ran the bounded full Renda suite there: **69 tests passed**. No live service, model/GPU, official judge, or network PR write was performed. The passing count does not address the findings below.

## Findings

### Important — operator admission has no per-head durable reservation; concurrent lanes can start duplicate judges

**Locations:** `tools/aien-local-ci/renda/renda-harness:347-374` (`judging`, process grep, cheap admission, then `Popen`).

The checks at 347-351 are observations, not synchronization. Two processes can both observe no judge and no `aien-local-ci once <repo>` process; both then complete the same read-only planner/crumb preflight and independently call `Popen`. The `publish` lock only serializes generic READY publication and does not cover this operator admission path. The shared `judging(sha)` check is also only a process-list substring check. This violates the stated no-duplicate-judge constraint and can consume duplicate GPU/service capacity (especially when different Renda instances own the same repository or converge on one head).

**Counterexample:** start two `operator_run(repo, sha)` calls simultaneously with `judging` returning false and the process grep returning empty until both checks complete; with valid plan/metadata and non-dry mode both calls reach `Popen`. There is no lock file or atomic per-head claim between the check and launch. The service regression suite has no concurrent-admission test.

### Medium — READY acknowledgement is accepted by shape only and is trivially forgeable

**Locations:** `delivery.py:76-94`.

Publication treats any JSON file in `queue.root/acks` with the matching repository/state/item tuple as the exact acknowledgement. It does not validate the returned `ack_path`, `provenance_sha256`, input provenance, or any authenticated/append-only acknowledgement token. A file containing the matching fields can be fabricated by any writer to that state directory, so the check does not establish that `seed_ready` actually acknowledged the intake. This is a gap if the required “exact acknowledged READY” is intended as an auditable control rather than a same-user filesystem convention.

**Counterexample:** create `acks/fake.json` with schema `renda-intake-ack/1`, repository equal to the configured full name, state `ACCEPTED_LOCAL_NOT_PUBLISHED`, `remote_eligible: false`, and one matching `{ref, head, base, number:null, remote_observed_at:null}` item. `assessment()` accepts it without a corresponding `seed_ready` receipt or provenance validation (assuming the remaining local review/receipt evidence is valid). Tests only delete a genuine ack; they do not inject a forged one.

### Low — planner parser does not fail closed on malformed terminal output sections

**Location:** `delivery.py:49-60`.

`admit_plan()` validates one terminal verdict and scans `UNMAPPED:` only up to the next non-whitespace line. A malformed output can therefore carry an unmapped path (or `Removed gates:` section) and still be admitted if it says `RUNNABLE` with code 0. The official planner normally indents unmapped paths and emits a nonzero/NOT_RUN verdict, so this is primarily a malformed-output/defense-in-depth gap rather than a demonstrated ordinary planner path.

**Reproduction (executed on the frozen source):**
```
admit_plan({'code': 0, 'output': 'UNMAPPED:\nhidden.c\nPlan verdict: RUNNABLE\n'})
# (True, 'planner RUNNABLE')
admit_plan({'code': 0, 'output': 'REMOVED:\n  deleted.c\nPlan verdict: RUNNABLE\n'})
# (True, 'planner RUNNABLE')
```
`test_operator_cheap_admission_rejects_unmapped_or_missing_plan` covers only the canonical indented unmapped format, and no removed-section case.

## Assessment

I would not deploy without closing the duplicate operator-launch race. The acknowledgement issue needs an explicit threat-model decision; if filesystem writers other than the trusted intake path are in scope, it also needs closure before publication. The parser finding is lower priority but straightforward to make fail-closed. No critical finding demonstrated that a receipt or hosted official gate is waived; the signed receipt verifier and the existing merge/gate paths were left intact in the inspected code.
