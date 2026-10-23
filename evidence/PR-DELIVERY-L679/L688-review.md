# L688 independent corrective review

## Verdict

**No blocking finding in the explicit-old-PR-base correction or current-main/head-lease boundary reviewed.** The correction distinguishes an existing PR's leased old base from the candidate's qualified current-main base without weakening approval or source-ref checks. No live PR, model, judge, queue, or source operation was performed.

## Audit

Reviewed the frozen candidate package's two-file correction (`successor.py`, `test_successor.py`) against the L683 finding and inspected surrounding `delivery.py` intake, qualification, and remote-ref boundaries.

- `expected_base` is accepted only as a full 40-character lowercase SHA in the explicit target schema; absent it, legacy targets continue to require the candidate base.
- The exact target object is checked against the hash-bound original READY input, so changing `expected_base` in queue state alone does not confer authority.
- PR snapshot validation still requires the exact open, non-draft PR number, same-repository head and `main` base refs. With an expected old head, observed PR base SHA must equal the explicit `expected_base`; with candidate head already present, it must equal candidate `base` (idempotent readback).
- Candidate qualification still verifies the exact-source receipt and retained operator verdict. `refs_match()` still requires the candidate source branch and remote `main` to equal the READY head/base pins, and merge-tree validation remains in place.
- Before the push, the implementation rechecks PR snapshot/head, ownership, qualification, and current refs; the push retains the exact `--force-with-lease=refs/heads/<ref>:<expected_head>`. Exact PR-head and remote-branch readback are required afterward. Hosted and merge checks remain mandatory; no merge authorization is created.
- Synthetic tests cover old-base success, unexpected-base refusal, target tampering, moved/closed/forked target, failed gate, exact lease, and readback failure.

## Verification

Ran the full Mac candidate package suite, with `RENDA_CANDIDATE` set to its harness, Homebrew OpenSSL 3 on `PATH`, and scratch `TMPDIR`:

`python3 -m unittest discover -s /Users/drakestapleton/.hermes/cache/scratch/pr-repair-L679/candidate -p 'test_*.py' -v`

**Result: 119 tests passed, 16.810 seconds, OK.** This includes all eight successor tests and the full package suite.

## Scope / limitations

Read-only code and synthetic-control review only; did not contact GitHub or validate live PR state. No source files, PRs, model/judge calls, or ledger state changed. Review report only was created.