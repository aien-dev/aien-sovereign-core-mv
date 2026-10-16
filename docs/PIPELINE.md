# The Spark pipeline: how a pull request gets tested and merged

Since 2026-10-16 the DGX Spark is the only test judge for this repository (ADR 0033 Decision 11,
phase 3, agreed by Drake on 2026-10-15). GitHub's heavy test jobs are retired one by one as a
Spark gate is proven to cover each of them.

## What happens to every pull request

1. The Spark worker (`aien-test worker --github`, built from `tools/aien-test` in this repository) sees the
   new head, runs the gates in `tests/gates/` that the change touches plus `.aien/always`, inside
   the Spark sandbox, and signs a bundle with the Spark key. It posts the commit status
   `AIEN Verification (Spark)`.
2. The pipeline (`tools/aien-local-ci/pipeline.sh` in this repository, every 2 minutes)
   checks the bundle against `.aien/verify.pub` on `main` and publishes it to
   `aien-dev/aien-receipts` at `aien-sovereign-core/<sha>/bundle.json`.
3. It then reruns the `AIEN Verification` workflow on that head. That workflow downloads the
   bundle and verifies it on GitHub. Until the bundle is published the check is red with
   `NOT_RUN`; that is expected and clears on the rerun.

## How to get a pull request merged

Add the label **`ready-to-merge`** when the pull request is finished. The pipeline merges it
(merge commit) only when all of these hold for the current head:

- the label is exactly `ready-to-merge` and the pull request is not a draft;
- the Spark bundle for that head verifies as PASS, including every gate in `.aien/always`;
- the published copy in aien-receipts is byte-identical to the Spark's copy;
- every required check on `main` is green on that head, and no `AIEN Verification` run is pending.

Pushing a new commit moves the head; the pipeline tests the new head before it merges anything.
Without the label nothing is merged. Remove the label to stop a merge.

To add the label from a terminal use the REST call (`gh pr edit --add-label` can fail with a
Projects (classic) error, as it does on aienos):
`gh api -X POST repos/aien-dev/aien-sovereign-core/issues/<number>/labels -f 'labels[]=ready-to-merge'`.

## Changes to the pipeline itself

A head that changes `.aien/`, `tests/gates/`, `.github/workflows/`, `.forgejo/` or
`tools/aien-test/` is `BLOCKED_OPERATOR`: the worker does not run it on its own. The operator runs
it by hand on the Spark (`aien-local-ci once aien-sovereign-core <sha> --operator`); the signed bundle then
records the operator run and the checks pass.

## Where to look

- Spark state: `aien-local-ci status`, journal `~/.local/state/aien-local-ci/journal.jsonl`,
  pipeline journal `~/.local/state/aien-local-ci/pipeline.jsonl`.
- Published bundles: <https://github.com/aien-dev/aien-receipts/tree/main/aien-sovereign-core>.

## Which gate covers which hosted step

Each manifest in `tests/gates/` names, in its `Mirrors hosted:` line, the hosted step it replaces;
`tests/gates/run.sh` runs the same command. The heavy hosted job `Workspace Verification &
Invariant Audit` (`.github/workflows/ci.yml`, job `verify`) is covered step by step:
SC-NO-PYTHON, SC-FMT, SC-RELEASE-GATE-NEGATIVES (five release scripts), SC-AIEN-VERIFY-TESTS,
SC-PINS-SELFTEST, SC-NATIVE-IDENTITY, SC-INSTALL, SC-SCORING-V5, SC-CHECK, SC-CLIPPY, SC-TEST,
SC-STRICT-MODE, SC-ZERO-CUDA, SC-PREFLIGHT-GUARD, SC-AGENT-PREFLIGHT, SC-ZERO-DISK-SECRETS and
SC-ANTI-SLOP. Known gap: `scripts/agent-preflight.sh` scans the whole tree (secret-looking files,
dashes in the diff), but SC-AGENT-PREFLIGHT runs only when its listed inputs change and is not in
`.aien/always`, so a commit touching only other paths skips that scan.
