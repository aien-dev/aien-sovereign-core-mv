# L593 PR delivery fastpath

This extends the local-authoritative L522 queue and L545 website lane. It does not replace either service or create another PR writer.

## Generic READY lifecycle

1. Intake remains an explicit, exact-repository local READY record with an acknowledged full head/base and literal branch name. Historical `refs/remotes/origin/...` values are refused, not silently rewritten.
2. Local progress performs the existing full-diff review and host planner. No truncated diff is approved. The review identity includes source/base/diff/model/policy and delivery code.
3. Missing qualification can enter the existing operator runner, at most one local READY admission per repository per pass. Planner refusals and pinned crumb failures stop before admission. Known FAIL is retained, never retried.
4. Generic publication requires the existing signed-bundle verifier to declare the exact-source receipt eligible, including a PASS plan without unmapped or removed paths. Then Renda rechecks exact remote branch/main, ownership, pause/hold and existing PR identity before guarded creation. An exact open PR is reused; closed/moved/malformed PRs are refused. Read-back must confirm the exact PR/head/base before writing a publication record.
5. PR publication is not merge approval. Current official Spark and hosted checks, required branch rules and the original merge contract remain unchanged.

The old README's statement that READY intake never converts to PR describes L522 alone; this qualified optional publication adapter is the subsequent L593 delivery stage. The site lane retains its distinct Mac-signed qualification policy.

## Cheap admission and noise reduction

The existing planner must return RUNNABLE, or the exact operator-only pipeline-configuration refusal. Malformed, unmapped, removed or ambiguous plans never start a judge. When CRUMB-VERIFY remains required, metadata preflight runs an already-installed exact pinned tool in a detached candidate worktree. It never compiles or executes a candidate-supplied Makefile or substitutes a different crumb version. Missing tooling is an explicit hold, not a waiver.

Hosted reconciliation requires eligible local signed evidence and identical published bundle bytes. One dispatch is reserved durably per exact source/evidence/run identity. An unchanged failed attempt is not repeatedly rerun. Refused/ambiguous dispatch stays visible pending correction.

Reports emit changed queue observations and bounded 15-minute health heartbeats rather than repeated full-queue dumps. A disappearing observation is labelled 'no longer reported', not 'resolved'. Retained operator verdicts are not counted as new judge actions. Successful existing-PR reconciliation is not counted as a new publication.

## Reproduction

```sh
export RENDA_CANDIDATE="$PWD/tools/aien-local-ci/renda/renda-harness"
python3 -m unittest discover -s tools/aien-local-ci/renda -p 'test_*.py' -v
```

Mac tests require Homebrew OpenSSL 3 first on PATH because system LibreSSL lacks the Ed25519 recipe used by existing site tests. The tests use synthetic external responses and real temporary local git repositories; they are not live PR publication evidence. Original RED logs and exact Spark receipts are retained in the evidence packet. Full core workspace/model and official Spark qualification remain separate requirements, never inferred from this service regression suite.
