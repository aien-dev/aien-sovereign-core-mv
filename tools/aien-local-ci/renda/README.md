# Renda local queue (L522)

`renda-harness` retains the L509 desk and its seven reliability regressions.
`local_queue.py` runs pinned local work before considering remote effects.
Tracking issue: [705](https://github.com/aien-dev/aien-sovereign-core-mv/issues/705).

The versioned queue is under `$RENDA_STATE/renda-queue` (default
`~/.local/state/aien-local-ci/renda-queue`). Each PR record names repository,
number, exact head, exact observed base, provenance, and remote observation time.
Local READY records have no PR number, remote observation time, or remote state.
There is a maximum of 800 records per repository. Capacity holds new intake and
preserves existing work, including failures. No automatic cache purge occurs.

One per-repository flock protects sync and a durable 300-second reservation.
One GraphQL query fetches up to 100 PRs and their exact-head contexts. A second
page is permitted; incomplete PR or context pagination holds publication. Failures
back off exponentially, honor Retry-After and guard reset diagnostics, and retain
the last observed pins. Guard exit 98 waits at least 900 seconds. The local model
and local source work do not depend on a successful sync.

Each pass admits at most two new local plans per repository. Source objects must
already exist in a configured trusted pipeline or Renda clone. Missing source
stays unavailable. The adapter saves the real git diff, creates its own detached
worktree, and invokes the existing `aien-test plan`. It never runs a gate from this
path. Local review identities cover repository/PR/head, base, diff, model ID, and
the source hashes of both adapter and desk policy. An unchanged refusal is retained.
Oversized diffs remain available locally and require a separate complete review.

Local receipts are read from the existing bundle store. Public keys and required
gates come from the trusted source base. `aien-test verify-bundle` performs signature,
repository, commit, and gate verification on a private byte-exact snapshot. Receipt
repository identity comes from the trusted clone origin, preserving the existing
canonical versus migrated GitHub repository distinction. A signed PASS with an
absent, blocked, unmapped, or NOT_RUN plan is ineligible. Receipt diagnostics are
refreshed even when the model review is cached.

Local intent files under `renda-queue/pending` are diagnostic requests for the
existing desk. They grant no authority and are separate from the signed pipeline
outbox. The existing receipt publisher remains asynchronous. Remote desk work runs
only after a successful sync in that pass. It then re-reads exact PR head/base,
open/draft state, ownership and holds before effects. Merge additionally requires
fresh hosted/Spark/required contexts, a verified local receipt and PASS plan, and
the existing match-head merge constraint. Scripted rebases retain the existing
crumb helper and inject a final fresh-state check at its push boundary. Failed
remote closes retain block state. Failed local heads remain failed when remote
status publication is absent.

## Website lane: exact identity, Mac qualification, and guarded PR intake

The site lane key `aienos.com` maps only to `aien-dev/aienos.com`. The kernel lane
`aienos` remains `aien-dev/aienos-mv`; the registry refuses `aienos.com-mv` and
other unregistered identities. READY import writes an exact local acknowledgement
with no PR number or remote timestamp. Local diff planning and full-diff review
continue without GitHub access and do not run the Spark `aien-test` planner/gates.

Website PR publication requires a detached Ed25519 signature over the exact
canonical UTF-8 receipt bytes: `json.dumps(value, sort_keys=True,
separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')`,
with no final newline. The base64 signature is stored separately. The receipt
schema is `renda-site-qualification/1`, repository is `aien-dev/aienos.com`,
commit and tree are full 40-hex IDs, recipe is
`aienos.com/mac-site-tests-build-collector-v1`, `qualified` is true, and
`platform` is `macOS`. The required exact argv rows are the two site-builder
Cargo commands and `python3 scripts/test_renda_metrics.py`; every row also has
exit code 0 and a 64-hex SHA-256 of its full log. The detached signature is
verified only with the deployment-pinned PEM public key at the path named by
`RENDA_SITE_QUALIFICATION_PUBKEY`. Never put private keys or a replacement public
key in receipt JSON or source. Receipt files live at
`$RENDA_STATE/renda-queue/site-qualification/<commit>.json` and `.sig`.

Before opening a PR, Renda rechecks the trusted clone origin, immutable source
commit/tree/diff, complete local review, signature, remote branch SHA and latest
`main` SHA, merge-tree cleanliness, and existing PR identity. A malformed/closed/
moved PR or any changed ref is held; an existing exact open PR is reused. PR
PR creation uses the harness's inherited guarded `gh` transport, then reads back
the exact PR and requires `OPEN` plus `CLEAN`. Before recording publication,
Renda reads branch protection and active rulesets; every currently required
status context must succeed in the exact PR's status rollup. No configured check
is not a qualification exemption because the signed Mac recipe remains mandatory.
The acknowledgement distinguishes `PR_OPEN_CLEAN_PUBLICATION_PENDING_REVIEW` from
local intake. This path never merges a website PR and never claims a Spark gate;
the existing desk controls merge review.

Tests use explicitly synthetic repositories, model replies, and transport responses:

```sh
python3 tools/aien-local-ci/renda/test_local_queue.py
RENDA_CANDIDATE="$PWD/tools/aien-local-ci/renda/renda-harness" \
  python3 tools/aien-local-ci/renda/test_reliability.py
python3 tools/aien-local-ci/renda/test_install.py
python3 tools/aien-local-ci/renda/test_admission.py
python3 tools/aien-local-ci/renda/measure_fixture.py --baseline /path/to/L509-renda-harness
```

The fixed fixture has six failing PRs, ten passes, and 60 seconds between passes.
L509 makes 70 reads; the adapter makes two. Under injected guard refusal, attempts
are 10 versus one. These are fixture measurements, not a live performance claim.

The authentic retained-metadata proof is reproducible without executing gates:

```sh
python3 tools/aien-local-ci/renda/prove_offline.py \
  --retained /path/to/retained/renda-queue/omega.json \
  --state /path/to/new-private-proof-state
```

An explicit local READY import is also offline:

```sh
RENDA_STATE=/path/to/state tools/aien-local-ci/renda/renda-harness \
  --lanes omega --seed-ready /path/to/ready.json
```

The import is a JSON list of `{repository, ref, sha, base}` records. Preserve the
original READY text and its digest alongside that import. A local READY declaration
is never converted to a remote PR.

`install.py` checks the candidate release manifest hash, both live input hashes,
and release tests before installing immutable release files. It backs up live
files and atomically replaces executable links. It never signals workers. Existing
long-lived L509 processes require their owner's orderly clean start. Until a new
PID and startup/pass log identify this release, the state is
`INSTALLED_PENDING_CLEAN_START`. Future cron starts use both regression suites.

Workspace/model tests and full `scripts/agent-preflight.sh` are NOT_RUN in this
delivery: the latter launches the workspace suite, while this brief excludes other
workers' full model tests. Branch guard, Python regressions, real host planning,
real receipt verification, syntax checks, and diff checks are executed. The sole
PR desk must run the required full preflight before any merge. This worker opens
no PR and changes no lane ownership or L480 closure state.
