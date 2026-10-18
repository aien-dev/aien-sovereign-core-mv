# Local verification with aien-test

ADR 0033 Slices E and H: the planner, the trusted worker, the status and
verify commands, and the sandbox seam. Everything here runs on the Spark,
offline, with no outside service in the loop. GitHub only receives a commit
status when it can be reached; it is never asked for permission.

## Commands

| Command | What it does | Exit |
|---|---|---|
| `aien-test plan [--base REF] [--repo NAME] [--operator] [--allow-dirty]` | Which gates BASE..HEAD needs, by section, with the graph digest. BASE defaults to `origin/main`, then `main`. | 0 runnable, 2 NOT_RUN (unmapped path or removed gate), 4 BLOCKED_OPERATOR |
| `aien-test test changed [--base REF]` | Prints the plan, then runs the non-blocked selection through the normal runner. | the campaign verdict (PASS 0, FAIL 1, NOT_RUN 2, BLOCKED 3/4, MISSING 5) |
| `aien-test test crate:NAME` | The gates covering a workspace crate (from `cargo metadata`) and every crate that depends on it. | campaign verdict |
| `aien-test status [--state-dir DIR]` | Pools, GPU lock holder, aien-proof slots, quiet flag, worker heartbeat, pending triggers and outbox, newest bundle per repository. Read-only. | 0 |
| `aien-test verify COMMIT [--repo NAME] [--base REF] [--state-dir DIR] [--release]` | The independent check of a bundle, printed as the fixed report block below. | 0 only on `MERGE ELIGIBILITY: READY`, else 1 |
| `aien-test worker [--once REPO SHA] [--poll SECS] [--repos A,B] [--github] [--iterations N] [--state-dir DIR] [--key FILE] [--clones DIR] [--wt-root DIR] [--wt-tool FILE] [--base REF] [--sandbox user\|none --dev-unsandboxed]` | The trusted worker: one head with `--once`, or the loop. Runs gates in the user sandbox unless told otherwise. | `--once`: the plan verdict; loop: 0; 2 on a refused sandbox mode |
| `aien-test run GATE --sandbox user`, `aien-test test ... --sandbox user` | Any run, with every gate program started through the exec helper. | as before |

`--release` (run, test, verify): qualification mode. No cache reuse, the tree
must be clean even with `--allow-dirty`, and in `verify` a cached result of a
gate whose `cache_scope` is `hardware` cannot count.

## The plan

Selection is the set of gates whose watched paths (declared `inputs`, the
D3 implied inputs, the manifest file, the golden file) cover a changed path,
plus the always set, closed over dependents and then dependencies. Sections:

- **Required**: runs on this machine now.
- **Cached**: an evidence receipt already settles it (`plan` only; `verify` takes cached entries from the bundle).
- **Needs Spark**: pool `qemu`, or `arch`/`bare_metal` requirements.
- **Needs GB10**: pool `gb10`.
- **Advisory**: exactness ADVISORY; never required, never moves the verdict.
- **Blocked**: `PIPELINE_CONFIG_CHANGED` or `OPERATOR_STEP_PENDING`. A gate
  with `requires: operator` is `OPERATOR_STEP_PENDING` in every automatic run;
  under `--operator` (`plan`, `test`, `worker`) it takes its normal section and
  runs, with its own timeout, and its receipt counts like any other. The
  operator starting the run by hand is the attestation the manifest asks for
  (ADR 0033 Decision 2); nothing else lifts it.

A changed path no gate covers is listed under `UNMAPPED` and the verdict is
`NOT_RUN UNMAPPED_PATH`. A gate that exists at BASE and is gone at HEAD is
`NOT_RUN GATE_REMOVED`. Both are refusals, not failures: nothing vouches for
that change.

### Trust rules A1 and A4

- **A1**: the always set (`.aien/always`), the unmapped allow-list
  (`.aien/unmapped-ok`) and the trusted keys (`.aien/verify.pub`) are read
  from BASE through git objects, never from the working tree or from HEAD. A
  change under test cannot widen its own allow-list or add its own key.
- **A4**: a change under `.aien/`, `tests/gates/`, `.github/workflows/`,
  `.forgejo/` or `tools/aien-test/` is a pipeline configuration change. The
  whole plan is `BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED` unless the operator
  passes `--operator`.

`graph_digest` is sha256 over the canonical `[{gate, check_id, depends_on}]`
of the selected gates, sorted by gate. `verify` recomputes it from the
manifests at COMMIT and rejects a bundle whose digest differs.

## Manifests: `run.exec`

`run.exec` is either a repo-relative path or a bare program name:

- A name **with a slash** (`tools/run.sh`) is resolved against the repository
  root. Absolute paths and any `..` component are a manifest reject
  (`BAD_MANIFEST`, exit 1), never NOT_RUN and never a silent run. A path that
  leaves the repository through a symlink (`tools/run.sh -> /bin/true`) is
  treated as absent: `MISSING_IMPLEMENTATION`, never run, never PASS.
- A name **without a slash** (`crumb`, `true`) is a program on a PATH. Under
  `--sandbox none` that is the runner's own PATH. Under `--sandbox user` it
  is the exec helper's trusted PATH
  (`/var/lib/aien-run/cargo/bin:/usr/local/bin:/usr/bin:/bin`, the
  `TRUSTED_PATH` constant); the runner resolves the name there itself and
  hands the helper the full path, so the program hashed into the receipt and
  the cache key (`binary_sha256`) is the program that runs. A name not found
  there is `MISSING_IMPLEMENTATION`.

This is what lets the shared gate template say `exec: crumb` with
`args: [verify, .]`: INFRA installs a pinned `crumb` at `/usr/local/bin/crumb`
(root-owned, on the trusted PATH). Proven end to end on 2026-10-11: a gate
running `crumb verify .` as `aien-run` in a worker checkout passed.

## The sandbox seam (C8)

`--sandbox user` rewrites every gate launch into

```
sudo -n /usr/local/sbin/aien-run-exec --cwd CWD --timeout SECS --profile host|qemu|gb10 [--env CARGO_TARGET_DIR=...] -- PROGRAM ARGS
```

Only `CARGO_TARGET_DIR` is passed through. Under `--sandbox user` the
receipt's `core.machine` also carries `sandbox_helper_sha256` (sha256 of
the installed helper), so it is in `machine_digest` and the cache key: a
result made under another helper (other network, mounts or limits) is
never reused. The runner's own timeout and
kill rules are unchanged; a gb10 timeout stays advisory. For `run` and
`test`, `--sandbox none` (the default) leaves the launch exactly as it was.

What the helper requires (INFRA, read from the installed script): `--cwd`
canonical, and inside `/var/lib/aien-run` only a run directory
`wt/<repo>/<40-hex>` or below it; host and qemu checks get a private `/tmp`, so a tree
under `/tmp` is invisible to them; `CARGO_TARGET_DIR` only as
`/var/lib/aien-run/target/<repo>/pr` or `.../main` (created on demand).
Inside a run everything is read-only except the run's own run directory (file
ownership inside it still applies: siblings are not writable), its one
target dir and a private `/tmp`. host and qemu checks have no network but
their own loopback: `lo` is up inside their private network namespace, so a
test may start a server on 127.0.0.1 and talk to it; the host's own
127.0.0.1 services stay out of reach. When the helper refuses a call it exits 2
and its stderr starts with `aien-run-exec:`; nothing ran. The runner records
such a run with no exit status, so it is FAIL and can never meet
`expects.exit`, not even `exit: 2` (`runner::sandbox_refused`).

## The worker and the sandbox

The worker runs other people's code, so it uses `--sandbox user` unless told
otherwise. `--sandbox none` runs gates as the worker's own user, who can read
the signing key: the worker refuses it (exit 2) unless `--dev-unsandboxed` is
given, and refuses it with `--github` in every case. The library has the same
rule (`Config.allow_unsandboxed`, default false). Under `--sandbox user` the
worker passes `CARGO_TARGET_DIR=/var/lib/aien-run/target/<repo>/main` for a
head already on the base line and `.../pr` for any other head (A2).

The gates of one head share that target dir, so the runner holds it for one
gate at a time: every gate that runs takes an exclusive `flock` on
`<state>/locks/target/<sanitised target dir>.lock` (the `CARGO_TARGET_DIR` it
is given, or else the worktree's own `target/`) for its whole run, build and
execution, and releases it when its process has exited. Every gate, not only
those whose manifest names cargo inputs: SC-INQUISITOR builds with cargo and
lists no `Cargo.toml`, and the runner cannot see inside a script. The lock is
the worker's, never the gate's. Without it two gates built in one target dir
at once and one relinked a test binary the other was executing (sc fix/521:
SC-AGENT-PREFLIGHT under SC-TEST, `current_exe: NotFound`, a false FAIL).
Gates given different target dirs still run at the same time. The fixed lock
order of one gate is: pool slot, target-dir lock, `/tmp/aien-gb10.lock`,
aien-proof `slots/gpu.lock`; the worker's target priming takes only the
target-dir lock. Known limit: a gb10 gate is never killed on timeout, so a
hung one holds the target dir until it exits on its own.

Under `--sandbox user` every checkout is made with INFRA's worktree tool
(`--wt-tool`, default `<clones>/aien-sovereign-core/tools/aien-local-ci/wt.sh`;
without it the worker refuses to run). `wt.sh sync` refreshes the bare mirror
`/var/lib/aien-run/mirror/<repo>.git` from the trusted clone (a head no branch
names is then fetched into the mirror by its id), and `wt.sh add` checks the
head out at `/var/lib/aien-run/wt/<repo>/<sha>/<repo>` (the run directory
`wt/<repo>/<sha>/` holds it and its siblings): files owned by `aien-run`,
the `.git` link root-owned and pointing into the mirror. An older tree for
the same head is removed first, never reused. The worker opens that tree
with its own git only after checking the link (a regular root-owned file
nobody else can write, pointing into `<mirror>/worktrees/`), and then only
through git's command-line `safe.directory` for that one path
(`process::trust_worktree`): every git setting and hook git reads there is
the operator's mirror, never the checked-out code. `wt.sh remove` deletes
the run directory (checkout and siblings) afterwards.

Before it fetches or runs anything the worker refuses a repository name that
is not a plain name (letters, digits, `.`, `_`, `-`, not starting with a
dot), a commit that is not 40 lowercase hex, and a signing key whose mode
gives group or other any access (it must be 0600).

## Sibling checkouts

Some gates build or read a neighbouring repository: `tests/gates/run.sh`
calls `need_sibling <dir>` or `pin <name> <repo> <sha>` and reads `../<dir>`,
the way the hosted job checks that repository out beside the one under test.
The one rule:

1. **Where.** A sibling is always `../<dir>` of the checkout under test. The
   worker runs a head in a run directory `<wt-root>/<repo>/<sha>/`: the
   checkout at `<repo>/`, each sibling at `<dir>/`, so `..` from the checkout
   is the run directory. Under `--sandbox user` the exec helper gets
   `--cwd <wt-root>/<repo>/<sha>/<repo>`.
2. **What.** A sibling is a read-only checkout of a trusted clone
   `<clones>/<source>` at `refs/remotes/origin/<ref>` as that clone has it
   when the run starts. The worker never fetches it for the run and never
   uses the network for it. Under `--sandbox user` it is a worktree of the
   `wt.sh` mirror (`wt.sh sync <source>`, then `wt.sh sibling`), owned by the
   operator with no group or other write, so `aien-run` can read the files
   and their git history (`git -C ../<dir> rev-parse HEAD` works) but cannot
   change them or add entries to the run directory. Under `--sandbox none` it
   is a `git worktree add --detach` from the clone.
3. **Which.** The operator's sibling map decides, never the change under
   test: `tools/aien-local-ci/siblings` in the trusted clone of
   aien-sovereign-core (`--siblings FILE` overrides), one line
   `<repo> <dir> <source-repo> <ref>`. Names are plain names, the ref a plain
   branch name; one malformed line, or an explicit map that cannot be read,
   refuses every run before anything is checked out. A repository the map
   does not name gets no siblings. A sibling that cannot be made (no
   `origin/<ref>` in its clone, a `wt.sh` failure) is a journal note and its
   gates fail on the missing directory; they never pass without it.
4. **Recorded.** The journal names every sibling with its commit
   (`sibling ../<dir>: <source> origin/<ref> <sha>`), and `run.sh` records
   the commit it used in the receipt as an `AIEN-OBS` observation.
5. **Other commits.** A gate that needs a sibling at a commit other than its
   head (a lock file, a pinned protocol version) materialises that commit
   from the sibling's object store into a writable, ignored place:
   `$CARGO_TARGET_DIR/aien-pins/<name>` (aien-sovereign-core) or a
   gitignored build directory, never inside the sibling and never in a
   tracked path. The mirror holds every branch of the trusted clone, so a
   commit on any branch is there.
6. **Manifests.** A gate that needs a sibling is `requires: host` (or a
   stronger pool), never `portable`: `portable` means the tree alone, which
   is what the hosted thin gate runs.
7. **Cache.** A sibling lies outside the tree the cache key covers, so the
   worker records each sibling made (`name` = its `dir`) and each operator
   pin (`name` = `pin:<source>`) with its commit in the receipt's `pins`,
   which is part of the cache key. The worker's evidence directory is per
   head (`evidence/<repo>/<sha>/`), so a result is reused only when the same
   commit runs again with every sibling at the same commit; a sibling that
   moved runs every gate fresh. A declared sibling that cannot be made turns
   the cache off for the head (`no_cache`). Results are never reused across
   commits in the worker: gates read files they do not declare as `inputs`
   (SC-TEST reads the root `README.md` and `COVENANT.md`), so a key built
   from declared inputs alone could hand a commit another commit's PASS.

`wt.sh remove` takes the whole run directory away (checkout and siblings)
and prunes every mirror's worktree list. An older `wt.sh` that still makes
the flat layout (`<wt-root>/<repo>/<sha>` is the checkout) keeps working;
the worker then journals that no siblings were provided.


### Pinned sibling commits and model checkpoints

Some gates check out a pinned commit of a sibling (`pin <dir> <repo> <sha>` in
`tests/gates/run.sh`, cloned from `../<repo>`). The sandbox has no network,
so a pinned commit that is on no fetched branch must already be in the
sibling. `tools/aien-local-ci/pins` (beside `siblings`, read from the
operator's trusted sc clone, never from the head) lists them as
`<repo> <source-repo> <sha40>`. Before the siblings are made, the worker
(`ensure_pins`, trusted side, provisioning time only) fetches a missing pinned
commit by its sha from the trusted clone's own `origin` and holds it as
`refs/remotes/aien-pins/<sha>`, so `wt.sh sync` copies it into the mirror and
git never prunes it. A pin that cannot be made present is a journal note; the
gate that needs it then fails, never PASS. Pins read from lock files (omega's
`omega.lock` and the locks inside it) follow main and are not listed.

Model gates (PREFILL-E2E-2 and its eight mutant gates, `requires: host`) read
the TinyLlama-1.1B-Chat-v1.0 checkpoint at `$HOME/models/TinyLlama-1.1B-Chat-v1.0`
(`model.safetensors`, `tokenizer.json`). Inside the sandbox `$HOME` is
`/var/lib/aien-run`; `aien-run-exec` mounts the sudo caller's `~/models`
read-only at `/var/lib/aien-run/models` (empty root-owned mount point made by
`install.sh`), so the default path resolves there. The script records the
weights' sha256 as the observation `model_sha256`. Without the checkpoint it
prints `AIEN-OBS not_run_reason "model_missing"` and exits 77: the verdict is
NOT_RUN with reason `model_missing`, never PASS. Exit 77 with a plain
lowercase `not_run_reason` is the general rule (`verdict::SELF_NOT_RUN_EXIT`);
without that observation, or when a manifest itself expects 77, it is an
ordinary exit.

The mutant gates apply their patch to a private copy of the tree in `/tmp`,
never to the checkout: the gates of one head run in parallel in one checkout,
and a patched file there would fail every other gate's clean-tree check.

`wt.sh sync` also copies the trusted clone's `origin` URL into the mirror, so
`git remote get-url origin` in a sandbox checkout gives the GitHub URL, as on a
GitHub runner (the sc tier-1 test f62 reads it). Nothing fetches through it:
the sandbox has no network and cannot write the mirror.

## The worker

For one head `REPO SHA`:

1. The refusals above (names, key mode, sandbox mode).
2. `git fetch --all` in the trusted clone `<clones>/<repo>` (default
   `~/workspace/<repo>`); its checkout is never touched.
3. Under `--sandbox user`: `wt.sh sync` and `wt.sh add` (see above). Under
   `--sandbox none`: `git worktree add --detach <wt-root>/<repo>/<sha>/<repo>`
   from the clone (default `/var/lib/aien-run/wt`). Then the siblings of
   `<repo>` from the sibling map (see "Sibling checkouts"), each journaled
   with its commit.
4. `crumb claim` on the worktree when it has a `.crumb` (a missing one is a
   journal note, not an error).
5. Plan against BASE: `origin/main` (`--base` overrides), or the head's first
   parent when the head is already on that line (the tip of main or an older
   main commit). `verify` applies the same rule, so both compute one plan.
6. Run the non-blocked selection with the pools and the sandbox; receipts go
   to `evidence/<repo>/<sha>/`.
7. Build the bundle: one `checks[]` entry per planned gate with status
   `run`, `cached`, `blocked` or `not_run`; `tree_digest` is sha256 of the
   commit's tree id; the executor is the key's fingerprint with class
   `spark`. Sign with `~/.aien/verify.key` and write
   `bundles/<repo>/<sha>/bundle.json` (canonical bytes). The bundle also
   carries a `plan` member (`verdict`, `reason`, `unmapped`, `removed`,
   `blocked`): the bundle algebra says PASS when no required check failed,
   which is vacuously true for an unmapped change with no checks. The plan
   member is signed with the rest, so a reader can tell the vacuous case
   apart: `verify-bundle` reports such a bundle as NOT_RUN (see below).
8. Run the independent check on the bundle just written, exactly as
   `aien-test verify` does (keys at BASE, signature, graph digest, check ids,
   receipts and their manifests, crumb). The journal records `verify: READY`
   or `BLOCKED` with the reasons.
9. Heartbeat (`worker.heartbeat`), journal line (`journal.jsonl`), `crumb
   close aien-worker . worker test`, remove the worktree (`wt.sh remove`
   under `--sandbox user`). A crumb failure is
   a journal note, never a verdict change.

With `--github` the commit status `AIEN Verification (Spark)` is posted:
`success` only when step 8 said READY (which needs the bundle and the plan to
say PASS), `failure` when the bundle or the plan says FAIL, `error`
otherwise; the description names the bundle path. When `gh` cannot reach
GitHub the status is queued as `outbox/<seq>.<repo>.<sha>.json`, where `<seq>`
is a 20-digit number that only grows, and the journal says `publication:
pending`. Every later round of a loop started with `--github` replays the
outbox first, in queue order (a loop without `--github` leaves it queued); the
first status GitHub refuses stops the replay, so a newer status never
overtakes an older one. A newer status for the same head replaces the queued
one and moves to the end. A file that is not a status is renamed to `*.bad`
and journaled. GitHub being down never fails a local verification.

The loop (`worker` without `--once`) serves trigger files
`trigger/<repo>.<sha>` oldest first (by modification time), then, with
`--github`, the open pull request heads and `main` of each `--repos` entry
that has no bundle yet, then sleeps `--poll` seconds. `--iterations N`
bounds the rounds (tests). Trigger names that are not a plain repository
name and a 40-hex commit are ignored.

What starts a run, and what does not (checked 2026-10-11 on the Spark):

| trigger | how | covers today |
|---|---|---|
| operator | `aien-local-ci trigger <repo> <sha>` writes `trigger/<repo>.<sha>`; `aien-local-ci once` runs in the foreground | any repository with a trusted clone |
| Forgejo push | `tools/aien-local-ci/forgejo-webhook.sh install <repo>` puts a post-receive hook in the Forgejo bare repository; a push there writes the trigger file, no network involved | installed for `aien-sovereign-core` only; Forgejo holds no copy of omega, aienos, physics or aien-architecture, and the copies it holds stopped at September commits because nothing pushes to them |
| GitHub poll | the loop started with `--github` (`aien-local-ci start --github`) lists open pull request heads and `main` of every `--repos` entry each round | all nine repositories, but it also posts the `AIEN Verification (Spark)` status, so it is off until the operator switches it on |

So a loop started without `--github` never picks up a new pull request head
or a new `main` commit by itself: someone has to write a trigger (or push to
Forgejo). Merging a pull request on GitHub does not wake it.

`--jobs host=N,qemu=M` and `--no-cache` apply to the runs the worker starts,
as they do for `aien-test test`. `--no-cache` runs every selected gate again
(a head reuses only its own earlier results, see "Sibling checkouts"). `--jobs host=1`
runs one host gate at a time: omega's gates share one checkout's `build/`
and pin clones, and on main 771af52 ten of them failed from parallel races
("Text file busy" on `build/omegatool`, concurrent clones of
`build/pins/aienos-src`); its operator runs use
`aien-local-ci once omega <sha> --operator --jobs host=1` until the gates or
a per-repository pool setting make them safe in parallel.

## verify-bundle and the vacuous PASS

`aien-test verify-bundle` (the GitHub thin gate's command) prints and exits
with the gate verdict, not the raw algebra: a bundle with no required check,
or whose signed `plan.verdict` is not PASS, is `NOT_RUN` (exit 2) with the
reason in brackets, even though its `verdict` member says PASS.

## The verify report

```
AIEN LOCAL VERIFICATION
Repository: github.com/aien-dev/<repo>
Commit: <40 hex>
Changed files: N
Affected gates: N (names) | none
Executed gates: ...
Cached gates: ...
Failed gates: ...
Blocked gates: ...
Crumb status: CURRENT | STALE
Evidence status: VERIFIED | INVALID
Hardware status: QUALIFIED | NOT_REQUIRED | BLOCKED
Final verdict: PASS | FAIL | NOT_RUN | BLOCKED
Queue time: N ms | n/a
Build time: n/a
Test time: N ms | n/a
Total time: N ms | n/a
MERGE ELIGIBILITY: READY | BLOCKED
```

Lines after the block (`Evidence note`, `Crumb note`, `Verdict reason`,
`Unmapped`, `Problem`, `Bundle`) say why. Checks, in order: the bundle's
shape and signature against the keys at BASE (several keys allowed, one hex
per line, `#` comments; the key the bundle names must be one of them, else
`UNKNOWN_KEY`), repository, commit, executor fingerprint, every required
gate present, the verdict algebra; then the graph digest against a fresh
plan (BASE by the worker's rule above); then every entry's `check_id` must
be the one the plan computes at COMMIT (`CHECK_ID_MISMATCH`); then every
`run` entry's receipt must exist under `evidence/<repo>/<commit>/` with the
same gate, commit and verdict (an altered receipt no longer matches its
digest and is reported missing) and with the `manifest_digest` of the gate's
manifest at COMMIT (`MANIFEST_MISMATCH`); then the plan judge. Every
rejection is a `Problem` or `Evidence note` line and leaves READY out of
reach; no path defaults to PASS. Crumb status comes from `crumb verify .` in a temporary
detached worktree at COMMIT, removed afterwards. Queue time is the sum of
`pool_wait_ms`, test time the sum of `duration_ms` over the executed
receipts, total time the bundle's `finished_utc - started_utc`; build time
is not measured yet.

`READY` requires: evidence VERIFIED, final verdict PASS, crumb CURRENT, no
unmapped path, no blocked gate, no problem line.

## File layout

```
~/.aien/verify.key                       the worker's signing seed, mode 0600
<repo>/.aien/verify.pub                  trusted public keys, read at BASE
<repo>/.aien/always                      gates that always run, read at BASE
<repo>/.aien/unmapped-ok                 paths no gate needs to cover, read at BASE
~/.local/state/aien-local-ci/
  evidence/<repo>/<sha>/<digest>.json    receipts of that run
  bundles/<repo>/<sha>/bundle.json       the signed bundle
  trigger/<repo>.<sha>                   a run request (removed when served)
  outbox/<seq>.<repo>.<sha>.json         a commit status waiting for GitHub, in queue order
  journal.jsonl                          one line per finished head
  worker.heartbeat                       pid, phase, repo, sha, utc
/var/lib/aien-run/wt/<repo>/<sha>/       the run directory of one head (operator, 0755)
  <repo>/                                the detached checkout under test
  <dir>/                                 a read-only sibling checkout (../<dir> from the checkout)
<clones>/aien-sovereign-core/tools/aien-local-ci/siblings   the operator's sibling map
/var/lib/aien-run/target/<repo>/pr|main  cargo target dir per trust class (sandbox user)
/usr/local/sbin/aien-run-exec            the exec helper (INFRA)
```

`<repo>` is the last segment of the repository name
(`github.com/aien-spark/aien-sovereign-core` gives `aien-sovereign-core`).

## Tests

`cargo test -p aien-test` covers the planner (selection, sections, A1, A4,
removed gates, verdict algebra), the sandbox rewrite, status parsing, the
gate verdict of `verify-bundle`, the worker's name and sandbox rules, and in
`tests/local_ci.rs` the pipeline end to end on a throwaway repository: a
worker run that `verify` reports READY, then the refusals: an altered
receipt, a bad signature, a bundle presented for another commit, a required
check left out of a re-signed bundle, a hardware-scoped cached result under
`--release`, an executor whose key is not in `verify.pub`, a signing key
another uid cannot read, a signing key with mode 0644, an entry for another
check id, a receipt made from another manifest, `--sandbox none` without
`--dev-unsandboxed` or with `--github` (library and CLI), unsafe repository
names and commit ids, a status that is never `success` when the independent
check is not READY, GitHub down then back (fake `gh`), an outbox that
replays in queue order and stops at the first refusal, a trigger file served
by the loop, a head on main checked against its first parent, a bare
`exec: true` running, escaping exec paths refused and a symlinked exec that
leaves the tree never run, and, when the helper is installed, a gate and a
whole worker run in the user sandbox through `wt.sh` (in
`/var/lib/aien-run/wt/aien-test-selftest`, because the sandbox does not share
`/tmp`), a wrong target dir refused by the helper with exit 2 and never
counted as the program's exit, only a root-owned `wt.sh` link trusted, and
the user sandbox refused without the worktree tool, and siblings: a gate
that reads `../sib` fails without the map and passes with it, the run
directory and both worktree lists are clean afterwards, a malformed or
unreadable map refuses the run, and in the user sandbox the sibling is
readable but neither it nor the run directory is writable.
`mutations/run_mutants.sh` must
kill every line of `mutations/aien-test.mutants`.

## Accepted limits (ENGINE-2 review, 2026-10-11)

- A head that no branch names is fetched into the `wt.sh` mirror by its id;
  its objects stay there unreferenced until git prunes them. The mirror is
  readable by `aien-run`, like every other object in it.
- While a worktree is trusted, the worker sets `GIT_CONFIG_COUNT` and
  `GIT_CONFIG_KEY_n` / `GIT_CONFIG_VALUE_n` on the git and crumb processes it
  starts; a value of these variables in the worker's own environment is
  replaced for those processes.
- The `main` target dir is not wiped after a FAIL or timeout yet (A2 asks for
  it); the dir is owned by `aien-run`, so the wipe has to run through the
  helper or `sudo -n rm -rf`. Until then a poisoned `main` target dir is
  cleared by INFRA's `gc.sh` or by hand.
- `--dev-unsandboxed` runs are not marked in the bundle (the executor class
  stays `spark`); only the journal and the command line show it. Never use
  it on a machine whose bundles are published.
- A `cached` entry is trusted through the bundle signature; `verify` does not
  reopen the receipt it was resolved from.
- `verify` has no `--operator`: a pipeline-configuration change is never
  READY, also after the operator ran it by hand with `worker --operator`. The
  operator merges such a change himself (A1).
- A head whose base is a root commit with no parent cannot be planned on the
  base line (the worker reports an error, never PASS).
- Build time is `n/a` in the report. Total time comes from the bundle's
  whole-second timestamps, so a run inside one second reports `0 ms`; the
  per-gate test time is exact.
- A sibling is as fresh as its trusted clone: the worker fetches the clone
  of the repository it serves, not the sibling's clone (SANDBOX lane,
  2026-10-11). The commit used is journaled, so a stale sibling is visible.
- The sibling map and `wt.sh` are read from the trusted clone of
  aien-sovereign-core; after a change to either merges, that clone's
  checkout must be moved to the new main by the operator before the worker
  uses it (the worker never touches a clone's checkout).
- Five aien-test self-tests (`a_wrong_target_dir_is_refused_by_the_helper`,
  `user_sandbox_when_the_helper_exists`, `worker_in_the_user_sandbox_is_ready`,
  `worker_in_the_user_sandbox_reads_a_read_only_sibling`,
  `worker_provides_a_sibling_checkout_and_removes_it`) start a sandbox or
  provision sibling worktrees themselves, which `aien-run` cannot do inside the
  sandbox. They skip only when `AIEN_SANDBOX_NESTED=1`, which the sandbox
  wrapper sets for every program it runs (`sandbox::NESTED_MARKER`); they run
  on GitHub runners and in operator runs. SC-TEST and SC-AGENT-PREFLIGHT record
  the skip as the observation `nested_sandbox_skips`.
- The model mount exposes the whole of the operator's `~/models` read-only to
  every sandboxed gate (all model files, and anything else kept there).
