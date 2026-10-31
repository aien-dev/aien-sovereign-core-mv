# aien-local-ci: host sandbox for the sovereign local pipeline (lane INFRA)

Owner: lane INFRA of the sovereign local pipeline mission (inventory and contracts in
`~/handoffs/2026-10-11-sovereign-pipeline-M1-inventory.md`, contracts C3, C7, C8, A2, A3).
Everything here is bash. No Python. Host systemd units are used only for transient scopes; nothing here touches
AIENOS.

## Files

| file | role |
|---|---|
| `aien-run-exec` | source of truth for `/usr/local/sbin/aien-run-exec`, the C8 exec contract |
| `install.sh` | idempotent host install (user, dirs, caches, rustup proxies, trusted `crumb`, exec script, sudoers) |
| `wt.sh` | trusted-side mirror + worktree management under `/var/lib/aien-run` |
| `forgejo-webhook.sh` | push trigger: Forgejo post-receive hook that writes the worker trigger file (A3) |
| `health.sh` | one-screen health (Forgejo, user, exec, crumb, home layout, caches, worktrees, scopes, disk, GPU lock, quiet flag) |
| `gc.sh` | worktree age + target-dir quota cleanup, dry run by default |
| `tests/run.sh` | proofs of the contract on this host (33 checks with two wt.sh worktrees present) |
| `pipeline.sh` | publish verified bundles to aien-receipts, merge `ready-to-merge` PRs that merge cleanly (no Spark gate), tag full-scope Spark PASS commits on main as `alpha/YYYYMMDD.N` |
| `aien-alpha` | source of `~/.local/bin/aien-alpha`: cut an Alpha (full-scope Spark judge on a main commit, publish, tag); `aien-alpha list` |
| `tests/pipeline_test.sh` | offline proofs of `pipeline.sh` with fake `gh`, fake `aien-test` and throwaway git repositories (75 checks) |
| `rebase-train.sh` | moves `ready-to-merge` PRs that conflict only in `.crumb` files (or are behind) onto main, recompiles crumbs with the pinned tool, force-with-lease pushes; labels `needs-rebase` and comments once otherwise |
| `cron-tick.sh` | source of the cron entry (`*/5`): crumb drift guard, keeps the trigger-only worker alive, runs `rebase-train.sh`, then `pipeline.sh round` |
| `desk-loop.sh` | the same train + round in a 90 s loop while `$STATE/KEEP-RUNNING` exists |
| `aien-local-ci` | source of the operator front door `~/.local/bin/aien-local-ci` (status, start/stop named worker lanes, trigger, verify, once, plan, health, log) |
| `tests/cli_args_test.sh` | offline launcher argument and lane parsing controls with stubbed Git, worker dispatch, process discovery and signals; run with `bash tools/aien-local-ci/tests/cli_args_test.sh` |
| `crumb-shim.sh` | the operator's `~/.local/bin/crumb`: runs the crumb tool the repository's own CRUMB-VERIFY gate uses (crumb.lock pin, own tree for aien-architecture, else the sandbox tool), so author, readiness check and gate never disagree |
| `crumb-doctor` | drift check, cron (via cron-tick.sh): gate, readiness and hosted crumb tools agree on every main and tool-changing PR; worker sibling maps match origin/main; `--heal` keeps this clone's work tree on origin/main |
| `tests/crumb_shim_test.sh`, `tests/crumb_doctor_test.sh` | offline proofs of both on throwaway fixtures (17 and 7 checks) |
| `tests/crumb_real_test.sh` | Spark-only red/green on real trees and on the real failing heads of 2026-10-16..18 (19 checks) |

## C8 exec contract

```
sudo -n /usr/local/sbin/aien-run-exec --cwd DIR --timeout SECS --profile host|qemu|gb10 [--env K=V]... -- PROGRAM [ARGS...]
```

Chain (all on the privileged side, then drop): `systemd-run --scope` under `aien-run.slice` with `MemoryMax`,
`MemorySwapMax=0`, `TasksMax`, `CPUQuota`, `DevicePolicy` -> `unshare -m [-n -i]` -> mount setup (below) ->
`prlimit --core=0 --nofile=65536` -> `timeout --signal=TERM --kill-after=10 SECS` (host|qemu) -> `setpriv --reuid
aien-run --regid aien-run --init-groups --reset-env --no-new-privs` -> `env -i -C DIR PATH HOME CARGO_HOME RUSTUP_HOME
[K=V...] PROGRAM`. After PROGRAM exits the scope is killed and stopped, so a crashed check leaves no strays.
Unprivileged user namespaces are refused on this host, which is why the namespaces are created as root before
dropping privileges.

Exit codes: PROGRAM's own code passes through; 124 = timeout (host|qemu); 137 = killed (OOM or SIGKILL);
143 = the wrapper itself got TERM/INT/HUP (the scope is killed); 2 = refused by the wrapper before anything ran.

Mount setup inside the private mount namespace (every profile): the whole sandbox home `/var/lib/aien-run` is
read-only, except the worktree that contains `--cwd` and the target dir named by `CARGO_TARGET_DIR`, which stay
writable. Private empty tmpfs on `/var/tmp`, `/var/crash`, `/var/lib/BrlAPI` (the world-writable places on the root
disk). host|qemu also get private `/tmp`, `/dev/shm`, `/run/lock`, `/run/screen`, a fresh IPC namespace and a fresh
`/dev/mqueue`.

| profile | memory | tasks | cpu | network | IPC, /tmp, /dev/shm | devices | timeout |
|---|---|---|---|---|---|---|---|
| host | 48G | 4096 | 1600% (16 cores) | none (fresh netns, only its own `lo`, up) | private | closed (null, zero, random, tty, ptmx) | yes |
| qemu | 48G | 4096 | 1600% | none (as host) | private | closed + `/dev/kvm` (aien-run is in group kvm) | yes |
| gb10 | 96G | 4096 | 1800% | host network | shared with the host | all (`/dev/nvidia*` are 0666; `/dev/dri/*` via groups video, render) | NO kill |

Argument rules (exit 2 when broken):
* `--cwd` absolute, existing, canonical (no symlink, no `..`, no trailing slash). Inside `/var/lib/aien-run` only the
  home itself or `wt/<repo>/<40-hex sha>[/...]`.
* `--env K=V`: plain names; `PATH`, `HOME`, `CARGO_HOME`, `RUSTUP_HOME`, `LD_*`, `BASH_ENV`, `ENV` refused; each name at
  most once. `CARGO_TARGET_DIR` must be exactly `/var/lib/aien-run/target/<repo>/pr/<sha40>` or `.../main/<sha40>` (trust class, A2),
  the worktree's own repo and sha when `--cwd` is in one; it is created on demand (parents root-owned, head dir aien-run).
* `PROGRAM` may not contain `=` or start with `-` (otherwise `env` would read it as one more variable assignment).
* The wrapper refuses to run if the home, `cargo`, `cargo/bin`, `rustup`, `target`, `/usr/local/bin` or `/usr/bin`
  is not root-owned or is group/other-writable.

Optional flags `--memory-max`, `--tasks-max`, `--cpu-quota` override the profile defaults (tests use `--memory-max 64M`).
Sandbox PATH is `/var/lib/aien-run/cargo/bin:/usr/local/bin:/usr/bin:/bin` (all root-owned). A bare program name
resolves only through it; a file of the same name inside the worktree is never used.

## Trusted crumb on the sandbox PATH

`/usr/local/bin/crumb` (root 0755) is built by `install.sh` from aien-architecture commit
`94fffbcf88492c5a1c54211b5126c8a6c370a98d` (`tools/crumb/crumb.c`, git blob `841f5317...`, checked before building;
source read from the git object store of `AIEN_ARCH_CLONE`, default `~/workspace/aien-architecture`) with
`cc -O2 -Wall -std=c11`. With gcc 13.3.0 the binary's sha256 is `22015e6b156e038e...`, identical to the operator's
`~/.local/bin/crumb` on 2026-10-11. To move the pin, change `CRUMB_COMMIT` and `CRUMB_BLOB` in `install.sh` in a PR.

The worker's readiness check does not trust any PATH `crumb`: it builds the tool each repository names in `crumb.lock`
(aien-architecture: the tool at the commit under test) and caches it under `<state>/crumb-tool/<pin>/` (sc#602,
details in `tools/aien-test/docs/LOCAL_VERIFICATION.md`).

## Operator crumb (shim) and crumb-doctor (2026-10-19)

Four places used to pick a crumb build four different ways: each repo's CRUMB-VERIFY gate (crumb.lock pin, own
tree, or this sandbox tool), the worker's readiness check (`aien-test verify` runs the bare name `crumb`, so
`~/.local/bin/crumb`), the required hosted "AIEN Verification" (its workflow pin), and whoever ran `crumb compile`.
`~/.local/bin/crumb` was a fixed 94fffbcf build. Effects seen 2026-10-16..18: every PASS head in omega, aienos and
physics (pinned d0f0b95d) published as `BLOCKED PASS` / error; every RFC-0003 Am1 migration head read "N of N crumbs
stale"; authors compiled sc with d0f0b95d and omega with 94fffbcf and the gates called the result stale (12 of 34
STALE gate failures). Separately, the poll workers read `siblings` from this clone's WORK TREE, which nothing had
moved since 2026-10-11, so synapse ran CRUMB-VERIFY without `../aien-architecture` (30 failed judgements, exit 3).

- Install the shim: `install -m 0755 tools/aien-local-ci/crumb-shim.sh ~/.local/bin/crumb` (keep the old binary as
  `crumb.bak-<ts>-<sha8>`). It never falls back silently: a malformed or unbuildable pin exits 97.
- Install the doctor: `install -m 0755 tools/aien-local-ci/crumb-doctor ~/.local/bin/crumb-doctor`. cron-tick.sh
  runs `crumb-doctor --heal-only` every tick and `crumb-doctor --heal --alert` at most every 15 minutes in the
  background; failures become a `CRUMB-DRIFT` line in the aien-watch ALERT (shown by the gh guard and the chat hook).
- Before merging any PR that moves a crumb pin (crumb.lock, the hosted workflow pin, tools/crumb, install.sh), run
  `crumb-doctor --repos <repo>`: every consumer must give the same verdict on the PR head.


## Egress decision

No nftables rule for uid aien-run. host and qemu checks run in a fresh network namespace with no interface but its own loopback, which is
stronger than a packet filter (no DNS). The namespace's own loopback is brought up (SANDBOX lane, 2026-10-11:
aien-protocols tests start an HTTP server on 127.0.0.1 and failed with "Network is unreachable" while `lo` was down);
it reaches only processes inside the same run, never the host's 127.0.0.1 services (tests/run.sh 4b: Forgejo on
127.0.0.1:3443 is "refused" from inside). gb10 is the only profile with network: it keeps the host
network per the contract; a later nft rule for uid aien-run would be the way to close that and is deliberately not
added by this lane.

## Layout under /var/lib/aien-run

| path | owner / mode | purpose |
|---|---|---|
| `/var/lib/aien-run` | root:aien-run 0750, ACL `u:drakestapleton:r-x` | HOME of aien-run; no skeleton dotfiles |
| `cargo/` | root 0755 | CARGO_HOME (cargo works with it read-only: `--locked --offline` builds need no writes there) |
| `cargo/registry`, `cargo/git` | root, read-only | seeded once from `~/.cargo` (1.9 GB + 1.3 GB; the git cache is needed because `Cargo.lock` has 3 git sources) |
| `registry-refresh.sh <Cargo.lock>` | operator, sudo | adds to `cargo/registry` the crates.io packages a trusted lockfile names that the judge lacks: copied from `~/.cargo`, sha256 checked against the lockfile, unpacked into `registry/src` with cargo's `.cargo-ok` marker (the read-only judge cargo cannot unpack a `.crate` itself), index-cache entry copied, one journal line; a `.crate` the judge has without its source is unpacked after the same check; never deletes or replaces a crate or source directory (#622). `--dry-run` only lists |
| `cargo/bin` | root | the operator's `rustup` binary + proxy symlinks; no second rustup install |
| `rustup/` | root, read-only | `settings.toml` (default 1.98.1) + `toolchains/1.98.1-aarch64-unknown-linux-gnu` copied (631 MB), plus `toolchains/nightly-aarch64-unknown-linux-gnu` with miri for SC-MIRI |
| `.cache/miri` | root, read-only | miri sysroot built once by `install.sh` against the copied nightly (110 MB); `cargo +nightly miri setup` in the sandbox finds it and writes nothing (#600) |
| `.gitconfig` | root 0644 | `safe.directory = *` (git 2.43 has no path globs; aien-run reaches nothing else) |
| `/tmp/aien-gb10.lock` | root 0664, ACL `u:drakestapleton:rw-`, `u:aien-run:rw-` | the one GB10 chip lock for operator runs and sandboxed gb10 gates; root ownership gets past `fs.protected_regular=2` in sticky `/tmp`; `/etc/tmpfiles.d/aien-gb10.conf` recreates it at boot (#627); a gb10 gate waits for it in the kernel queue up to 30 min by default (`aien-test --gpu-wait`, between the live server's requests), then is NOT_RUN `gpu_lock`; the wait is in each receipt (`volatile.gpu_wait`) |
| `mirror/<repo>.git` | drakestapleton 0755 | bare mirror of the trusted clone, refreshed by `wt.sh sync`; unwritable by aien-run |
| `wt/<repo>/<sha>` | root dir drakestapleton:aien-run 1775, `.git` root 0644, files aien-run | checkouts; aien-run can add files, cannot replace `.git` or write the mirror |
| `target/`, `target/<repo>/` | root 0755 | parents cannot be swapped by aien-run |
| `target/<repo>/{pr,main}/<sha40>` | aien-run | one per head inside its trust class (A2); a run can write only its own; `wt.sh remove` deletes it with the worktree (a shared dir let cargo link one head's build into another, sc #605) |

## Trigger file contract (A3)

`~/.local/state/aien-local-ci/trigger/<repo>.<sha>` with one line `<ref> <old> <new> <unix-time>`, written by the
Forgejo post-receive hook (`forgejo-webhook.sh install <repo>`), consumed and deleted by `aien-test worker`.
forgejo-runner stays off (its user unit is disabled); Forgejo is git hosting + PR UI + status display only.

## Review (INFRA-2, 2026-10-11)

Fixed in this branch (each has a test in `tests/run.sh`):
1. The sandbox could write its own home and CARGO_HOME (both were owned by aien-run): a check could plant
   `/var/lib/aien-run/.cargo/config.toml` or `cargo/config.toml` (for example a `build.rustc-wrapper`), or rename the
   root-owned `cargo/registry` or `cargo/bin` aside and put poisoned crates or a fake `cargo` in their place. Every later
   run, including a `main` qualification, would have used them. Now the home, `cargo/` and `target/` are root-owned and
   the whole home is read-only inside every run.
2. Cross trust class writes: a PR run could write `target/<repo>/main` and every other worktree (all owned by
   aien-run). Now only the run's own worktree and the one target dir it names are writable.
3. `/tmp`, `/var/tmp`, `/var/crash` were shared with the operator: a check could read the operator's temporary files
   and leave files behind for later runs or for root crash tooling. Now private (see the profile table for gb10).
4. SysV shared memory, semaphores and POSIX queues were shared with the host; host|qemu now get a fresh IPC namespace.
5. `PROGRAM` of the form `NAME=VALUE` was read by `env` as one more variable, which bypassed the `--env` refusals
   (for example `LD_PRELOAD`). Now refused.
6. `CARGO_TARGET_DIR` and `--cwd` were not checked: a typo or a symlink planted by an earlier run inside a worktree
   could point a run at another worktree or trust class. Now canonical paths only, target dir only `pr|main`.
7. `install.sh` staged the sudoers line in a predictable `/tmp/aien-sudoers.<pid>` before installing it as root.
   (`fs.protected_regular=2` on this host already blocked the race; now a private `mktemp -d` directory.)
8. The wrapper ran with the caller's PATH via `/usr/bin/env bash`; it now uses `/bin/bash` and pins its own PATH.
9. Skeleton dotfiles (`.bashrc`, `.profile`, `.config/autostart`, `Desktop`) in the home were aien-run-writable;
   removed.
10. The qemu profile allowed `/dev/kvm` in the cgroup policy, but aien-run was not in group `kvm` (`/dev/kvm` is
    root:kvm 0660), so qemu checks could not use KVM. install.sh now adds the group; host still cannot open it.

Checked and found sound (no change):
* aien-run cannot read `~/.aien/verify.key`, `~/.config/gh/hosts.yml`, `~/.ssh/id_ed25519` or list `~/workspace`
  (0600 files, 0700 workspace; tests 10). The trusted clone lives under `~/workspace`; the sandbox only sees the mirror.
* sudoers: one exact command path, no wildcards; arguments are free by design and validated by the script; sudo's
  `env_reset` plus `setpriv --reset-env` plus `env -i` clear the caller's environment (test 2).
* `setpriv --no-new-privs`: setuid programs (sudo, crontab, at) gain nothing inside a run; the scope kills every
  child at the end, so nothing outlives a run.
* Repo-controlled files (`.cargo/config.toml`, `build.rs`, `rust-toolchain.toml`, test code) run as aien-run only and
  cannot change the exec contract, which reads nothing from the tree. A toolchain pin other than 1.98.1 fails offline.
* The environment whitelist is minimal: PATH, HOME, CARGO_HOME, RUSTUP_HOME (cargo and the rustup proxy need the last
  two to find the sandbox caches) plus explicit `--env` pairs.
* gb10 host network is the only network exception (`unshare -n` for host and qemu only).
* Operator git refuses to run inside a sandbox worktree ("dubious ownership": the `.git` file is root-owned and the
  files are aien-run-owned), so a `.git` directory planted by a check cannot run hooks or config as the operator.

Accepted risks (recorded, not changed by this lane):
* gb10 keeps host network, host `/tmp`, `/dev/shm` and IPC: a chip gate can reach local services on 127.0.0.1 (Forgejo
  3443/2222, Cortex 18080 need credentials it cannot read) and the internet, and can leave files in `/tmp`.
  Closing it needs either a nft rule for uid aien-run or a netns for gb10 after the chip gates are proven inside one.
* If the caller kills the wrapper during a gb10 run, the trap kills the chip run too (no strays). The engine must not
  signal a gb10 run; it only marks its advisory timeout in the receipt.
* The operator account has `(ALL) NOPASSWD: ALL` in sudoers, so the narrow `aien-local-ci` rule is documentation of
  intent, not a limit, until that broad rule is removed (operator decision, outside this lane).
* The operator's home is 0755 (ACL user atlas `--x`); its secrets are 0600 and verified unreadable.
* Same-SHA reuse: a PR run and a `main` qualification of the same commit share one worktree path. The engine should
  remove and re-add the worktree (fresh checkout) before a `main` qualification.
* `git` write operations as aien-run inside a worktree fail (the index lives in the read-only mirror). Gates must not
  need them.
* A run's HOME is read-only: a gate that writes to `$HOME` fails; write to `/tmp` (private for host|qemu) instead.
* Worktree root dirs inherit setgid from the mirror layout (`drwxrwsr-t`); harmless.

## Stages: merge, then Alpha (ADR 0033 amended 2026-10-24)

Drake, 2026-10-24: "Lift the requirement to test on the Spark in order to be merged ... There should instead be tags
that clearly label when a certain build gets to the Alpha stage. That is the stage that must be tested on the Spark.
We must not limit our development process by these tests."

| stage | what it takes | who does it |
|---|---|---|
| merge to `main` | label `ready-to-merge`, not a draft, merges without conflicts | `pipeline.sh merge` (cron) |
| Alpha (`alpha/YYYYMMDD.N` tag on a main commit) | full-scope Spark qualification: every gate of the repository, GPU gates included, signed bundle PASS, `verify-bundle --require-full` exit 0 | `aien-alpha REPO` runs the judge; `pipeline.sh alpha` tags |

No pull request is judged on the Spark, no status check is required on GitHub (the `main-protect` rulesets keep only
no-deletion, no-force-push and PR-only), and nothing ever moves `main` except a PR merge. The hosted `AIEN` workflow
is advisory on PRs (crumb verify, portable smoke set) and verifies every pushed `alpha/*` tag against its published
bundle.

```
aien-alpha aien-sovereign-core            # qualify origin/main's head; exit 0 only once it carries an alpha tag
aien-alpha omega <sha-on-main>            # any commit already on main
aien-alpha list                           # every repository's alpha tags, newest first
```

## Publish, merge, alpha (pipeline.sh)

```
tools/aien-local-ci/pipeline.sh publish|merge|alpha|round [--repos a,b]     # round = publish, merge, alpha
bash tools/aien-local-ci/tests/pipeline_test.sh                           # offline proofs, a few seconds
```

It never reads `~/.aien/verify.key` and never signs: it only moves bundles the worker already signed, merges, and
tags. One instance at a time (`flock` on `$STATE/pipeline.lock`; a second one exits 0 at once, or waits up to
`AIEN_LOCK_WAIT` seconds when set, as `aien-alpha` does). Every action and refusal is one JSON line in
`$STATE/pipeline.jsonl`. Exit 0 when the round ran (refusals included), 2 on a usage or configuration error.
Overrides: `AIEN_STATE`, `AIEN_CLONES`, `AIEN_RECEIPTS`, `AIEN_RECEIPTS_BRANCH`, `AIEN_RECEIPTS_URL`, `AIEN_TEST_BIN`,
`AIEN_GH_OWNER` (default `aien-dev`), `AIEN_REPOS`, `AIEN_TAGGER_NAME`, `AIEN_TAGGER_EMAIL`, `GH`, `GIT`. GitHub calls
use the canonical slug (`gh api repos/aien-dev/<repo> --jq .full_name`; e.g. `aien-dev/aegis-runtime-mv`), cached in
`$STATE/slugs/<repo>` for 24 h (`AIEN_SLUG_TTL`); a failed refresh falls back to the older cached name. Each lookup
journals `action=slug` with `result=cached`, `refreshed` or `stale-cache`.

* **publish**: for each `$STATE/bundles/<repo>/<sha>/bundle.json` of a repository whose trusted clone has a key in
  `origin/main:.aien/verify.pub` (several keys allowed, one per line, `#` comments, same rule as aien-test): skip when
  `plan.blocked` is not null; run `verify-bundle` with `--repo github.com/aien-dev/<repo> --commit <sha>` taken from
  the path, once per key; publish when any key gives 0 PASS or 1 FAIL, never 2 NOT_RUN or 3 REJECTED. Publishing
  copies the bytes to `<receipts>/<repo>/<sha>/bundle.json` when absent or when its sha256 differs (replacement by
  digest), one commit per round listing every sha, then `pull --rebase` and push, 3 attempts, never forced. A failed
  push drops only that round's own commit; the next round recomputes it.
* **merge**: open PRs labeled exactly `ready-to-merge`, not draft, head unchanged since listing, GitHub mergeable
  state `clean`, `unstable` (some non-required check failed), `behind` or `has_hooks`. `dirty` (conflicts) and
  `blocked` are refused, `unknown` waits for a later round. Then `gh pr merge N -R <slug> --merge
  --match-head-commit <sha>`. Never `--admin`. No bundle, no status check, no up-to-date rule, no limit per round.
* **alpha**: per repository, the bundles in receipts `origin/main` whose signed plan says `"scope": "full"` (one
  `git grep`), restricted to commits on the trusted clone's `origin/main`; the newest by committer time that
  `verify-bundle --require-full --require <origin/main:.aien/always>` passes (exit 0) is the candidate. It is tagged
  unless it already has an `alpha/*` tag or the newest existing `alpha/*` tag already contains it. The tag is
  annotated, named `alpha/<UTC date>.<n>` (n from 1 per day), and its message names the commit, the bundle sha256
  and the receipt URL; it is pushed with git. A failed push deletes the local tag. Never moves a branch.

Merge pause: while `$STATE/MERGE-PAUSE` exists (Drake, 2026-10-18: "stop all agents from trying to merge"), the
merge step journals `result=paused` and merges nothing; publish and alpha keep running. `rebase-train.sh` keeps
rebasing (that is not merging) and never adds `ready-to-merge` or enables auto-merge, paused or not. No script creates
or deletes the file: only Drake lifts the pause.

## Rebase train (rebase-train.sh, before every round)

```
tools/aien-local-ci/rebase-train.sh [--dry-run] [--repos a,b] [--pr N]   # --dry-run: local work only, prints the plan
```

Why: every merge to main rewrites the root `.crumb` (generated by the pinned crumb compiler), so every other open PR
turned CONFLICTING and `pipeline.sh merge` refused it as `dirty` forever (2026-10-18: 11 of 12 `ready-to-merge` PRs in
sovereign-core; the four probed conflicted in the root `.crumb` only). The train does what a person would do.

For each repo (same list as `pipeline.sh`), each open PR labeled exactly `ready-to-merge`, not draft, head branch in
the same repository (no forks), that GitHub reports `CONFLICTING` or `BEHIND`:

1. fetch `pull/N/head` into `refs/aien-train/N` of the trusted clone, scratch worktree under `$STATE/.train.*`
   (hooks off: `core.hooksPath=/dev/null`); both removed on exit, also on error;
2. skip (journal only) when the head commit's committer time is under 10 minutes old (`AIEN_TRAIN_MIN_AGE`, seconds;
   a proxy for "pushed recently": GitHub's push time costs an extra API call), when the PR touches a trust-surface
   file (`.aien/verify.pub`, `.github/workflows/*`; never rebased by the train), or when nothing but `.crumb` files
   would be left after the rebase (the changes are already in main);
3. `git rebase origin/<base>`; conflicts in `.crumb` files (root or nested) take the base's copy; any other
   conflicting file aborts the rebase;
4. `crumb compile .` with the tool built from aien-architecture at the base branch's pin (`crumb.lock`, else the
   `aien-architecture @ <sha>` pin in `.github/workflows/aien-verification.yml`; `make -C tools/crumb crumb` as the
   workflow does, cached in `$STATE/crumb-tools/<pin>/crumb`); a change is committed as
   `crumb: recompile after rebase onto main`; then `crumb verify .` must pass;
5. `git push --force-with-lease=refs/heads/<branch>:<old head>` to the PR branch; a `needs-rebase` label is removed.

A conflict outside `.crumb`, or a recompile that changes anything but `.crumb` files, adds the label `needs-rebase`
(created once per repo, recorded in `$STATE/train-labels`) and posts one comment that starts with
`> *This was generated by AI during triage.*` and lists the files. One comment per head sha: recorded in
`$STATE/train-commented` and marked `<!-- aien-rebase-train head=<sha> -->` in the comment, checked with one
`gh pr view` before posting. Every decision is one JSON line in `$STATE/train.jsonl` (`utc, repo, pr, head, action,
reason`); `--dry-run` writes none, pushes, labels and comments nothing. One instance at a time (`flock` on
`$STATE/train.lock`). API budget per run: one `gh pr list` per repo (GraphQL; the label is filtered locally because
`gh pr list --label` returned nothing on 2026-10-18), and only for a blocked PR one slug lookup (shared disk cache,
see pipeline.sh), one label call, one
`gh pr view` and one comment. Pushes go through git. Exit 0 when the run finished (skips included), 2 on a usage or
configuration error. Overrides: `AIEN_STATE`, `AIEN_CLONES`, `AIEN_ARCH_CLONE`, `AIEN_GH_OWNER`, `AIEN_REPOS`,
`AIEN_TRAIN_MIN_AGE`, `GH`, `GIT`.

GitHub rate-limit guard: every `gh` call of `pipeline.sh` and `rebase-train.sh` goes through `ghc`. When the shared
guard `~/.local/bin/gh` refuses with exit 98 (allowance reserved or rate-limit marker set), the run makes no further
GitHub call and every journal line after it carries `deferred: "gh guard exit 98: try next round"` (in train.jsonl
the same words close the reason). Never a retry loop, never a FAIL: the next tick tries again. Neither script uses
`gh run watch` (the guard's exit 97). Pushes use git, whose credential helper is `/usr/bin/gh auth git-credential`.

## Cron tick (cron-tick.sh)

Installed as `~/.local/lib/aien-local-ci/cron-tick.sh` with one crontab line (no systemd):

```
*/5 * * * * $HOME/.local/lib/aien-local-ci/cron-tick.sh >> $HOME/.local/state/aien-local-ci/cron-tick.log 2>&1
```

Each tick: the crumb drift guard, `aien-local-ci start` (a trigger-only worker: no `--github`, so no pull request is
polled, judged or given a status; a no-op while it runs), `rebase-train.sh`, `pipeline.sh round`. With
`$STATE/KEEP-RUNNING` present, `desk-loop.sh` runs the train and the round every 90 s instead.
`touch ~/.local/state/aien-local-ci/PAUSE` stops the tick.

The Spark judges two kinds of work only: Alpha candidates (`aien-alpha`, foreground `worker --once --full
--operator`) and trigger files written by hand (`aien-local-ci trigger REPO SHA`). The per-PR worker lanes of
2026-10-18 (`--github`, lanes main/small/big/core) are retired with the merge gate they fed.

## Operate

```
tools/aien-local-ci/install.sh               # idempotent; AIEN_ARCH_CLONE=<aien-architecture clone> for crumb
tools/aien-local-ci/tests/run.sh             # all proofs, about 10 s
tools/aien-local-ci/wt.sh sync aien-sovereign-core ~/workspace/aien-sovereign-core
tools/aien-local-ci/wt.sh add aien-sovereign-core <sha>
tools/aien-local-ci/health.sh
tools/aien-local-ci/gc.sh [--apply] [--days 3] [--target-quota-gb 120]
```

Measured 2026-10-11 (sovereign-core 5ef7720, 20 cores, caches seeded): cold `cargo build --workspace --all-targets
--locked --offline` as aien-run 65.0 s (target dir 27 GB); warm rebuild after touching one crate 2.2 s. After the
read-only home change: full workspace build of worktree 62542eb into the existing `pr` target 49.9 s (new worktree
path, so workspace crates rebuilt; dependencies reused), exit 0.
