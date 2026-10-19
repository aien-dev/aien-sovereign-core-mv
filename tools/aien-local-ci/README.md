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
| `pipeline.sh` | ADR 0033 phase 3: publish verified bundles to aien-receipts, rerun the GitHub thin gate, merge `ready-to-merge` PRs behind guards |
| `tests/pipeline_test.sh` | offline proofs of `pipeline.sh` with fake `gh`, fake `aien-test` and throwaway git repositories (56 checks) |
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
| `cargo/bin` | root | the operator's `rustup` binary + proxy symlinks; no second rustup install |
| `rustup/` | root, read-only | `settings.toml` (default 1.98.1) + `toolchains/1.98.1-aarch64-unknown-linux-gnu` copied (631 MB) |
| `.gitconfig` | root 0644 | `safe.directory = *` (git 2.43 has no path globs; aien-run reaches nothing else) |
| `/tmp/aien-gb10.lock` | root 0664, ACL `u:drakestapleton:rw-`, `u:aien-run:rw-` | the one GB10 chip lock for operator runs and sandboxed gb10 gates; root ownership gets past `fs.protected_regular=2` in sticky `/tmp`; `/etc/tmpfiles.d/aien-gb10.conf` recreates it at boot (#627) |
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

## Publish, rerun, merge (pipeline.sh, ADR 0033 phase 3)

```
tools/aien-local-ci/pipeline.sh publish|rerun|merge|round [--repos a,b]     # round = publish, rerun, merge
bash tools/aien-local-ci/tests/pipeline_test.sh                           # offline proofs, a few seconds
```

It never reads `~/.aien/verify.key` and never signs: it only moves bundles the worker already signed and asks
GitHub to act. One instance at a time (`flock` on `$STATE/pipeline.lock`; a second one exits 0 at once). Every action
and refusal is one JSON line in `$STATE/pipeline.jsonl`. Exit 0 when the round ran (refusals included), 2 on a usage
or configuration error. Overrides: `AIEN_STATE`, `AIEN_CLONES`, `AIEN_RECEIPTS`, `AIEN_RECEIPTS_BRANCH`,
`AIEN_TEST_BIN`, `AIEN_GH_OWNER` (default `aien-dev`), `AIEN_REPOS`, `GH`, `GIT`. GitHub calls use the canonical slug
(`gh api repos/aien-dev/<repo> --jq .full_name`, so `aienos` becomes `aien-spark/aienos`).

* **publish**: for each `$STATE/bundles/<repo>/<sha>/bundle.json` of a repository whose trusted clone has a key in
  `origin/main:.aien/verify.pub` (several keys allowed, one per line, `#` comments, same rule as aien-test): skip when
  `plan.blocked` is not null; run `verify-bundle` with `--repo github.com/aien-dev/<repo> --commit <sha>` taken from
  the path, once per key; publish when any key gives 0 PASS or 1 FAIL, never 2 NOT_RUN or 3 REJECTED. Publishing
  copies the bytes to `<receipts>/<repo>/<sha>/bundle.json` when absent or when its sha256 differs (replacement by
  digest), one commit per round listing every sha, then `pull --rebase` and push, 3 attempts, never forced. A failed
  push drops only that round's own commit (`reset --hard origin/main` in the pipeline's receipts clone, which must be
  clean and on its branch before the round starts, otherwise the round refuses); the next round recomputes it.
* **rerun**: for each open PR whose head has a published bundle, once per (repo, sha, digest) recorded in
  `$STATE/pipeline-reruns`: the latest `AIEN Verification` run for that head is rerun when it is completed and its
  latest attempt started (`run_started_at`, else `created_at`) before the publication (push time in
  `$STATE/pipeline-published`, else the receipts commit time). A queued or running run waits for a later round.
* **merge**: only open PRs labeled exactly `ready-to-merge`, not draft, mergeable state not `dirty`, head unchanged
  since listing. All guards required: the local bundle for the head exists; its sha256 equals the published copy at
  receipts `origin/main`; `verify-bundle` exits 0 with a key from `origin/<base>:.aien/verify.pub` and `--require`
  from `origin/<base>:.aien/always` (as the thin gate does); the base branch has required status checks and every one
  is present on the head with only `success` (check runs and commit statuses); no `AIEN Verification` run for the head
  is queued or in progress. Then `gh pr merge N -R <slug> --merge --match-head-commit <sha>`. Never `--admin`. A base
  branch without required status checks (or whose protection cannot be read) is refused, not merged.

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
