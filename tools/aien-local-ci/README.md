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
| `cron-tick.sh` | the cron entry point (every 2 min): worker keepalive + one `pipeline.sh round`, thinned out while quiet (below); installed by `install.sh` to `~/.local/lib/aien-local-ci/` |
| `tests/cron_tick_test.sh` | offline proofs of `cron-tick.sh` (PAUSE, QUIET-SINCE, WAKE, stale heartbeat, garbage markers; 17 checks) |
| `prime` | warm-standby prime map: which cargo builds of main seed `target/<repo>/pr` |
| `cold-warm-proof.sh` | periodic proof that a warm pr target gives the same verdicts and receipt cores as a cold one |

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
  most once. `CARGO_TARGET_DIR` must be exactly `/var/lib/aien-run/target/<repo>/pr` or `.../main` (trust class, A2);
  it is created on demand (parent root-owned, class dir aien-run).
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
| `mirror/<repo>.git` | drakestapleton 0755 | bare mirror of the trusted clone, refreshed by `wt.sh sync`; unwritable by aien-run |
| `wt/<repo>/<sha>` | root dir drakestapleton:aien-run 1775, `.git` root 0644, files aien-run | checkouts; aien-run can add files, cannot replace `.git` or write the mirror |
| `target/`, `target/<repo>/` | root 0755 | parents cannot be swapped by aien-run |
| `target/<repo>/{pr,main}` | aien-run | per trust class (A2); a run can write only the one it names; `main` is wiped after FAIL/timeout by the engine |

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

## Power saving and warm standby (2026-10-16)

Markers, all in `~/.local/state/aien-local-ci/` (the merge train and anything else may read them):

| marker | written by | meaning |
|---|---|---|
| `QUIET-SINCE` | the worker, only with `--github` | one line `<epoch-secs> <utc>`: the time of the last activity the worker saw. Present = nothing moved since then. Removed the moment anything moves. Absent, empty or not a number = **not quiet** (readers fail open to full cadence). |
| `WAKE` | anyone (`touch`) | "full cadence now". Never removed by a reader: each reader compares its mtime with the last one it acted on (the worker in memory, `cron-tick.sh` against `cron-last-round`). |
| `PAUSE` | the operator | unchanged: `cron-tick.sh` exits at once (no keepalive, no round). PAUSE present -> absent wakes both the worker and the next tick. |
| `cron-last-round` | `cron-tick.sh` | mtime = start of its last `pipeline.sh round`. |
| `warm/<repo>.json` | the worker | `{"class":"pr","complete":true,"primed_from_main":<sha>,"utc":..}`: `target/<repo>/pr` was primed from that main. Removed before priming starts and by `--cold`, `gc.sh` and the proof; written only when every prime line exited 0. |

Activity (worker): a trigger file, a GitHub poll that differs from the previous one for a repository (new head,
`ready-to-merge` added or removed, main moved), a head run that had not already failed to run in this loop, a wake.
Priming is never activity.

Cheap check vs expensive work. The cheap check never backs off. Every `--poll` (60 s, as installed) the worker makes
ONE GraphQL call for all enrolled repositories (`gh api graphql`: per repository the main head and up to 50 open pull
request heads with their labels). A repository the answer does not cover (no main line, or a second page of pull
requests or labels: `hasNextPage`) is polled the old way (two REST calls), and a failed probe polls every repository the old way; a partial answer never reads as "no pull
requests". Measured on 2026-10-16 for the 11 enrolled repositories (`worker --probe`, 5 runs): about 1.0 s wall and
0.07 CPU-s (user+sys) per probe, against about 13.4 s wall and 1.4 CPU-s for the 22 REST calls it replaces; at 60 s
that is about 0.1 % of one core. So a new head or label is seen within one poll (60 s), quiet or not, and its gates
start in that round. Only the expensive work backs off: retrying a head that already failed to run (a fetch and a
sibling sync) waits the backed-off interval (60 s, doubling to `--backoff-cap` 900 s once nothing moved for
`--quiet-after` 600 s), and `cron-tick.sh` thins pipeline rounds while QUIET-SINCE is old. A label seen while quiet
removes QUIET-SINCE at once, so the next cron tick (at most 2 min) runs the merge round. The worker sleeps in 5 s
slices that only stat WAKE, PAUSE and the trigger dir and rewrite the heartbeat (no network, no process), so WAKE,
PAUSE removal or a trigger file ends the sleep within 5 s.

`cron-tick.sh`: the keepalive runs every tick. The round is skipped only while QUIET-SINCE is at least 600 s old,
the worker heartbeat is younger than 1200 s (a dead or hung worker never silences the pipeline) and the last round
started less than 600 s ago; so while quiet there is still one round every 10 min. Each round logs its own child
CPU (`round rc=.. child_cpu user=..s sys=..s`).

Fetching: a run fetches its trusted clone only when `git ls-remote origin`, mapped through `remote.origin.fetch`,
shows a ref (or any tag) the clone does not have at the same sha, or the commit is missing; anything unclear
fetches as before. At most once per clone per round. Sibling clones are refreshed the same way once per round, and
`wt.sh sync` runs only when the clone's refs changed since the last sync in that round.

Warm standby: when nothing waits and the quiet flag is clear (`quietlock check` rc 0, else `~/workspace/.spark-quiet`
absent), the worker primes one repository whose `warm/<repo>.json` does not name the main head polled this round:
it checks out that main head (asserted on the base line and equal to the polled head) with its siblings and runs
each `prime` line through the exec helper, profile `host`, `CARGO_TARGET_DIR=/var/lib/aien-run/target/<repo>/pr`.
Main is trusted, so its build may seed the lower class; nothing ever writes the `main` class from priming, and a pr
build never reaches `main`. Every 5 s it checks the trigger dir, WAKE, PAUSE and the quiet flag, and once a minute
GitHub (the same one-call probe) for a head without a bundle that has not already failed to run in this loop (a
head that errored never gets a bundle, so it must not stop priming every round); any of them sends SIGTERM to the
helper (its trap stops the scope). Only a new head, WAKE, PAUSE removal or a trigger file counts as activity, and
then the loop runs the next round at once instead of sleeping; the quiet flag, PAUSE present or the disk bound do
neither. Everything priming runs (cargo, du, the wipe) runs as `nice -n 19 ionice -c 3` inside the sandbox, below
the gates. Never gb10, never QEMU, never a booted VM.

Disk bound. Priming takes the target dir lock, `<state>/locks/target/<sanitised target dir>.lock` (the flock the
runner holds around a cargo gate, sc #584). It retries every 5 s and gives up on any stop reason, and never holds the
GPU or proof locks. Holding it, priming measures `du -sk target/<repo>/{pr,main}` (inside the sandbox, as aien-run)
and the filesystem's free space, then:
* total (main + pr) at most `--target-quota-gb` (120, as gc.sh) and at least `--disk-floor-gb` (50) free: prime;
* over, but main alone fits and emptying pr restores the floor: empty pr (the lowest class, a cache the next run
  rebuilds), then prime;
* otherwise (main alone over the quota, or the floor even with pr empty): do not prime, and leave pr as it is (what
  PR runs put there is gc.sh's to bound). The outcome is `cold` in the journal, that main is not tried again, and no
  bundle claims warm until space is freed.

While priming, free space is checked every 5 s and the repository total once a minute; going over, or a check that
cannot measure, stops the build, empties pr and leaves it cold. After the last line the total is measured again under the same rule. The worker log
carries the disk before and after (`PRIME <repo> disk before: ...`).

Every bundle carries a signed top-level member `target`: `{"class":"main|pr|unsandboxed","warm":bool,
"primed_from_main":sha|null,"primed_utc":..,"cold":bool}`. `warm` = the complete record existed and the pr dir was
not empty when the run started. It means "primed from main <sha>"; the pr dir may also hold earlier PR builds, as it
always has. `bundle::validate` ignores members it does not know and the signature covers every member, so this
needs no schema bump; `target` is the first such member, on purpose.

`worker --once REPO SHA --cold` empties `target/<repo>/pr` from inside the sandbox first (pr heads only; the main
class is never emptied); `worker --once REPO MAIN --prime-only [--cold]` primes by hand. `cold-warm-proof.sh <repo>
<pr-head>` runs warm (empty, prime, run) then cold (empty, run), both `--no-cache` into their own state dirs, and
compares per gate the verdict and the receipt core minus the stdout/stderr digests. It refuses unless PAUSE exists
and the live worker is stopped, because it shares the pr dir. Not built: a crumb tool cache (the gate builds it in
`$TMP` from its pin in about 3 s, inside the repos' `run.sh`) and a boot image per main sha (aienos `ck_gates.sh`
builds its images itself; both need repo-side gate changes).
