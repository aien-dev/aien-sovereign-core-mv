# Headstart M0: stable Rust build baselines

Tracking issue: #532. This directory holds the instructions and the record
schema for the stable-toolchain build baselines that later Headstart work
(#536) compares against. It is measurement only: the runner never edits
`Cargo.toml`, `rust-toolchain.toml`, CI, compiler flags or the caller's
checkout.

| Path | What it is |
| --- | --- |
| `scripts/bench/headstart-aien.sh` | the runner (bash) |
| `scripts/bench/headstart-summarize.sh` | summarizer and Cargo timings extractor (bash + jq) |
| `scripts/bench/headstart-selftest.sh` | self-test on a throwaway fixture workspace, with negative cases |
| `benchmarks/headstart/schema.json` | JSON Schema for `manifest.json` and each `samples.jsonl` line |
| `benchmarks/data/headstart/baseline-<date>-<short>/` | one run: raw samples, manifest, summary, timings extract, log tails |

No Python is used anywhere. Requirements: bash, git, jq, awk, sed, pgrep,
cargo through rustup, and GNU `/usr/bin/time` for max RSS and CPU% (when it
is missing the samples record `time_v.available: false` instead of numbers).

## How a run works

1. `setup`: `git worktree add --detach $HS_WORK/src <commit>` outside the
   caller checkout (refused when `HS_WORK` is inside it), its own
   `CARGO_TARGET_DIR=$HS_WORK/target`, then `cargo fetch --locked` once so the
   cargo registry and git caches are warm. Writes `manifest.json` (commit,
   `rustc -vV`, `cargo -V`, arch, nproc, CPU models, MemTotal/MemAvailable,
   features, profiles, target dir, jobs, build env, date).
2. `timings`: one cold `cargo build --workspace --timings`. The `UNIT_DATA`
   block of `cargo-timing.html` is cut out with sed and validated with jq
   (`timings/units.json`), the observed critical path and the upper-bound
   estimate are computed into `timings/critical-path.json`, the package graph
   from `cargo metadata --filter-platform <host> --locked --offline` goes to
   `timings/graph.json`, the realistic estimate (list-schedule replay) goes
   to `timings/replay.json`, and the report is kept gzipped.
3. `cold`: interleaved rounds. Each round runs, in order: cold check (arm A or
   B), cold build workspace, cold build `spark-harness`, cold test
   `--no-run`, cold check (the other arm), cold build `spark-supervisor`, cold
   build `aien-runtime`. The A/B arm order swaps every round; the two arms of
   the same command are the A/A noise estimate. "Cold" means the target dir
   is deleted before every attempt while the registry stays warm. If a cold
   workspace sample in round 1 exceeds 1200 s the run drops to 3 rounds and
   says so in `manifest.protocol.samples_reduced`.
4. `incr`: prime `cargo check --workspace --all-targets`, three no-op checks,
   then for each series a 10-step body-only edit and a 10-step
   interface-changing edit (each step reverts the previous edit first, so
   every step is a fresh one-edit change). After every series the edit is
   reverted, `git diff --quiet` is verified (abort when not clean) and an
   untimed-for-stats `incr_settle` check brings the target dir back to the
   pristine source. Two series per kind run as body, iface, body, iface.
5. `cleanup`: deletes the target dir, the raw logs and the worktree.

Before every sample the runner first waits while the host's shared quiet
flag is held by another job (`quietlock check`, tool name `HS_QUIETLOCK_BIN`;
disable with `HS_QUIETLOCK=0`). The wait is capped at `HS_QUIET_WAIT_CAP_S`
(default 3600 s) so a stale flag cannot hang the run; on the cap the runner
logs a line and proceeds. Each sample records `quiet_flag_checked`,
`quiet_flag_wait_s` and `quiet_flag_wait_capped`. The manifest's
`protocol.quiet_flag` records `enabled`, `tool`, `available` (the tool was
found), `respected` (enabled and available) and `wait_cap_s`, so a host
without the tool reads `respected: false`. Then the runner waits until the
1-minute load average is below `HS_LOAD_MAX` (cap `HS_WAIT_CAP_S`), records
load before and after, and lists foreign compiler processes at the start,
every `HS_MON_INTERVAL_S` seconds (default 3) while the command runs, and once
more right after it ends (`foreign.at_end`). A foreign process is a `cargo`
process, or a `rustc` / `clippy-driver` / `rustdoc` process with
`--crate-name` (an actual compile), that is not in the runner's process tree.

A sample is `contaminated: true` when `contamination_reasons` is non-empty:
`foreign_at_start`, `foreign_during`, `foreign_at_end`, `load_wait_capped`
(the load never dropped below the limit within the cap) or
`quiet_flag_wait_capped`. Contaminated samples are re-run up to
`HS_MAX_RERUNS` times; earlier attempts stay in `samples.jsonl` with
`final: false`.

Every command's exit status is recorded. A failed build is a failure
(`ok: false`, with the last 80 log lines kept); it is never counted in a
median and never replaced by a smaller workload.

## A/B stages for one toolchain with two arm environments (M5, #536)

The same runner measures one toolchain with a feature off and on. It takes
the toolchain from `HS_CARGO` (replaces the leading `cargo` of every scenario
command) and `HS_RUSTC` (exported as `RUSTC` to every sample), and the arm
environments from `HS_ARM_ENV_OFF` and `HS_ARM_ENV_ON` (space-separated
`KEY=VALUE` words, both required and different, or the stage aborts). Each
arm has its own `CARGO_TARGET_DIR` (`<target>-<arm>`), so arms never share
build state. Every sample records `cargo_bin` and `env_words`; `setup`
records the sha256 of cargo, rustc and the `librustc_driver-*.so` in the
rustc sysroot under `manifest.toolchain.binaries`.

- `ab`: `HS_SAMPLES` rounds of cold check, cold build and cold
  `test --no-run`, OFF and ON alternating (the arm that goes first swaps
  every round), plus an A/A pair: cold check twice with the OFF environment
  (arms A and B, `aa_noise` in the summary). `HS_AB_JOBS="4 8 16"` adds cold
  workspace builds at those `-j` values, `HS_AB_JOBS_SAMPLES` rounds each.
- `abincr`: the `incr` edit series, but both arms check the same source after
  every edit (OFF first on odd steps, ON first on even steps), each on its own
  incremental target dir. A failed check is recorded with its exit status and
  log tail and the series continues; it is never counted as a time.
- `abverify`: per arm, a cold `cargo check --workspace --all-targets
  --message-format=json` and `cargo test -p <HS_VERIFY_TEST_PKGS> --lib
  --bins --tests -- --test-threads=1` (no doctests: those run `rustdoc`,
  and a stage-1 `compiler library` build ships none, so the rustup rustdoc
  would be mixed in), compared in `verify.json`: exit codes, diagnostics
  normalized for event order (each `compiler-message` reduced to level, code,
  message and primary spans, then sorted) and test result lines (timing
  suffix removed). With `HS_VERIFY_ENV_ON` set, one more cold
  `cargo build --workspace` runs with the ON environment plus those words.
  `verdict` is `match` only when everything is equal, both arms exit 0 and
  that extra build exits 0.

With `HS_HOLD=1` every sample also holds the host quiet flag for itself:
after the usual `quietlock check` queue wait, the runner takes it with
`quietlock hold --owner <HS_HOLD_OWNER> --minutes <HS_HOLD_MINUTES> --reason
<HS_HOLD_REASON>`, keeps it through the load wait and the timed command, and
releases it before the next sample, so other jobs (chip gates, a paused
tokenizer) get the flag between samples. A refused hold (exit 75: another
job took the flag first) queues again within `HS_QUIET_WAIT_CAP_S`; a hold
that is not taken marks the sample contaminated (`quiet_hold_not_taken`).
Each sample records `quiet_hold` (requested, held, hold id, seconds to
acquire, refusals) and the manifest lists `protocol.quiet_hold.samples_held`
and `samples_unheld`.

A failed sample's log tail starts with the error lines of the full log
(cargo prints its manifest warnings last, so the last lines alone can miss
the cause). When rustc panics it leaves `rustc-ice-*.txt` in the worktree
root; the runner appends each report's head (up to the backtrace) to the
tail, deletes the file and records `ice_reports`.

The summary gains an `ab` section (only when OFF/ON samples exist, so older
summaries reproduce byte for byte): per scenario the OFF and ON wall time
stats, `delta_s` (ON minus OFF), `saving_pct` ((OFF - ON) / OFF), failures
per arm, median max RSS and median CPU seconds (user + sys) per arm and
their deltas.

## M5 result (patched-20261015-4c83e38): REJECTED, Headstart stays disabled

**Verdict: REJECTED.** The promotion gate fails on the crash below.
Further timing was stopped by Drake's decision (via the orchestrator,
2026-10-16), so speed cells beyond the clean held samples are NOT_MEASURED.
The gate needs zero unexplained crashes; Headstart ON crashes the compiler
on every incremental edit, so it fails whatever the speed.
No "Optional local invocation" section is added. Default builds are
unchanged: nothing in this repository enables `-Zheadstart` or
`-Zearly-metadata`.

The toolchain was local only: built under a disposable directory outside
the repository, never vendored or committed (the upstream patches carry no
license), never installed into rustup or `~/.cargo/bin`, and never used by
CI, release, signing or spark-rsi. Only digests and our own scripts are kept
(`toolchain/digests.txt`, `toolchain/*.sh`).

Pins: upstream headstart `ff81bfecda2116d54776a7fec538fc2f4e349d1f`, rustc
submodule `6006fd03947bd7b04ae5cb5f7f9bb66b44dc8c6e`, cargo submodule
`4f3fb2428282fb2467c4bb5be0163d2f42a31491`; sha256 of the stage-1 rustc,
`librustc_driver-*.so`, cargo and each patch in `toolchain/digests.txt` and
`manifest.toolchain.binaries`. AIEN source: `4c83e38`.

**Crash (observed, the gate failure).** In the `abincr` stage every check
after an edit failed with exit 101 when ON, and every one passed when OFF:

| check after edit | OFF exit 0 | ON exit 101 |
|---|---|---|
| body edit (20 steps) | 20 of 20 | 20 of 20 |
| interface edit (20 steps) | 20 of 20 | 20 of 20 |
| settle after revert (4) | 4 of 4 | 4 of 4 |

No-op checks and the cold prime passed in both arms. The run left 46 rustc
ICE reports, all the same panic (`logs/abincr-run1-ice-heads.txt`). One
untimed diagnostic (`diag-once/`: prime, the step-1 body edit, one
incremental ON check, `-j 8`, under `quietlock run`) reproduced it; full
stderr in `diag-once/incr-on.stderr.log`:

```
thread 'rustc' (586269) panicked at compiler/rustc_metadata/src/rmeta/encoder.rs:2837:29:
assertion `left == right` failed: early and full metadata disagree on the SVH
note: compiler flags: --crate-type lib -Z early-metadata -C embed-bitcode=no -C debuginfo=2 -C incremental=[REDACTED]
error: could not compile `aien-runtime` (lib)
```

The crate that panics is `aien-runtime`, which depends on the edited
`cortex-rs` (`crates/aien-runtime/Cargo.toml`). This matches the upstream
note `docs/ice-early-metadata-svh.md` at the pinned commit: same assertion
and file, during incremental `cargo check` runs, "It's not fixed yet", and
"Every later build fails the same way, even with the edit reverted", which
is what the settle checks show.

**Correctness of cold builds (observed).** `verify.json` verdict `match`:
same exit codes, same normalized diagnostics, same results for the 115
tests of `spark-harness` and `aien-proof` in both arms, and the
`-Zearly-metadata-verify` cold build exited 0.

**Cold timings (observed, INDICATIVE: n=2 to 3 clean held samples per
arm, under the planned 5).** Same patched toolchain, OFF vs ON, default
`-j` (20), per-sample quiet-flag hold, arms alternating. The rerun was
stopped by Drake's decision during round 3; the remaining rounds are
NOT_MEASURED.

| workload | OFF median (n) | ON median (n) | saving | max RSS | CPU (user+sys) |
|---|---|---|---|---|---|
| cold `check --workspace --all-targets` | 38.117 s (2) | 31.643 s (3) | 16.98 % | -3.05 % | +1.92 % |
| cold `build --workspace` | 52.793 s (2) | 42.596 s (3) | 19.31 % | +0.00 % | +4.62 % |
| cold `test --workspace --no-run` | 79.933 s (3) | 73.493 s (2) | 8.06 % | -0.12 % | +6.76 % |
| A/A cold check (OFF vs OFF) | 37.895 s (3) | 37.566 s (2) | 0.87 % | | |

The OFF and ON ranges do not overlap in any of the three workloads
(`summary.json` `.ab[].wall_s_off/on`). One OFF check was contaminated and
is excluded. NOT_MEASURED (stopped by decision): jobs 4/8/16 (NOT_RUN),
rounds 4 and 5 of the cold A/B, every incremental ON timing (no ON check
after an edit succeeded). NOT_RUN: the upstream unit tests (`x test
tests/run-make/early-metadata tests/ui/rmeta`, cargo headstart tests). The
upstream check scripts (smoke, check-swap, check-errors, check-incremental
check and build) all exited 0 (`upstream-tests/results.txt`).

**Stable reference (M0, kept separate).** The stable 1.98.1 medians are
20.367 s (cold check), 31.054 s (cold build) and 56.387 s (test --no-run).
The patched stage-1 compiler is a different, slower build (OFF takes 1.42
to 1.87 times the stable median, ON 1.30 to 1.55 times), so patched and
stable times are not comparable. The M0 schedule replay **estimated** 14.5 % (realistic) and
15.7 % (upper) for the cold build; that is an estimate, separate from the
19.31 % observed above on the patched toolchain.

The first cold A/B run (before the per-sample hold) was stopped after
round 1 on a contaminated host; its samples are kept apart in
`samples-ab-run1-interrupted-unheld.jsonl` and used nowhere.

**Next:** the M0 critical path shows the `aws-lc-sys` build script (run) at
13.08 s of the 31.5 s cold build (`baseline-20261015-4c83e38/timings/critical-path.json`).
The follow-up is a native fix of that bottleneck, in a separate issue.

## Running it

From the repository root, with the stable toolchain pinned by
`rust-toolchain.toml`:

```bash
bash scripts/bench/headstart-selftest.sh            # about a minute, must print 0 failed
HS_COMMIT=<sha> \
HS_BODY_FILE=<repo-relative .rs file> HS_BODY_ANCHOR='<exact fn opening line>' \
HS_IFACE_FILE=<repo-relative lib.rs> HS_EDIT_REASON='<why these targets>' \
bash scripts/bench/headstart-aien.sh                # setup timings cold incr cleanup
```

Stages can be run separately (`setup timings`, then `cold incr cleanup`)
against the same `HS_WORK` and `HS_OUT`, which is how the edit targets were
chosen from the measured critical path before the incremental stage. The
other knobs are listed at the top of the runner.

## Known quirks of the committed run (baseline-20261015-4c83e38)

That run used the earlier runner (commit afa4316). Its data is kept as
recorded; what it can and cannot show:

- Foreign compiles were checked at sample start and then every 30 s
  (`protocol.foreign_monitor_interval_s: 30`), with no end-of-sample check.
  74 of the 84 counted samples finished before the first tick
  (`foreign.monitor_ticks: 0`), so for them only the start was checked. The
  data supports "no foreign compile at sample start, nor at any 30 s tick
  that occurred", not "none during the sample".
- `load_wait_capped` did not mark a sample contaminated then. Three attempts
  hit the cap; all three were also contaminated by a foreign process and
  re-run, so no counted sample has `load_wait_capped: true`. The highest
  `loadavg_before` (1-minute) of a counted sample is 1.99.
- The quiet-flag wait was uncapped and the manifest has only `respected:
  true`; there is no `available` field. The tool was present on that host:
  one sample waited 20 s for the flag (`quiet_flag_wait_s`), which only a
  working `quietlock check` can produce.
- The timings stage ran before the quiet-flag check was added, so its three
  samples have no `quiet_flag_wait_s`.
- `timings/graph.json` and `timings/replay.json` were derived after the run,
  from `cargo metadata --filter-platform aarch64-unknown-linux-gnu --locked
  --offline` in a checkout whose `Cargo.lock`, `Cargo.toml` files and crate
  sources are identical to 4c83e38, and from the committed `units.json`. No
  build was re-run. `critical-path.json` was regenerated from the same
  `units.json` by the current summarizer; only its `estimate` block changed.
- In that run both settle checks of a series shared one `sample_id` (step 0),
  so the first settle log tail of each series was overwritten (85 tail files
  for 87 lines). Settle samples are not measurements. The runner now numbers
  them step 1 (after body) and 2 (after iface), and the self-test checks that
  sample ids are unique.
- Log tails are named `*.tail.log`, which the root `.gitignore` ignores; they
  were committed with `git add -f`.

## Reproducing the summary

`summary.json` is a pure function of `samples.jsonl`:

```bash
D=benchmarks/data/headstart/baseline-<date>-<short>
bash scripts/bench/headstart-summarize.sh samples $D/samples.jsonl | cmp - $D/summary.json && echo identical
```

Per scenario and arm it reports `wall_s` (counted samples: final attempt and
exit 0) and `wall_s_clean` (counted and not contaminated), each with n,
median, min, max, MAD (median absolute deviation) and sample stdev, plus
failures with exit status, superseded attempts, median max RSS and median
CPU%. `aa_noise` gives the A/A median difference in seconds and percent.

Max RSS comes from `/usr/bin/time -v`, which reports `ru_maxrss` of the
waited process tree: per getrusage(2) that is the largest single process, not
the sum over parallel rustc jobs.

## Critical path and the estimate

`critical-path.json` walks back from the last unit to finish along the edge
that unblocked each unit: cargo records, per unit, which units became
runnable when it finished (`unblocked_units`) or when its metadata was ready
(`unblocked_rmeta_units`), and `sections[frontend].end` marks when the
metadata was ready. That chain is the observed critical path of that one
build (scheduling delays included).

Two estimates, both labeled ESTIMATE and both covering the cold
`cargo build --workspace` only, of what a metadata-pipelining compiler change
(removing the wait for a dependency's metadata) could save on that build:

- upper bound (`critical-path.json`, `estimate`) = min(sum of predecessor
  frontend time on rmeta-gated links of the chain, build end minus floor),
  where floor = max(longest single unit, sum of unit durations / jobs).
  Full-gated links (build scripts, proc macros, linked binaries) need
  finished artifacts and are not shortened. This ignores that the critical
  path may move to another chain.
- realistic estimate (`replay.json`) = a list-schedule replay of the
  measured unit durations at the build's job count (nproc), with unit edges
  rebuilt from the package graph (`graph.json`). Stable cargo has no unit
  graph (`--unit-graph` is unstable), so the edges follow cargo's rules: a
  build-script compile needs its build-deps' libs; a build-script run needs
  its own compile and the build-script run of any normal dependency that
  declares `links`; a lib needs its own build-script run and its normal
  dependencies' metadata (full artifact for proc macros); a bin needs its
  own lib and its normal dependencies in full. Units start in observed start
  order whenever a job slot is free. `baseline` treats a metadata edge as
  ready at the dependency's frontend end; `no_rmeta_wait` treats it as ready
  when the dependency starts. The saving is `baseline - no_rmeta_wait`, both
  from the same model, and `model_error_pct` compares `baseline` with the
  observed build end. Feature variants of one package are grouped
  conservatively (every variant waits for every variant of its dependency).

For the committed run: observed build 31.50 s, replay baseline 31.35 s
(model error -0.48%), no-metadata-wait 26.79 s, realistic saving 4.56 s
(14.5%); upper bound 4.94 s (15.7%).
