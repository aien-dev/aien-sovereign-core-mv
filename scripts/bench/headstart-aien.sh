#!/usr/bin/env bash
# Headstart M0 (#532): reproducible stable-toolchain build baselines for the
# AIEN monorepo. Measurement only: this script never edits Cargo.toml,
# rust-toolchain.toml, CI, compiler flags or the caller's checkout.
#
# Usage:
#   scripts/bench/headstart-aien.sh [stage ...]
# Stages (default: setup timings cold incr cleanup):
#   setup    create a detached git worktree of HS_COMMIT under HS_WORK, warm
#            the cargo registry (cargo fetch --locked), write manifest.json
#   timings  one cold `cargo build --workspace --timings`, extract unit data
#            and the observed critical path (timings/)
#   cold     interleaved rounds of the cold scenarios (A/A on cold check)
#   incr     10-step body-only and interface-changing edit series on
#            `cargo check --workspace --all-targets`, reverted and verified
#   summary  rebuild summary.json from samples.jsonl
#   cleanup  remove the benchmark worktree and its target directory
# Headstart M5 (#536) A/B stages, for one toolchain run with two arm
# environments (HS_ARM_ENV_OFF, HS_ARM_ENV_ON, both required):
#   ab       interleaved cold check/build/test --no-run, arm OFF vs ON, plus
#            an A/A pair (arms A and B, both with the OFF environment);
#            optional extra cold builds at HS_AB_JOBS job counts
#   abincr   the incr edit series with OFF and ON on separate target dirs,
#            both arms built after every edit
#   abverify correctness: normalized diagnostics, exit codes and test
#            results of OFF vs ON, plus one ON build with HS_VERIFY_ENV_ON
#            (writes verify.json)
#
# Environment (defaults in brackets): HS_COMMIT [HEAD of this checkout],
# HS_WORK [$HOME/workspace/headstart-m0-<short>], HS_OUT
# [benchmarks/data/headstart/baseline-<date>-<short>], HS_SAMPLES [5],
# HS_LOAD_MAX [2], HS_WAIT_CAP_S [600], HS_MON_INTERVAL_S [3],
# HS_QUIETLOCK [1], HS_QUIETLOCK_BIN [quietlock], HS_QUIET_WAIT_CAP_S [3600],
# HS_MAX_RERUNS [2], HS_SLOW_COLD_S [1200], HS_INCR_STEPS [10],
# HS_INCR_SERIES [2], HS_BODY_FILE, HS_BODY_ANCHOR, HS_IFACE_FILE,
# HS_EDIT_REASON. Toolchain and A/B knobs: HS_CARGO [cargo], HS_RUSTC
# [unset: cargo picks rustc], HS_ARM_ENV_OFF, HS_ARM_ENV_ON (space-separated
# KEY=VALUE words), HS_AB_JOBS [empty], HS_AB_JOBS_SAMPLES [3],
# HS_VERIFY_TEST_PKGS [spark-harness], HS_VERIFY_ENV_ON [empty]. Quiet-flag
# hold: HS_HOLD [0], HS_HOLD_OWNER [headstart-runner], HS_HOLD_MINUTES [5],
# HS_HOLD_REASON [headstart timing sample]. HS_KEEP_LOG [0] keeps full logs.
# See benchmarks/headstart/README.md.
set -euo pipefail

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${HS_REPO:-$(git -C "$SELF_DIR" rev-parse --show-toplevel)}
SUMMARIZE="$SELF_DIR/headstart-summarize.sh"

HS_COMMIT=$(git -C "$REPO" rev-parse "${HS_COMMIT:-HEAD}")
SHORT=${HS_COMMIT:0:7}
HS_WORK=${HS_WORK:-$HOME/workspace/headstart-m0-$SHORT}
HS_DATE=${HS_DATE:-$(date -u +%Y%m%d)}
HS_OUT=${HS_OUT:-$REPO/benchmarks/data/headstart/baseline-$HS_DATE-$SHORT}
HS_SAMPLES=${HS_SAMPLES:-5}
HS_LOAD_MAX=${HS_LOAD_MAX:-2}
HS_WAIT_CAP_S=${HS_WAIT_CAP_S:-600}
HS_MON_INTERVAL_S=${HS_MON_INTERVAL_S:-3}
HS_MAX_RERUNS=${HS_MAX_RERUNS:-2}
HS_SLOW_COLD_S=${HS_SLOW_COLD_S:-1200}
HS_INCR_STEPS=${HS_INCR_STEPS:-10}
HS_INCR_SERIES=${HS_INCR_SERIES:-2}
HS_QUIETLOCK=${HS_QUIETLOCK:-1}
HS_QUIETLOCK_BIN=${HS_QUIETLOCK_BIN:-quietlock}
HS_QUIET_WAIT_CAP_S=${HS_QUIET_WAIT_CAP_S:-3600}
# Per-sample hold of the quiet flag (HS_HOLD=1): take it after the queue wait,
# keep it through the load wait and the timed command, release it after.
HS_HOLD=${HS_HOLD:-0}
HS_HOLD_OWNER=${HS_HOLD_OWNER:-headstart-runner}
HS_HOLD_MINUTES=${HS_HOLD_MINUTES:-5}
HS_HOLD_REASON=${HS_HOLD_REASON:-headstart timing sample}
# Edit targets (repo-relative paths). The anchor is a fixed string that must
# match exactly one line, the opening line of a function body.
HS_BODY_FILE=${HS_BODY_FILE:-}
HS_BODY_ANCHOR=${HS_BODY_ANCHOR:-}
HS_IFACE_FILE=${HS_IFACE_FILE:-}
HS_EDIT_REASON=${HS_EDIT_REASON:-}
# Toolchain under test. HS_CARGO replaces the leading `cargo` of every
# scenario command; HS_RUSTC, when set, is exported as RUSTC to each sample.
HS_CARGO=${HS_CARGO:-cargo}
HS_RUSTC=${HS_RUSTC:-}
HS_AB_JOBS=${HS_AB_JOBS:-}
HS_AB_JOBS_SAMPLES=${HS_AB_JOBS_SAMPLES:-3}
HS_VERIFY_TEST_PKGS=${HS_VERIFY_TEST_PKGS:-spark-harness}
HS_VERIFY_ENV_ON=${HS_VERIFY_ENV_ON:-}

SRC="$HS_WORK/src"
TARGET="$HS_WORK/target"
RAWLOG="$HS_WORK/logs"
SAMPLES="$HS_OUT/samples.jsonl"
MANIFEST="$HS_OUT/manifest.json"
RUN_ID=${HS_RUN_ID:-headstart-m0-$HS_DATE-$SHORT}
TIME_BIN=/usr/bin/time
MON_PID=""
EDITED_FILE=""
HOLD_PID=""
HOLD_FILE=""

say() { printf '[headstart %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { say "ABORT: $*"; exit 1; }

# Release a per-sample quiet-flag hold (see hold_acquire): removing the want
# file ends the held waiter, and quietlock releases the flag when it exits.
hold_release() {
  [ -n "$HOLD_PID" ] || return 0
  rm -f "$HOLD_FILE.want"
  wait "$HOLD_PID" 2>/dev/null || true
  HOLD_PID=""
  rm -f "$HOLD_FILE"
}

cleanup_on_exit() {
  local rc=$?
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true
  hold_release
  if [ -n "$EDITED_FILE" ] && [ -d "$SRC" ]; then
    git -C "$SRC" checkout -- "$EDITED_FILE" 2>/dev/null || true
    say "reverted in-flight edit of $EDITED_FILE"
  fi
  exit "$rc"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT TERM

need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }
need git; need jq; need awk; need sed; need "$HS_CARGO"; need pgrep; need sha256sum

now_ns() { date +%s%N; }
secs_between() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", (b - a) / 1e9 }'; }
loadavg_json() { awk '{ printf "[%s,%s,%s]", $1, $2, $3 }' /proc/loadavg; }
load1() { awk '{ print $1 }' /proc/loadavg; }
mem_avail_kb() { awk '/^MemAvailable:/ { print $2 }' /proc/meminfo; }

# PIDs of this runner and every descendant (our own cargo/rustc).
own_pids() {
  ps -e -o pid=,ppid= | awk -v root="$$" '
    { parent[$1] = $2; pids[NR] = $1 }
    END {
      own[root] = 1; changed = 1
      while (changed) { changed = 0
        for (i in pids) { p = pids[i]; if (!(p in own) && (parent[p] in own)) { own[p] = 1; changed = 1 } } }
      for (p in own) print p
    }'
}

# Foreign rustc/cargo/clippy-driver/rustdoc processes as a JSON array of
# "pid:comm:crate" strings. Excludes our process tree and anything whose
# command line mentions our work directory.
foreign_json() {
  local own line pid comm crate out=()
  own=" $(own_pids | tr '\n' ' ') "
  while read -r pid; do
    [ -n "$pid" ] || continue
    case "$own" in *" $pid "*) continue ;; esac
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    [ -n "$comm" ] || continue
    line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$line" in *"$HS_WORK"*) continue ;; esac
    crate=$(printf '%s' "$line" | sed -n 's/.*--crate-name \([^ ]*\).*/\1/p')
    # A compiler process counts only when it compiles a crate; this skips
    # idle probes such as an orphaned `rustc --version`.
    [ "$comm" != cargo ] && [ -z "$crate" ] && continue
    out+=("$pid:$comm:${crate:--}")
  done < <(pgrep -x 'rustc|cargo|clippy-driver|rustdoc' || true)
  if [ ${#out[@]} -eq 0 ]; then echo '[]'; else printf '%s\n' "${out[@]}" | head -40 | jq -R . | jq -cs .; fi
}

# Respect the host's shared quiet flag (`quietlock check` exits non-zero while
# another job holds it for a quiet measurement). Capped at HS_QUIET_WAIT_CAP_S
# so a stale flag cannot hang the run; a capped wait is logged and the sample
# is marked contaminated. Prints "<waited_s> <capped> <checked>", where
# checked is false when the check is disabled or quietlock is not installed.
quietlock_available() { [ "$HS_QUIETLOCK" = 1 ] && command -v "$HS_QUIETLOCK_BIN" >/dev/null 2>&1; }
wait_quietlock() {
  local start waited
  start=$(date +%s)
  if ! quietlock_available; then echo "0 false false"; return; fi
  while ! "$HS_QUIETLOCK_BIN" check >/dev/null 2>&1; do
    waited=$(( $(date +%s) - start ))
    if [ "$waited" -ge "$HS_QUIET_WAIT_CAP_S" ]; then
      say "quiet flag still held after ${waited}s (cap ${HS_QUIET_WAIT_CAP_S}s): proceeding, sample marked contaminated"
      echo "$waited true true"; return
    fi
    sleep 5
  done
  echo "$(( $(date +%s) - start )) false true"
}

# hold_acquire: with HS_HOLD=1, take the quiet flag for one sample with
# `quietlock hold`. The held command is a small waiter that writes the hold id
# to $HOLD_FILE, then waits until hold_release removes $HOLD_FILE.want (or
# this runner dies). A refusal (exit 75: someone took the flag after our
# queue wait) queues again behind the holder, within HS_QUIET_WAIT_CAP_S.
# Sets HOLD_HELD (true|false), HOLD_ID, HOLD_S and HOLD_REFUSED; it must not
# run in a subshell, because the waiter has to outlive this call.
hold_acquire() {
  local start rc waited
  HOLD_HELD=false; HOLD_ID=""; HOLD_S=0; HOLD_REFUSED=0
  [ "$HS_HOLD" = 1 ] && quietlock_available || return 0
  start=$(date +%s)
  HOLD_FILE="$RAWLOG/.quiet-hold.$$"
  while :; do
    rm -f "$HOLD_FILE"; : > "$HOLD_FILE.want"
    "$HS_QUIETLOCK_BIN" hold --owner "$HS_HOLD_OWNER" --minutes "$HS_HOLD_MINUTES" --reason "$HS_HOLD_REASON" -- \
      sh -c 'printf "%s\n" "${QUIETLOCK_HOLD:-unknown}" > "$1"
             while [ -e "$2" ] && kill -0 "$3" 2>/dev/null; do sleep 0.2; done' \
      hold-waiter "$HOLD_FILE" "$HOLD_FILE.want" "$$" >> "$RAWLOG/quietlock-hold.log" 2>&1 &
    HOLD_PID=$!
    while [ ! -s "$HOLD_FILE" ] && kill -0 "$HOLD_PID" 2>/dev/null; do sleep 0.1; done
    if [ -s "$HOLD_FILE" ]; then
      HOLD_HELD=true; HOLD_ID=$(head -1 "$HOLD_FILE"); HOLD_S=$(( $(date +%s) - start )); return 0
    fi
    rc=0; wait "$HOLD_PID" || rc=$?
    HOLD_PID=""; rm -f "$HOLD_FILE.want"
    HOLD_S=$(( $(date +%s) - start ))
    if [ "$rc" -ne 75 ]; then
      say "quietlock hold failed (exit $rc, see $RAWLOG/quietlock-hold.log): sample runs unheld, marked contaminated"
      return 0
    fi
    HOLD_REFUSED=$((HOLD_REFUSED + 1))
    if [ "$HOLD_S" -ge "$HS_QUIET_WAIT_CAP_S" ]; then
      say "quiet flag hold refused $HOLD_REFUSED times in ${HOLD_S}s (cap ${HS_QUIET_WAIT_CAP_S}s): sample runs unheld, marked contaminated"
      return 0
    fi
    sleep 1
    wait_quietlock > /dev/null
  done
}

# Files rustc leaves in the worktree root when it panics (internal compiler
# error). harvest_ice <tail file> appends each report's head (the panic lines,
# up to the backtrace) to the tail file, removes the reports so the next
# sample starts clean, and prints how many there were.
harvest_ice() {
  local out=$1 n=0 f
  for f in "$SRC"/rustc-ice-*.txt; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    { printf '== %s (head, up to the backtrace) ==\n' "${f##*/}"
      awk '/^stack backtrace:/ { exit } NR > 12 { exit } { print }' "$f" | LC_ALL=C tr -cd '\11\12\15\40-\176'
    } >> "$out"
    rm -f "$f"
  done
  echo "$n"
}

# Wait until 1-minute loadavg < HS_LOAD_MAX, at most HS_WAIT_CAP_S seconds.
# Prints "<waited_s> <capped:true|false>".
wait_quiet() {
  local start waited
  start=$(date +%s)
  while awk -v l="$(load1)" -v m="$HS_LOAD_MAX" 'BEGIN { exit !(l >= m) }'; do
    waited=$(( $(date +%s) - start ))
    if [ "$waited" -ge "$HS_WAIT_CAP_S" ]; then echo "$waited true"; return; fi
    sleep 5
  done
  echo "$(( $(date +%s) - start )) false"
}

# Parse a /usr/bin/time -v report into JSON.
time_v_json() {
  local f=$1
  if [ ! -s "$f" ]; then echo '{"available":false,"reason":"no /usr/bin/time report"}'; return; fi
  awk -F': ' '
    /Maximum resident set size/ { rss = $2 }
    /Percent of CPU this job got/ { cpu = $2; sub(/%/, "", cpu) }
    /User time \(seconds\)/ { usr = $2 }
    /System time \(seconds\)/ { sys = $2 }
    /Elapsed \(wall clock\) time/ { el = $NF }
    /Exit status/ { ex = $2 }
    END {
      if (cpu == "?" || cpu == "") cpu = "null"
      printf "{\"available\":true,\"max_rss_kb\":%s,\"cpu_percent\":%s,\"user_s\":%s,\"sys_s\":%s,\"elapsed\":\"%s\",\"exit_status\":%s,\"max_rss_scope\":\"ru_maxrss of the waited cargo tree: the largest single process, not a sum (getrusage(2))\"}",
        (rss == "" ? "null" : rss), cpu, (usr == "" ? "null" : usr), (sys == "" ? "null" : sys), el, (ex == "" ? "null" : ex)
    }' "$f"
}

ensure_clean() {
  local where=$1
  git -C "$SRC" diff --quiet || die "worktree not clean after $where: $(git -C "$SRC" diff --stat | tail -1)"
  git -C "$SRC" diff --cached --quiet || die "worktree index not clean after $where"
}

# Environment words for an arm: HS_ARM_ENV_<arm> (space-separated
# KEY=VALUE), empty when unset. Arms A and B of the M0 stages have none.
arm_env() { local v="HS_ARM_ENV_$1"; printf '%s' "${!v:-}"; }

# run_sample <scenario> <arm> <kind> <round> <step> <aa_pair|-> <prep> -- cargo args...
# prep: "cold" removes the target dir before each attempt, "warm" keeps it.
# CUR_TARGET (set by a stage) overrides the target dir; CUR_ENV_EXTRA adds
# environment words after the arm's; HS_KEEP_LOG=1 keeps the full log in
# $RAWLOG/<sample id>.log for the stage to parse.
run_sample() {
  local scenario=$1 arm=$2 kind=$3 round=$4 step=$5 aa=$6 prep=$7
  shift 8
  local cmd=("$@") cmdline attempt=0 final tgt=${CUR_TARGET:-$TARGET}
  cmdline="${cmd[*]}"
  local runcmd=("${cmd[@]}") envw=() extra=() envline
  [ "${runcmd[0]}" = cargo ] && runcmd[0]=$HS_CARGO
  [ -n "$HS_RUSTC" ] && envw+=("RUSTC=$HS_RUSTC")
  read -ra extra <<< "$(arm_env "$arm") ${CUR_ENV_EXTRA:-}"
  envw+=("${extra[@]}")
  envline="${envw[*]}"
  while :; do
    attempt=$((attempt + 1))
    local sid="${scenario}-${arm}-r${round}-s${step}-a${attempt}"
    [ "$prep" = cold ] && rm -rf "$tgt"
    local wq waited capped lb la fstart fend mon t0 t1 rc wall tv tf log tail_rel contaminated maxseen ticks memb
    local qw qwait qcapped qchecked reasons ice
    qw=$(wait_quietlock); read -r qwait qcapped qchecked <<< "$qw"
    hold_acquire
    wq=$(wait_quiet); waited=${wq% *}; capped=${wq#* }
    lb=$(loadavg_json); memb=$(mem_avail_kb)
    fstart=$(foreign_json)
    mon="$RAWLOG/$sid.mon"; : > "$mon"
    ( while sleep "$HS_MON_INTERVAL_S"; do foreign_json >> "$mon"; done ) &
    MON_PID=$!
    tf="$RAWLOG/$sid.time"; log="$RAWLOG/$sid.log"; rm -f "$tf"
    say "sample $sid: ${cmdline} (waited ${waited}s, load $lb)"
    t0=$(now_ns)
    set +e
    if [ -x "$TIME_BIN" ]; then
      (cd "$SRC" && CARGO_TARGET_DIR="$tgt" env "${envw[@]}" "$TIME_BIN" -v -o "$tf" "${runcmd[@]}") > "$log" 2>&1
    else
      (cd "$SRC" && CARGO_TARGET_DIR="$tgt" env "${envw[@]}" "${runcmd[@]}") > "$log" 2>&1
    fi
    rc=$?
    set -e
    t1=$(now_ns)
    fend=$(foreign_json)
    kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; MON_PID=""
    hold_release
    la=$(loadavg_json)
    wall=$(secs_between "$t0" "$t1")
    if [ -x "$TIME_BIN" ]; then tv=$(time_v_json "$tf"); else tv='{"available":false,"reason":"/usr/bin/time not found"}'; fi
    ticks=$(wc -l < "$mon")
    maxseen=$( { echo "$fstart"; cat "$mon"; echo "$fend"; } | jq -s 'map(length) | max')
    reasons=$(jq -cn --argjson s "$fstart" --argjson e "$fend" --slurpfile m "$mon" \
      --argjson lc "$capped" --argjson qc "$qcapped" --arg hr "$HS_HOLD" --argjson hh "$HOLD_HELD" \
      '[ (if ($s | length) > 0 then "foreign_at_start" else empty end),
         (if any($m[]; length > 0) then "foreign_during" else empty end),
         (if ($e | length) > 0 then "foreign_at_end" else empty end),
         (if $lc then "load_wait_capped" else empty end),
         (if $qc then "quiet_flag_wait_capped" else empty end),
         (if $hr == "1" and ($hh | not) then "quiet_hold_not_taken" else empty end) ]')
    contaminated=false; [ "$reasons" != '[]' ] && contaminated=true
    tail_rel="logs/$sid.tail.log"
    local nt=30; [ "$rc" -ne 0 ] && nt=80
    : > "$HS_OUT/$tail_rel"
    # A failure's cause can sit far above the last lines (cargo prints its
    # manifest warnings last), so failed samples keep the error lines first.
    if [ "$rc" -ne 0 ]; then
      { echo '== error lines (first 30 of the full log) =='
        grep -m 30 -E "^error|panicked at|internal compiler error|^thread '|^note: (please attach|compiler flags|we would appreciate)" "$log" || true
        echo "== last $nt lines =="
      } | LC_ALL=C tr -cd '\11\12\15\40-\176' >> "$HS_OUT/$tail_rel"
    fi
    tail -n "$nt" "$log" | LC_ALL=C tr -cd '\11\12\15\40-\176' >> "$HS_OUT/$tail_rel"
    ice=$(harvest_ice "$HS_OUT/$tail_rel")
    final=true
    if [ "$contaminated" = true ] && [ "$attempt" -le "$HS_MAX_RERUNS" ]; then final=false; fi
    jq -cn \
      --arg run_id "$RUN_ID" --arg sid "$sid" --arg scenario "$scenario" --arg arm "$arm" \
      --arg kind "$kind" --argjson round "$round" --argjson step "$step" --argjson attempt "$attempt" \
      --arg aa "$aa" --arg prep "$prep" --arg cmdline "$cmdline" --arg cwd "$SRC" --arg target "$tgt" \
      --arg envline "$envline" --arg cargo_bin "$HS_CARGO" \
      --arg started "$(date -u -d "@$((t0 / 1000000000))" +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson wall "$wall" --argjson tv "$tv" --argjson rc "$rc" \
      --argjson lb "$lb" --argjson la "$la" --argjson waited "$waited" --argjson capped "$capped" \
      --argjson qwait "$qwait" --argjson qcapped "$qcapped" --argjson qchecked "$qchecked" \
      --argjson fend "$fend" --argjson reasons "$reasons" --argjson memb "${memb:-null}" --argjson fstart "$fstart" --argjson maxseen "$maxseen" \
      --argjson ticks "$ticks" --argjson contaminated "$contaminated" --argjson final "$final" \
      --arg tail "$tail_rel" --arg edit "${HS_CUR_EDIT:-}" --argjson ice "$ice" \
      --arg hr "$HS_HOLD" --argjson hh "$HOLD_HELD" --arg hid "$HOLD_ID" --argjson hs "$HOLD_S" --argjson hrf "$HOLD_REFUSED" \
      --slurpfile mon <(cat "$mon") \
      '{schema: "headstart-sample/v1", run_id: $run_id, sample_id: $sid, scenario: $scenario, arm: $arm,
        kind: $kind, round: $round, step: $step, attempt: $attempt,
        aa_pair: (if $aa == "-" then null else $aa end), prep: $prep, edit: (if $edit == "" then null else $edit end),
        command: $cmdline, cwd: $cwd, cargo_target_dir: $target, started_at: $started,
        cargo_bin: $cargo_bin, env_words: $envline,
        wall_s: $wall, time_v: $tv, exit_status: $rc, ok: ($rc == 0),
        loadavg_before: $lb, loadavg_after: $la,
        quiet_flag_checked: $qchecked, quiet_flag_wait_s: $qwait, quiet_flag_wait_capped: $qcapped,
        quiet_hold: {requested: ($hr == "1"), held: $hh, id: (if $hid == "" then null else $hid end),
                     acquire_s: $hs, refused: $hrf},
        load_wait_s: $waited, load_wait_capped: $capped,
        mem_available_kb_before: $memb,
        foreign: {at_start: $fstart, at_end: $fend, monitor_ticks: $ticks, max_seen: $maxseen,
                  seen: ([$fstart[], ($mon[] | .[]), $fend[]] | unique | .[0:40])},
        contaminated: $contaminated, contamination_reasons: $reasons, ice_reports: $ice,
        final: $final, log_tail: $tail}' >> "$SAMPLES"
    [ "${HS_KEEP_LOG:-0}" = 1 ] || rm -f "$log"
    LAST_LOG=$log
    say "  -> rc=$rc wall=${wall}s contaminated=$contaminated final=$final"
    LAST_WALL=$wall
    LAST_RC=$rc
    [ "$final" = true ] && break
  done
}

# ---------------------------------------------------------------- stages

stage_setup() {
  say "setup: commit $HS_COMMIT, work $HS_WORK, out $HS_OUT"
  [ -x "$TIME_BIN" ] || say "WARNING: $TIME_BIN not found; max RSS and CPU% will be recorded as unavailable"
  case "$HS_WORK/" in "$REPO/"*) die "HS_WORK must be outside the checkout ($REPO)";; esac
  mkdir -p "$HS_WORK" "$RAWLOG" "$HS_OUT/logs" "$HS_OUT/timings"
  if [ ! -d "$SRC/.git" ] && [ ! -f "$SRC/.git" ]; then
    git -C "$REPO" worktree add --detach "$SRC" "$HS_COMMIT" >&2
  fi
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$HS_COMMIT" ] || die "worktree HEAD is not $HS_COMMIT"
  ensure_clean setup
  local df_line t0 t1 frc
  df_line=$(df -h "$HS_WORK" | tail -1)
  say "disk: $df_line"
  t0=$(now_ns)
  set +e
  (cd "$SRC" && "$HS_CARGO" fetch --locked) > "$RAWLOG/fetch.log" 2>&1
  frc=$?
  set -e
  t1=$(now_ns)
  [ "$frc" -eq 0 ] || die "cargo fetch --locked failed (rc=$frc), see $RAWLOG/fetch.log"
  local rustc_vv cargo_v cpu_models cfgs
  rustc_vv=$(cd "$SRC" && "${HS_RUSTC:-rustc}" -vV)
  cargo_v=$(cd "$SRC" && "$HS_CARGO" -V)
  cpu_models=$(lscpu 2>/dev/null | awk -F': *' '/^Model name/ { print $2 }' | jq -R . | jq -cs .)
  cfgs=$(for f in "$HOME/.cargo/config.toml" "$HOME/.cargo/config" "$SRC/.cargo/config.toml" "$SRC/.cargo/config"; do
           if [ -e "$f" ]; then echo "$f"; fi; done | jq -R . | jq -cs .)
  jq -n \
    --arg run_id "$RUN_ID" --arg commit "$HS_COMMIT" --arg short "$SHORT" \
    --arg origin "$(git -C "$REPO" remote get-url origin 2>/dev/null || echo unknown)" \
    --arg src "$SRC" --arg target "$TARGET" --arg rustc_vv "$rustc_vv" --arg cargo_v "$cargo_v" \
    --arg toolchain_file "$(cat "$SRC/rust-toolchain.toml" 2>/dev/null || echo '<absent>')" \
    --arg arch "$(uname -m)" --arg kernel "$(uname -r)" --argjson nproc "$(nproc)" \
    --argjson memtotal "$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)" --argjson memavail "$(mem_avail_kb)" \
    --argjson cpu_models "$cpu_models" --argjson cfgs "$cfgs" --arg df "$df_line" \
    --arg date "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson fetch_s "$(secs_between "$t0" "$t1")" \
    --arg rustflags "${RUSTFLAGS-<unset>}" --arg wrapper "${RUSTC_WRAPPER-<unset>}" \
    --arg jobs_env "${CARGO_BUILD_JOBS-<unset>}" --arg incr_env "${CARGO_INCREMENTAL-<unset>}" \
    --argjson time_v "$([ -x "$TIME_BIN" ] && echo true || echo false)" \
    --argjson samples "$HS_SAMPLES" --argjson loadmax "$HS_LOAD_MAX" --argjson waitcap "$HS_WAIT_CAP_S" \
    --argjson mon "$HS_MON_INTERVAL_S" --argjson reruns "$HS_MAX_RERUNS" --argjson slow "$HS_SLOW_COLD_S" \
    '{schema: "headstart-manifest/v1", run_id: $run_id, created_at: $date,
      source: {commit: $commit, short: $short, origin: $origin, worktree: $src,
               worktree_kind: "git worktree add --detach (outside the caller checkout)"},
      toolchain: {rustc_vV: $rustc_vv, cargo_V: $cargo_v, rust_toolchain_toml: $toolchain_file,
                  cargo_config_files: $cfgs,
                  env: {RUSTFLAGS: $rustflags, RUSTC_WRAPPER: $wrapper, CARGO_BUILD_JOBS: $jobs_env, CARGO_INCREMENTAL: $incr_env}},
      host: {arch: $arch, kernel: $kernel, nproc: $nproc, cpu_models: $cpu_models,
             mem_total_kb: $memtotal, mem_available_kb_at_setup: $memavail, disk_at_setup: $df},
      build: {features: "default (no --features / --all-features / --no-default-features flags)",
              profiles: {check: "dev", build: "dev", "test --no-run": "test (inherits dev)"},
              target_dir: $target, jobs: "cargo default (-j not passed, CARGO_BUILD_JOBS as recorded) = nproc",
              cold_definition: "CARGO_TARGET_DIR removed before every cold attempt; ~/.cargo/registry and ~/.cargo/git kept warm",
              registry_cache: {state: "warm", how: "cargo fetch --locked ran once in setup before any sample", fetch_wall_s: $fetch_s}},
      protocol: {samples_per_scenario: $samples, load1_max: $loadmax, load_wait_cap_s: $waitcap,
                 foreign_monitor_interval_s: $mon, max_reruns_when_contaminated: $reruns,
                 slow_cold_threshold_s: $slow,
                 foreign_definition: "cargo processes, or rustc / clippy-driver / rustdoc processes with --crate-name (an actual compile), outside this runner process tree and whose command line does not mention the benchmark work dir",
                 contaminated_definition: "any of: a foreign process at sample start, at any monitor tick or in the end-of-sample snapshot; the load wait hit its cap; the quiet-flag wait hit its cap (see contamination_reasons)",
                 time_v_available: $time_v},
      scenarios: [], stages: []}' > "$MANIFEST"
  record_binaries
  say "manifest written: $MANIFEST"
}

# sha256 of the cargo and rustc under test. rustc is a small launcher; the
# compiler itself is lib/librustc_driver-*.so in its sysroot, so that is
# digested too.
record_binaries() {
  local cbin rbin sysroot drv files=()
  cbin=$(command -v "$HS_CARGO")
  rbin=$(command -v "${HS_RUSTC:-rustc}")
  sysroot=$(cd "$SRC" && "${HS_RUSTC:-rustc}" --print sysroot 2>/dev/null || true)
  files=("$cbin" "$rbin")
  for drv in "$sysroot"/lib/librustc_driver-*.so; do [ -e "$drv" ] && files+=("$drv"); done
  manifest_add '.toolchain.binaries = {cargo: $c, rustc: $r, sysroot: $s,
      sha256: ($sums | split("\n") | map(select(length > 0) | split("  ") | {key: .[1], value: .[0]}) | from_entries),
      arm_env: {OFF: $off, ON: $on}, verify_env_on: $ver,
      note: "with HS_RUSTC unset, rustc is the rustup proxy and sysroot the toolchain selected by rust-toolchain.toml"}' \
    --arg c "$cbin" --arg r "$rbin" --arg s "$sysroot" --arg sums "$(sha256sum "${files[@]}")" \
    --arg off "$(arm_env OFF)" --arg on "$(arm_env ON)" --arg ver "$HS_VERIFY_ENV_ON"
}

manifest_add() { # manifest_add <jq filter> [jq args...]
  local filter=$1; shift
  local tmp="$MANIFEST.tmp"
  jq "$@" "$filter" "$MANIFEST" > "$tmp" && mv "$tmp" "$MANIFEST"
}

stage_begin() {
  manifest_add '.stages += [{stage: $s, started_at: $t, script: "scripts/bench/headstart-aien.sh"}]' \
    --arg s "$1" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
stage_end() {
  manifest_add '(.stages[] | select(.stage == $s and .ended_at == null)) |= . + {ended_at: $t}' \
    --arg s "$1" --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # Which samples (all stages so far) ran under a per-sample quiet-flag hold.
  if [ -s "$SAMPLES" ]; then
    manifest_add '.protocol.quiet_hold += {
        samples_held: [$x[] | select(.quiet_hold.held == true) | .sample_id],
        samples_unheld: [$x[] | select(.quiet_hold.held != true) | .sample_id]}' \
      --slurpfile x "$SAMPLES"
  fi
}
scenario_add() { # scenario_add <id> <kind> <command>
  manifest_add 'if any(.scenarios[]; .id == $id) then . else .scenarios += [{id: $id, kind: $k, command: $c, cwd: "worktree root"}] end' \
    --arg id "$1" --arg k "$2" --arg c "$3"
}

require_setup() {
  [ -s "$MANIFEST" ] || die "no manifest at $MANIFEST; run the setup stage first"
  [ -d "$SRC" ] || die "no worktree at $SRC; run the setup stage first"
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$HS_COMMIT" ] || die "worktree HEAD moved"
  mkdir -p "$RAWLOG" "$HS_OUT/logs" "$HS_OUT/timings"
  local qa=false; quietlock_available && qa=true
  manifest_add '.protocol.quiet_flag = ((.protocol.quiet_flag // {first_stage: $s}) + {
      enabled: ($q == "1"), tool: $bin, available: $qa, respected: ($q == "1" and $qa),
      wait_cap_s: $cap,
      how: "before each sample, wait while `quietlock check` reports the host quiet flag held by another job, at most wait_cap_s; a capped wait is logged and marks the sample contaminated; per-sample quiet_flag_checked / quiet_flag_wait_s / quiet_flag_wait_capped"})' \
    --arg q "$HS_QUIETLOCK" --arg s "$CUR_STAGE" --arg bin "$HS_QUIETLOCK_BIN" \
    --argjson qa "$qa" --argjson cap "$HS_QUIET_WAIT_CAP_S"
  manifest_add '.protocol.quiet_hold = ((.protocol.quiet_hold // {}) + {
      enabled: ($h == "1"), owner: $o, minutes: $m, reason: $r,
      how: "with enabled, each sample runs: the quiet_flag queue wait, then `quietlock hold --owner <owner> --minutes <minutes> --reason <reason>` around the load wait and the timed command, released before the next sample; a refused hold queues again (within wait_cap_s); a hold not taken marks the sample contaminated (quiet_hold_not_taken). Per-sample quiet_hold {requested, held, id, acquire_s, refused}; samples_held / samples_unheld list sample ids"})' \
    --arg h "$HS_HOLD" --arg o "$HS_HOLD_OWNER" --argjson m "$HS_HOLD_MINUTES" --arg r "$HS_HOLD_REASON"
}

stage_timings() {
  require_setup; stage_begin timings
  local c=(cargo build --workspace --timings)
  scenario_add timings_build_ws timings "${c[*]}"
  run_sample timings_build_ws A timings 1 0 - cold -- "${c[@]}"
  ensure_clean timings
  local html="$TARGET/cargo-timings/cargo-timing.html"
  if [ "$LAST_RC" -eq 0 ] && [ -s "$html" ]; then
    "$SUMMARIZE" timings "$html" > "$HS_OUT/timings/units.json"
    "$SUMMARIZE" critical "$HS_OUT/timings/units.json" "$(nproc)" > "$HS_OUT/timings/critical-path.json"
    local host
    host=$(cd "$SRC" && rustc -vV | sed -n 's/^host: //p')
    (cd "$SRC" && cargo metadata --format-version 1 --filter-platform "$host" --locked --offline) > "$RAWLOG/metadata.json"
    "$SUMMARIZE" graph "$RAWLOG/metadata.json" > "$HS_OUT/timings/graph.json"
    "$SUMMARIZE" replay "$HS_OUT/timings/units.json" "$HS_OUT/timings/graph.json" "$(nproc)" > "$HS_OUT/timings/replay.json"
    say "replay estimate: $(jq -c '{baseline_model_end_s, no_rmeta_wait_model_end_s, realistic_saving_s, model_error_pct}' "$HS_OUT/timings/replay.json")"
    sed -n 's#.*<td>Max concurrency:</td><td>\(.*\)</td>.*#\1#p; s#.*<td>Total time:</td><td>\(.*\)</td>.*#\1#p' "$html" \
      | jq -R . | jq -s '{max_concurrency: .[0], total_time: .[1], source: "cargo-timing.html summary table"}' \
      > "$HS_OUT/timings/report-header.json"
    gzip -9 -c "$html" > "$HS_OUT/timings/cargo-timing.html.gz"
    say "critical path: $(jq -r '[.observed.chain[].name] | join(" -> ")' "$HS_OUT/timings/critical-path.json")"
  else
    say "timings build failed (rc=$LAST_RC); no critical path extracted"
  fi
  stage_end timings
}

stage_cold() {
  require_setup; stage_begin cold
  local ids=(cold_check_ws cold_build_ws cold_test_norun_ws cold_build_spark_harness cold_build_spark_supervisor cold_build_aien_runtime)
  declare -A cmd=(
    [cold_check_ws]="cargo check --workspace --all-targets"
    [cold_build_ws]="cargo build --workspace"
    [cold_test_norun_ws]="cargo test --workspace --no-run"
    [cold_build_spark_harness]="cargo build -p spark-harness"
    [cold_build_spark_supervisor]="cargo build -p spark-supervisor"
    [cold_build_aien_runtime]="cargo build -p aien-runtime"
  )
  local id; for id in "${ids[@]}"; do scenario_add "$id" cold "${cmd[$id]}"; done
  local n=$HS_SAMPLES round=1
  while [ "$round" -le "$n" ]; do
    # A/A: cold_check_ws runs as arms A and B, first and middle of the round,
    # with the arm order swapped every round. Other scenarios sit between.
    local first=A second=B
    [ $((round % 2)) -eq 0 ] && { first=B; second=A; }
    local order=("cold_check_ws:$first" cold_build_ws cold_build_spark_harness cold_test_norun_ws "cold_check_ws:$second" cold_build_spark_supervisor cold_build_aien_runtime)
    local item sid arm words
    for item in "${order[@]}"; do
      sid=${item%%:*}; arm=A; aa=-
      if [ "$sid" = cold_check_ws ]; then arm=${item#*:}; aa=cold_check_ws; fi
      read -ra words <<< "${cmd[$sid]}"
      run_sample "$sid" "$arm" cold "$round" 0 "$aa" cold -- "${words[@]}"
      ensure_clean "$sid round $round"
      case "$sid" in
        *_ws)
          if [ "$round" -eq 1 ] && [ "$n" -gt 3 ] && [ "$LAST_RC" -eq 0 ] \
             && awk -v w="$LAST_WALL" -v s="$HS_SLOW_COLD_S" 'BEGIN { exit !(w > s) }'; then
            n=3
            say "cold workspace sample took ${LAST_WALL}s (> ${HS_SLOW_COLD_S}s): reducing to 3 rounds"
            manifest_add '.protocol.samples_reduced = {to: 3, reason: $r}' \
              --arg r "$sid took ${LAST_WALL}s in round 1, above slow_cold_threshold_s"
          fi ;;
      esac
    done
    "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
    round=$((round + 1))
  done
  manifest_add '.protocol.cold_rounds_run = $n' --argjson n "$n"
  stage_end cold
}

apply_edit() { # apply_edit body|iface <k>
  local kind=$1 k=$2 f n
  if [ "$kind" = body ]; then
    f=$HS_BODY_FILE
    git -C "$SRC" checkout -- "$f"
    n=$(grep -nF -- "$HS_BODY_ANCHOR" "$SRC/$f" | cut -d: -f1 || true)
    [ "$(printf '%s\n' "$n" | grep -c .)" -eq 1 ] || die "body anchor must match exactly one line in $f"
    sed -i "${n}a\\    let _headstart_m0_probe = std::hint::black_box(${k}u64);" "$SRC/$f"
    HS_CUR_EDIT="body: $f line $((n + 1)) inserted 'let _headstart_m0_probe = std::hint::black_box(${k}u64);'"
  else
    f=$HS_IFACE_FILE
    git -C "$SRC" checkout -- "$f"
    printf '\n/// Headstart M0 benchmark probe (never committed).\npub fn headstart_m0_probe_%s() -> u64 {\n    %s\n}\n' "$k" "$k" >> "$SRC/$f"
    HS_CUR_EDIT="iface: $f appended 'pub fn headstart_m0_probe_${k}() -> u64'"
  fi
  EDITED_FILE=$f
  ! git -C "$SRC" diff --quiet || die "edit $kind step $k produced no change"
}

revert_edit() {
  [ -n "$EDITED_FILE" ] && git -C "$SRC" checkout -- "$EDITED_FILE"
  EDITED_FILE=""; HS_CUR_EDIT=""
}

stage_incr() {
  require_setup; stage_begin incr
  [ -n "$HS_BODY_FILE" ] && [ -n "$HS_BODY_ANCHOR" ] && [ -n "$HS_IFACE_FILE" ] \
    || die "set HS_BODY_FILE, HS_BODY_ANCHOR and HS_IFACE_FILE (see README)"
  [ -f "$SRC/$HS_BODY_FILE" ] || die "no such file in worktree: $HS_BODY_FILE"
  [ -f "$SRC/$HS_IFACE_FILE" ] || die "no such file in worktree: $HS_IFACE_FILE"
  manifest_add '.incremental = {command: "cargo check --workspace --all-targets", steps: $s, series_per_kind: $n,
      order: "body series, iface series, repeated (ABAB); untimed-for-stats settle check after every revert",
      body: {file: $bf, anchor: $ba, edit: "insert `let _headstart_m0_probe = std::hint::black_box(<k>u64);` after the anchor line; previous step reverted first"},
      iface: {file: $if, edit: "append `pub fn headstart_m0_probe_<k>() -> u64 { <k> }` (new pub item, new name every step); previous step reverted first"},
      reason: $why}' \
    --argjson s "$HS_INCR_STEPS" --argjson n "$HS_INCR_SERIES" --arg bf "$HS_BODY_FILE" \
    --arg ba "$HS_BODY_ANCHOR" --arg if "$HS_IFACE_FILE" --arg why "$HS_EDIT_REASON"
  local c=(cargo check --workspace --all-targets)
  scenario_add incr_prime prime "${c[*]}"
  scenario_add incr_noop_check incremental "${c[*]}"
  scenario_add incr_body_check incremental "${c[*]}"
  scenario_add incr_iface_check incremental "${c[*]}"
  scenario_add incr_settle settle "${c[*]}"
  run_sample incr_prime A prime 0 0 - warm -- "${c[@]}"
  [ "$LAST_RC" -eq 0 ] || die "prime check failed; incremental scenarios cannot run"
  local i; for i in 1 2 3; do run_sample incr_noop_check A incremental 0 "$i" - warm -- "${c[@]}"; done
  ensure_clean "noop"
  local series kind k settle_step
  for series in $(seq 1 "$HS_INCR_SERIES"); do
    settle_step=0
    for kind in body iface; do
      settle_step=$((settle_step + 1))
      for k in $(seq 1 "$HS_INCR_STEPS"); do
        apply_edit "$kind" "$((series * 100 + k))"
        run_sample "incr_${kind}_check" A incremental "$series" "$k" - warm -- "${c[@]}"
      done
      revert_edit
      ensure_clean "incr_${kind} series $series"
      # step 1 = settle after the body series, 2 = after the iface series
      run_sample incr_settle A settle "$series" "$settle_step" - warm -- "${c[@]}"
    done
  done
  ensure_clean incr
  "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
  stage_end incr
}

# ------------------------------------------------ M5 (#536) A/B stages
# One toolchain, two arm environments. A sample of arm X runs with the words
# of HS_ARM_ENV_X in its environment and its own target dir ($TARGET-X), so
# the arms never share build state.

require_ab_arms() {
  [ -n "$(arm_env OFF)" ] && [ -n "$(arm_env ON)" ] \
    || die "set HS_ARM_ENV_OFF and HS_ARM_ENV_ON (the A/B stages need both arm environments)"
  [ "$(arm_env OFF)" != "$(arm_env ON)" ] || die "HS_ARM_ENV_OFF and HS_ARM_ENV_ON are identical"
}

ab_sample() { # ab_sample <scenario> <arm> <kind> <round> <step> <aa|-> <prep> <target suffix> -- cmd...
  local scenario=$1 arm=$2 kind=$3 round=$4 step=$5 aa=$6 prep=$7 suffix=$8
  shift 9
  CUR_TARGET="$TARGET-$suffix"
  run_sample "$scenario" "$arm" "$kind" "$round" "$step" "$aa" "$prep" -- "$@"
  CUR_TARGET=""
}

stage_ab() {
  require_setup; require_ab_arms; stage_begin ab
  # The A/A pair runs twice with the OFF environment: the noise floor of the
  # very toolchain and environment that the ON arm is compared against.
  HS_ARM_ENV_A=$(arm_env OFF); HS_ARM_ENV_B=$HS_ARM_ENV_A
  declare -A cmd=(
    [ab_cold_check_ws]="cargo check --workspace --all-targets"
    [ab_cold_build_ws]="cargo build --workspace"
    [ab_cold_test_norun_ws]="cargo test --workspace --no-run"
    [ab_aa_check_ws]="cargo check --workspace --all-targets"
  )
  local id j
  for id in ab_cold_check_ws ab_cold_build_ws ab_cold_test_norun_ws ab_aa_check_ws; do scenario_add "$id" ab "${cmd[$id]}"; done
  for j in $HS_AB_JOBS; do
    case "$j" in *[!0-9]*|"") die "HS_AB_JOBS must list positive integers, got '$j'";; esac
    cmd[ab_cold_build_ws_j$j]="cargo build --workspace -j $j"
    scenario_add "ab_cold_build_ws_j$j" ab "${cmd[ab_cold_build_ws_j$j]}"
  done
  manifest_add '.protocol.ab = {arms: {OFF: $off, ON: $on, A: $off, B: $off},
      rounds: $n, jobs_rounds: $jn, jobs: $jobs,
      order: "per round: check X, build Y, test-no-run X, A/A first, check Y, build X, test-no-run Y, A/A second; round 1 X=OFF Y=ON first=A, swapped every round; job-count builds after the main rounds, OFF/ON order swapped every round",
      target_dirs: "each arm has its own CARGO_TARGET_DIR (<target>-<arm>), removed before every cold attempt"}' \
    --arg off "$(arm_env OFF)" --arg on "$(arm_env ON)" --argjson n "$HS_SAMPLES" \
    --argjson jn "$HS_AB_JOBS_SAMPLES" --arg jobs "$HS_AB_JOBS"
  local round x y a b item sid arm aa words
  for round in $(seq 1 "$HS_SAMPLES"); do
    x=OFF; y=ON; a=A; b=B
    [ $((round % 2)) -eq 0 ] && { x=ON; y=OFF; a=B; b=A; }
    for item in "ab_cold_check_ws:$x" "ab_cold_build_ws:$y" "ab_cold_test_norun_ws:$x" "ab_aa_check_ws:$a" \
                "ab_cold_check_ws:$y" "ab_cold_build_ws:$x" "ab_cold_test_norun_ws:$y" "ab_aa_check_ws:$b"; do
      sid=${item%%:*}; arm=${item#*:}; aa=-
      [ "$sid" = ab_aa_check_ws ] && aa=ab_aa_check_ws
      read -ra words <<< "${cmd[$sid]}"
      ab_sample "$sid" "$arm" ab "$round" 0 "$aa" cold "$arm" -- "${words[@]}"
      ensure_clean "$sid $arm round $round"
    done
    "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
  done
  for j in $HS_AB_JOBS; do
    for round in $(seq 1 "$HS_AB_JOBS_SAMPLES"); do
      x=OFF; y=ON; [ $((round % 2)) -eq 0 ] && { x=ON; y=OFF; }
      read -ra words <<< "${cmd[ab_cold_build_ws_j$j]}"
      for arm in "$x" "$y"; do
        ab_sample "ab_cold_build_ws_j$j" "$arm" ab "$round" 0 - cold "$arm" -- "${words[@]}"
        ensure_clean "ab_cold_build_ws_j$j $arm round $round"
      done
    done
    "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
  done
  rm -rf "$TARGET-OFF" "$TARGET-ON" "$TARGET-A" "$TARGET-B"
  stage_end ab
}

stage_abincr() {
  require_setup; require_ab_arms; stage_begin abincr
  [ -n "$HS_BODY_FILE" ] && [ -n "$HS_BODY_ANCHOR" ] && [ -n "$HS_IFACE_FILE" ] \
    || die "set HS_BODY_FILE, HS_BODY_ANCHOR and HS_IFACE_FILE (see README)"
  [ -f "$SRC/$HS_BODY_FILE" ] || die "no such file in worktree: $HS_BODY_FILE"
  [ -f "$SRC/$HS_IFACE_FILE" ] || die "no such file in worktree: $HS_IFACE_FILE"
  manifest_add '.ab_incremental = {command: "cargo check --workspace --all-targets", steps: $s, series_per_kind: $n,
      order: "prime OFF then ON (each on its own fresh target dir), 3 no-op checks per arm, then body and iface series as in incr; after every edit both arms check the same source, OFF first on odd steps and ON first on even steps; settle check per arm after every revert",
      body: {file: $bf, anchor: $ba}, iface: {file: $if}, reason: $why,
      failure_rule: "a failed ON (or OFF) check is recorded with its exit status and log tail and the series continues; it is never counted as a time"}' \
    --argjson s "$HS_INCR_STEPS" --argjson n "$HS_INCR_SERIES" --arg bf "$HS_BODY_FILE" \
    --arg ba "$HS_BODY_ANCHOR" --arg if "$HS_IFACE_FILE" --arg why "$HS_EDIT_REASON"
  local c=(cargo check --workspace --all-targets) arm id
  for id in abincr_prime abincr_noop_check abincr_body_check abincr_iface_check abincr_settle; do
    scenario_add "$id" abincr "${c[*]}"
  done
  for arm in OFF ON; do
    rm -rf "$TARGET-incr-$arm"
    ab_sample abincr_prime "$arm" prime 0 0 - warm "incr-$arm" -- "${c[@]}"
    if [ "$arm" = OFF ] && [ "$LAST_RC" -ne 0 ]; then die "OFF prime check failed; the A/B incremental stage cannot run"; fi
  done
  local i; for i in 1 2 3; do
    for arm in OFF ON; do ab_sample abincr_noop_check "$arm" incremental 0 "$i" - warm "incr-$arm" -- "${c[@]}"; done
  done
  ensure_clean "abincr noop"
  local series kind k settle_step order
  for series in $(seq 1 "$HS_INCR_SERIES"); do
    settle_step=0
    for kind in body iface; do
      settle_step=$((settle_step + 1))
      for k in $(seq 1 "$HS_INCR_STEPS"); do
        apply_edit "$kind" "$((series * 100 + k))"
        order="OFF ON"; [ $((k % 2)) -eq 0 ] && order="ON OFF"
        for arm in $order; do
          ab_sample "abincr_${kind}_check" "$arm" incremental "$series" "$k" - warm "incr-$arm" -- "${c[@]}"
        done
      done
      revert_edit
      ensure_clean "abincr_${kind} series $series"
      for arm in OFF ON; do
        ab_sample abincr_settle "$arm" settle "$series" "$settle_step" - warm "incr-$arm" -- "${c[@]}"
      done
    done
  done
  ensure_clean abincr
  "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
  rm -rf "$TARGET-incr-OFF" "$TARGET-incr-ON"
  stage_end abincr
}

# Diagnostics normalized for event order: compiler-message records reduced to
# level, code, message and primary spans, then sorted (a multiset compare).
norm_diagnostics() {
  { grep '^{' "$1" || true; } | jq -c 'select(.reason == "compiler-message") | .message
    | {level, code: (.code.code // null), message,
       spans: [.spans[] | select(.is_primary) | {file_name, line_start, column_start}]}' | LC_ALL=C sort
}

stage_abverify() {
  require_setup; require_ab_arms; stage_begin abverify
  local vd="$HS_OUT/verify" arm pkgs=() p
  mkdir -p "$vd"
  for p in $HS_VERIFY_TEST_PKGS; do pkgs+=(-p "$p"); done
  [ ${#pkgs[@]} -gt 0 ] || die "HS_VERIFY_TEST_PKGS is empty"
  local check=(cargo check --workspace --all-targets --message-format=json)
  local test=(cargo test "${pkgs[@]}" --lib --bins --tests -- --test-threads=1)
  scenario_add verify_check_json verify "${check[*]}"
  scenario_add verify_test verify "${test[*]}"
  declare -A crc trc
  HS_KEEP_LOG=1
  for arm in OFF ON; do
    rm -rf "$TARGET-verify-$arm"
    ab_sample verify_check_json "$arm" verify 1 0 - cold "verify-$arm" -- "${check[@]}"
    crc[$arm]=$LAST_RC
    norm_diagnostics "$LAST_LOG" > "$vd/diagnostics-$arm.jsonl"
    { grep '^{' "$LAST_LOG" || true; } | jq -r '.reason' | LC_ALL=C sort | uniq -c | awk '{ print $2, $1 }' > "$vd/message-reasons-$arm.txt"
    ensure_clean "verify check $arm"
    ab_sample verify_test "$arm" verify 1 0 - warm "verify-$arm" -- "${test[@]}"
    trc[$arm]=$LAST_RC
    { grep -E '^test result:|^error' "$LAST_LOG" || true; } | sed 's/; finished in .*//' > "$vd/test-results-$arm.txt"
    ensure_clean "verify test $arm"
  done
  local vrc=null venv="$HS_VERIFY_ENV_ON"
  if [ -n "$venv" ]; then
    scenario_add verify_build_extra verify "cargo build --workspace (ON arm plus HS_VERIFY_ENV_ON)"
    CUR_ENV_EXTRA=$venv
    rm -rf "$TARGET-verify-ONX"
    ab_sample verify_build_extra ON verify 1 0 - cold "verify-ONX" -- cargo build --workspace
    CUR_ENV_EXTRA=""
    vrc=$LAST_RC
    ensure_clean "verify build extra"
  fi
  HS_KEEP_LOG=0
  rm -f "$RAWLOG"/verify_*.log
  rm -rf "$TARGET-verify-OFF" "$TARGET-verify-ON" "$TARGET-verify-ONX"
  local d_off d_on
  d_off=$(sha256sum < "$vd/diagnostics-OFF.jsonl" | cut -d' ' -f1)
  d_on=$(sha256sum < "$vd/diagnostics-ON.jsonl" | cut -d' ' -f1)
  jq -n --argjson coff "${crc[OFF]}" --argjson con "${crc[ON]}" --argjson toff "${trc[OFF]}" --argjson ton "${trc[ON]}" \
    --arg doff "$d_off" --arg don "$d_on" \
    --argjson noff "$(wc -l < "$vd/diagnostics-OFF.jsonl")" --argjson non "$(wc -l < "$vd/diagnostics-ON.jsonl")" \
    --rawfile troff "$vd/test-results-OFF.txt" --rawfile tron "$vd/test-results-ON.txt" \
    --argjson vrc "$vrc" --arg venv "$venv" --arg pk "$HS_VERIFY_TEST_PKGS" \
    '($troff | split("\n") | map(select(length > 0))) as $tl_off
     | ($tron | split("\n") | map(select(length > 0))) as $tl_on
     | {schema: "headstart-verify/v1",
        check_command: "cargo check --workspace --all-targets --message-format=json",
        test_command: ("cargo test -p " + ($pk | split(" ") | join(" -p ")) + " --lib --bins --tests -- --test-threads=1 (no doctests: they run rustdoc, which the toolchain under test may not ship)"),
        normalization: "compiler-message records reduced to {level, code, message, primary spans}, sorted; test result lines with the timing suffix removed",
        arms: {OFF: {check_exit: $coff, diagnostics: $noff, diagnostics_sha256: $doff, test_exit: $toff, test_results: $tl_off},
               ON: {check_exit: $con, diagnostics: $non, diagnostics_sha256: $don, test_exit: $ton, test_results: $tl_on}},
        check_exit_equal: ($coff == $con), diagnostics_equal: ($doff == $don),
        test_exit_equal: ($toff == $ton), test_results_equal: ($tl_off == $tl_on),
        verify_build: (if $venv == "" then null else {env: $venv, command: "cargo build --workspace", exit: $vrc} end)}
     | .verdict = (if .check_exit_equal and .diagnostics_equal and .test_exit_equal and .test_results_equal
                      and (.verify_build == null or .verify_build.exit == 0)
                      and $coff == 0 and $toff == 0 then "match" else "MISMATCH" end)' > "$HS_OUT/verify.json"
  say "verify: $(jq -c '{verdict, check_exit_equal, diagnostics_equal, test_exit_equal, test_results_equal, verify_build}' "$HS_OUT/verify.json")"
  stage_end abverify
}

stage_summary() {
  "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
  say "summary written: $HS_OUT/summary.json"
}

stage_cleanup() {
  say "cleanup: removing $TARGET (and per-arm $TARGET-*) and worktree $SRC"
  rm -rf "$TARGET" "$TARGET"-*
  if [ -d "$SRC" ]; then
    ensure_clean cleanup
    git -C "$REPO" worktree remove --force "$SRC"
  fi
  rm -rf "$RAWLOG"
  rmdir "$HS_WORK" 2>/dev/null || true
  [ -s "$MANIFEST" ] && manifest_add '.cleanup = {at: $t, removed: [$w]}' \
    --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg w "$HS_WORK"
  return 0
}

CUR_STAGE=""
STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(setup timings cold incr cleanup)
mkdir -p "$HS_OUT"
for s in "${STAGES[@]}"; do
  case "$s" in
    setup|timings|cold|incr|ab|abincr|abverify|summary|cleanup) CUR_STAGE=$s; "stage_$s" ;;
    *) die "unknown stage: $s" ;;
  esac
done
[ -s "$SAMPLES" ] && "$SUMMARIZE" samples "$SAMPLES" > "$HS_OUT/summary.json"
say "done: ${STAGES[*]}"
