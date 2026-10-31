#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DRA-65 chip window. The single chip queue (~/handoffs/pdra4-hive/chip/run-queue.sh) runs it through the slices
# chip/DRA-65/s01-golden.sh (steps 1-5), s02-time-single.sh (6a 6b), s03-time-batch.sh (6c 6d); each carries
# the header `# pdra4-gb10: none`, so the runner opens `spark-window open ... --gb10 none --quiet exclusive
# --lease 20` (18091 and Flint stopped by the runner's preflight) and the slice runs
#   spark-window run bash window-dra65.sh run BIN OUT STEP [STEP ...]
# BIN = the server binary built from this branch, OUT = a receipt root shared by every lease (step N writes
# OUT/N/). The server's chip reservation (aien-serve-http chip_reservation.rs) only accepts /tmp/aien-gb10.lock
# held by one of its ancestors, so this script takes `flock -o` on it itself (the lock stays in flock(1), an
# ancestor; no descendant inherits the fd) and starts every server as its child, in its own session
# (setsid, pid file OUT/.server.<run pid>.pid), with AIEN_GB10_LOCK_HELD=1 and the affinity reset of the live start scripts.
# DRA65_LOCK_WAIT=N: wait at most N s for the lock (default 0: no wait). A TERM/INT/HUP to this script ends
# flock(1): the lock is released and the watchdog stops the server group and the run.
# Exit 3 = NOT_RUN, chip untouched: no SPARK_WINDOW, a window that holds the chip lock itself
# (SPARK_WINDOW_DIR/gb10 exclusive|shared), the lock held by someone else, a listener on
# 18091/18300/18310 or the test port, a compute app on the chip, the binary or model missing, a green step
# without passing red steps (listed before it, or their step receipts in OUT/1 and OUT/2), or the steps'
# estimated time longer than the lease has left.
# Lease: deadline = SPARK_WINDOW_DIR/held + lease * SPARK_LEASE_UNIT_S (default 60), or held + DRA65_CAP_S when
# that is sooner (queue slices: 1060, under the runner's 1140 s). Every server wait and
# probe is bounded by the time left (timeout, curl -m). A watchdog in its own session stops the server group
# at the deadline (and this script if it has not ended 40 s later), or at once when this script or the lock
# holder dies; the lock is then released.
# A step that ends after the deadline or with SPARK_WINDOW_DIR/expired present is INCONCLUSIVE (rc 2).
#   window-dra65.sh plan     (no window needed: the steps, their server env and estimated durations)
# Every receipt line carries the window id and the binary sha256.
#
# Probes (an already running server on PORT; `run` calls them, the host checks them against a mock door):
#   window-dra65.sh golden PORT OUT red   BIN_SHA256 LABEL
#   window-dra65.sh golden PORT OUT green BIN_SHA256 LABEL RED_RECEIPT RED_RECEIPT [...]
#       36 cases (32-prompt set, 3 long prompts, turn 2 restoring the long-2048 snapshot) vs the pinned goldens.
#       green refuses to run unless it is handed at least two red receipts (verdict PASS, distinct labels) of
#       the same server binary sha256, each next to a step-receipt.json of `run` with rc 0, the same
#       bin_sha256, a lock holder and the lease not expired: red before green.
#       With DRA65_SERVER_LOG (set by `run`), turn2 also needs the door's snapshot-restore log line for that
#       request (single: `reuse Snapshot(`, qwen38_engine.rs; batch: `tok (snapshot, row`, qwen36_batch.rs).
#   window-dra65.sh time PORT OUT BIN_SHA256 LABEL
#       TTFT and end-to-end tok/s, ~512 and 8210-token prompts, 3 runs each.
# Per case: SAME (ids equal), DIFF (a flip inside the common length, or more ids than the golden),
# SHORT (a strict prefix of the golden: no flip shown), NO_IDS (HTTP error, missing or empty ids),
# NO_SNAP (turn2 without the snapshot-restore log line).
# golden red:   PASS = at least one DIFF and no NO_IDS/SHORT/NO_SNAP; those -> INCONCLUSIVE; else FAIL.
# golden green: PASS = all SAME; any NO_IDS -> INCONCLUSIVE; else (NO_SNAP included) FAIL.
# Exit 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE or refused, 3 = NOT_RUN, 4 = interrupted. time: 0 only if every run
# has HTTP 200, a first content delta and usage with completion_tokens > 0; else 2. `run` stops at the first
# step that does not exit 0 and returns its code.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); SELF=$HERE/$(basename "$0")
die() { echo "window-dra65: $*" >&2; exit 2; }
not_run() { echo "window-dra65: NOT_RUN: $*" >&2; exit 3; }
# left_s [CAP]: seconds to DRA65_DEADLINE (unset: CAP), at most CAP; returns 1 when none are left.
left_s() {
  local cap=${1:-999999} l
  if [ -n "${DRA65_DEADLINE:-}" ]; then l=$((DRA65_DEADLINE - $(date +%s))); [ "$l" -lt "$cap" ] || l=$cap; else l=$cap; fi
  echo "$l"; [ "$l" -gt 0 ]
}

# step -> door, server env, probe, estimated seconds. MODELED (Wolfram, README "Window"): decode 0.0201 s/token,
# chunked prompts 250 tok/s (8210: 196.8), stepped prompts 0.0201 s/token; upper bounds at max_tokens; each
# includes ~33 s server start.
step_def() {
  ENVS=(); case $1 in
  1)  DOOR=single; ENVS=(AIEN_Q36_TEST_MUTANT=4);                                PROBE="golden red mutant4";       EST=120 ;;
  2)  DOOR=single; ENVS=(AIEN_OMEGA_Q36_PF_MUTANT=2);                            PROBE="golden red pf2";           EST=120 ;;
  3)  DOOR=single; ENVS=(AIEN_OMEGA_Q36_PF_MUTANT=2 AIEN_Q36_PREFILL_CHUNK=0);   PROBE="golden green pf2-chunk0";  EST=289 ;;
  4)  DOOR=single;                                                               PROBE="golden green single";      EST=120 ;;
  5)  DOOR=batch;                                                                PROBE="golden green batch";       EST=120 ;;
  6a) DOOR=single;                                                               PROBE="time single";              EST=195 ;;
  6b) DOOR=single; ENVS=(AIEN_Q36_PREFILL_CHUNK=0);                              PROBE="time single-chunk0";       EST=590 ;;
  6c) DOOR=batch;                                                                PROBE="time batch";               EST=195 ;;
  6d) DOOR=batch;  ENVS=(AIEN_Q36_PREFILL_MIN=1);                                PROBE="time batch-min1";          EST=195 ;;
  *) return 1 ;;
  esac
}
# red_ok OUT STEP SHA: that red step's run receipt passed for this binary inside the lease.
red_ok() {
  jq -e --arg b "$3" '.rc == 0 and .bin_sha256 == $b and (.lock_holder | length) > 0 and .lease_expired == false
    and .result.case == "golden" and .result.expect == "red" and .result.verdict == "PASS"' "$1/$2/step-receipt.json" > /dev/null 2>&1
}

case ${1:-} in
plan)
  printf '%-4s %-7s %-55s %-24s %s\n' step door "server env" probe est_s
  for s in 1 2 3 4 5 6a 6b 6c 6d; do
    step_def $s; printf '%-4s %-7s %-55s %-24s %s\n' $s $DOOR "${ENVS[*]:--}" "$PROBE" $EST
  done
  echo "Steps 1-2 (red) come first; 3-5 need them earlier in the same run or their step receipts in OUT/1 and OUT/2."
  echo "Queue slices (chip/DRA-65, cap 1060 s each): s01-golden = 1 2 3 4 5 (769 s), s02-time-single = 6a 6b (785 s), s03-time-batch = 6c 6d (390 s)."
  exit 0 ;;
_watch) # _watch MAIN HOLDER DEADLINE PIDFILE (own session; started by run)
  M=$2; H=$3; DL=$4; PF=$5
  while kill -0 "$M" 2>/dev/null && kill -0 "$H" 2>/dev/null && [ "$(date +%s)" -lt "$DL" ]; do sleep 1; done
  why="deadline"; kill -0 "$M" 2>/dev/null || why="run shell $M gone"; kill -0 "$H" 2>/dev/null || why="lock holder $H gone"
  pg=$(cat "$PF" 2>/dev/null)
  if [ -n "$pg" ] && kill -0 -- "-$pg" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) watchdog: $why: TERM server group $pg"
    kill -TERM -- "-$pg" 2>/dev/null
    for _ in $(seq 20); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pg" 2>/dev/null; rm -f "$PF"
  fi
  # At the deadline the run shell's own bounds (probe timeout, stop_server) end the step as INCONCLUSIVE and it
  # exits: give it 40 s to write that receipt. Its exit ends flock(1), which releases the lock.
  if [ "$why" = deadline ]; then for _ in $(seq 40); do kill -0 "$M" 2>/dev/null || break; sleep 1; done; fi
  kill -0 "$M" 2>/dev/null && { echo "$(date -u +%FT%TZ) watchdog: $why: TERM run shell $M"; kill -TERM "$M"; }
  exit 0 ;;
run)
  [ -n "${SPARK_WINDOW:-}" ] || not_run "no SPARK_WINDOW: run as 'spark-window run bash $0 run ...' inside a --quiet exclusive window"
  [ $# -ge 4 ] || die "usage: run BIN OUT STEP [STEP ...]"
  # DRA65_TEST_LOCK (host check only): another lock file; the binary must be a script (a mock door, never the
  # ELF server), the chip and live-port checks are skipped and board lines go to OUT/board.log.
  LOCK=${DRA65_TEST_LOCK:-/tmp/aien-gb10.lock}; TEST=${DRA65_TEST_LOCK:+1}
  # A window that holds the chip lock itself (--gb10 exclusive|shared): its holder is never our ancestor, so the
  # server would refuse; and we would wait on the lock until the window closes. Refuse before any wait.
  g=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/gb10" 2>/dev/null)
  case $g in exclusive|shared) not_run "window $SPARK_WINDOW holds the chip lock (gb10 $g): open it with --gb10 none (queue slice header '# pdra4-gb10: none')" ;; esac
  if [ "${DRA65_LOCKED:-}" != "$SPARK_WINDOW" ]; then
    # flock(1) holds the lock in its own process (-o: closed before exec, so no descendant inherits it) and
    # runs us as its child: the holder is an ancestor of this shell and of every server it starts.
    # DRA65_LOCK_WAIT=N (seconds, queue slices): wait at most N s for the lock; unset or 0: do not wait.
    lw=${DRA65_LOCK_WAIT:-0}; case $lw in ''|*[!0-9]*) die "DRA65_LOCK_WAIT=$lw: whole seconds" ;; esac
    wflag=(-n); [ "$lw" -gt 0 ] && wflag=(-w "$lw")
    # A signal to this shell (the queue runner's timeout) ends flock(1): the lock is released and the
    # watchdog, seeing its holder gone, stops the server group and the run shell.
    DRA65_LOCKED=$SPARK_WINDOW flock "${wflag[@]}" -o -E 75 "$LOCK" bash "$SELF" "$@" & FL=$!
    trap 'kill -TERM $FL 2>/dev/null' INT TERM HUP
    while :; do wait $FL; rc=$?; kill -0 $FL 2>/dev/null || break; done
    [ $rc = 75 ] && not_run "$LOCK is held by another process (waited $lw s); this script takes it itself (open the window with --gb10 none)"
    exit $rc
  fi
  BIN=$(readlink -f "$2"); OUT=$(readlink -m "$3"); shift 3; STEPS=("$@")
  WIN=$SPARK_WINDOW
  # 1. An exclusive flock on LOCK held by an ancestor of this shell (/proc/locks: "N: FLOCK ADVISORY WRITE pid maj:min:ino").
  ino=$(stat -c %i "$LOCK") || not_run "$LOCK missing"
  holders=$(awk -v ino="$ino" '$2 == "FLOCK" && $4 == "WRITE" { split($6, a, ":"); if (a[3] == ino) print $5 }' /proc/locks | sort -u)
  anc=" "; p=$$
  while [ "$p" -gt 1 ]; do anc="$anc$p "; p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break; done
  HOLDER=; for h in $holders; do case "$anc" in *" $h "*) HOLDER=$h ;; esac; done
  [ -n "$HOLDER" ] || not_run "$LOCK held by pid(s) '${holders}', none an ancestor of $$ ($anc)"
  if [ -n "$TEST" ]; then
    [ -x "$BIN" ] && [ "$(head -c 4 "$BIN" | od -An -c | tr -d ' ')" != '177ELF' ] \
      || not_run "test lock $LOCK held by ancestor $HOLDER; DRA65_TEST_LOCK only runs a script mock door, never the server binary"
  fi
  [ -x "$BIN" ] || not_run "binary $BIN missing"
  SHA=$(sha256sum < "$BIN" | cut -c1-64)
  # 2. Steps known; each green step after both red steps in this run, or their passing step receipts in OUT.
  total=0; seen1=0; seen2=0
  for s in "${STEPS[@]}"; do
    step_def "$s" || die "unknown step $s (see: $0 plan)"
    [ -n "$TEST" ] && EST=5
    total=$((total + EST))
    case $s in
    1) seen1=1 ;; 2) seen2=1 ;;
    3|4|5) { [ $seen1 = 1 ] || red_ok "$OUT" 1 "$SHA"; } && { [ $seen2 = 1 ] || red_ok "$OUT" 2 "$SHA"; } \
      || not_run "green step $s needs red steps 1 and 2 before it in this run, or their passing step receipts in $OUT/1 and $OUT/2 for bin $SHA" ;;
    esac
  done
  # 3. The lease has room for the steps' estimates.
  lease=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/lease" 2>/dev/null); held=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/held" 2>/dev/null)
  [ -n "$lease" ] && [ -n "$held" ] || not_run "no lease/held in SPARK_WINDOW_DIR='${SPARK_WINDOW_DIR:-}'"
  [ -e "$SPARK_WINDOW_DIR/expired" ] && not_run "the lease has expired ($SPARK_WINDOW_DIR/expired)"
  heldt=$(date -d "$held" +%s) || not_run "cannot read held time '$held'"
  DEADLINE=$(( heldt + lease * ${SPARK_LEASE_UNIT_S:-60} ))
  # DRA65_CAP_S=N (queue slices): also end N s after `held`, before the queue runner's own cap stops the slice.
  cap=${DRA65_CAP_S:-}; case $cap in '') ;; *[!0-9]*) die "DRA65_CAP_S=$cap: whole seconds" ;;
    *) [ $((heldt + cap)) -ge "$DEADLINE" ] || DEADLINE=$((heldt + cap)) ;; esac
  export DRA65_DEADLINE=$DEADLINE
  left=$(( DEADLINE - $(date +%s) ))
  [ "$total" -le "$left" ] || not_run "steps ${STEPS[*]} estimate $total s, the lease has $left s left"
  # 4. Live 35B, Flint and the chip idle; model, port, tools there.
  PORT=${DRA65_TEST_PORT:-18192}; [ -n "$TEST" ] || PORT=18192
  MODEL=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
  for t in curl jq ss flock setsid timeout sha256sum; do command -v $t > /dev/null || not_run "$t missing"; done
  [ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "a listener is on $PORT"
  if [ -z "$TEST" ]; then
    command -v nvidia-smi > /dev/null || not_run "nvidia-smi missing"
    for port in 18091 18300 18310; do [ -z "$(ss -ltnH "sport = :$port")" ] || not_run "a listener is on $port"; done
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) || not_run "nvidia-smi failed: $apps"
    [ -z "$apps" ] || not_run "compute apps on the chip: $apps"
    [ -f "$MODEL/config.json" ] || not_run "model $MODEL missing"
  fi
  mkdir -p "$OUT" || not_run "cannot create $OUT"
  PIDF=$OUT/.server.$$.pid   # per run: a previous run's watchdog never sees this run's server
  log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log" >&2; }
  board() {
    if [ -n "$TEST" ]; then echo "$(date -u +%FT%TZ) DRA65Prefill: $*" >> "$OUT/board.log"
    else echo "$(date -u +%FT%TZ) DRA65Prefill: $*" >> "$HOME/handoffs/BOARD.md"; fi
  }
  SERVER=
  stop_server() { # TERM the server's process group (its own session), KILL after 20 s
    [ -n "$SERVER" ] || return 0
    kill -TERM -- "-$SERVER" 2>/dev/null
    for _ in $(seq 20); do kill -0 -- "-$SERVER" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; rm -f "$PIDF"; SERVER=
  }
  # start_server LOG: DOOR and ENVS from step_def; the live start-35b(-batch).sh flags on a loopback test port.
  start_server() {
    local flags=(--max-context 131072) aff cpus lim l
    [ "$DOOR" = batch ] && flags=(--max-context 49152 --batch-slots 8)
    # As start-35b(-batch).sh: undo an inherited affinity (a core5/nice19 parent) so the server gets every core
    # (test mode: cores 0-9, the host budget).
    cpus=$(cat /sys/devices/system/cpu/online); [ -n "$TEST" ] && cpus=0-9
    aff=(taskset -c "$cpus"); "${aff[@]}" true 2> /dev/null || aff=()
    # A background subshell is not a group leader, so setsid(1) does not fork: SERVER is the session and group id.
    (cd "$(dirname "$BIN")" && exec setsid "${aff[@]}" env AIEN_GB10_LOCK_HELD=1 AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-$PORT.lock \
      AIEN_Q36_STROKE=0 "${ENVS[@]}" "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
      --max-tokens-cap 16384 --thinking off "${flags[@]}") > "$1" 2>&1 < /dev/null &
    SERVER=$!; echo "$SERVER" > "$PIDF"
    lim=$(left_s 300) || { log "no lease time left to start the server"; stop_server; return 1; }
    for _ in $(seq "$lim"); do
      kill -0 "$SERVER" 2>/dev/null || { log "server exited during load: $1"; wait "$SERVER" 2>/dev/null; rm -f "$PIDF"; SERVER=; return 1; }
      curl -s -m 2 127.0.0.1:$PORT/health > /dev/null && break; sleep 1
    done
    curl -s -m 2 127.0.0.1:$PORT/health > /dev/null || { log "server not healthy in $lim s"; stop_server; return 1; }
    [ "$(ps -o sid= -p "$SERVER" | tr -d ' ')" = "$SERVER" ] || { log "server $SERVER is not its own session leader"; stop_server; return 1; }
    l=$(ss -ltnH "sport = :$PORT" | awk '{print $4}')
    if [ -z "$l" ] || echo "$l" | grep -qv '^127\.0\.0\.1:'; then log "listener check FAILED on $PORT: $l"; stop_server; return 1; fi
  }
  trap 'stop_server; board "GPU-END dra-65 window-dra65.sh window $WIN steps ${STEPS[*]} -> $OUT (rc ${RC:-?})"' EXIT
  trap 'log "interrupted"; RC=4; exit 4' INT TERM
  setsid bash "$SELF" _watch $$ "$HOLDER" "$DEADLINE" "$PIDF" >> "$OUT/watch.log" 2>&1 < /dev/null &
  board "GPU-START dra-65 window-dra65.sh window $WIN steps ${STEPS[*]} (est $total s, lease left $left s; gb10 lock held by our pid $HOLDER; bin $SHA)"
  log "window $WIN lock holder $HOLDER bin $BIN sha256 $SHA steps ${STEPS[*]} est $total s left $left s deadline $(date -u -d @$DEADLINE +%FT%TZ)"
  RC=0
  for s in "${STEPS[@]}"; do
    step_def "$s"; [ -n "$TEST" ] && EST=5
    D=$OUT/$s; mkdir -p "$D"; t0=$(date +%s); why=
    log "step $s: $DOOR door, env '${ENVS[*]:-}', $PROBE, est $EST s"
    if [ $((t0 + EST)) -gt "$DEADLINE" ]; then
      ready=-1; RC=3; why="not started: est $EST s, $((DEADLINE - t0)) s left"; : > "$D/probe.out"
    elif start_server "$D/server.log"; then
      ready=$(( $(date +%s) - t0 ))
      read -r kind a1 a2 <<<"$PROBE"
      if [ "$kind" = golden ]; then
        args=(golden $PORT "$D" "$a1" "$SHA" "$a2")
        [ "$a1" = green ] && args+=("$OUT/1/golden-receipt.json" "$OUT/2/golden-receipt.json")
      else
        args=(time $PORT "$D" "$SHA" "$a1")
      fi
      if lim=$(left_s); then
        DRA65_SERVER_LOG=$D/server.log timeout -k 10 "$lim" bash "$SELF" "${args[@]}" > "$D/probe.out"; RC=$?
        case $RC in 124|137) RC=2; why="probe stopped at the lease deadline" ;; esac
      else
        RC=2; why="no lease time left for the probe"
      fi
      stop_server
    else
      ready=-1; RC=3; why="server did not come up"
    fi
    el=$(( $(date +%s) - t0 )); expired=false
    if [ -e "$SPARK_WINDOW_DIR/expired" ] || [ "$(date +%s)" -ge "$DEADLINE" ]; then
      expired=true; [ $RC = 3 ] || { RC=2; why="lease expired before the step ended${why:+ ($why)}"; }
    fi
    pj=$(tail -n 1 "$D/probe.out" 2>/dev/null | jq -c . 2>/dev/null); [ -n "$pj" ] || pj=null
    case $RC in 0) v=$(jq -r '.verdict // "PASS"' <<<"$pj") ;; 1) v=FAIL ;; 2) v=INCONCLUSIVE ;; 3) v=NOT_RUN ;; *) v=INCONCLUSIVE ;; esac
    jq -cn --arg s "$s" --arg w "$WIN" --arg b "$BIN" --arg h "$SHA" --arg d "$DOOR" --arg e "${ENVS[*]:-}" --arg p "$PROBE" \
      --argjson est $EST --argjson ready $ready --argjson el $el --argjson rc $RC --arg hold "$HOLDER" --arg at "$(date -u +%FT%TZ)" \
      --argjson probe "$pj" --argjson x $expired --arg v "$v" --arg why "$why" --argjson dl "$DEADLINE" \
      '{step:$s, window:$w, bin:$b, bin_sha256:$h, door:$d, env:$e, probe:$p, est_s:$est, server_ready_s:$ready, elapsed_s:$el,
        rc:$rc, verdict:$v, why:$why, lease_expired:$x, deadline:$dl, lock_holder:$hold, at:$at, result:$probe}' \
      | tee "$D/step-receipt.json"
    [ $RC = 0 ] || { log "step $s rc $RC ($v${why:+: $why}): stopping"; exit $RC; }
  done
  exit 0 ;;
esac

[ -n "${SPARK_WINDOW:-}" ] || not_run "no SPARK_WINDOW: the probes run from '$0 run' inside a window"
WIN=$SPARK_WINDOW
MODE=${1:-}; PORT=${2:-}; OUT=${3:-}
[ -n "$MODE" ] && [ -n "$PORT" ] && [ -n "$OUT" ] || die "usage: see the header"
URL=http://127.0.0.1:$PORT
(cd "$REPO" && sha256sum -c --quiet "$HERE/SHA256SUMS") || die "pinned inputs changed"
mkdir -p "$OUT"
MODEL=$(curl -s -m 10 "$URL/v1/models" | jq -r '.data[0].id // empty' 2>/dev/null)
[ -n "$MODEL" ] || die "no model on $URL"
DEF=$(jq -c .defaults "$HERE/set32.json")
SET=$HERE/set32.json; LONG=$HERE/long-goldens.json
req() { # <messages json> <max_tokens> [stream]
  jq -cn --arg m "$MODEL" --argjson msg "$1" --argjson n "$2" --argjson d "$DEF" --argjson s "${3:-false}" \
    '$d + {model:$m, messages:$msg, max_tokens:$n} + (if $s then {stream:true, stream_options:{include_usage:true}} else {} end)'
}
user() { jq -cn --arg u "$1" '[{role:"user",content:$u}]'; }
# ask <messages> <max_tokens> <raw file>: prints the HTTP code (000 without lease time left)
ask() {
  local m; m=$(left_s 1800) || { : > "$3"; echo 000; return; }
  curl -s -m "$m" -o "$3" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$(req "$1" "$2")"
}
judge() { # <http code> <raw file> <golden ids json>: prints SAME|DIFF|SHORT|NO_IDS
  local got; [ "$1" = 200 ] || { echo NO_IDS; return; }
  got=$(jq -c '.usage.aien_token_ids | select(type == "array" and length > 0)' "$2" 2>/dev/null)
  [ -n "$got" ] || { echo NO_IDS; return; }
  jq -rn --argjson g "$got" --argjson w "$3" '
    ([$g, $w] | map(length) | min) as $m
    | if $g == $w then "SAME"
      elif ([range(0; $m) | select($g[.] != $w[.])] | length) > 0 then "DIFF"
      elif ($g | length) > ($w | length) then "DIFF"
      else "SHORT" end'
}

case $MODE in
golden)
  EXPECT=${4:-}; BIN=${5:-}; LABEL=${6:-}
  case $EXPECT in red|green) ;; *) die "golden needs red|green";; esac
  [[ $BIN =~ ^[0-9a-f]{64}$ ]] || die "BIN_SHA256 must be the server binary's sha256"
  [ -n "$LABEL" ] || die "LABEL missing"
  if [ "$EXPECT" = green ]; then
    # A red receipt counts only next to run's step receipt for it: rc 0, same binary, lock held, lease not expired.
    shift 6; reds=$(for r in "$@"; do
      d=$(dirname "$r"); red_ok "$(dirname "$d")" "$(basename "$d")" "$BIN" || continue
      jq -r --arg b "$BIN" 'select(.case == "golden" and .expect == "red" and .verdict == "PASS" and .bin_sha256 == $b) | .label' "$r" 2>/dev/null
    done | sort -u | wc -l)
    [ "$reds" -ge 2 ] || die "green refused: needs >= 2 red PASS receipts (distinct labels) for binary $BIN, each with a passing step-receipt.json of 'run', got $reds"
  fi
  : > "$OUT/golden.jsonl"; declare -A n=([SAME]=0 [DIFF]=0 [SHORT]=0 [NO_IDS]=0 [NO_SNAP]=0)
  rec() { jq -cn --arg id "$1" --arg v "$2" --arg h "$3" --argjson w "$4" --arg win "$WIN" --arg b "$BIN" --arg sn "${6:-}" \
    --slurpfile r <(jq -c . "$5" 2>/dev/null || echo null) \
    '{id:$id, verdict:$v, http:$h, got:($r[0].usage.aien_token_ids // null), golden:$w, window:$win, bin_sha256:$b}
     + (if $sn == "" then {} else {snapshot_log:$sn} end)' >> "$OUT/golden.jsonl"; }
  for i in $(seq 0 31); do
    u=$(jq -r ".prompts[$i].user" "$SET"); m=$(jq -r ".prompts[$i].max_tokens" "$SET"); w=$(jq -c ".prompts[$i].golden_ids" "$SET")
    h=$(ask "$(user "$u")" "$m" "$OUT/p$i.raw"); v=$(judge "$h" "$OUT/p$i.raw" "$w"); n[$v]=$((n[$v]+1)); rec "p$i" "$v" "$h" "$w" "$OUT/p$i.raw"
  done
  # turn2 right after long-2048: the door restores the snapshot it took at long-2048's answer header; the
  # server log lines written during the turn2 request must show it (single and batch door wording).
  SLOG=${DRA65_SERVER_LOG:-}
  for name in long-256 long-8192 long-2048 turn2; do
    c=$(jq -c --arg n "$name" '.prompts[] | select(.name == $n)' "$LONG"); w=$(jq -c .golden_ids <<<"$c"); m=$(jq -r .max_tokens <<<"$c")
    if [ "$name" = turn2 ]; then msg=$(jq -c . "$REPO/$(jq -r .messages <<<"$c")"); else msg=$(user "$(cat "$REPO/$(jq -r .file <<<"$c")")"); fi
    before=0; [ -n "$SLOG" ] && before=$(wc -l < "$SLOG")
    h=$(ask "$msg" "$m" "$OUT/$name.raw"); v=$(judge "$h" "$OUT/$name.raw" "$w"); sn=
    if [ "$name" = turn2 ]; then
      [ -n "$SLOG" ] && sn=$(tail -n "+$((before + 1))" "$SLOG" | grep -E 'reuse Snapshot\(|tok \(snapshot, row' | tail -n 1)
      [ -n "$sn" ] || { sn="(none: ${SLOG:-no DRA65_SERVER_LOG})"; [ "$v" = NO_IDS ] || v=NO_SNAP; }
    fi
    n[$v]=$((n[$v]+1)); rec "$name" "$v" "$h" "$w" "$OUT/$name.raw" "$sn"
  done
  if [ "$EXPECT" = red ]; then
    if [ $((n[NO_IDS] + n[SHORT] + n[NO_SNAP])) -gt 0 ]; then verdict=INCONCLUSIVE; rc=2
    elif [ "${n[DIFF]}" -gt 0 ]; then verdict=PASS; rc=0; else verdict=FAIL; rc=1; fi
  else
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2
    elif [ "${n[SAME]}" -eq 36 ]; then verdict=PASS; rc=0; else verdict=FAIL; rc=1; fi
  fi
  jq -cn --arg e "$EXPECT" --arg v $verdict --arg b "$BIN" --arg l "$LABEL" --argjson s "${n[SAME]}" --argjson d "${n[DIFF]}" \
    --argjson sh "${n[SHORT]}" --argjson x "${n[NO_IDS]}" --argjson ns "${n[NO_SNAP]}" --arg port "$PORT" --arg at "$(date -u +%FT%TZ)" --arg win "$WIN" \
    '{case:"golden", expect:$e, verdict:$v, bin_sha256:$b, label:$l, window:$win, same:$s, diff:$d, short:$sh, no_ids:$x, no_snap:$ns, of:36, port:$port, at:$at}' \
    | tee "$OUT/golden-receipt.json"
  exit $rc ;;
time)
  BIN=${4:-}; LABEL=${5:-}
  [[ $BIN =~ ^[0-9a-f]{64}$ ]] || die "BIN_SHA256 must be the server binary's sha256"
  [ -n "$LABEL" ] || die "LABEL missing"
  # TTFT = first content delta - t0 (client clock); e2e tok/s = completion_tokens / (t_end - t_first).
  # Medians of the 3 runs go through Wolfram afterwards (README).
  : > "$OUT/time.jsonl"; ok=1
  for name in long-512 long-8192; do
    f=$HERE/$name.txt; [ -f "$f" ] || f=$REPO/evidence/merge-train/l369/$name.txt
    for run in 1 2 3; do
      o=$OUT/$name-r$run; : > "$o.sse"; t0=$(date +%s.%N); first=""
      if m=$(left_s 1800); then
        while IFS= read -r line; do
          [ -z "$first" ] && [[ $line =~ ^data:.*\"content\":\ *\"[^\"] ]] && first=$(date +%s.%N)
          printf '%s\n' "$line" >> "$o.sse"
        done < <(curl -s -N -m "$m" -w '\nHTTP %{http_code}\n' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
                   -d "$(req "$(user "$(cat "$f")")" 256 true)")
      fi
      t1=$(date +%s.%N)
      code=$(grep -o '^HTTP [0-9]*' "$o.sse" | tail -n 1 | cut -c6-)
      usage=$(grep '^data: {' "$o.sse" | sed 's/^data: //' | jq -c 'select(.usage != null) | .usage' 2>/dev/null | tail -n 1)
      good=$(jq -n --arg c "${code:-}" --arg f "$first" --argjson u "${usage:-null}" \
        '$c == "200" and $f != "" and ($u | type) == "object" and (($u.completion_tokens // 0) > 0)')
      [ "$good" = true ] || ok=0
      jq -cn --arg n "$name" --argjson r $run --arg b "$BIN" --arg l "$LABEL" --arg h "${code:-}" --arg t0 "$t0" --arg tf "$first" \
        --arg t1 "$t1" --argjson u "${usage:-null}" --argjson g "$good" --arg win "$WIN" \
        '{prompt:$n, run:$r, bin_sha256:$b, label:$l, window:$win, http:$h, ok:$g, t0:$t0, t_first:$tf, t_end:$t1, usage:$u}' | tee -a "$OUT/time.jsonl"
    done
  done
  if [ $ok = 1 ]; then jq -cn --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" '{case:"time", verdict:"RECORDED", window:$win, bin_sha256:$b, label:$l}'; exit 0; fi
  jq -cn --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" '{case:"time", verdict:"INCONCLUSIVE", window:$win, bin_sha256:$b, label:$l, why:"a run had no HTTP 200, no first content delta or no usage"}'; exit 2 ;;
*) die "mode plan|run|golden|time" ;;
esac
