#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DRA-86 chip window (G4: the shared prompt cache of the batch door, AIEN_Q36_PREFIX=1). Adapted from
# evidence/dra65-prefill-20261031/window-dra65.sh (#862): the same lock, lease, watchdog and receipt machinery.
# The single chip queue (~/handoffs/pdra4-hive/chip/run-queue.sh) runs it through the slices
# chip/DRA-86/s01-identity.sh (steps 1 2 3), s02-controls.sh (4a 4b 4c 5), s03-time-warm.sh (6a), s04-time-cold.sh (6b); each carries the
# header `# pdra4-gb10: none`, so the runner opens `spark-window open ... --gb10 none --quiet exclusive --lease 20`
# (18091 and Flint stopped by the runner's preflight) and the slice runs
#   spark-window run bash window-dra86.sh run BIN OUT STEP [STEP ...]
# BIN = the server binary built from this branch, OUT = a receipt root shared by every lease (step N writes OUT/N/).
# The server's chip reservation only accepts /tmp/aien-gb10.lock held by one of its ancestors, so this script takes
# `flock -o` on it itself and starts every server as its child, in its own session (setsid, pid file
# OUT/.server.<run pid>.pid), with AIEN_GB10_LOCK_HELD=1 and the affinity reset of the live start scripts.
# DRA86_LOCK_WAIT=N: wait at most N s for the lock (default 0). DRA86_CAP_S=N: end N s after `held` (slices: 1060).
# Exit 3 = NOT_RUN, chip untouched: no SPARK_WINDOW, a window that holds the chip lock itself, the lock held by
# someone else, a listener on 18091/18300/18310 or the test port, a compute app on the chip, the binary or model
# missing, a step without the steps it needs (below), or the steps' estimate longer than the lease has left.
# A step that ends after the deadline or with SPARK_WINDOW_DIR/expired present is INCONCLUSIVE (rc 2).
#   window-dra86.sh plan     (no window needed: the steps, their server env and estimated durations)
#
# Steps (all on the batch door, one request at a time, greedy, thinking off):
#   1  ref     prompt cache OFF (no AIEN_Q36_PREFIX): 35 conversations (set32 + l369 long-256/2048/8192), turn 1
#              and turn 2 (turn 1 + its answer + a fixed follow-up). Their ids are the references (same binary).
#              The log must say "prompt cache off" (the switch defaults off).
#   2  red     AIEN_Q36_PREFIX=1 AIEN_Q36_PREFIX_MUTANT=1 (a restored row continues one position early): PASS =
#              at least one turn 2 differs from step 1, every turn 2 restored (log line), no NO_IDS/SHORT.
#   3  green   AIEN_Q36_PREFIX=1 (default budget): PASS = every turn 1 and turn 2 equals step 1, every turn 2
#              restored from the cache and none refused, the log says "budget 16.0 GiB". Needs 1 and 2 (red).
#   4a write   AIEN_Q36_PREFIX=1 + sealed store OUT/4a/store (a key made for this run, never written down):
#              long-2048 turn 1 only; the store must hold its nodes. 4b and 4c must follow in the same run.
#   4b reload  same env: the store loads >= 1 node; long-2048 turn 2 is restored from it and equals step 1.
#   4c other   same env + AIEN_Q36_NS_PROBE=1 (a numeric env key: another namespace = another model/lib/env):
#              the store discards every node (0 loaded, >= 1 discarded); turn 2 reads cold and equals step 1.
#   5  budget  AIEN_Q36_PREFIX=1 AIEN_Q36_PREFIX_POOL_GB=0.5: the three long conversations; PASS = every save line
#              reports the conversation's resident bytes and a pool <= 512 MiB, at least one eviction, every
#              turn equal to step 1.
#   6a time    AIEN_Q36_PREFIX=1: turn 1 = ~32.8k-token prompt (long-8192 x 4), then 3 turn-2 runs adding ~1k new
#              tokens each (streamed): HTTP TTFT (client) and the door's restore ms + prefill ms per run.
#   6b time    prompt cache off: the same turn 1 and turn-2 run 1, read cold (the control TTFT).
# Probes (an already running server on PORT; `run` calls them; the host check runs them against a mock door):
#   window-dra86.sh conv PORT OUT KIND BIN_SHA256 LABEL [REF_JSONL]     KIND ref|red|green|write|reload|other|budget
#   window-dra86.sh time PORT OUT KIND BIN_SHA256 LABEL                 KIND warm|cold
# Per turn vs the reference: SAME, DIFF (a flip inside the common length, or more ids), SHORT (strict prefix),
# NO_IDS (HTTP error, missing or empty ids). Exit 0 = PASS/RECORDED, 1 = FAIL, 2 = INCONCLUSIVE or refused,
# 3 = NOT_RUN, 4 = interrupted. `run` stops at the first step that does not exit 0 and returns its code.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); SELF=$HERE/$(basename "$0")
die() { echo "window-dra86: $*" >&2; exit 2; }
not_run() { echo "window-dra86: NOT_RUN: $*" >&2; exit 3; }
left_s() {
  local cap=${1:-999999} l
  if [ -n "${DRA86_DEADLINE:-}" ]; then l=$((DRA86_DEADLINE - $(date +%s))); [ "$l" -lt "$cap" ] || l=$cap; else l=$cap; fi
  echo "$l"; [ "$l" -gt 0 ]
}

# step -> server env, probe, needs, estimated seconds. MODELED (Wolfram, PLAN.md): batch start 60 s, decode 0.0201
# s/token (short prompts stepped at the same rate), chunked prompts 250 tok/s, the ~32.8k prompt 120 tok/s
# (a deliberately slow bound), restore 0.5-1 s; upper bounds at max_tokens.
STEPS_ALL="1 2 3 4a 4b 4c 5 6a 6b"
step_def() {
  ENVS=(); STORE=0; NEEDS=""; case $1 in
  1)  PROBE="conv ref";                                                        EST=240 ;;
  2)  ENVS=(AIEN_Q36_PREFIX=1 AIEN_Q36_PREFIX_MUTANT=1); PROBE="conv red";     EST=180; NEEDS="1" ;;
  3)  ENVS=(AIEN_Q36_PREFIX=1);                         PROBE="conv green";   EST=180; NEEDS="1 2" ;;
  4a) ENVS=(AIEN_Q36_PREFIX=1); STORE=1;                PROBE="conv write";   EST=95;  NEEDS="1" ;;
  4b) ENVS=(AIEN_Q36_PREFIX=1); STORE=1;                PROBE="conv reload";  EST=85;  NEEDS="1 4a!" ;;
  4c) ENVS=(AIEN_Q36_PREFIX=1 AIEN_Q36_NS_PROBE=1); STORE=1; PROBE="conv other"; EST=85; NEEDS="1 4a! 4b!" ;;
  5)  ENVS=(AIEN_Q36_PREFIX=1 AIEN_Q36_PREFIX_POOL_GB=0.5); PROBE="conv budget"; EST=200; NEEDS="1" ;;
  6a) ENVS=(AIEN_Q36_PREFIX=1);                         PROBE="time warm";    EST=355 ;;
  6b)                                                   PROBE="time cold";    EST=620 ;;
  *) return 1 ;;
  esac
}
# step_ok OUT STEP SHA: that step's run receipt passed for this binary inside the lease.
step_ok() {
  jq -e --arg b "$3" '.rc == 0 and .bin_sha256 == $b and (.lock_holder | length) > 0 and .lease_expired == false' \
    "$1/$2/step-receipt.json" > /dev/null 2>&1
}

case ${1:-} in
plan)
  printf '%-4s %-62s %-12s %-10s %s\n' step "server env (batch door, --batch-slots 8)" probe needs est_s
  for s in $STEPS_ALL; do
    step_def $s; e="${ENVS[*]:-}"; [ $STORE = 1 ] && e="$e AIEN_Q36_PREFIX_DIR=OUT/4a/store AIEN_KV_STORE_KEY=(run key)"
    printf '%-4s %-62s %-12s %-10s %s\n' $s "${e:--}" "$PROBE" "${NEEDS:--}" $EST
  done
  echo "needs: N = step N earlier in this run or its passing step receipt in OUT/N; N! = earlier in this run (the store key)."
  echo "Queue slices (chip/DRA-86, cap 1060 s each): s01-identity = 1 2 3 (600 s), s02-controls = 4a 4b 4c 5 (465 s), s03-time-warm = 6a (355 s), s04-time-cold = 6b (620 s)."
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
  if [ "$why" = deadline ]; then for _ in $(seq 40); do kill -0 "$M" 2>/dev/null || break; sleep 1; done; fi
  kill -0 "$M" 2>/dev/null && { echo "$(date -u +%FT%TZ) watchdog: $why: TERM run shell $M"; kill -TERM "$M"; }
  exit 0 ;;
run)
  [ -n "${SPARK_WINDOW:-}" ] || not_run "no SPARK_WINDOW: run as 'spark-window run bash $0 run ...' inside a --quiet exclusive window"
  [ $# -ge 4 ] || die "usage: run BIN OUT STEP [STEP ...]"
  # DRA86_TEST_LOCK (host check only): another lock file; the binary must be a script (a mock door, never the
  # ELF server), the chip and live-port checks are skipped and board lines go to OUT/board.log.
  LOCK=${DRA86_TEST_LOCK:-/tmp/aien-gb10.lock}; TEST=${DRA86_TEST_LOCK:+1}
  g=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/gb10" 2>/dev/null)
  case $g in exclusive|shared) not_run "window $SPARK_WINDOW holds the chip lock (gb10 $g): open it with --gb10 none (queue slice header '# pdra4-gb10: none')" ;; esac
  if [ "${DRA86_LOCKED:-}" != "$SPARK_WINDOW" ]; then
    lw=${DRA86_LOCK_WAIT:-0}; case $lw in ''|*[!0-9]*) die "DRA86_LOCK_WAIT=$lw: whole seconds" ;; esac
    wflag=(-n); [ "$lw" -gt 0 ] && wflag=(-w "$lw")
    DRA86_LOCKED=$SPARK_WINDOW flock "${wflag[@]}" -o -E 75 "$LOCK" bash "$SELF" "$@" & FL=$!
    trap 'kill -TERM $FL 2>/dev/null' INT TERM HUP
    while :; do wait $FL; rc=$?; kill -0 $FL 2>/dev/null || break; done
    [ $rc = 75 ] && not_run "$LOCK is held by another process (waited $lw s); this script takes it itself (open the window with --gb10 none)"
    exit $rc
  fi
  BIN=$(readlink -f "$2"); OUT=$(readlink -m "$3"); shift 3; STEPS=("$@")
  WIN=$SPARK_WINDOW
  ino=$(stat -c %i "$LOCK") || not_run "$LOCK missing"
  holders=$(awk -v ino="$ino" '$2 == "FLOCK" && $4 == "WRITE" { split($6, a, ":"); if (a[3] == ino) print $5 }' /proc/locks | sort -u)
  anc=" "; p=$$
  while [ "$p" -gt 1 ]; do anc="$anc$p "; p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break; done
  HOLDER=; for h in $holders; do case "$anc" in *" $h "*) HOLDER=$h ;; esac; done
  [ -n "$HOLDER" ] || not_run "$LOCK held by pid(s) '${holders}', none an ancestor of $$ ($anc)"
  if [ -n "$TEST" ]; then
    [ -x "$BIN" ] && [ "$(head -c 4 "$BIN" | od -An -c | tr -d ' ')" != '177ELF' ] \
      || not_run "test lock $LOCK held by ancestor $HOLDER; DRA86_TEST_LOCK only runs a script mock door, never the server binary"
  fi
  [ -x "$BIN" ] || not_run "binary $BIN missing"
  SHA=$(sha256sum < "$BIN" | cut -c1-64)
  # Steps known; every need met (earlier in this run, or a passing step receipt in OUT unless marked !).
  total=0; seen=" "
  for s in "${STEPS[@]}"; do
    step_def "$s" || die "unknown step $s (see: $0 plan)"
    [ -n "$TEST" ] && EST=5
    total=$((total + EST))
    for n in $NEEDS; do
      case $n in
      *!) case "$seen" in *" ${n%!} "*) ;; *) not_run "step $s needs step ${n%!} earlier in this run (the store and its key live for one run)" ;; esac ;;
      *) case "$seen" in *" $n "*) ;; *) step_ok "$OUT" "$n" "$SHA" || not_run "step $s needs step $n earlier in this run or its passing step receipt in $OUT/$n for bin $SHA" ;; esac ;;
      esac
    done
    seen="$seen$s "
  done
  lease=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/lease" 2>/dev/null); held=$(cat "${SPARK_WINDOW_DIR:-/nonexistent}/held" 2>/dev/null)
  [ -n "$lease" ] && [ -n "$held" ] || not_run "no lease/held in SPARK_WINDOW_DIR='${SPARK_WINDOW_DIR:-}'"
  [ -e "$SPARK_WINDOW_DIR/expired" ] && not_run "the lease has expired ($SPARK_WINDOW_DIR/expired)"
  heldt=$(date -d "$held" +%s) || not_run "cannot read held time '$held'"
  DEADLINE=$(( heldt + lease * ${SPARK_LEASE_UNIT_S:-60} ))
  cap=${DRA86_CAP_S:-}; case $cap in '') ;; *[!0-9]*) die "DRA86_CAP_S=$cap: whole seconds" ;;
    *) [ $((heldt + cap)) -ge "$DEADLINE" ] || DEADLINE=$((heldt + cap)) ;; esac
  export DRA86_DEADLINE=$DEADLINE
  left=$(( DEADLINE - $(date +%s) ))
  [ "$total" -le "$left" ] || not_run "steps ${STEPS[*]} estimate $total s, the lease has $left s left"
  PORT=${DRA86_TEST_PORT:-18192}; [ -n "$TEST" ] || PORT=18192
  MODEL=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
  for t in curl jq ss flock setsid timeout sha256sum od; do command -v $t > /dev/null || not_run "$t missing"; done
  [ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "a listener is on $PORT"
  if [ -z "$TEST" ]; then
    command -v nvidia-smi > /dev/null || not_run "nvidia-smi missing"
    for port in 18091 18300 18310; do [ -z "$(ss -ltnH "sport = :$port")" ] || not_run "a listener is on $port"; done
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) || not_run "nvidia-smi failed: $apps"
    [ -z "$apps" ] || not_run "compute apps on the chip: $apps"
    [ -f "$MODEL/config.json" ] || not_run "model $MODEL missing"
  fi
  mkdir -p "$OUT" || not_run "cannot create $OUT"
  PIDF=$OUT/.server.$$.pid
  log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log" >&2; }
  board() {
    if [ -n "$TEST" ]; then echo "$(date -u +%FT%TZ) DRA86PrefixCache: $*" >> "$OUT/board.log"
    else echo "$(date -u +%FT%TZ) DRA86PrefixCache: $*" >> "$HOME/handoffs/BOARD.md"; fi
  }
  # The sealed store's key: 32 random bytes for this run only, handed to the server's environment (the server
  # reads it once and removes it), never written to a file or a receipt.
  KEY=$(od -An -tx1 -N32 /dev/urandom | tr -d ' \n'); [ ${#KEY} = 64 ] || not_run "no store key"
  SERVER=
  stop_server() {
    [ -n "$SERVER" ] || return 0
    kill -TERM -- "-$SERVER" 2>/dev/null
    for _ in $(seq 20); do kill -0 -- "-$SERVER" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; rm -f "$PIDF"; SERVER=
  }
  # start_server LOG: ENVS (+ the store) from step_def; the live start-35b-batch.sh flags on a loopback test port.
  start_server() {
    local aff cpus lim l senv=()
    [ "$STORE" = 1 ] && senv=(AIEN_Q36_PREFIX_DIR="$OUT/4a/store" AIEN_KV_STORE_KEY="$KEY")
    cpus=$(cat /sys/devices/system/cpu/online); [ -n "$TEST" ] && cpus=0-9
    aff=(taskset -c "$cpus"); "${aff[@]}" true 2> /dev/null || aff=()
    (cd "$(dirname "$BIN")" && exec setsid "${aff[@]}" env AIEN_GB10_LOCK_HELD=1 AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-$PORT.lock \
      AIEN_Q36_STROKE=0 "${ENVS[@]}" "${senv[@]}" "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
      --max-tokens-cap 16384 --thinking off --max-context 49152 --batch-slots 8) > "$1" 2>&1 < /dev/null &
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
  trap 'stop_server; KEY=; board "GPU-END dra-86 window-dra86.sh window $WIN steps ${STEPS[*]} -> $OUT (rc ${RC:-?})"' EXIT
  trap 'log "interrupted"; RC=4; exit 4' INT TERM
  setsid bash "$SELF" _watch $$ "$HOLDER" "$DEADLINE" "$PIDF" >> "$OUT/watch.log" 2>&1 < /dev/null &
  board "GPU-START dra-86 window-dra86.sh window $WIN steps ${STEPS[*]} (est $total s, lease left $left s; gb10 lock held by our pid $HOLDER; bin $SHA)"
  log "window $WIN lock holder $HOLDER bin $BIN sha256 $SHA steps ${STEPS[*]} est $total s left $left s deadline $(date -u -d @$DEADLINE +%FT%TZ)"
  RC=0
  for s in "${STEPS[@]}"; do
    step_def "$s"; [ -n "$TEST" ] && EST=5
    D=$OUT/$s; mkdir -p "$D"; t0=$(date +%s); why=
    [ "$s" = 4a ] && { rm -rf "$D/store"; mkdir -m 700 "$D/store"; }
    log "step $s: batch door, env '${ENVS[*]:-}'$([ $STORE = 1 ] && echo " + store $OUT/4a/store"), $PROBE, est $EST s"
    if [ $((t0 + EST)) -gt "$DEADLINE" ]; then
      ready=-1; RC=3; why="not started: est $EST s, $((DEADLINE - t0)) s left"; : > "$D/probe.out"
    elif start_server "$D/server.log"; then
      ready=$(( $(date +%s) - t0 ))
      read -r kind a1 <<<"$PROBE"
      args=("$kind" $PORT "$D" "$a1" "$SHA" "dra86-$s")
      [ "$kind" = conv ] && [ "$a1" != ref ] && args+=("$OUT/1/conv.jsonl")
      if lim=$(left_s); then
        DRA86_SERVER_LOG=$D/server.log DRA86_STORE=$OUT/4a/store DRA86_RED=$OUT/2/conv-receipt.json \
          timeout -k 10 "$lim" bash "$SELF" "${args[@]}" > "$D/probe.out"; RC=$?
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
    e="${ENVS[*]:-}"; [ $STORE = 1 ] && e="$e AIEN_Q36_PREFIX_DIR=$OUT/4a/store AIEN_KV_STORE_KEY=(run key)"
    jq -cn --arg s "$s" --arg w "$WIN" --arg b "$BIN" --arg h "$SHA" --arg e "$e" --arg p "$PROBE" \
      --argjson est $EST --argjson ready $ready --argjson el $el --argjson rc $RC --arg hold "$HOLDER" --arg at "$(date -u +%FT%TZ)" \
      --argjson probe "$pj" --argjson x $expired --arg v "$v" --arg why "$why" --argjson dl "$DEADLINE" \
      '{step:$s, window:$w, bin:$b, bin_sha256:$h, door:"batch", env:$e, probe:$p, est_s:$est, server_ready_s:$ready, elapsed_s:$el,
        rc:$rc, verdict:$v, why:$why, lease_expired:$x, deadline:$dl, lock_holder:$hold, at:$at, result:$probe}' \
      | tee "$D/step-receipt.json"
    [ $RC = 0 ] || { log "step $s rc $RC ($v${why:+: $why}): stopping"; exit $RC; }
  done
  exit 0 ;;
esac

[ -n "${SPARK_WINDOW:-}" ] || not_run "no SPARK_WINDOW: the probes run from '$0 run' inside a window"
WIN=$SPARK_WINDOW
MODE=${1:-}; PORT=${2:-}; OUT=${3:-}; KIND=${4:-}; BIN=${5:-}; LABEL=${6:-}; REF=${7:-}
[ -n "$MODE" ] && [ -n "$PORT" ] && [ -n "$OUT" ] && [ -n "$KIND" ] || die "usage: see the header"
[[ $BIN =~ ^[0-9a-f]{64}$ ]] || die "BIN_SHA256 must be the server binary's sha256"
[ -n "$LABEL" ] || die "LABEL missing"
URL=http://127.0.0.1:$PORT
(cd "$REPO" && sha256sum -c --quiet "$HERE/SHA256SUMS") || die "pinned inputs changed"
mkdir -p "$OUT"
SLOG=${DRA86_SERVER_LOG:-}
MODEL=$(curl -s -m 10 "$URL/v1/models" | jq -r '.data[0].id // empty' 2>/dev/null)
[ -n "$MODEL" ] || die "no model on $URL"
SET=$HERE/set32.json; DEF=$(jq -c .defaults "$SET"); L369=$REPO/evidence/merge-train/l369
printf '%s' "Now say that again in fewer words." > "$OUT/follow.txt"; FOLLOW=$OUT/follow.txt
# Bodies go through files and stdin: a ~33k-token prompt is over Linux's 128 KiB limit for one argv string.
req() { # <messages file> <max_tokens> [stream]
  jq -cn --arg m "$MODEL" --slurpfile msg "$1" --argjson n "$2" --argjson d "$DEF" --argjson s "${3:-false}" \
    '$d + {model:$m, messages:$msg[0], max_tokens:$n} + (if $s then {stream:true, stream_options:{include_usage:true}} else {} end)'
}
user() { jq -cn --rawfile u "$1" '[{role:"user",content:$u}]'; }           # <user text file>
turn2() { jq -cn --rawfile u "$1" --rawfile a "$2" --rawfile f "$3" \
  '[{role:"user",content:$u},{role:"assistant",content:$a},{role:"user",content:$f}]'; }  # <user> <answer> <follow-up> files
ask() { # <messages file> <max_tokens> <raw file>: prints the HTTP code (000 without lease time left)
  local m; m=$(left_s 1800) || { : > "$3"; echo 000; return; }
  req "$1" "$2" | curl -s -m "$m" -o "$3" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @-
}
ids() { jq -c '.usage.aien_token_ids | select(type == "array" and length > 0)' "$1" 2>/dev/null; }
idsj() { local x; x=$(ids "$1"); echo "${x:-null}"; }
judge() { # <http code> <raw file> <reference ids json>: SAME|DIFF|SHORT|NO_IDS
  local got; [ "$1" = 200 ] || { echo NO_IDS; return; }
  got=$(ids "$2"); [ -n "$got" ] || { echo NO_IDS; return; }
  jq -rn --argjson g "$got" --argjson w "$3" '
    ([$g, $w] | map(length) | min) as $m
    | if $g == $w then "SAME"
      elif ([range(0; $m) | select($g[.] != $w[.])] | length) > 0 then "DIFF"
      elif ($g | length) > ($w | length) then "DIFF"
      else "SHORT" end'
}
nlines() { if [ -n "$SLOG" ] && [ -f "$SLOG" ]; then wc -l < "$SLOG"; else echo 0; fi; }
since() { [ -n "$SLOG" ] && [ -f "$SLOG" ] && tail -n "+$(($1 + 1))" "$SLOG"; }
# cases KIND: lines "name<TAB>max_tokens<TAB>source" (source: set32 index or an l369 file)
cases() {
  case $1 in
  ref|red|green) for i in $(seq 0 31); do printf 'p%s\t%s\tset:%s\n' $i "$(jq -r ".prompts[$i].max_tokens" "$SET")" $i; done
                 for n in long-256 long-2048 long-8192; do printf '%s\t24\tfile:%s\n' $n $n; done ;;
  write|reload|other) printf 'long-2048\t24\tfile:long-2048\n' ;;
  budget) for n in long-256 long-2048 long-8192; do printf '%s\t24\tfile:%s\n' $n $n; done ;;
  esac
}
text_of() { case $1 in set:*) jq -j ".prompts[${1#set:}].user" "$SET" ;; file:*) cat "$L369/${1#file:}.txt" ;; esac; }

case $MODE in
conv)
  case $KIND in ref|red|green|write|reload|other|budget) ;; *) die "conv KIND ref|red|green|write|reload|other|budget" ;; esac
  if [ "$KIND" != ref ]; then [ -f "$REF" ] || die "reference $REF missing (step 1 first)"; fi
  [ -n "$SLOG" ] || die "DRA86_SERVER_LOG unset: the verdicts read the door's log"
  : > "$OUT/conv.jsonl"
  declare -A n=([SAME]=0 [DIFF]=0 [SHORT]=0 [NO_IDS]=0 [REC]=0); t2n=0; restored=0; refused=0; d2=0
  while IFS=$'\t' read -r name mt src; do
    U=$OUT/$name.user.txt; A=$OUT/$name.a1.txt; text_of "$src" > "$U"; r1=null; r2=null
    [ "$KIND" = ref ] || { r1=$(jq -c --arg n "$name" 'select(.name == $n) | .ids1' "$REF"); r2=$(jq -c --arg n "$name" 'select(.name == $n) | .ids2' "$REF"); }
    [ -n "$r1" ] || r1=null; [ -n "$r2" ] || r2=null
    v1=-; h1=-; a1=
    if [ "$KIND" = reload ] || [ "$KIND" = other ]; then
      a1=$(jq -r --arg n "$name" 'select(.name == $n) | .text1' "$REF")
    else
      user "$U" > "$OUT/$name.t1.msg"; h1=$(ask "$OUT/$name.t1.msg" "$mt" "$OUT/$name.t1.raw")
      if [ "$KIND" = ref ]; then v1=REC; [ -n "$(ids "$OUT/$name.t1.raw")" ] && [ "$h1" = 200 ] || v1=NO_IDS
      else v1=$(judge "$h1" "$OUT/$name.t1.raw" "$r1"); fi
      n[$v1]=$((n[$v1] + 1)); a1=$(jq -r '.choices[0].message.content // empty' "$OUT/$name.t1.raw" 2>/dev/null)
    fi
    v2=-; h2=-; rl=; fl=; jl=
    if [ "$KIND" != write ]; then
      printf '%s' "$a1" > "$A"; turn2 "$U" "$A" "$FOLLOW" > "$OUT/$name.t2.msg"
      b=$(nlines); h2=$(ask "$OUT/$name.t2.msg" 16 "$OUT/$name.t2.raw"); seg=$(since "$b")
      rl=$(grep -E 'prompt cache restore: ' <<<"$seg" | tail -n 1); fl=$(grep -E 'prompt cache restore refused' <<<"$seg" | tail -n 1)
      jl=$(grep -E 'qwen36-batch: join: ' <<<"$seg" | tail -n 1)
      if [ "$KIND" = ref ]; then v2=REC; [ -n "$(ids "$OUT/$name.t2.raw")" ] && [ "$h2" = 200 ] || v2=NO_IDS
      else v2=$(judge "$h2" "$OUT/$name.t2.raw" "$r2"); fi
      n[$v2]=$((n[$v2] + 1)); t2n=$((t2n + 1)); [ -n "$rl" ] && restored=$((restored + 1)); [ -n "$fl" ] && refused=$((refused + 1))
      [ "$v2" = DIFF ] && d2=$((d2 + 1))
    fi
    jq -cn --arg n "$name" --arg v1 "$v1" --arg v2 "$v2" --arg h1 "$h1" --arg h2 "$h2" --arg a1 "$a1" --arg rl "$rl" --arg fl "$fl" --arg jl "$jl" \
      --arg win "$WIN" --arg b "$BIN" --argjson g1 "$(idsj "$OUT/$name.t1.raw")" --argjson g2 "$(idsj "$OUT/$name.t2.raw")" \
      '{name:$n, t1:$v1, t2:$v2, http1:$h1, http2:$h2, ids1:$g1, ids2:$g2, text1:$a1, restore_log:$rl, refused_log:$fl, join_log:$jl, window:$win, bin_sha256:$b}' \
      2>/dev/null >> "$OUT/conv.jsonl" || echo "{\"name\":\"$name\",\"t1\":\"NO_IDS\",\"t2\":\"NO_IDS\"}" >> "$OUT/conv.jsonl"
  done < <(cases "$KIND")
  all=$(cat "$SLOG" 2>/dev/null); why=
  bad=$((n[NO_IDS] + n[SHORT])); verdict=FAIL; rc=1
  case $KIND in
  ref)
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="a turn without ids"
    elif grep -q 'prompt cache off' <<<"$all"; then verdict=RECORDED; rc=0
    else why="no 'prompt cache off' line: the switch is not off by default"; fi ;;
  red)
    if [ $bad -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="NO_IDS/SHORT turns"
    elif ! grep -q 'prompt cache on: .*POSITION MUTANT' <<<"$all"; then verdict=INCONCLUSIVE; rc=2; why="the mutant is not on"
    elif [ $restored -lt $t2n ]; then verdict=INCONCLUSIVE; rc=2; why="$((t2n - restored)) turn 2 without a restore line"
    elif [ $d2 -gt 0 ]; then verdict=PASS; rc=0
    else why="no turn 2 differs: the gate cannot see the off-by-one"; fi ;;
  green)
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="a turn without ids"
    elif [ -n "${DRA86_RED:-}" ] && ! jq -e --arg b "$BIN" '.verdict == "PASS" and .kind == "red" and .bin_sha256 == $b' "$DRA86_RED" > /dev/null 2>&1; then
      verdict=INCONCLUSIVE; rc=2; why="no red PASS receipt for this binary at $DRA86_RED"
    elif ! grep -q 'prompt cache on: budget 16.0 GiB' <<<"$all"; then why="the default budget is not 16.0 GiB"
    elif [ "${n[SAME]}" -ne $((2 * t2n)) ]; then why="$((2 * t2n - n[SAME])) turns differ from the cold reference"
    elif [ $restored -ne $t2n ]; then why="$((t2n - restored)) turn 2 not restored from the cache"
    elif [ $refused -gt 0 ]; then why="$refused restore refused"
    else verdict=PASS; rc=0; fi ;;
  write)
    for _ in $(seq 60); do ls "$DRA86_STORE"/*.q36p > /dev/null 2>&1 && ! ls "$DRA86_STORE"/.*.tmp > /dev/null 2>&1 && break; sleep 1; done
    files=$(ls "$DRA86_STORE"/*.q36p 2>/dev/null | wc -l)
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="no ids"
    elif ! grep -q 'q36 prefix: store .* open: 0 nodes loaded' <<<"$all"; then why="the store did not open empty"
    elif [ "${n[SAME]}" -ne 1 ]; then why="turn 1 differs from the reference"
    elif [ "$files" -lt 1 ]; then why="no sealed node in $DRA86_STORE"
    else verdict=RECORDED; rc=0; why="$files sealed nodes"; fi ;;
  reload|other)
    ol=$(grep -E 'q36 prefix: store .* open: ' <<<"$all" | tail -n 1)
    ld=$(sed -nE 's/.* open: ([0-9]+) nodes loaded .*/\1/p' <<<"$ol"); dc=$(sed -nE 's/.*\), ([0-9]+) discarded.*/\1/p' <<<"$ol")
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="no ids"
    elif [ -z "$ol" ]; then why="no store open line"
    elif [ "${n[SAME]}" -ne 1 ]; then why="turn 2 differs from the cold reference"
    elif [ "$KIND" = reload ]; then
      if [ "${ld:-0}" -ge 1 ] && [ $restored = 1 ]; then verdict=PASS; rc=0; else why="loaded ${ld:-?}, restored $restored (want >= 1 and 1)"; fi
    else
      if [ "${ld:-1}" = 0 ] && [ "${dc:-0}" -ge 1 ] && [ $restored = 0 ]; then verdict=PASS; rc=0
      else why="loaded ${ld:-?} discarded ${dc:-?} restored $restored (want 0, >= 1, 0)"; fi
    fi ;;
  budget)
    sv=$(grep -E 'prompt cache save: ' <<<"$all")
    ns=$(grep -c . <<<"$sv"); nr=$(grep -c 'conversation resident [0-9]* nodes [0-9.]* MiB' <<<"$sv")
    over=$(sed -nE 's/.*; pool [0-9]+ nodes ([0-9.]+) MiB of .*/\1/p' <<<"$sv" | awk '$1 > 512.0' | wc -l)
    ev=$(sed -nE 's/.* ([0-9]+) evicted, .*/\1/p' <<<"$sv" | sort -n | tail -n 1)
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2; why="a turn without ids"
    elif ! grep -q 'prompt cache on: budget 0.5 GiB' <<<"$all"; then why="the budget is not 0.5 GiB"
    elif [ -z "$sv" ]; then why="no save line"
    elif [ "$nr" -ne "$ns" ]; then why="$((ns - nr)) save lines without the conversation's resident bytes"
    elif [ "$over" -gt 0 ]; then why="$over save lines report a pool over 512 MiB"
    elif [ "${ev:-0}" -lt 1 ]; then why="nothing evicted"
    elif [ "${n[SAME]}" -ne 6 ]; then why="$((6 - n[SAME])) turns differ from the cold reference"
    else verdict=PASS; rc=0; why="$ns saves, max evicted $ev"; fi ;;
  esac
  jq -cn --arg k "$KIND" --arg v $verdict --arg b "$BIN" --arg l "$LABEL" --arg why "$why" --argjson s "${n[SAME]}" --argjson d "${n[DIFF]}" \
    --argjson sh "${n[SHORT]}" --argjson x "${n[NO_IDS]}" --argjson rc "${n[REC]}" --argjson t2 $t2n --argjson rs $restored --argjson rf $refused \
    --arg port "$PORT" --arg at "$(date -u +%FT%TZ)" --arg win "$WIN" \
    '{case:"conv", kind:$k, verdict:$v, why:$why, bin_sha256:$b, label:$l, window:$win, same:$s, diff:$d, short:$sh, no_ids:$x, recorded:$rc,
      turn2:$t2, restored:$rs, refused:$rf, port:$port, at:$at}' | tee "$OUT/conv-receipt.json"
  exit $rc ;;
time)
  case $KIND in warm) runs="1 2 3" ;; cold) runs="1" ;; *) die "time KIND warm|cold" ;; esac
  # turn 1: ~32.8k tokens (long-8192 x 4); turn 2 adds "Part r." + long-256 x 4 (~1k tokens).
  L32=$OUT/l32.txt; for _ in 1 2 3 4; do cat "$L369/long-8192.txt"; printf '\n\n'; done > "$L32"
  : > "$OUT/time.jsonl"; ok=1
  user "$L32" > "$OUT/t1.msg"; h1=$(ask "$OUT/t1.msg" 16 "$OUT/t1.raw")
  jq -j '.choices[0].message.content // empty' "$OUT/t1.raw" > "$OUT/a1.txt" 2>/dev/null
  p1=$(jq -r '.usage.prompt_tokens // empty' "$OUT/t1.raw" 2>/dev/null)
  [ "$h1" = 200 ] && [ -s "$OUT/a1.txt" ] || ok=0
  for run in $runs; do
    o=$OUT/t2-r$run; : > "$o.sse"
    { printf 'Part %s.\n' "$run"; for _ in 1 2 3 4; do cat "$L369/long-256.txt"; printf '\n\n'; done; } > "$o.new.txt"
    turn2 "$L32" "$OUT/a1.txt" "$o.new.txt" > "$o.msg"; req "$o.msg" 16 true > "$o.body"   # built before t0
    b=$(nlines); t0=$(date +%s.%N); first=""
    if m=$(left_s 1800); then
      while IFS= read -r line; do
        [ -z "$first" ] && [[ $line =~ ^data:.*\"content\":\ *\"[^\"] ]] && first=$(date +%s.%N)
        printf '%s\n' "$line" >> "$o.sse"
      done < <(curl -s -N -m "$m" -w '\nHTTP %{http_code}\n' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
                 --data-binary "@$o.body")
    fi
    t1=$(date +%s.%N); seg=$(since "$b")
    code=$(grep -o '^HTTP [0-9]*' "$o.sse" | tail -n 1 | cut -c6-)
    usage=$(grep '^data: {' "$o.sse" | sed 's/^data: //' | jq -c 'select(.usage != null) | .usage' 2>/dev/null | tail -n 1)
    rl=$(grep -E 'prompt cache restore: ' <<<"$seg" | tail -n 1); jl=$(grep -E 'qwen36-batch: join: ' <<<"$seg" | tail -n 1)
    ll=$(grep -E 'qwen36-batch: leave: ' <<<"$seg" | tail -n 1)
    rms=$(sed -nE 's/.* MiB in ([0-9]+) ms, conversation resident.*/\1/p' <<<"$rl"); pms=$(sed -nE 's/.*, prefill ([0-9]+) ms,.*/\1/p' <<<"$ll")
    reuse=$(sed -nE 's/.* reuse ([0-9]+) tok .*/\1/p' <<<"$jl"); pt=$(sed -nE 's/.*join: prompt ([0-9]+) tok,.*/\1/p' <<<"$jl")
    good=$(jq -n --arg c "${code:-}" --arg f "$first" --argjson u "${usage:-null}" --arg k "$KIND" --arg rl "$rl" --arg p "$pms" \
      '$c == "200" and $f != "" and ($u | type) == "object" and (($u.completion_tokens // 0) > 0) and $p != ""
       and (if $k == "warm" then $rl != "" else $rl == "" end)')
    [ "$good" = true ] || ok=0
    jq -cn --arg k "$KIND" --argjson r $run --arg b "$BIN" --arg l "$LABEL" --arg h "${code:-}" --arg t0 "$t0" --arg tf "$first" \
      --arg t1 "$t1" --argjson u "${usage:-null}" --argjson g "$good" --arg win "$WIN" --arg rms "$rms" --arg pms "$pms" \
      --arg reuse "$reuse" --arg pt "$pt" --arg p1 "$p1" --arg rl "$rl" --arg ll "$ll" \
      '{kind:$k, run:$r, bin_sha256:$b, label:$l, window:$win, http:$h, ok:$g, t0:$t0, t_first:$tf, t_end:$t1,
        http_ttft_s:(if $tf == "" then null else (($tf | tonumber) - ($t0 | tonumber)) end),
        turn1_prompt_tok:($p1 | tonumber? // null), turn2_prompt_tok:($pt | tonumber? // null), reused_tok:($reuse | tonumber? // null),
        door_restore_ms:($rms | tonumber? // null), door_prefill_ms:($pms | tonumber? // null), restore_log:$rl, leave_log:$ll, usage:$u}' \
      | tee -a "$OUT/time.jsonl"
  done
  if [ $ok = 1 ]; then jq -cn --arg k "$KIND" --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" '{case:"time", kind:$k, verdict:"RECORDED", window:$win, bin_sha256:$b, label:$l}'; exit 0; fi
  jq -cn --arg k "$KIND" --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" \
    '{case:"time", kind:$k, verdict:"INCONCLUSIVE", window:$win, bin_sha256:$b, label:$l, why:"turn 1 failed, or a run had no HTTP 200, first delta, usage, leave line, or (warm) restore line / (cold) a restore"}'; exit 2 ;;
*) die "mode plan|run|conv|time" ;;
esac
