#!/usr/bin/env bash
# DRA-66 host time per token, chip window. Runs ONLY inside a spark-window opened by Main:
#
#   spark-window open dra66 --gb10 none --quiet exclusive --lease 20 --wait
#   spark-window run bash evidence/dra-66-host-time-20261031/window.sh
#   spark-window close
#
# Why --gb10 none: the server verifies an inherited chip reservation by requiring the flock
# holder of /tmp/aien-gb10.lock to be one of its ANCESTORS (crates/aien-serve-http/src/
# chip_reservation.rs, `detect`). spark-window's holder (`setsid spark-window _hold ID`) never is,
# so in a window that holds the chip the server would refuse to start (exit 1). spark-window's
# documented rule for such jobs is a window with --gb10 none; this script then takes
# /tmp/aien-gb10.lock itself through flock(1), which is the ancestor of every server it starts
# (AIEN_GB10_LOCK_HELD=1 is set by the script for its servers, after it verified the holder).
# The window's quiet lock (--quiet exclusive) is what makes the timings quiet.
#
# Refuses with NOT_RUN (exit 3) before any chip contact unless: SPARK_WINDOW and
# SPARK_WINDOW_DIR are set and name the open, held window; that window is gb10=none and
# quiet=exclusive; AIEN_GB10_LOCK_HELD is not already set; the chip lock comes free within 90 s;
# 18091/18300/18310 are down; no compute process is on the chip; the binary has the pinned sha256.
#
# Gates (every one writes $EV/gate-<name>.json with window id, binary sha256, sovereign-core
# commit, lock holder, start/end, its files' sha256 and its status). Each gate has a hard cap;
# a gate starts only if the lease has at least its cap left, otherwise it and the rest are
# NOT_RUN. Caps sum to 1055 s of the 1200 s lease (Wolfram, see PROTOCOL.md).
#   mutant4-red      cap 270 s  server with AIEN_Q36_TEST_MUTANT=4 must DIFFER from the golden
#                               ids; if it does not, the window stops (exit 4).
#   golden-identity  cap 240 s  honest server, per-token receipt on: golden IDENTICAL.
#   receipt-1conv    cap 180 s  3 sequential greedy replies of 256 tokens.
#   receipt-8conv    cap 120 s  8 concurrent greedy replies of 256 tokens.
#   seed-t07         cap  90 s  seed 1, seed 2, seed 1 again at T=0.7, 64 tokens.
#   checker          cap  35 s  server stopped; --check-decode-timeline exit 0, two --drop red
#                               controls exit 1, summary.wl for Main's Wolfram 0.5 ms gate.
# Everything lands in $EV; receipt.json collects every gate.
set -u
umask 022

W=/home/drakestapleton/work/dra-66
BIN=$W/bin-head/aien-serve-http
SHA_HEAD=3bd287beb5c74fc5c241307765f3a858c1a7e33e0d1d5ae2d766f2dae13e9c69 # sc 84a28c09 + omega a647b6a9 (PROTOCOL.md)
GOLD=$W/core/evidence/q36-host-20261020/gen-chat-max-24.json
MODEL=/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0
PORT=18190
LOCK=/tmp/aien-gb10.lock
BOARD=/home/drakestapleton/handoffs/BOARD.md
SELF=$(readlink -f "$0")
COUNT='Count from 1 to 400 in English words, one number per line, nothing else.'

ts() { date -u +%FT%TZ; }

# ---- outer: the window, then the chip lock through flock(1) ----
if [ -z "${DRA66_INNER:-}" ]; then
  refuse() { echo "NOT_RUN: $*"; exit 3; }
  [ -n "${SPARK_WINDOW:-}" ] && [ -n "${SPARK_WINDOW_DIR:-}" ] \
    || refuse "not inside a spark-window (spark-window open dra66 --gb10 none --quiet exclusive --lease 20 --wait; spark-window run bash $SELF)"
  [ "$(spark-window id 2> /dev/null)" = "$SPARK_WINDOW" ] || refuse "window $SPARK_WINDOW is not the open window holding its locks"
  [ -f "$SPARK_WINDOW_DIR/held" ] || refuse "window $SPARK_WINDOW has no held stamp"
  gb=$(cat "$SPARK_WINDOW_DIR/gb10" 2> /dev/null)
  q=$(cat "$SPARK_WINDOW_DIR/quiet" 2> /dev/null)
  [ "$gb" = none ] \
    || refuse "window gb10=$gb: its holder is not an ancestor of the server, which would refuse the chip (chip_reservation.rs detect); open with --gb10 none"
  [ "$q" = exclusive ] || refuse "window quiet=$q: the timing gates need --quiet exclusive"
  [ -z "${AIEN_GB10_LOCK_HELD:-}" ] || refuse "AIEN_GB10_LOCK_HELD=$AIEN_GB10_LOCK_HELD already set; this script takes the chip lock itself"
  lease=$(cat "$SPARK_WINDOW_DIR/lease")
  DRA66_LEASE_END=$(( $(stat -c %Y "$SPARK_WINDOW_DIR/held") + lease * 60 ))
  DRA66_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  export DRA66_INNER=1 DRA66_LEASE_END DRA66_STAMP
  echo "$(ts) window $SPARK_WINDOW (gb10 $gb, quiet $q, lease $lease min, $(( DRA66_LEASE_END - $(date +%s) )) s left); taking $LOCK (at most 90 s)"
  flock -w 90 -E 75 "$LOCK" bash "$SELF"
  rc=$?
  [ "$rc" = 75 ] && refuse "$LOCK did not come free in 90 s"
  exit "$rc"
fi

# ---- inner: child of flock(1), which holds the chip lock ----
EV=/home/drakestapleton/handoffs/dra-66-host-time/window-$DRA66_STAMP
SERVER_PID=
HOLDER=
BIN_SHA=
BIN_COMMIT=
G_T0=$(ts)
G_S0=$(date +%s)
T0=$G_S0

say() { echo "$(ts) $*" | tee -a "$EV/window.log"; }
left() { echo $(( DRA66_LEASE_END - $(date +%s) )); }
elapsed() { echo $(( $(date +%s) - T0 )); }

# gate NAME STATUS DETAIL [FILE...]: the gate's receipt, stamped with window, binary and lock.
gate() {
  local name=$1 status=$2 detail=$3 files
  shift 3
  files=$(for f in "$@"; do [ -f "$f" ] && sha256sum "$f"; done \
    | jq -R 'capture("^(?<sha256>[0-9a-f]{64})  (?<path>.*)$")' | jq -s .)
  jq -n --arg gate "$name" --arg status "$status" --arg detail "$detail" \
    --arg window_id "$SPARK_WINDOW" --arg window_dir "$SPARK_WINDOW_DIR" \
    --arg bin "$BIN" --arg bin_sha256 "$BIN_SHA" --arg sc_commit "$BIN_COMMIT" \
    --arg holder "$HOLDER" --arg start "$G_T0" --arg end "$(ts)" \
    --argjson elapsed_s "$(( $(date +%s) - G_S0 ))" --argjson lease_left_s "$(left)" \
    --argjson files "$files" \
    '{schema: "aien.dra66-window-gate/1", gate: $gate, status: $status, detail: $detail,
      window_id: $window_id, window_dir: $window_dir, bin: $bin, bin_sha256: $bin_sha256,
      sc_commit: $sc_commit, gb10_lock_holder_pid: $holder, start: $start, end: $end,
      elapsed_s: $elapsed_s, lease_left_s: $lease_left_s, files: $files}' \
    > "$EV/gate-$name.json"
  say "gate $name: $status ($detail)"
}

# begin CAP NAME: start a gate only if the lease has CAP seconds left.
begin() {
  G_T0=$(ts)
  G_S0=$(date +%s)
  local l
  l=$(left)
  if [ "$l" -lt "$1" ]; then
    gate "$2" NOT_RUN "lease has $l s left, gate cap $1 s; this and later gates not run"
    exit 3
  fi
  say "gate $2 begins (cap $1 s, lease left $l s)"
}

not_run() {
  echo "NOT_RUN: $*"
  [ -d "$EV" ] && gate preconditions NOT_RUN "$*"
  exit 3
}

mkdir -p "$EV" || { echo "NOT_RUN: cannot create $EV"; exit 3; }

# ---- preconditions (no chip contact) ----
[ -e "$LOCK" ] || not_run "$LOCK missing"
ino=$(stat -c %i "$LOCK")
holders=$(awk -v ino="$ino" '$2 == "FLOCK" && $4 == "WRITE" { split($6, a, ":"); if (a[3] == ino) print $5 }' /proc/locks | sort -u)
anc=" "
p=$$
while [ "$p" -gt 1 ]; do
  anc="$anc$p "
  p=$(awk '{print $4}' "/proc/$p/stat" 2> /dev/null) || break
done
for h in $holders; do case "$anc" in *" $h "*) HOLDER=$h ;; esac; done
[ -n "$HOLDER" ] || not_run "$LOCK held exclusively by pid(s) '$holders', none an ancestor of $$ ($anc)"
if curl -s -m 5 127.0.0.1:18091/health > "$EV/health-18091-before.json"; then
  not_run "18091 answers /health; the live 35B must be stopped by Main first"
fi
for port in 18091 18300 18310; do
  [ -z "$(ss -ltnH "sport = :$port")" ] || not_run "a listener is still on $port"
done
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) \
  || not_run "nvidia-smi failed: $apps"
echo "$apps" > "$EV/compute-apps-before.txt"
[ -z "$apps" ] || not_run "compute apps present: $apps"
BIN_SHA=$(sha256sum < "$BIN" | cut -c1-64)
[ "$BIN_SHA" = "$SHA_HEAD" ] || not_run "$BIN sha256 $BIN_SHA, pinned $SHA_HEAD"
"$BIN" --version > "$EV/version.txt" 2>&1 || not_run "$BIN --version failed"
BIN_COMMIT=$(awk '/^sovereign-core commit:/ { print $3 }' "$EV/version.txt")
grep -q '^omega gpu engine: native' "$EV/version.txt" || not_run "$BIN has no native omega gpu engine"
[ -f "$GOLD" ] || not_run "golden $GOLD missing"
WANT=$(jq -c '.generated_ids' "$GOLD")
[ "$(echo "$WANT" | jq length)" = 24 ] || not_run "golden $GOLD has no 24 generated_ids"
[ -f "$MODEL/config.json" ] || not_run "model dir $MODEL missing"
[ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "port $PORT busy"
for t in curl jq ss sha256sum spark-window; do command -v $t > /dev/null || not_run "$t missing"; done
{
  echo "window $SPARK_WINDOW $SPARK_WINDOW_DIR"; echo "lock $LOCK holder $HOLDER; ancestors$anc"
  echo "bin $BIN $BIN_SHA commit $BIN_COMMIT"; echo "golden $GOLD $(sha256sum < "$GOLD" | cut -c1-64)"
  uname -r; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; cat /proc/loadavg
} > "$EV/preconditions.txt"
gate preconditions PASS "window $SPARK_WINDOW, lock holder $HOLDER (ancestor), $(left) s of lease left" \
  "$EV/preconditions.txt" "$EV/version.txt" "$EV/compute-apps-before.txt" "$GOLD"

# ---- server control ----
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM "$SERVER_PID" 2> /dev/null
  for _ in $(seq 30); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2> /dev/null
  wait "$SERVER_PID" 2> /dev/null
  SERVER_PID=
}
finish() {
  stop_server
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$EV/compute-apps-after.txt" 2>&1
  jq -s --arg window_id "$SPARK_WINDOW" --arg bin_sha256 "$BIN_SHA" --arg sc_commit "$BIN_COMMIT" \
    --argjson elapsed_s "$(elapsed)" \
    '{schema: "aien.dra66-window/1", window_id: $window_id, bin_sha256: $bin_sha256,
      sc_commit: $sc_commit, elapsed_s: $elapsed_s, gates: .}' \
    $(ls -tr "$EV"/gate-*.json 2> /dev/null) > "$EV/receipt.json"
  board "GPU-END dra-66 window.sh window $SPARK_WINDOW $EV (elapsed $(elapsed) s; compute apps after: $(tr '\n' ';' < "$EV/compute-apps-after.txt"))"
}
board() { echo "$(ts) DRA66Host: $*" >> "$BOARD"; }

# start_server LOG [ENV=VAL ...]: the live 18091 flags on a loopback test port, inheriting the
# chip lock this script's flock(1) holds; at most 180 s to health (measured loads 39-78 s).
start_server() {
  local log=$1
  shift
  env AIEN_GB10_LOCK_HELD=1 AIEN_Q36_STROKE=1 AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-$PORT.lock "$@" \
    "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
    --max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8 \
    > "$log" 2>&1 &
  SERVER_PID=$!
  local deadline=$(( $(date +%s) + 180 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    kill -0 "$SERVER_PID" 2> /dev/null || { say "server exited during load: $(tail -n 2 "$log" | tr '\n' ' ')"; SERVER_PID=; return 1; }
    curl -s -m 2 127.0.0.1:$PORT/health > /dev/null && break
    sleep 1
  done
  curl -s -m 2 127.0.0.1:$PORT/health > /dev/null || { say "server not healthy in 180 s"; stop_server; return 1; }
  grep -q "chip reservation inherited from ancestor pid $HOLDER" "$log" \
    || { say "server did not report the inherited reservation of pid $HOLDER"; stop_server; return 1; }
  local l
  l=$(ss -ltnH "sport = :$PORT" | awk '{print $4}')
  if [ -z "$l" ] || [ -n "$(echo "$l" | grep -v '^127\.0\.0\.1:')" ]; then
    say "listener check FAILED on $PORT: $l"
    stop_server
    return 1
  fi
  say "server pid $SERVER_PID up, listeners: $(echo $l)"
}

# chat OUT PROMPT MAX TEMP SEED CAP_S: one chat request with token ids; prints "tokens wall_s".
chat() {
  local out=$1 prompt=$2 max=$3 temp=$4 seed=$5 cap=$6 body wall
  body=$(jq -nc --arg p "$prompt" --argjson m "$max" --argjson t "$temp" --argjson s "$seed" \
    '{model:"qwen3.6-35b-a3b-fp8",messages:[{role:"user",content:$p}],temperature:$t,seed:$s,max_tokens:$m,stream:false,chat_template_kwargs:{enable_thinking:false},aien_return_token_ids:true}')
  wall=$(curl -s -m "$cap" -o "$out" -w '%{time_total}' -H 'content-type: application/json' \
    -d "$body" 127.0.0.1:$PORT/v1/chat/completions) || { echo "fail 0"; return 1; }
  echo "$(jq -r '.usage.completion_tokens // "fail"' "$out") $wall"
}

# golden LABEL: the gate2 request (24 greedy tokens, 60 s cap); prints IDENTICAL, DIFFER or ERROR.
golden() {
  local out=$EV/golden-$1.json got
  chat "$out" 'Write a C function that returns the larger of two ints.' 24 0 0 60 > /dev/null
  got=$(jq -c '.usage.aien_token_ids // empty' "$out" 2> /dev/null)
  if [ -z "$got" ]; then echo ERROR; elif [ "$got" = "$WANT" ]; then echo IDENTICAL; else echo DIFFER; fi
}

board "GPU-START dra-66 window.sh window $SPARK_WINDOW $EV (lock holder $HOLDER via flock; 18091 + Flint stopped; compute apps empty)"
trap 'finish' EXIT
trap 'say "interrupted"; exit 4' INT TERM

# ---- mutant4-red: the negative control goes first ----
begin 270 mutant4-red
start_server "$EV/server-mutant4.log" AIEN_Q36_TEST_MUTANT=4 \
  || { gate mutant4-red NOT_RUN "mutant server did not come up" "$EV/server-mutant4.log"; exit 3; }
m4=$(golden mutant4)
stop_server
if [ "$m4" = DIFFER ]; then
  gate mutant4-red PASS "mutant 4 DIFFERs from the golden (the control went red)" "$EV/golden-mutant4.json" "$EV/server-mutant4.log"
else
  gate mutant4-red FAIL "mutant 4 gave $m4, not DIFFER; no green below would count" "$EV/golden-mutant4.json" "$EV/server-mutant4.log"
  exit 4
fi

# ---- golden-identity: honest server, per-token receipt on ----
TOK=$EV/tokens.jsonl
begin 240 golden-identity
start_server "$EV/server-tokens.log" AIEN_Q36_DECODE_TIMELINE="$TOK" AIEN_Q36_DECODE_TIMELINE_KIND=token \
  || { gate golden-identity NOT_RUN "honest server did not come up" "$EV/server-tokens.log"; exit 3; }
grep -E 'per-token receipt' "$EV/server-tokens.log" | tee -a "$EV/window.log"
g=$(golden honest)
[ "$g" = IDENTICAL ] && st=PASS || st=FAIL
gate golden-identity "$st" "honest golden $g (must be IDENTICAL)" "$EV/golden-honest.json" "$GOLD"

# ---- receipt-1conv ----
begin 180 receipt-1conv
ok=PASS
for i in 1 2 3; do
  r=$(chat "$EV/solo-$i.json" "$COUNT" 256 0 0 60) || ok=FAIL
  say "solo $i: tokens wall_s = $r"
done
sleep 2 # the door closes the spell at idle
gate receipt-1conv "$ok" "3 sequential greedy replies of 256 tokens; receipt records so far $(wc -l < "$TOK" 2> /dev/null || echo 0)" \
  "$EV"/solo-*.json "$TOK"

# ---- receipt-8conv ----
begin 120 receipt-8conv
pids=()
for i in 1 2 3 4 5 6 7 8; do
  chat "$EV/eight-$i.json" "Count from $((i * 50)) to $((i * 50 + 300)) in English words, one number per line, nothing else." 256 0 0 110 \
    > "$EV/eight-$i.txt" &
  pids+=($!)
done
ok=PASS
for p in "${pids[@]}"; do wait "$p" || ok=FAIL; done
for i in 1 2 3 4 5 6 7 8; do say "eight $i: tokens wall_s = $(cat "$EV/eight-$i.txt")"; done
sleep 2
gate receipt-8conv "$ok" "8 concurrent greedy replies of 256 tokens; receipt records so far $(wc -l < "$TOK" 2> /dev/null || echo 0)" \
  "$EV"/eight-*.json "$TOK"

# ---- seed-t07 ----
begin 90 seed-t07
for s in 1 2 1b; do chat "$EV/seed-$s.json" "$COUNT" 64 0.7 "${s%b}" 30 > /dev/null; done
ids() { jq -c '.usage.aien_token_ids' "$EV/seed-$1.json" 2> /dev/null; }
if [ "$(ids 1)" = "$(ids 1b)" ] && [ "$(ids 1)" != "$(ids 2)" ] && [ -n "$(ids 1)" ] && [ "$(ids 1)" != null ]; then
  gate seed-t07 PASS "seed 1 repeats, seed 2 differs" "$EV"/seed-*.json
else
  gate seed-t07 FAIL "seed1 $(ids 1 | cut -c1-60) / seed2 $(ids 2 | cut -c1-60) / seed1b $(ids 1b | cut -c1-60)" "$EV"/seed-*.json
fi

# ---- checker: server stopped (the recorder writes its last record at finish) ----
begin 35 checker
stop_server
grep -E 'spell over|per-token receipt' "$EV/server-tokens.log" > "$EV/spells.txt"
"$BIN" --check-decode-timeline "$TOK" > "$EV/summary.wl" 2> "$EV/check.txt"
c0=$?
echo "check exit $c0 (must be 0)" >> "$EV/check.txt"
"$BIN" --check-decode-timeline "$TOK" --drop 0:record:0 > /dev/null 2> "$EV/red-drop-first.txt"
c1=$?
echo "red control (drop the first record) exit $c1 (must be 1)" >> "$EV/red-drop-first.txt"
n=$(wc -l < "$TOK" 2> /dev/null || echo 0)
"$BIN" --check-decode-timeline "$TOK" --drop $(( n / 2 )):record:0 > /dev/null 2> "$EV/red-drop-middle.txt"
c2=$?
echo "red control (drop record $(( n / 2 ))) exit $c2 (must be 1)" >> "$EV/red-drop-middle.txt"
[ "$c0" = 0 ] && [ "$c1" = 1 ] && [ "$c2" = 1 ] && st=PASS || st=FAIL
gate checker "$st" "$n records; check exit $c0, drop-first exit $c1, drop-middle exit $c2; Wolfram gate = first query of summary.wl" \
  "$TOK" "$EV/summary.wl" "$EV/check.txt" "$EV/red-drop-first.txt" "$EV/red-drop-middle.txt" "$EV/spells.txt" "$EV/server-tokens.log"
say "done in $(elapsed) s, lease left $(left) s"
