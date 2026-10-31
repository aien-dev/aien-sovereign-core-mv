#!/usr/bin/env bash
# DRA-66 host time per token, chip window. DO NOT RUN outside an exclusive window opened by Main.
#
# Assumes Main holds /tmp/aien-gb10.lock (an ancestor of this shell holds it, e.g.
#   flock -w 900 /tmp/aien-gb10.lock bash <this file>
# ) and has stopped the live 35B on 18091 and Flint (18300/18310). Any failed precondition
# prints NOT_RUN and exits 3 without touching the chip.
#
# Phase 1: golden identity through the HTTP door, negative control first: a server with
#          AIEN_Q36_TEST_MUTANT=4 (shared expert dropped) must DIFFER from the golden ids; only
#          then does the honest server's IDENTICAL count. A mutant that stays green stops the
#          window (exit 4).
# Phase 2: the honest server with the per-token receipt on
#          (AIEN_Q36_DECODE_TIMELINE=<file> AIEN_Q36_DECODE_TIMELINE_KIND=token):
#          the golden request, 1 conversation (3 sequential greedy replies), 8 conversations
#          (8 concurrent greedy replies), and a T=0.7 seed check (seed 1, seed 2, seed 1 again).
# Phase 3: the receipt checker (green), its two red controls (--drop 0:record:0 and a middle
#          record), and the Wolfram-ready summary.wl. Main evaluates summary.wl with the
#          Wolfram MCP afterwards; the DRA-66 gate is its first query
#          (median host_overhead_ns over clean 1-row beats <= 500000 ns).
# Everything lands in $EV.
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
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EV=/home/drakestapleton/handoffs/dra-66-host-time/window-$STAMP
T0=$(date +%s)
COUNT='Count from 1 to 400 in English words, one number per line, nothing else.'
SERVER_PID=

ts() { date -u +%FT%TZ; }
board() { echo "$(ts) DRA66Host: $*" >> "$BOARD"; }
say() { echo "$(ts) $*" | tee -a "$EV/window.log"; }
not_run() {
  echo "NOT_RUN: $*"
  [ -d "$EV" ] && echo "$(ts) NOT_RUN: $*" >> "$EV/window.log"
  exit 3
}
elapsed() { echo $(( $(date +%s) - T0 )); }

mkdir -p "$EV" || not_run "cannot create $EV"

# ---- preconditions (no chip contact) ----
# 1. The gb10 lock is held, and by an ancestor of this shell.
[ -e "$LOCK" ] || not_run "$LOCK missing"
ino=$(stat -c %i "$LOCK")
holders=$(awk -v ino="$ino" '{ split($6, a, ":"); if (a[3] == ino) print $5 }' /proc/locks | sort -u)
[ -n "$holders" ] || not_run "$LOCK is not held by anyone; take it first (flock -w 900 $LOCK bash $0)"
anc=" "
p=$$
while [ "$p" -gt 1 ]; do
  anc="$anc$p "
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break
done
held_by_us=no
for h in $holders; do case "$anc" in *" $h "*) held_by_us=yes ;; esac; done
[ "$held_by_us" = yes ] || not_run "$LOCK held by pid(s) $holders, none an ancestor of $$ ($anc)"
# 2. The live 35B and Flint are stopped.
if curl -s -m 5 127.0.0.1:18091/health > "$EV/health-18091-before.json"; then
  not_run "18091 answers /health; the live 35B must be stopped by Main first"
fi
for port in 18091 18300 18310; do
  [ -z "$(ss -ltnH "sport = :$port")" ] || not_run "a listener is still on $port"
done
# 3. No compute process on the chip.
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) \
  || not_run "nvidia-smi failed: $apps"
echo "$apps" > "$EV/compute-apps-before.txt"
[ -z "$apps" ] || not_run "compute apps present: $apps"
# 4. Exact binary, golden, model, port, tools.
echo "$SHA_HEAD  $BIN" | sha256sum -c --status 2>/dev/null || not_run "$BIN sha256 mismatch"
[ -f "$GOLD" ] || not_run "golden $GOLD missing"
WANT=$(jq -c '.generated_ids' "$GOLD")
[ "$(echo "$WANT" | jq length)" = 24 ] || not_run "golden $GOLD has no 24 generated_ids"
[ -f "$MODEL/config.json" ] || not_run "model dir $MODEL missing"
[ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "port $PORT busy"
for t in curl jq ss sha256sum; do command -v $t > /dev/null || not_run "$t missing"; done

{
  echo "stamp $STAMP"; echo "lock holders $holders"; echo "ancestors$anc"
  echo "bin $BIN $SHA_HEAD"; echo "golden $GOLD $(sha256sum < "$GOLD" | cut -c1-64)"; uname -r
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  cat /proc/loadavg
} > "$EV/preconditions.txt"

# ---- server control ----
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM "$SERVER_PID" 2> /dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 1; done
  kill -KILL "$SERVER_PID" 2> /dev/null
  wait "$SERVER_PID" 2> /dev/null
  SERVER_PID=
}
finish() {
  stop_server
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$EV/compute-apps-after.txt" 2>&1
  board "GPU-END dra-66 window.sh $EV (elapsed $(elapsed) s; compute apps after: $(tr '\n' ';' < "$EV/compute-apps-after.txt"))"
}

# start_server LOG [ENV=VAL ...]: the live 18091 flags on a loopback test port, fail closed.
start_server() {
  local log=$1
  shift
  env AIEN_Q36_STROKE=1 AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-$PORT.lock "$@" \
    "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
    --max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8 \
    > "$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 300); do
    kill -0 "$SERVER_PID" 2> /dev/null || { say "server exited during load: $log"; SERVER_PID=; return 1; }
    curl -s -m 2 127.0.0.1:$PORT/health > /dev/null && break
    sleep 1
  done
  curl -s -m 2 127.0.0.1:$PORT/health > /dev/null || { say "server not healthy in 300 s"; stop_server; return 1; }
  local l
  l=$(ss -ltnH "sport = :$PORT" | awk '{print $4}')
  if [ -z "$l" ] || [ -n "$(echo "$l" | grep -v '^127\.0\.0\.1:')" ]; then
    say "listener check FAILED on $PORT: $l"
    stop_server
    return 1
  fi
  say "server pid $SERVER_PID up, listeners: $(echo $l)"
}

# chat OUT PROMPT MAX TEMP SEED: one chat request with token ids; prints "tokens wall_s".
chat() {
  local out=$1 prompt=$2 max=$3 temp=$4 seed=$5 body wall
  body=$(jq -nc --arg p "$prompt" --argjson m "$max" --argjson t "$temp" --argjson s "$seed" \
    '{model:"qwen3.6-35b-a3b-fp8",messages:[{role:"user",content:$p}],temperature:$t,seed:$s,max_tokens:$m,stream:false,chat_template_kwargs:{enable_thinking:false},aien_return_token_ids:true}')
  wall=$(curl -s -m 600 -o "$out" -w '%{time_total}' -H 'content-type: application/json' \
    -d "$body" 127.0.0.1:$PORT/v1/chat/completions) || { echo "fail 0"; return 1; }
  echo "$(jq -r '.usage.completion_tokens // "fail"' "$out") $wall"
}

# golden LABEL: the gate2 request (24 greedy tokens); prints IDENTICAL, DIFFER or ERROR.
golden() {
  local out=$EV/golden-$1.json
  chat "$out" 'Write a C function that returns the larger of two ints.' 24 0 0 > /dev/null
  local got
  got=$(jq -c '.usage.aien_token_ids // empty' "$out" 2> /dev/null)
  if [ -z "$got" ]; then echo ERROR; elif [ "$got" = "$WANT" ]; then echo IDENTICAL; else echo DIFFER; fi
}

board "GPU-START dra-66 window.sh $EV (lock held by $holders; 18091 + Flint stopped; compute apps empty)"
trap 'finish' EXIT
trap 'say "interrupted"; exit 4' INT TERM

# ---- phase 1: golden identity, mutant 4 first ----
say "phase 1a: mutant 4 must differ from the golden"
start_server "$EV/server-mutant4.log" AIEN_Q36_TEST_MUTANT=4 || not_run "mutant server did not come up"
m4=$(golden mutant4)
stop_server
say "mutant 4 golden: $m4 (must be DIFFER)"
[ "$m4" = DIFFER ] || { say "STOP: mutant 4 did not go red ($m4); no green below would count"; exit 4; }

# ---- phase 2: honest server, per-token receipt on ----
TOK=$EV/tokens.jsonl
say "phase 2: honest server, token receipt $TOK"
start_server "$EV/server-tokens.log" AIEN_Q36_DECODE_TIMELINE="$TOK" AIEN_Q36_DECODE_TIMELINE_KIND=token \
  || not_run "honest server did not come up"
grep -E 'per-token receipt' "$EV/server-tokens.log" | tee -a "$EV/window.log"
g=$(golden honest)
say "honest golden: $g (must be IDENTICAL)"
say "phase 2a: 1 conversation, 3 sequential greedy replies of 256 tokens"
for i in 1 2 3; do
  say "solo $i: tokens wall_s = $(chat "$EV/solo-$i.json" "$COUNT" 256 0 0)"
done
sleep 2 # the door closes the spell at idle
say "phase 2b: 8 conversations, 8 concurrent greedy replies of 256 tokens"
pids=()
for i in 1 2 3 4 5 6 7 8; do
  chat "$EV/eight-$i.json" "Count from $((i * 50)) to $((i * 50 + 300)) in English words, one number per line, nothing else." 256 0 0 \
    > "$EV/eight-$i.txt" &
  pids+=($!)
done
wait "${pids[@]}"
for i in 1 2 3 4 5 6 7 8; do say "eight $i: tokens wall_s = $(cat "$EV/eight-$i.txt")"; done
sleep 2
say "phase 2c: T=0.7 seeds through the door (seed 1, seed 2, seed 1 again; 64 tokens)"
for s in 1 2 1b; do chat "$EV/seed-$s.json" "$COUNT" 64 0.7 "${s%b}" > /dev/null; done
ids() { jq -c '.usage.aien_token_ids' "$EV/seed-$1.json"; }
if [ "$(ids 1)" = "$(ids 1b)" ] && [ "$(ids 1)" != "$(ids 2)" ] && [ "$(ids 1)" != null ]; then
  say "seed check: PASS (seed 1 repeats, seed 2 differs)"
else
  say "seed check: FAIL (seed1 $(ids 1 | cut -c1-80) / seed2 $(ids 2 | cut -c1-80) / seed1b $(ids 1b | cut -c1-80))"
fi
sleep 2
stop_server
grep -E 'spell over|per-token receipt' "$EV/server-tokens.log" > "$EV/spells.txt"
say "spell lines: $(wc -l < "$EV/spells.txt")"

# ---- phase 3: the receipt checker ----
say "phase 3: checker"
"$BIN" --check-decode-timeline "$TOK" > "$EV/summary.wl" 2> "$EV/check.txt"
echo "check exit $? (must be 0)" >> "$EV/check.txt"
"$BIN" --check-decode-timeline "$TOK" --drop 0:record:0 > /dev/null 2> "$EV/red-drop-first.txt"
echo "red control (drop the first record) exit $? (must be 1)" >> "$EV/red-drop-first.txt"
n=$(wc -l < "$TOK" 2> /dev/null || echo 0)
"$BIN" --check-decode-timeline "$TOK" --drop $(( n / 2 )):record:0 > /dev/null 2> "$EV/red-drop-middle.txt"
echo "red control (drop record $(( n / 2 ))) exit $? (must be 1)" >> "$EV/red-drop-middle.txt"
tail -n 1 "$EV/check.txt" "$EV/red-drop-first.txt" "$EV/red-drop-middle.txt" | tee -a "$EV/window.log"
say "records $n; Wolfram gate: evaluate $EV/summary.wl (first query = DRA-66 gate)"
say "done in $(elapsed) s"
