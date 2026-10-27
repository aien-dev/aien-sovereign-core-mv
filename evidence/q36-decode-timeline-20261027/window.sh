#!/usr/bin/env bash
# q36 decode timeline window (#785). DO NOT RUN outside an exclusive window opened by Drake.
#
# Assumes Main holds /tmp/aien-gb10.lock (an ancestor of this shell holds it, e.g.
#   flock -w 900 /tmp/aien-gb10.lock bash /home/drakestapleton/handoffs/DecodeTimeline/window.sh
# ) and has stopped the live 35B on 18091. Any failed precondition prints NOT_RUN and exits 3
# without touching the chip.
#
# Phase 1: head binary, AIEN_Q36_DECODE_TIMELINE on, 200 greedy tokens on a fixed prompt, then
#          the checker (green), its two red controls, and the Wolfram-ready summary.
# Phase 2: zero-cost chip A/B, A = bf51e6d6 (main + lock restore only) vs B = head with the switch
#          unset (PROTOCOL.md Amendment 1 in the evidence
#          folder): 4 server blocks in a seeded order, 1 warm-up + 10 requests each.
# Everything lands in $EV. Wolfram computes the medians afterwards from $EV/*.wl.
set -u
umask 022

W=/home/drakestapleton/work/DecodeTimeline
BIN_HEAD=$W/bin-head/aien-serve-http
BIN_BASE=$W/bin-a/aien-serve-http
SHA_HEAD=6eca898f1628ac93b16ac5f0ab7772d310b8b51bdbd5e4b236e7306369ad9c10 # sc f551b1e2 + omega a647b6a9
SHA_BASE=e7c2e61a814421a8b22046a408affbe4fc19c3f5e552726e56359152a50bcab7 # sc bf51e6d6 + omega a647b6a9
MODEL=/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0
PORT=18190
LOCK=/tmp/aien-gb10.lock
BOARD=/home/drakestapleton/handoffs/BOARD.md
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EV=/home/drakestapleton/handoffs/DecodeTimeline/window-$STAMP
BUDGET_S=840 # stop starting new work after 14 min; the hold limit is 15
T0=$(date +%s)
PROMPT='Count from 1 to 400 in English words, one number per line, nothing else.'
SERVER_PID=

ts() { date -u +%FT%TZ; }
board() { echo "$(ts) DecodeTimeline: $*" >> "$BOARD"; }
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
# 2. The live 35B is stopped: no health answer, no listener on 18091.
if curl -s -m 5 127.0.0.1:18091/health > "$EV/health-18091-before.json"; then
  not_run "18091 answers /health; the live 35B must be stopped by Main first"
fi
[ -z "$(ss -ltnH 'sport = :18091')" ] || not_run "a listener is still on 18091"
# 3. No compute process on the chip.
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1) \
  || not_run "nvidia-smi failed: $apps"
echo "$apps" > "$EV/compute-apps-before.txt"
[ -z "$apps" ] || not_run "compute apps present: $apps"
# 4. Exact binaries.
echo "$SHA_HEAD  $BIN_HEAD" | sha256sum -c --status 2>/dev/null || not_run "$BIN_HEAD sha256 mismatch"
echo "$SHA_BASE  $BIN_BASE" | sha256sum -c --status 2>/dev/null || not_run "$BIN_BASE sha256 mismatch"
# 5. Model, ports, tools.
[ -f "$MODEL/config.json" ] || not_run "model dir $MODEL missing"
[ -z "$(ss -ltnH "sport = :$PORT")" ] || not_run "port $PORT busy"
for t in curl jq ss python3 flock; do command -v $t > /dev/null || not_run "$t missing"; done

{
  echo "stamp $STAMP"; echo "lock holders $holders"; echo "ancestors$anc"
  echo "head $SHA_HEAD"; echo "base $SHA_BASE"; uname -r
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
  curl -s -m 5 127.0.0.1:18091/health > "$EV/health-18091-after.json"
  board "GPU-END window.sh $EV (elapsed $(elapsed) s; compute apps after: $(tr '\n' ';' < "$EV/compute-apps-after.txt"))"
}

# start_server BIN LOG [ENV=VAL ...]: loopback only, fail closed.
start_server() {
  local bin=$1 log=$2
  shift 2
  env AIEN_Q36_STROKE=1 AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-$PORT.lock "$@" \
    "$bin" --engine omega --model-dir "$MODEL" --port $PORT --bind 127.0.0.1 \
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
  say "server pid $SERVER_PID up ($bin), listeners: $(echo $l)"
}

# request OUT MAX: one greedy chat request; prints "tokens wall_s 18091_in_flight".
request() {
  local out=$1 max=$2 inflight
  inflight=$(curl -s -m 2 127.0.0.1:18091/health | jq -r '.in_flight // "down"' 2> /dev/null)
  [ -n "$inflight" ] || inflight=down
  local body
  body=$(jq -nc --arg p "$PROMPT" --argjson m "$max" \
    '{model:"qwen3.6-35b-a3b-fp8",messages:[{role:"user",content:$p}],temperature:0,max_tokens:$m,stream:false}')
  local wall
  wall=$(curl -s -m 300 -o "$out" -w '%{time_total}' -H 'content-type: application/json' \
    -d "$body" 127.0.0.1:$PORT/v1/chat/completions) || { echo "fail 0 $inflight"; return 1; }
  echo "$(jq -r '.usage.completion_tokens // "fail"' "$out") $wall $inflight"
}

board "GPU-START window.sh $EV (lock held by $holders; 18091 stopped; compute apps empty)"
trap 'finish' EXIT
trap 'say "interrupted"; exit 4' INT TERM

# ---- phase 1: the timeline ----
TL=$EV/timeline.jsonl
say "phase 1: timeline"
if start_server "$BIN_HEAD" "$EV/server-timeline.log" AIEN_Q36_DECODE_TIMELINE="$TL"; then
  r=$(request "$EV/timeline-reply.json" 200)
  say "timeline request: tokens wall_s 18091_in_flight = $r"
  sleep 2 # the door flushes the last record at idle
  stop_server
  n=$(wc -l < "$TL" 2> /dev/null || echo 0)
  tok=${r%% *}
  say "records $n (one per decode step: expect completion_tokens - 1 = $(( ${tok:-1} - 1 )) when the reply is a single run)"
  grep -E 'decode timeline' "$EV/server-timeline.log" | tee -a "$EV/window.log"
  "$BIN_HEAD" --check-decode-timeline "$TL" > "$EV/summary.wl" 2> "$EV/check.txt"
  echo "check exit $?" >> "$EV/check.txt"
  "$BIN_HEAD" --check-decode-timeline "$TL" --drop 5:dev:3 > /dev/null 2> "$EV/red-drop-dev.txt"
  echo "red control (drop one launch) exit $? (must be 1)" >> "$EV/red-drop-dev.txt"
  "$BIN_HEAD" --check-decode-timeline "$TL" --drop 7:record:0 > /dev/null 2> "$EV/red-drop-record.txt"
  echo "red control (drop one record) exit $? (must be 1)" >> "$EV/red-drop-record.txt"
  tail -n 1 "$EV/check.txt" "$EV/red-drop-dev.txt" "$EV/red-drop-record.txt" | tee -a "$EV/window.log"
else
  say "phase 1 NOT_RUN: server did not come up"
fi

# ---- phase 2: zero-cost chip A/B (switch unset) ----
order=$(python3 -c '
import itertools, random
orders = sorted(set(itertools.permutations("AABB")))
print("".join(random.Random(785).choice(orders)))')
say "phase 2: A/B order $order (seed 785)"
echo "block,arm,i,tokens,wall_s,in_flight_18091,text_sha256" > "$EV/ab.csv"
ref_text=
b=0
for arm in $(echo "$order" | fold -w1); do
  b=$((b + 1))
  if [ "$(elapsed)" -gt "$BUDGET_S" ]; then
    say "block $b ($arm) NOT_RUN: time budget $BUDGET_S s used"
    continue
  fi
  bin=$BIN_HEAD
  [ "$arm" = A ] && bin=$BIN_BASE
  start_server "$bin" "$EV/server-ab-$b-$arm.log" || { say "block $b ($arm) NOT_RUN: server"; continue; }
  request "$EV/ab-$b-$arm-warm.json" 200 > /dev/null
  for i in $(seq 10); do
    r=$(request "$EV/ab-$b-$arm-$i.json" 200)
    text=$(jq -r '.choices[0].message.content // ""' "$EV/ab-$b-$arm-$i.json" | sha256sum | cut -c1-16)
    [ -n "$ref_text" ] || ref_text=$text
    echo "$b,$arm,$i,$(echo "$r" | tr ' ' ','),$text" >> "$EV/ab.csv"
    [ "$text" = "$ref_text" ] || say "CORRECTNESS: block $b $arm $i text $text != $ref_text (A/B void)"
  done
  grep 'spell over' "$EV/server-ab-$b-$arm.log" > "$EV/spells-$b-$arm.txt"
  stop_server
done

# Wolfram-ready A/B lists: tokens/s per kept sample (18091 in_flight must be "down" or 0).
python3 - "$EV/ab.csv" > "$EV/ab.wl" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
kept = {"A": [], "B": []}
dropped = 0
for r in rows:
    if r["tokens"] in ("fail", "") or r["in_flight_18091"] not in ("down", "0"):
        dropped += 1
        continue
    kept[r["arm"]].append(f'{r["tokens"]}/{r["wall_s"]}')
print("(* tokens/s = completion_tokens / wall_s; discarded samples: %d *)" % dropped)
print("abA = {%s};" % ", ".join(kept["A"]))
print("abB = {%s};" % ", ".join(kept["B"]))
print("{Length[abA], Length[abB], N[Median[abA]], N[Median[abB]], N[Median[abB]/Median[abA]], N[Median[abB]/Median[abA]] >= 99/100}")
PY
say "A/B lists in $EV/ab.wl ($(grep -c . "$EV/ab.csv") csv lines); summary in $EV/summary.wl"
say "done in $(elapsed) s"
