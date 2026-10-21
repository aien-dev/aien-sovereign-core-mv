#!/bin/bash
# L375 chip gate: shared prompt cache on the 35B batch door. Loads an extra 35B ONLY under q36-extra
# (port 18097, 1 row, never 18091). Phases: B pool+store (cold, second chat, cold-vs-restored, tool
# change, goldens), C restart on the same store (first chat after restart, restored-from-disk words),
# D position mutant (must go red). Usage: gate.sh <dir holding aien-serve-http>
set -uo pipefail
BIN=$1
EV=$HOME/workspace/sc-q36-prefix/evidence/l375
OD=$EV/run-$(date -u +%Y%m%dT%H%MZ); mkdir -p "$OD"
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
PORT=18097
NEED=${Q36_EXTRA_NEED_GB:-44}
POOL_GB=${POOL_GB:-6}
STORE=$(mktemp -d /tmp/l375-store-XXXXXX); chmod 700 "$STORE"
KEY=$(openssl rand -hex 32)   # a test key, held in this shell only
cd "$EV"
./mkreq.sh "$OD/req1.json" "In one sentence, which tool would you use to read a file?" > /dev/null
./mkreq.sh "$OD/req2.json" "Name two tools that could list a directory." > /dev/null
./mkreq.sh "$OD/req3.json" "In one sentence, which tool would you use to read a file?" 7 > /dev/null
./mkreq.sh "$OD/req4.json" "What is the date today, according to your instructions?" > /dev/null
./mkreq.sh "$OD/g1.json" "Write a haiku about a lighthouse." -1 4 > /dev/null
./mkreq.sh "$OD/g2.json" "Explain what a hash table is in two sentences." -1 4 > /dev/null
sha256sum "$BIN/aien-serve-http" > "$OD/bin.sha256"
SPID=""
start() {
  local tag=$1; shift
  echo "$(date -u +%FT%TZ) phase $tag: waiting for the q36-extra slot" | tee -a "$OD/gate.log"
  Q36_EXTRA_WAIT=7200 Q36_EXTRA_NEED_GB=$NEED q36-extra env AIEN_Q36_STROKE=0 AIEN_Q36_PREFIX_POOL_GB=$POOL_GB "$@" \
    "$BIN/aien-serve-http" --engine omega --model-dir "$M" --port $PORT --max-context 32768 \
    --max-tokens-cap 512 --thinking off --batch-slots 1 > "$OD/server-$tag.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 1200); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { echo "$(date -u +%FT%TZ) phase $tag: up" | tee -a "$OD/gate.log"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "phase $tag: server exited" | tee -a "$OD/gate.log"; tail -5 "$OD/server-$tag.log" >> "$OD/gate.log"; return 1; }
    sleep 3
  done
  return 1
}
stop() {
  # Let the store writer finish (no temp files, file count steady) before stopping.
  local prev=-1 n
  for i in $(seq 1 120); do
    n=$(ls "$STORE" | grep -c .); ls -a "$STORE" | grep -q '\.tmp$' || { [ "$n" = "$prev" ] && break; }
    prev=$n; sleep 2
  done
  kill $SPID; wait $SPID 2>/dev/null
  echo "$(date -u +%FT%TZ) stopped, store holds $(ls "$STORE" | grep -c q36p) nodes, $(du -sh "$STORE" | cut -f1)" | tee -a "$OD/gate.log"
}
ask() {
  ./ask.sh $PORT "$OD/$1.json" "$2" "$OD" >> "$OD/gate.log"
  if [ ! -s "$OD/$2.ids" ]; then
    echo "ABORT: $2 gave no tokens: $(head -c 300 "$OD/$2.sse")" | tee -a "$OD/gate.log"; kill $SPID; wait $SPID 2>/dev/null; rm -rf "$STORE"; exit 1
  fi
}

start B AIEN_Q36_PREFIX_DIR="$STORE" AIEN_KV_STORE_KEY="$KEY" || exit 1
ask req1 b1; ask req2 b2; ask req1 b1r; ask req3 b3; ask req1 b1x
ask g1 g1a; ask g2 g2a; ask g1 g1b; ask g2 g2b
stop
start C AIEN_Q36_PREFIX_DIR="$STORE" AIEN_KV_STORE_KEY="$KEY" || exit 1
ask req4 c4; ask req1 c1
stop
start D AIEN_Q36_PREFIX_MUTANT=1 || exit 1
ask g1 d1a; ask g2 d2; ask g1 d1b
stop
rm -rf "$STORE"
same() { cmp -s "$OD/$1.ids" "$OD/$2.ids" && [ -s "$OD/$1.ids" ]; }
{
  echo "== verdict"
  same b1 b1r && echo "PASS cold req1 == restored req1 (pool, same server)" || echo "FAIL cold req1 != restored req1"
  same b1 b1x && echo "PASS cold req1 == req1 after the tool-change chat" || echo "FAIL req1 after tool change differs"
  same b1 c1  && echo "PASS cold req1 == req1 restored from disk after restart" || echo "FAIL req1 from disk differs"
  same g1a g1b && echo "PASS golden g1 cold == restored" || echo "FAIL golden g1"
  same g2a g2b && echo "PASS golden g2 cold == restored" || echo "FAIL golden g2"
  same g1a d1a && echo "PASS mutant server cold g1 == cold g1" || echo "FAIL mutant server cold g1 differs"
  same d1a d1b && echo "FAIL mutant NOT caught (restored one position early, same words)" || echo "PASS mutant red: restore one position early changes the words"
  echo "== times"; cat "$OD/times.txt"
  echo "== door log"; grep -h "join:\|prompt cache restore\|store .* open\|prompt cache on\|tool marks\|tool order" "$OD"/server-*.log | sed 's/^aien-serve-http: //' | head -60
} | tee "$OD/verdict.txt"
