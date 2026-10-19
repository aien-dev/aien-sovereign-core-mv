#!/usr/bin/env bash
# Two small follow-up sessions on test port 18191 (never the live 18091), same binary as run.sh.
# usage: run-extra.sh repeat <outdir>   one-at-a-time door: prompt 3 three times, prompt 0 once (does the single door repeat itself?)
#        run-extra.sh mem <outdir>      batch door loaded with 1, 2, 4 slots, no requests: memory per slot from the load log (MemAvailable before/after the slots, VmRSS)
# Correctness / memory only: no timings published, so no quiet lock. The server takes the GPU lock itself.
set -euo pipefail
MODE=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/workspace/targets/q38-batch/bin-6a440c6d/aien-serve-http}
MODEL=$HOME/models/qwen3.8-27b-fp8-017b9c7a
PORT=18191; URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
NEED_KB=$((48*1024*1024))
waitmem() { while :; do a=$(awk '/MemAvailable/{print $2}' /proc/meminfo); [ "$a" -ge $NEED_KB ] && { echo "$(date -u +%FT%TZ) MemAvailable ${a} kB >= need, go"; return; }; echo "$(date -u +%FT%TZ) MemAvailable ${a} kB, waiting"; sleep 60; done; }
start() { # start <tag> <extra args...>
  local tag=$1; shift
  waitmem
  { date -u +%FT%TZ; sha256sum "$BIN"; cat /proc/loadavg; grep MemAvailable /proc/meminfo; } > "$OUT/env-$tag.txt"
  echo "nice -n 10 $BIN --engine omega --model-dir $MODEL --port $PORT --max-context 4096 --max-tokens-cap 16384 --thinking off $*" > "$OUT/server-$tag.cmd"
  setsid nohup nice -n 10 "$BIN" --engine omega --model-dir "$MODEL" --port $PORT --max-context 4096 --max-tokens-cap 16384 --thinking off "$@" > "$OUT/server-$tag.log" 2>&1 < /dev/null &
  SPID=$!
  for _ in $(seq 900); do
    curl -sf $URL/health > "$OUT/health-$tag.json" 2>/dev/null && break
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -5 "$OUT/server-$tag.log"; return 1; }
    sleep 2
  done
  { grep VmHWM /proc/$SPID/status; grep VmRSS /proc/$SPID/status; grep MemAvailable /proc/meminfo; } > "$OUT/vm-$tag.txt"
}
stop() { kill -TERM "$SPID" 2>/dev/null || true; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; }
if [ "$MODE" = repeat ]; then
  start single
  trap stop EXIT
  k=0
  for i in 3 3 3 0; do
    k=$((k+1))
    for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
      "$HERE/bench.sh" $URL "$OUT" "rep$k-p$i-a$try" 1 256 "$i"
      grep -q "finish none" "$OUT/rep$k-p$i-a$try.summary" || { echo "rep$k-p$i rep$k-p$i-a$try $i" >> "$OUT/good.txt"; break; }
    done
  done
  curl -s $URL/health > "$OUT/health-end.json"
else
  for s in 1 2 4; do
    start "slots$s" --batch-slots $s
    echo "slots $s: $(grep 'engine ready' "$OUT/server-slots$s.log" | cut -c1-400)" >> "$OUT/mem.txt"
    stop
    sleep 5
  done
fi
echo done
