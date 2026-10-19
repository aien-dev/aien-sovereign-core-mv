#!/usr/bin/env bash
# One measurement session on test port 18191 (never the live 18091).
# usage: run.sh <mode> <outdir>
#   mode single : the one-at-a-time door (no --batch-slots): single-stream greedy references, 8 prompts alone
#   mode batch  : --batch-slots 8: n=1 (each prompt alone), then n=2,4,8 for ROUNDS rounds
# Env: MAXT (max_tokens, default 256), ROUNDS (default 3), CTX (max-context, default 4096)
# The server takes /tmp/aien-gb10.lock itself at load and per busy spell; this script holds no lock.
set -euo pipefail
MODE=$1; OUT=$2; MAXT=${MAXT:-256}; ROUNDS=${ROUNDS:-3}; CTX=${CTX:-4096}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/workspace/targets/q38-batch/bin-6a440c6d/aien-serve-http}
MODEL=$HOME/models/qwen3.8-27b-fp8-017b9c7a
PORT=18191; URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
{ date -u +%FT%TZ; sha256sum "$BIN"; cat /proc/loadavg; grep MemAvailable /proc/meminfo; } > "$OUT/env.txt"
ARGS=(--engine omega --model-dir "$MODEL" --port $PORT --max-context "$CTX" --max-tokens-cap 16384 --thinking off)
[ "$MODE" = batch ] && ARGS+=(--batch-slots 8)
echo "nice -n 10 $BIN ${ARGS[*]}" > "$OUT/server.cmd"
setsid nohup nice -n 10 "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!; echo $SPID > "$OUT/server.pid"
stop() { kill -TERM "$SPID" 2>/dev/null || true; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; }
trap stop EXIT
for _ in $(seq 900); do
  curl -sf $URL/health > "$OUT/health-start.json" 2>/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done
grep VmHWM /proc/$SPID/status > "$OUT/vm-after-load.txt"; grep VmRSS /proc/$SPID/status >> "$OUT/vm-after-load.txt"
grep MemAvailable /proc/meminfo >> "$OUT/vm-after-load.txt"
if [ "$MODE" = single ]; then
  for i in 0 1 2 3 4 5 6 7; do "$HERE/bench.sh" $URL "$OUT" single-p$i 1 "$MAXT" $i; done
else
  for i in 0 1 2 3 4 5 6 7; do "$HERE/bench.sh" $URL "$OUT" n1-p$i 1 "$MAXT" $i; done
  for r in $(seq "$ROUNDS"); do
    for n in 2 4 8; do "$HERE/bench.sh" $URL "$OUT" n$n-r$r "$n" "$MAXT"; done
  done
fi
curl -s $URL/health > "$OUT/health-end.json"
grep VmHWM /proc/$SPID/status > "$OUT/vm-end.txt"; grep VmRSS /proc/$SPID/status >> "$OUT/vm-end.txt"
echo done
