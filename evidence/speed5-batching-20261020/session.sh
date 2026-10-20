#!/usr/bin/env bash
# One server session on test port 18191 (never the live 18091), run inside an open spark-window.
# usage: session.sh <outdir> <call>...   each call is LABEL=i,j,k (prompt indexes sent at once)
# Env: SLOTS (0 = one-at-a-time door, else --batch-slots), KERNEL (tree|rowpair -> AIEN_Q38_BATCH_KERNEL),
#      FP8HEAD (1 -> AIEN_Q38_FP8_HEAD=1), MAXT (default 256), CTX (default 4096), BIN, MINMEM_GIB (default 50).
# Every call goes through bench.sh; the server.log lines it produced are cut into <label>.log, so
# summarize.sh never has to guess which spell belongs to which call. A call that comes back with an
# empty answer (the door gave up on the GPU lock) is repeated as LABEL-aK, every attempt kept.
set -euo pipefail
OUT=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
SLOTS=${SLOTS:-32}; MAXT=${MAXT:-256}; CTX=${CTX:-4096}; MINMEM_GIB=${MINMEM_GIB:-50}
BIN=${BIN:?BIN=path of aien-serve-http}
MODEL=$HOME/models/qwen3.8-27b-fp8-017b9c7a
PORT=18191; URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"
# wait for memory (a refused chip allocation is not a result)
while :; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$a" -ge "$MINMEM_GIB" ] && break
  echo "$(date -u +%FT%TZ) waiting: MemAvailable $a GiB < $MINMEM_GIB" >> "$OUT/memwait.log"; sleep 30
done
{ date -u +%FT%TZ; sha256sum "$BIN"; cat /proc/loadavg; grep MemAvailable /proc/meminfo; env | grep -E '^(SLOTS|KERNEL|FP8HEAD|MAXT|CTX|SPARK_WINDOW)=' || true; } > "$OUT/env.txt"
ARGS=(--engine omega --model-dir "$MODEL" --port $PORT --max-context "$CTX" --max-tokens-cap 16384 --thinking off)
[ "$SLOTS" -gt 0 ] && ARGS+=(--batch-slots "$SLOTS")
export AIEN_Q38_BATCH_KERNEL=${KERNEL:-tree}
[ "${FP8HEAD:-0}" = 1 ] && export AIEN_Q38_FP8_HEAD=1
echo "AIEN_Q38_BATCH_KERNEL=$AIEN_Q38_BATCH_KERNEL AIEN_Q38_FP8_HEAD=${AIEN_Q38_FP8_HEAD:-} nice -n 10 $BIN ${ARGS[*]}" > "$OUT/server.cmd"
setsid nohup nice -n 10 "$BIN" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!; echo $SPID > "$OUT/server.pid"
stop() { kill -TERM "$SPID" 2>/dev/null || true; for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; }
trap stop EXIT
for _ in $(seq 900); do
  curl -sf $URL/health > "$OUT/health-start.json" 2>/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died" >> "$OUT/calls.txt"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done
{ grep -E 'VmHWM|VmRSS' /proc/$SPID/status; grep MemAvailable /proc/meminfo; } > "$OUT/vm-after-load.txt"
for call in "$@"; do
  L=${call%%=*}; IDX=$(echo "${call#*=}" | tr ',' ' ')
  for k in 1 2 3 4 5 6; do
    a0=$(wc -l < "$OUT/server.log")
    echo "start $L-a$k $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) mem $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB" >> "$OUT/calls.txt"
    # shellcheck disable=SC2086
    "$HERE/bench.sh" $URL "$OUT" "$L-a$k" "$MAXT" $IDX
    sleep 2
    sed -n "$((a0 + 1)),\$p" "$OUT/server.log" > "$OUT/$L-a$k.log"
    echo "end   $L-a$k $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/calls.txt"
    grep -q "finish none" "$OUT/$L-a$k.summary" || { echo "$L $L-a$k" >> "$OUT/good.txt"; break; }
    [ "$k" = 6 ] && echo "$L FAILED after 6 attempts" >> "$OUT/good.txt"
  done
done
curl -s $URL/health > "$OUT/health-end.json" || true
grep -E 'VmHWM|VmRSS' /proc/$SPID/status > "$OUT/vm-end.txt"
echo "session done $(date -u +%FT%TZ)" >> "$OUT/calls.txt"
