#!/bin/bash
# L354 test server: the batch door on 127.0.0.1:$PORT (env PORT, default 18094; never 18091), detached. Env passes through
# (AIEN_Q36_TEST_ROW_SWAP=k = the negative control). Usage: serve-batch.sh <bin-sha8> <slots> <max_context> <tag>
# Stop: kill $(cat ~/.local/state/aien-q36/batch-$PORT.pid)
set -euo pipefail
PORT=${PORT:-18094}
S=$HOME/.local/state/aien-q36; BIN=bin-${1:?sha8}; SL=${2:?slots}; CTX=${3:?ctx}; TAG=${4:?tag}
cd "$S/$BIN"; sha256sum -c --quiet SHA256SUMS
ss -ltn | grep -q "127.0.0.1:$PORT " && { echo "$PORT already in use"; exit 2; }
echo "=== $(date -u +%FT%TZ) start batch door $TAG ($BIN, $SL slots, ctx $CTX, swap=${AIEN_Q36_TEST_ROW_SWAP:-none}) MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB" >> "$S/batch-$PORT.log"
setsid nohup ./aien-serve-http --engine omega --model-dir "$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0" \
  --port $PORT --max-context $CTX --max-tokens-cap 16384 --thinking off --batch-slots $SL >> "$S/batch-$PORT.log" 2>&1 < /dev/null &
echo $! > "$S/batch-$PORT.pid"; echo 1000 > /proc/$(cat "$S/batch-$PORT.pid")/oom_score_adj 2>/dev/null || true  # out of memory: the kernel takes this test server, never 18091
echo "pid $(cat "$S/batch-$PORT.pid"), log $S/batch-$PORT.log"
