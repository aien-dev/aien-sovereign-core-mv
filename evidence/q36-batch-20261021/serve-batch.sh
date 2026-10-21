#!/bin/bash
# L354 test server: the batch door on 127.0.0.1:$PORT (env PORT, default 18094; never 18091), detached. Env passes through
# (AIEN_Q36_TEST_ROW_SWAP=k = the negative control). Loads under q36-extra (L366: one extra 35B at a time, memory
# checked inside its lock, oom_score_adj 1000 so the kernel takes this server, never 18091). Usage: serve-batch.sh <bin-sha8> <slots> <max_context> <tag>
# Stop: kill $(cat ~/.local/state/aien-q36/batch-$PORT.pid)
set -euo pipefail
PORT=${PORT:-18094}
S=$HOME/.local/state/aien-q36; BIN=bin-${1:?sha8}; SL=${2:?slots}; CTX=${3:?ctx}; TAG=${4:?tag}
cd "$S/$BIN"; sha256sum -c --quiet SHA256SUMS
ss -ltn | grep -q "127.0.0.1:$PORT " && { echo "$PORT already in use"; exit 2; }
echo "=== $(date -u +%FT%TZ) start batch door $TAG ($BIN, $SL slots, ctx $CTX, swap=${AIEN_Q36_TEST_ROW_SWAP:-none}) MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB" >> "$S/batch-$PORT.log"
# Inside the slot: load only while 18091 is up and loaded (12:00Z: q36-extra admitted an extra while 18091 was restarting).
setsid nohup $HOME/.local/bin/q36-extra bash -c 'curl -s -m 5 http://127.0.0.1:18091/v1/models | grep -q "\"id\"" || { echo "18091 not serving: not loading the extra" >&2; exit 75; }; exec "$@"' extra ./aien-serve-http --engine omega --model-dir "$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0" \
  --port $PORT --max-context $CTX --max-tokens-cap 16384 --thinking off --batch-slots $SL >> "$S/batch-$PORT.log" 2>&1 < /dev/null &
echo $! > "$S/batch-$PORT.pid"
echo "pid $(cat "$S/batch-$PORT.pid"), log $S/batch-$PORT.log"
