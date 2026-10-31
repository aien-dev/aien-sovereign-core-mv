#!/bin/bash
# swap.sh <new-start-script>: stop keepalive, SIGTERM the live 27B (drains the reply in flight),
# install the new start script (old kept as .bak-<ts>), start, wait for /v1/models. Prints timestamps.
set -uo pipefail
S=$HOME/.local/state/aien-q38; NEW=$1; TS=$(date -u +%Y%m%dT%H%M%SZ)
ts() { date -u +%FT%T.%3NZ; }
echo "$(ts) stop keepalive $(cat $S/keepalive.pid)"; kill "$(cat $S/keepalive.pid)"
# Shortest downtime: wait (up to 10 min) for a moment with no client connected, so the drain is empty.
for i in $(seq 1 3000); do [ "$(ss -Htn state established "( sport = :18091 )" | wc -l)" = 0 ] && break; sleep 0.2; done
echo "$(ts) open client connections: $(ss -Htn state established "( sport = :18091 )" | wc -l)"
OLD=$(cat $S/serve.pid)
echo "$(ts) SIGTERM old server $OLD"; T0=$(date +%s.%N); kill -TERM "$OLD"
while kill -0 "$OLD" 2>/dev/null; do sleep 0.2; done
T1=$(date +%s.%N); echo "$(ts) old server exited"
cp -p $S/start-27b.sh $S/start-27b.sh.bak-$TS && cp "$NEW" $S/start-27b.sh && echo "$(ts) start script swapped (backup start-27b.sh.bak-$TS)"
$S/start-27b.sh
until curl -s -m 2 127.0.0.1:18091/v1/models | grep -q qwen3.8-27b-fp8; do
  kill -0 "$(cat $S/serve.pid)" 2>/dev/null || { echo "$(ts) NEW SERVER DIED"; exit 1; }
  sleep 0.5; done
T2=$(date +%s.%N); echo "$(ts) new server answers /v1/models, pid $(cat $S/serve.pid)"
echo "drain_s=$(echo "$T1 - $T0" | bc) load_s=$(echo "$T2 - $T1" | bc) down_s=$(echo "$T2 - $T0" | bc)"
