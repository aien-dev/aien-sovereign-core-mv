#!/bin/bash
# Keeps port 18091 served: every 60 s, if the 27B server's pid is gone and nothing listens on
# 18091, run start-27b.sh. At most 3 restarts per hour (a failing load does not loop). Started
# detached by the q38-serve chat; stop with: kill $(cat ~/.local/state/aien-q38/keepalive.pid)
S=$HOME/.local/state/aien-q38
starts=()
while sleep 60; do
  kill -0 "$(cat "$S/serve.pid" 2>/dev/null)" 2>/dev/null && continue
  ss -ltn | grep -q '127.0.0.1:18091 ' && continue
  now=$(date +%s); recent=()
  for t in "${starts[@]}"; do [ $((now - t)) -lt 3600 ] && recent+=("$t"); done
  starts=("${recent[@]}")
  if [ ${#starts[@]} -ge 3 ]; then echo "$(date -u +%FT%TZ) keepalive: 3 restarts this hour, not retrying" >> "$S/keepalive.log"; continue; fi
  echo "$(date -u +%FT%TZ) keepalive: 18091 dead, starting the 27B" >> "$S/keepalive.log"
  "$S/start-27b.sh" >> "$S/keepalive.log" 2>&1
  starts+=("$now")
done
