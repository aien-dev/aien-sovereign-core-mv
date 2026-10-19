#!/bin/bash
# one.sh NAME "ENV..." TOOL ARGS... : one chip run under the GPU lock, ground + command recorded.
# v3 (2026-10-19): wait for 58 GiB free, then queue on the lock like everyone else (v2's non-blocking tries lost
# every race to blocking waiters); if the memory was taken while queued, the tool's own guard refuses in seconds,
# the lock is released at once, and we go round again (at most 40 rounds).
set -u; NAME="$1"; E="$2"; shift 2; TS=$(date -u +%Y%m%dT%H%M%SZ)
EV=$HOME/workspace/ab-w4-work/runs/$NAME-$TS; mkdir -p "$EV"; cd "$EV"
B="$1"; shift
echo "flock /tmp/aien-gb10.lock env $E nice $B $*" > command.txt
room() { [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $((58*1024*1024)) ]; }
for round in $(seq 40); do
  until room; do sleep 30; done
  { date -u; echo "env: $E"; sha256sum "$B" ${LIB:-}; grep MemAvailable /proc/meminfo; echo "server pid $(cat ~/.local/state/aien-q38/serve.pid)"; } > ground.txt
  flock /tmp/aien-gb10.lock env $E nice "$B" "$@" > run.log 2>&1; rc=$?
  grep -q 'memory guard' run.log || break
  echo "round $round refused by memory guard at $(date -u +%H:%M:%SZ)" >> refusals.txt
done
echo "rc=$rc" >> run.log
{ date -u; grep MemAvailable /proc/meminfo; } >> ground.txt
tail -3 run.log | cut -c1-400 > summary.txt
