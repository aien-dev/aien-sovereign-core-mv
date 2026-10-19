#!/bin/bash
# one.sh NAME "ENV..." TOOL ARGS... : one chip run under the GPU lock, ground + command recorded, binary copied.
# The lock is taken only when 58 GiB is free at that moment (non-blocking try, 30 s between tries): waiting on a
# blocking flock let the lock holder use the memory, so the run then hit its own memory guard (2026-10-19).
set -u; NAME="$1"; E="$2"; shift 2; TS=$(date -u +%Y%m%dT%H%M%SZ)
EV=$HOME/workspace/ab-w4-work/runs/$NAME-$TS; mkdir -p "$EV"; cd "$EV"
B="$1"; shift
echo "flock -n /tmp/aien-gb10.lock env $E nice $B $*" > command.txt
room() { [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $((58*1024*1024)) ]; }
while :; do
  if room; then
    { date -u; echo "env: $E"; sha256sum "$B" ${LIB:-}; grep MemAvailable /proc/meminfo; echo "server pid $(cat ~/.local/state/aien-q38/serve.pid)"; } > ground.txt
    flock -n -E 75 /tmp/aien-gb10.lock env $E nice "$B" "$@" > run.log 2>&1; rc=$?
    [ $rc -ne 75 ] && break
  fi
  sleep 30
done
echo "rc=$rc" >> run.log
{ date -u; grep MemAvailable /proc/meminfo; } >> ground.txt
tail -3 run.log | cut -c1-400 > summary.txt
