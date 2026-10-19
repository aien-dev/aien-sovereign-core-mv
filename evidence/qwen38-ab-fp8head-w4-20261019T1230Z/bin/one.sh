#!/bin/bash
# one.sh NAME "ENV..." TOOL ARGS... : one chip run under the GPU lock, ground + command recorded, binary copied.
set -u; NAME="$1"; E="$2"; shift 2; TS=$(date -u +%Y%m%dT%H%M%SZ)
EV=$HOME/workspace/ab-w4-work/runs/$NAME-$TS; mkdir -p "$EV"; cd "$EV"
B="$1"; shift
{ date -u; echo "env: $E"; sha256sum "$B" ${LIB:-}; grep MemAvailable /proc/meminfo; echo "server pid $(cat ~/.local/state/aien-q38/serve.pid)"; } > ground.txt
echo "flock /tmp/aien-gb10.lock env $E nice $B $*" > command.txt
flock /tmp/aien-gb10.lock env $E nice "$B" "$@" > run.log 2>&1; echo "rc=$?" >> run.log
{ date -u; grep MemAvailable /proc/meminfo; } >> ground.txt
tail -3 run.log | cut -c1-400 > summary.txt
