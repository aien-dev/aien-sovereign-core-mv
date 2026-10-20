#!/bin/bash
# one.sh NAME "ENV..." TOOL ARGS... : one chip run inside the open spark-window, under the GB10 lock, wrapped by
# spark-receipt (wall J); ground (binary + lib sha, MemAvailable, window id) and the command recorded. No retries:
# a failure is published as it is.
set -u; NAME="$1"; E="$2"; shift 2; TS=$(date -u +%Y%m%dT%H%M%SZ)
EV=$HOME/workspace/speed1-work/runs/$NAME-$TS; mkdir -p "$EV"; cd "$EV"
B="$1"; shift
echo "[window holds the chip lock: ${AIEN_GB10_LOCK_HELD:-0}] env $E nice $B $*" > command.txt
{ date -u; echo "window: ${SPARK_WINDOW:-none}"; echo "env: $E"; sha256sum "$B" $LIB; grep MemAvailable /proc/meminfo; } > ground.txt
LK="flock /tmp/aien-gb10.lock"; [ "${AIEN_GB10_LOCK_HELD:-}" = 1 ] && LK=""   # the window holds the chip lock
spark-receipt "l125-$NAME" --idle 10 -- $LK env $E nice "$B" "$@" > run.log 2>&1; rc=$?
echo "rc=$rc" >> run.log
{ date -u; grep MemAvailable /proc/meminfo; } >> ground.txt
grep -o 'receipt: .*' run.log | tail -1 > receipt-path.txt
tail -4 run.log | cut -c1-400 > summary.txt
exit $rc
