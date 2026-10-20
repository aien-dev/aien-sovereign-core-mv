#!/bin/bash
# job.sh NAME "ENV..." TOOL ARGS... : one chip job under the GPU lock and a wall-power receipt; ground + command recorded.
# Runs inside an open spark-window (the driver opens it). Output in runs/NAME-<UTC>/.
set -u; NAME="$1"; E="$2"; shift 2; TS=$(date -u +%Y%m%dT%H%M%SZ)
EV=$HOME/workspace/speed7-work/runs/$NAME-$TS; mkdir -p "$EV"; cd "$EV"
B="$1"; shift
{ date -u; echo "env: $E"; echo "window: ${SPARK_WINDOW:-none}"; sha256sum "$B" $HOME/workspace/omega-speed7-fp8kv/lib-kv8/libomega_gpu.a;
  git -C $HOME/workspace/omega-speed7-fp8kv rev-parse HEAD; git -C $HOME/workspace/sc-speed7-fp8kv rev-parse HEAD;
  grep MemAvailable /proc/meminfo; echo "loadavg: $(cat /proc/loadavg)"; ps -eo pcpu,etime,comm --sort=-pcpu | head -6; } > ground.txt
# the window holds the chip lock (AIEN_GB10_LOCK_HELD=1): a plain flock here would wait for our own window
if [ "${AIEN_GB10_LOCK_HELD:-}" = 1 ]; then LK=""; else LK="flock /tmp/aien-gb10.lock"; fi
echo "$LK spark-receipt l131-$NAME -- env $E $B $*" > command.txt
# retry only the known channel refusal (RM_ALLOC 0xcec0 status 0x51 = NV_ERR_NO_MEMORY at channel create, GPU memory
# pressure from resident servers; see ledger L019 / L064 / L125), up to 6 tries 60 s apart; any other failure stands
for try in 1 2 3 4 5 6; do
  $LK spark-receipt "l131-$NAME" -- env $E nice "$B" "$@" > run.log 2>&1; rc=$?
  echo "rc=$rc try=$try" >> run.log
  [ $rc != 0 ] && grep -aq "class 0xcec0 status 0x51" run.log || break
  cp run.log run-try$try.log; echo "$NAME try $try: channel refusal 0x51, retry in 60 s" >> $HOME/workspace/speed7-work/runs/retries.log; sleep 60
done
{ date -u; grep MemAvailable /proc/meminfo; } >> ground.txt
grep -a "receipt:" run.log | tail -1 > receipt.txt
grep -aE "QWEN38|PASS|FAIL|chip context|memory guard|^chip " run.log | tail -12 | cut -c1-400 > summary.txt
echo "$NAME rc=$rc $EV" >> $HOME/workspace/speed7-work/runs/jobs.log
