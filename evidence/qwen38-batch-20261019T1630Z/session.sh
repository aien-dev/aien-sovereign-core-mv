#!/bin/bash
# Waits for memory headroom, then runs the single-door references and the batch session, each under spark-receipt.
E=$HOME/workspace/sc-batch/evidence/qwen38-batch-20261019T1630Z
NEED_KB=$((48*1024*1024))
waitmem() { while :; do a=$(awk "/MemAvailable/{print \$2}" /proc/meminfo); [ "$a" -ge $NEED_KB ] && { echo "$(date -u +%FT%TZ) MemAvailable ${a} kB >= need, go"; return; }; echo "$(date -u +%FT%TZ) MemAvailable ${a} kB, waiting"; sleep 60; done; }
TS=$(date -u +%Y%m%dT%H%MZ)
for mode in ${MODES:-single batch}; do
  if [ $mode = single ]; then export QUIET=0; else export QUIET=1 CORRECT_FIRST=1; fi
  waitmem
  spark-receipt q38-batch-$mode-$TS -- "$E/run.sh" $mode "$E/$mode-$TS"
  echo "$(date -u +%FT%TZ) $mode exit $?"
  cp ~/yardsticks/receipts/*-q38-batch-$mode-$TS.json "$E/$mode-$TS/" 2>/dev/null
done
