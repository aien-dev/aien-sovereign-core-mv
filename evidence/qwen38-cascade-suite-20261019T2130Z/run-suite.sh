#!/bin/sh
# One suite = one model upload, many gen runs (qwen38_spec suite). $1 = suite name (plan-<name>.txt in this folder).
# Pinned binary built from e166f57f with omega 37034f03 (lib sha 6e905209...).
# Waits for MemAvailable >= 46 GiB (the tool needs ~32 GiB + 12 GiB kept free), then takes quiet (exclusive) and the GPU lock
# for the whole suite. A refusal for memory is kept as <name>.refused-<i>.log.txt and retried (at most 30 times).
set -u
cd $HOME/workspace/sc-cascade
E=evidence/qwen38-cascade-suite-20261019T2130Z
BIN=$HOME/workspace/targets/sc-cascade/pinned/qwen38_spec-e166f57f
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
. $HOME/workspace/targets/sc-cascade/env.sh
name=$1
i=0
while :; do
  while [ $(awk "/MemAvailable/{print \$2}" /proc/meminfo) -lt 48234496 ]; do sleep 30; done
  {
  echo "bin sha $(sha256sum $BIN | cut -c1-64) (built from e166f57f)"
  echo "lib ~/workspace/targets/q38-f7-omega/libomega_gpu.a sha 6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff (omega 37034f03)"
  echo "spark-receipt cascade-suite-$name -- ~/.local/bin/quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN suite M $E $E/plan-$name.txt"
  echo "queued $(date -u +%FT%TZ) load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)"
  } >> $E/command-$name.txt
  ~/.local/bin/spark-receipt cascade-suite-$name -- ~/.local/bin/quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN suite $M $E $E/plan-$name.txt > $E/suite-$name.log.txt 2>&1
  rc=$?
  echo "end $(date -u +%FT%TZ) rc $rc load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)" >> $E/command-$name.txt
  cp ~/yardsticks/receipts/$(date +%F)-cascade-suite-$name.json $E/receipt-$name.json 2>/dev/null
  if grep -q refusing $E/suite-$name.log.txt && [ $i -lt 30 ]; then
    i=$((i + 1)); mv $E/suite-$name.log.txt $E/suite-$name.refused-$i.log.txt; sleep 120; continue
  fi
  break
done
