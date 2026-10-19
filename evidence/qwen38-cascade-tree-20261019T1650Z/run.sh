#!/bin/sh
# Cascade tree measurements. $1 = name, rest = qwen38_spec gen flags.
# Pinned binary built from 7ec06216 with omega 37034f03 (lib sha 6e905209...).
# Waits for MemAvailable >= 46 GiB (the tool needs ~32 GiB + 12 GiB kept free) before taking the locks;
# published timings run under ~/.local/bin/quiet-measure (quiet lock exclusive) then the GPU lock, held only for the run.
# A run the tool refuses for memory is kept as <name>.refused-<i>.log.txt and retried (at most 20 times).
set -u
cd $HOME/workspace/sc-cascade
E=evidence/qwen38-cascade-tree-20261019T1650Z
BIN=$HOME/workspace/targets/sc-cascade/pinned/qwen38_spec-7ec06216
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
name=$1; shift
i=0
while :; do
  while [ $(awk "/MemAvailable/{print \$2}" /proc/meminfo) -lt 48234496 ]; do sleep 30; done
  {
  echo "bin sha $(sha256sum $BIN | cut -c1-64) (built from 7ec06216)"
  echo "lib ~/workspace/targets/q38-f7-omega/libomega_gpu.a sha 6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff (omega 37034f03)"
  echo "spark-receipt cascade-$name -- ~/.local/bin/quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN gen M $E/$name.json $*"
  echo "queued $(date -u +%FT%TZ) load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)"
  } >> $E/command-$name.txt
  ~/.local/bin/spark-receipt cascade-$name -- ~/.local/bin/quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN gen $M $E/$name.json "$@" > $E/$name.log.txt 2>&1
  rc=$?
  echo "end $(date -u +%FT%TZ) rc $rc load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)" >> $E/command-$name.txt
  cp ~/yardsticks/receipts/$(date +%F)-cascade-$name.json $E/receipt-$name.json 2>/dev/null
  if grep -q refusing $E/$name.log.txt && [ $i -lt 20 ]; then
    i=$((i + 1)); mv $E/$name.log.txt $E/$name.refused-$i.log.txt; sleep 120; continue
  fi
  break
done
