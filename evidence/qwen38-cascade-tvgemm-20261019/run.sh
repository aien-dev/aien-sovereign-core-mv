#!/bin/sh
# Tree-verify GEMM in the cascade (sc cascade/tree-verify-wire e2080d1e). $1 = name, rest = qwen38_spec gen flags.
# Pinned binary = release build of e2080d1e linking omega cascade/tree-verify-gemm lib (cascade-dev/libomega_gpu.a sha 6222ffa4...).
# Waits for MemAvailable >= 46 GiB (model ~32 GiB + 12 GiB kept free) before taking the locks; timings under quiet-measure then the GPU lock.
# A run refused for memory is kept as <name>.refused-<i>.log.txt and retried (at most 30 times).
set -u
cd $HOME/workspace/sc-tvwire
E=evidence/qwen38-cascade-tvgemm-20261019
BIN=$HOME/workspace/targets/cascade-sc/pinned/qwen38_spec-e2080d1e
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
name=$1; shift
i=0
while :; do
  while [ $(awk "/MemAvailable/{print \$2}" /proc/meminfo) -lt 48234496 ]; do sleep 30; done
  { echo "bin sha $(sha256sum $BIN | cut -c1-64)"; echo "spark-receipt tvw-$name -- quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN gen M $E/$name.json $*"; echo "queued $(date -u +%FT%TZ) load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)"; } >> $E/command-$name.txt
  ~/.local/bin/spark-receipt tvw-$name -- ~/.local/bin/quiet-measure nice -n 10 flock /tmp/aien-gb10.lock $BIN gen $M $E/$name.json "$@" > $E/$name.log.txt 2>&1
  rc=$?
  echo "end $(date -u +%FT%TZ) rc $rc load $(cat /proc/loadavg) $(grep MemAvailable /proc/meminfo)" >> $E/command-$name.txt
  cp ~/yardsticks/receipts/$(date +%F)-tvw-$name.json $E/receipt-$name.json 2>/dev/null
  if grep -q refusing $E/$name.log.txt && [ $i -lt 30 ]; then i=$((i + 1)); mv $E/$name.log.txt $E/$name.refused-$i.log.txt; sleep 120; continue; fi
  break
done
