#!/bin/sh
# Chain measurements for cascade/mtp-chain. $1 = name, rest = qwen38_spec gen flags.
# Pinned binary built from ca870265 with omega 37034f03 (lib sha 6e905209...). Each run takes the chip lock on its own.
set -u
cd $HOME/workspace/sc-cascade
E=evidence/qwen38-cascade-chain-20261019T1640Z
BIN=$HOME/workspace/targets/sc-cascade/pinned/qwen38_spec-ca870265
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
name=$1; shift
{
echo "bin sha $(sha256sum $BIN | cut -c1-64) (built from ca870265)"
echo "lib ~/workspace/targets/q38-f7-omega/libomega_gpu.a sha 6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff (omega 37034f03)"
echo "spark-receipt cascade-$name -- nice -n 10 flock /tmp/aien-gb10.lock $BIN gen M $E/$name.json $*"
echo "start $(date -u +%FT%TZ) load $(cut -d\" \" -f1-3 /proc/loadavg) MemAvailable_kB $(grep MemAvailable /proc/meminfo | tr -s \" \" | cut -d\" \" -f2)"
} >> $E/command-$name.txt
~/.local/bin/spark-receipt cascade-$name -- nice -n 10 flock /tmp/aien-gb10.lock $BIN gen $M $E/$name.json "$@" > $E/$name.log.txt 2>&1
rc=$?
echo "exit $rc $(date -u +%FT%TZ)" >> $E/command-$name.txt
cp ~/yardsticks/receipts/$(date +%F)-cascade-$name.json $E/receipt-$name.json 2>/dev/null
