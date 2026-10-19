#!/bin/sh
# One cascade run: $1 = name, rest = qwen38_spec gen flags. Records the command, binary sha and the log.
set -u
E=evidence/qwen38-cascade-20261019T1611Z
BIN=/home/drakestapleton/workspace/targets/sc-cascade/release/qwen38_spec
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
name=$1; shift
echo "bin sha $(sha256sum $BIN | cut -c1-64)" >> $E/command-$name.txt
echo "lib ~/workspace/targets/q38-f7-omega/libomega_gpu.a sha 6e90520913032ee8653ebbe325c485d329daa0a749840303a44b68b97ca17bff (omega 37034f03)" >> $E/command-$name.txt
echo "flock /tmp/aien-gb10.lock nice -n 10 $BIN gen $M $E/$name.json $*" >> $E/command-$name.txt
date -u +%FT%TZ >> $E/command-$name.txt
flock /tmp/aien-gb10.lock nice -n 10 $BIN gen $M $E/$name.json "$@" > $E/$name.log.txt 2>&1
echo "exit $? $(date -u +%FT%TZ)" >> $E/command-$name.txt
