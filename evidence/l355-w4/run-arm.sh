#!/bin/bash
# L355 full verdict arm (frozen w4-gate.md): forced 512 on the five goldens + AGPL score, in parallel.
#   [STORE=<store path>] run-arm.sh <name> <drift arm w4e|w4et> <AIEN_Q36_W4_FP8_LAYERS>   -> runs/forced-<name>-<p>.json, runs/score-<name>.json
set -u
NAME=$1; ARM=$2; FP8=$3
B=${BIN:-$HOME/workspace/targets/l355/sc/release/q36_w4_drift}; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
O=$HOME/workspace/sc-l355/evidence/l355-w4/runs
export AIEN_Q36_W4_STORE=${STORE:-default} AIEN_Q36_W4_FP8_LAYERS=$FP8
echo "$(date -u +%FT%TZ) [arm $NAME] start ($ARM, fp8 layers $FP8)" >> $O/run.out
for p in p1 p2 p3 p4 chat-max-24; do
  quiet-build nice -n 10 $B forced $M $R/drift-$p.json 512 $ARM $O/forced-$NAME-$p.json > $O/forced-$NAME-$p.log.txt 2>&1 &
done
quiet-build nice -n 10 $B score $M $G/passage-agpl.txt 400 $ARM $O/score-$NAME.json > $O/score-$NAME.log.txt 2>&1 &
wait
echo "$(date -u +%FT%TZ) [arm $NAME] end" >> $O/run.out
