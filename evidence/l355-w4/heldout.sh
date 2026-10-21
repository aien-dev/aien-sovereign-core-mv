#!/bin/bash
# L355 held-out prompts (w4-gate.md, GPTQ arm, part B): runs one arm on h1..h5, 512 steps, two at a time.
#   heldout.sh fp8                 -> runs/forced-fp8-h<i>.json (greedy: the FP8 arm's own ids are the reference)
#   heldout.sh <name> <store|default>  -> runs/forced-<name>-h<i>.json (drift arm w4e on that store, forced on the fp8 ids)
set -u
NAME=$1; STORE=${2:-}
B=$HOME/workspace/targets/l355/sc/release/q36_w4_drift; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
D=$HOME/workspace/sc-l355/evidence/l355-w4; O=$D/runs
echo "$(date -u +%FT%TZ) [heldout $NAME] start (store ${STORE:-none})" >> $O/run.out
one() {
  local i=$1
  if [ "$NAME" = fp8 ]; then
    quiet-build nice -n 10 $B forced $M $D/heldout/h$i.json 512 fp8 $O/forced-fp8-h$i.json > $O/forced-fp8-h$i.log.txt 2>&1
  else
    AIEN_Q36_W4_STORE=$STORE quiet-build nice -n 10 $B forced $M $O/forced-fp8-h$i.json 512 w4e $O/forced-$NAME-h$i.json > $O/forced-$NAME-h$i.log.txt 2>&1
  fi
  echo "$(date -u +%FT%TZ) [heldout $NAME] h$i exit $?" >> $O/run.out
}
one 1 & one 2 & wait
one 3 & one 4 & wait
one 5
echo "$(date -u +%FT%TZ) [heldout $NAME] end" >> $O/run.out
