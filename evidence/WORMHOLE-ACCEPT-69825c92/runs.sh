#!/bin/bash
# L080 quality runs, each step its own quiet-build so measurements can slip in between.
set -u
B=~/workspace/targets/wormhole/release/wormhole_accept
FN=~/models/qwen3.8-flash-next-nvfp4-fc694b54
Q=~/models/qwen3.8-27b-fp8-017b9c7a
I=~/wormhole-scratch/inputs; R=~/wormhole-scratch/runs; mkdir -p $R
export RAYON_NUM_THREADS=8
step() { local name=$1; shift; [ -s $R/$name.json ] && { echo "skip $name"; return; }
  echo "== $name $(date -Is)"; ~/.local/bin/quiet-build nice -n 5 "$@" > $R/$name.log 2>&1; echo "rc $? $(date -Is)"; tail -2 $R/$name.log; }
for s in code-golden prose codelong; do
  ids=$(jq -r ".ids|map(tostring)|join(\" \")" $I/q38-$s.json); pl=$(jq .prompt_len $I/q38-$s.json)
  step fnscore-$s $B fn-score $FN $R/fnscore-$s.json $pl $ids
  $B accept $I/q38-$s.json $R/fnscore-$s.json $R/accept-fn-drafts-27b-$s.json | tail -4
done
for s in code-golden prose codelong; do
  pids=$(jq -r ".ids[:.prompt_len]|map(tostring)|join(\" \")" $I/q38-$s.json)
  step fngen-$s $B fn-gen $FN $R/fngen-$s.json 128 $pids
done
for s in code-golden prose codelong; do
  ids=$(jq -r ".ids|map(tostring)|join(\" \")" $R/fngen-$s.json); pl=$(jq .prompt_len $R/fngen-$s.json)
  step q38score-fngen-$s $B q38-score $Q $R/q38score-fngen-$s.json $pl $ids
  $B accept $R/fngen-$s.json $R/q38score-fngen-$s.json $R/accept-27b-drafts-fn-$s.json | tail -4
done
for s in code-golden prose codelong; do
  ids=$(jq -r ".ids|map(tostring)|join(\" \")" $I/q38-$s.json); pl=$(jq .prompt_len $I/q38-$s.json)
  step q38score-chip-$s $B q38-score $Q $R/q38score-chip-$s.json $pl $ids
done
echo ALL DONE $(date -Is)
