#!/bin/bash
# L355 W4 trunk (+experts) host greedy goldens for the chip identity gate (32 tokens each), then convert.
set -u
B=$HOME/workspace/targets/l355/sc/release/q36_w4_drift; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-l355/evidence/q36-host-20261020; O=$HOME/workspace/sc-l355/evidence/l355-w4/runs
export AIEN_Q36_W4_STORE=default
for p in p1 p2 p3 p4 chat-max-24; do
  n=32; [ $p = chat-max-24 ] && n=24
  echo "$(date -u +%FT%TZ) [w4et-gold] start $p" >> $O/run.out
  quiet-build nice -n 10 $B forced $M $G/gen-$p.json $n w4et $O/golden-w4et-$p.json > $O/golden-w4et-$p.log.txt 2>&1
  echo "$(date -u +%FT%TZ) [w4et-gold] end $p exit $?" >> $O/run.out
  jq -c '{prompt_ids, generated_ids: .ref_ids, source: "q36_w4_drift forced w4et greedy self-reference (L355)", store: .store}' $O/golden-w4et-$p.json > $O/../gold-w4et-$p.json
done
