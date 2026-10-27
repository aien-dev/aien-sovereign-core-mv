#!/bin/bash
# L356 timing A/B after the swap (Drake: time it afterwards): q36_levers with the a|b overlap OFF then ON, same window,
# same command, same prompt (gen-p1) and goldens; identity checked in both runs (logits FNV hashes must match between them).
set -u
T=$HOME/workspace/targets/levers/sc/release/q36_levers
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-levers/evidence/q36-host-20261020
E=$HOME/workspace/sc-levers/evidence/levers-q36-20261021
spark-window open l356-time-ab --ledger L356 --lease 40 --wait || exit 3
for ab in 0 1; do
  AIEN_OMEGA_Q36_OVERLAP_AB=$ab spark-window run $T $M $E/time-ab$ab.json $G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json > $E/time-ab$ab.log 2>&1
  echo "ab=$ab rc=$?" >> $E/time-ab.rc
done
spark-window close
echo DONE >> $E/time-ab.rc
