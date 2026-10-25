#!/bin/bash
# gates.sh: identity gate (fp8b screen, Row bound): greedy decode with the two-pass head choosing every token on the
# 5 goldens. x1 = the proven bound (must PASS); x0.5 and x0 = shrunken bounds (slack receipts); x-1 = the proven set
# minus the full argmax (must FAIL).
set -u
B=$HOME/workspace/targets/hum-exact-head/sc/release/q36_exact_head
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-exact-head/evidence/q36-host-20261020
cd $HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
rm -f gate-fp8b-row-x*.json gate-fp8b-row-x*.txt gates.done
GOLD="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json"
for sc in 1 -1 0.5 0; do
  quiet-build nice -n 19 $B gate $M fp8b row $sc gate-fp8b-row-x$sc.json $GOLD > gate-fp8b-row-x$sc.txt 2>&1
  echo "gate x$sc rc $?" >> gate-fp8b-row-x$sc.txt
done
echo DONE > gates.done
