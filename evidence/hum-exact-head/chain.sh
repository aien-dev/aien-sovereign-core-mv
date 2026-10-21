#!/bin/bash
# chain.sh: after analyze v2, export the chip inputs, then the identity gate (fp8b, Row bound) at x1 and the mutants.
set -u
B=$HOME/workspace/targets/hum-exact-head/sc/release/q36_exact_head
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-exact-head/evidence/q36-host-20261020
E=$HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
cd $E
while [ ! -f analyze.done ]; do sleep 20; done
quiet-build nice -n 19 $B export $M $HOME/workspace/targets/hum-exact-head/chip > export.txt 2>&1; echo "export rc $?" >> export.txt
GOLD="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json"
for sc in 1 0.5 0.02 0; do
  quiet-build nice -n 19 $B gate $M fp8b row $sc gate-fp8b-row-x$sc.json $GOLD > gate-fp8b-row-x$sc.txt 2>&1
  echo "gate x$sc rc $?" >> gate-fp8b-row-x$sc.txt
done
echo DONE > chain.done
