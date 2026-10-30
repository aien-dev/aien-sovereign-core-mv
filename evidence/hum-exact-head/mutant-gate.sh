#!/bin/bash
# mutant-gate.sh: after gates.sh, rebuild (empty-set fix) and rerun the drop-winner gate x-1 (must FAIL).
set -u
E=$HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
B=$HOME/workspace/targets/hum-exact-head/sc/release/q36_exact_head
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-exact-head/evidence/q36-host-20261020
cd $E
while [ ! -f gates.done ]; do sleep 20; done
~/handoffs/hum-exact-head/build.sh build --release -p aien-inference-abi --bin q36_exact_head > mutant-build.txt 2>&1 || exit 1
quiet-build nice -n 19 $B gate $M fp8b row -1 gate-fp8b-row-x-1.json $G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json > gate-fp8b-row-x-1.txt 2>&1
echo "gate x-1 rc $?" >> gate-fp8b-row-x-1.txt
echo DONE > mutant.done
