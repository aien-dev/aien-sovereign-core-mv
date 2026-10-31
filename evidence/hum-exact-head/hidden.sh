#!/bin/bash
# hidden.sh: dump the final normed hidden vectors of the 35B host reference (L314 step 1).
set -e
B=$HOME/workspace/targets/hum-exact-head/sc/release/q36_exact_head
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-exact-head/evidence/q36-host-20261020
O=$HOME/workspace/targets/hum-exact-head/hidden.bin
cd $HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
rm -f $O
quiet-build nice -n 19 $B hidden $M $O $G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-max-24.json
quiet-build nice -n 19 $B passage $M $O $G/passage-agpl.txt 400 5
quiet-build nice -n 19 $B passage $M $O $G/passage-readme.txt 400 6
echo DONE > hidden.done
