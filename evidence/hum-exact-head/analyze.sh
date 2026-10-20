#!/bin/bash
# analyze.sh: |C| distribution, bytes/token, two-pass argmax == full argmax, mutant sweep, top-20 draws (L314 step 1+3).
set -e
B=$HOME/workspace/targets/hum-exact-head/sc/release/q36_exact_head
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
cd $HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
quiet-build nice -n 19 $B analyze $M $HOME/workspace/targets/hum-exact-head/hidden.bin analyze.json
echo DONE > analyze.done
