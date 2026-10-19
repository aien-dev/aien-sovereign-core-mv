#!/bin/bash
# suite/run.sh: W4 chip suite at omega 7ed31249 (default shape), graph --w4 and FP8, then chain-b
cd $(dirname $0); sha256sum gpu_w4_api_test gpu_q38_graph_test > ground.txt
flock /tmp/aien-gb10.lock nice ./gpu_w4_api_test --chip --fp8 --reps 15 > w4chip.log 2>&1; echo "rc=$?" >> w4chip.log
flock /tmp/aien-gb10.lock nice ./gpu_q38_graph_test --w4 --w4-head --reps 10 --tl-reps 11 > graph-w4.log 2>&1; echo "rc=$?" >> graph-w4.log
flock /tmp/aien-gb10.lock nice ./gpu_q38_graph_test --fp8-head --reps 10 --tl-reps 11 > graph-fp8.log 2>&1; echo "rc=$?" >> graph-fp8.log
echo DONE > done
~/workspace/ab-w4-work/bin/chain-b.sh > ../chain-b.out 2>&1
