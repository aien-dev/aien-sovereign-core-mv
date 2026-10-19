#!/bin/bash
# scale rule BEST (host survey: rel RMS 0.11267 vs OCP 0.11672) on chip quality dumps; waits for chain-c and for memory room
cd ~/workspace/ab-w4-work; until [ -f runs/chain-c.done ]; do sleep 30; done
AB=$HOME/workspace/targets/ab-w4/sc-t/release/qwen38_ab; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; S=$PWD/seq
export LIB=$HOME/workspace/targets/ab-w4/lib-c/libomega_gpu.a
declare -A V=( [Wb]="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1 AIEN_Q38_W4_RULE=best" [WWb]="AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1 AIEN_Q38_W4_RULE=best" )
room() { until [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $((58*1024*1024)) ]; do sleep 60; done; }
for job in "WWb golden" "WWb code" "WWb arch" "WWb license" "Wb code" "Wb arch"; do set -- $job
  for try in 1 2 3; do room; bin/one.sh dumpD-$2-$1 "${V[$1]}" $AB run $M $S/$2.json out
    d=$(ls -d runs/dumpD-$2-$1-* | tail -1); [ -f $d/out.json ] && break; done; done
echo DONE > runs/chain-d.done
