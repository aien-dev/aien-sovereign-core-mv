#!/bin/bash
# retry the W4 quality runs the memory guard refused (other agents' servers held the memory): wait for room first
cd ~/workspace/ab-w4-work; until [ -f runs/chain-b.done ]; do sleep 30; done
AB=$HOME/workspace/targets/ab-w4/sc-b/release/qwen38_ab; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; S=$PWD/seq
export LIB=$HOME/workspace/targets/ab-w4/lib-b/libomega_gpu.a
declare -A V=( [W]="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1" [WW]="AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1" )
room() { until [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $((58*1024*1024)) ]; do sleep 60; done; }
for job in "dumpC-prose-W W prose" "dumpC-arch-WW WW arch"; do set -- $job; room; bin/one.sh $1 "${V[$2]}" $AB run $M $S/$3.json out; done
for s in prose codelong; do for v in W WW; do [ -f runs/genB-$s-$v-*/out.json ] && continue; room; bin/one.sh genC-$s-$v "${V[$v]}" $AB run $M $S/${s}_prompt.json out --gen 256; done; done
echo DONE > runs/chain-c.done
