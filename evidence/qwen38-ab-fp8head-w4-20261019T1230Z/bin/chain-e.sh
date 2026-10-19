#!/bin/bash
# after chain-d: retry every quality run that still has no out.json (memory guard refusals happen when memory is
# taken while a run waits for the GPU lock); up to 20 tries each, waiting for memory room before each try
cd ~/workspace/ab-w4-work; until [ -f runs/chain-d.done ]; do sleep 60; done
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; S=$PWD/seq
B=$HOME/workspace/targets/ab-w4/sc-b/release/qwen38_ab; T=$HOME/workspace/targets/ab-w4/sc-t/release/qwen38_ab
W="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1"; WW="AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1"
room() { until [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $((58*1024*1024)) ]; do sleep 60; done; }
have() { ls runs/$1-*/out.json >/dev/null 2>&1; }
job() { # name-glob runname lib bin env args...
  local g=$1 n=$2 l=$3 b=$4 e=$5; shift 5
  for try in $(seq 20); do { have "$g" || have "$n"; } && return; room; LIB=$l bin/one.sh $n "$e" $b "$@"; done; }
LB=$HOME/workspace/targets/ab-w4/lib-b/libomega_gpu.a; LC=$HOME/workspace/targets/ab-w4/lib-c/libomega_gpu.a
have dumpB-prose-W || job dumpC-prose-W dumpE-prose-W $LB $B "$W" run $M $S/prose.json out
have dumpB-arch-WW || job dumpC-arch-WW dumpE-arch-WW $LB $B "$WW" run $M $S/arch.json out
for s in prose codelong; do for v in W WW; do e=$W; [ $v = WW ] && e=$WW
  have genB-$s-$v || have genC-$s-$v || job genE-$s-$v genE-$s-$v $LB $B "$e" run $M $S/${s}_prompt.json out --gen 256; done; done
for j in "WWb golden" "WWb code" "WWb arch" "WWb license" "Wb code" "Wb arch"; do set -- $j
  e="$WW AIEN_Q38_W4_RULE=best"; [ $1 = Wb ] && e="$W AIEN_Q38_W4_RULE=best"
  have dumpD-$2-$1 || job dumpE-$2-$1 dumpE-$2-$1 $LC $T "$e" run $M $S/$2.json out; done
echo DONE > runs/chain-e.done
