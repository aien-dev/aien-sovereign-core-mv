#!/bin/bash
# L125 session, second start in the same window 20261020T082430Z-0d26f5 (first start aborted: S runs refused by a Rust
# null-head check, fixed in sc; runs kept under runs/aborted-1). Lease: speed F/S first, then drift, energy, and the one
# convert-at-load run (L) last because it loads in ~13 min on this loaded box.
set -u; W=$HOME/workspace/speed1-work; cd $W
T=$HOME/workspace/targets/speed1; FULL=$T/sc/release/qwen38_omega_full; AB=$T/sc/release/qwen38_ab
export LIB=$T/omega/libomega_gpu.a
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; S=$HOME/workspace/ab-w4-work/seq; G=$S/golden-full.json
declare -A V=( [F]="AIEN_Q38_W4_STORE=off AIEN_Q38_FP8_HEAD=1" [L]="AIEN_Q38_W4_STORE=off AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1" [S]="AIEN_L125_ARM=S" [H0]="AIEN_Q38_W4_STORE=off" )
echo "session2 start $(date -u +%FT%TZ) window ${SPARK_WINDOW:-none}" >> $W/session.log
bin/one.sh chip-graph-w4only "" $T/omega/gpu_q38_graph_test --w4 --w4-head --reps 5 --tl-reps 3
ORD=( "F S" "S F" "F S" "S F" )
for r in 1 2 3 4; do for v in ${ORD[$((r-1))]}; do bin/one.sh speed-$v-r$r "${V[$v]}" $FULL $M 64 $G report.json; done; done
for s in golden license prose codelong code arch; do for v in S H0; do bin/one.sh dump-$s-$v "${V[$v]}" $AB run $M $S/$s.json out; done; done
for v in S F; do bin/one.sh gen-prose-$v "${V[$v]}" $AB run $M $S/prose_prompt.json out --gen 256; done
bin/one.sh dump-golden-L "${V[L]}" $AB run $M $S/golden.json out
echo "session2 end $(date -u +%FT%TZ)" >> $W/session.log
