#!/bin/bash
# Task B: whole model, F = FP8 weights + FP8 head (Task A winner), W = MXFP4 weights + FP8 head, WW = MXFP4 weights + MXFP4 head.
# Speed: qwen38_omega_full golden, 4 rounds, order rotated. Quality: qwen38_ab on the 6 texts + 2 free runs (W, WW).
cd ~/workspace/ab-w4-work
FULL=$HOME/workspace/targets/ab-w4/sc-b/release/qwen38_omega_full; AB=$HOME/workspace/targets/ab-w4/sc-b/release/qwen38_ab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; G=$PWD/seq/golden-full.json; S=$PWD/seq
export LIB=$HOME/workspace/targets/ab-w4/lib-b/libomega_gpu.a
declare -A V=( [F]="AIEN_Q38_FP8_HEAD=1" [W]="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1" [WW]="AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1" )
ORD=( "F W WW" "W WW F" "WW F W" "F WW W" )
for r in 1 2 3 4; do for v in ${ORD[$((r-1))]}; do bin/one.sh speedB-$v-r$r "${V[$v]}" $FULL $M 64 $G report.json; done; done
for s in golden license prose codelong code arch; do for v in W WW; do bin/one.sh dumpB-$s-$v "${V[$v]}" $AB run $M $S/$s.json out; done; done
for s in prose codelong; do for v in W WW; do bin/one.sh genB-$s-$v "${V[$v]}" $AB run $M $S/${s}_prompt.json out --gen 256; done; done
echo DONE > runs/chain-b.done
