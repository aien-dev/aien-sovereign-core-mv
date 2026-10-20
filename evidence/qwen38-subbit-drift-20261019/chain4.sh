#!/bin/bash
cd "$(dirname "$0")"
B=./qwen38_subbit.v2; C=~/models/qwen3.8-27b-fp8-017b9c7a; S=~/workspace/ab-w4-work/seq
run() { d=$1; shift; echo "$(date -u +%T) start $d" >> chain.log
  nice -n 5 $B $C $S $d "$@" --ref-dir ref --codebooks cb > $d.log 2>&1; echo "$(date -u +%T) done $d rc=$?" >> chain.log; }
run out-trellis_3_10_128_act050 trellis:3:10:128 --act calib/calib.act --act-exp 0.5
run out-trellis_4_10_128_act025 trellis:4:10:128 --act calib/calib.act --act-exp 0.25
