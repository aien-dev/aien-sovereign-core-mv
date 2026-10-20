#!/bin/bash
# L125 session 3, same window: rerun only the five dumps refused at channel setup in session 2 (RM 0x51, main context
# buffer needs one contiguous block; Cortex lesson 45a41b22). They never measured; originals kept in runs/aborted-2.
set -u; W=$HOME/workspace/speed1-work; cd $W
T=$HOME/workspace/targets/speed1; AB=$T/sc/release/qwen38_ab; export LIB=$T/omega/libomega_gpu.a
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; S=$HOME/workspace/ab-w4-work/seq
declare -A V=( [S]="AIEN_L125_ARM=S" [H0]="AIEN_Q38_W4_STORE=off" )
echo "session3 start $(date -u +%FT%TZ) window ${SPARK_WINDOW:-none}" >> $W/session.log
for sv in golden:H0 license:H0 prose:S prose:H0 arch:H0; do s=${sv%:*}; v=${sv#*:}
  echo "$(date -u +%FT%TZ) $s-$v buddy $(grep Normal /proc/buddyinfo | awk '{print $(NF-4),$(NF-3),$(NF-2)}')" >> $W/session.log
  bin/one.sh dump-$s-$v "${V[$v]}" $AB run $M $S/$s.json out; done
echo "session3 end $(date -u +%FT%TZ)" >> $W/session.log
