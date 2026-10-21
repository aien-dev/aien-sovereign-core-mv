#!/bin/bash
# L355 diagnostic (not a verdict): W4 experts on ONE quarter of the layers (others FP8), forced 256 steps on p3, p4, chat
# (the goldens that drifted most in the all-layer w4e arm). NLL rise per group says where the drift lives.
set -u
B=$HOME/workspace/targets/l355/sc/release/q36_w4_drift; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; O=$HOME/workspace/sc-l355/evidence/l355-w4/runs/sens
export AIEN_Q36_W4_STORE=default
grp=$1; case $grp in g0) fp8="10-39" ;; g1) fp8="0-9,20-39" ;; g2) fp8="0-19,30-39" ;; g3) fp8="0-29" ;; *) exit 2 ;; esac
for p in p4 chat-max-24 p3; do
  AIEN_Q36_W4_FP8_LAYERS=$fp8 nice -n 10 $B forced $M $R/drift-$p.json 256 w4e $O/$grp-$p.json > $O/$grp-$p.log.txt 2>&1
  echo "$(date -u +%FT%TZ) [sens] $grp $p exit $?" >> $O/../run.out
done
