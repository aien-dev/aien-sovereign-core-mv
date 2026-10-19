#!/bin/bash
# Task A: FP8 lm_head off (H0) vs on (H1). Speed: qwen38_omega_full golden, 4 interleaved rounds. Drift: qwen38_ab on 6 texts.
cd ~/workspace/ab-w4-work
FULL=$HOME/workspace/targets/ab-w4/sc/release/qwen38_omega_full; AB=$HOME/workspace/targets/ab-w4/sc/release/qwen38_ab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a; G=$PWD/seq/golden-full.json; S=$PWD/seq
export LIB=$HOME/workspace/targets/ab-w4/lib-a/libomega_gpu.a
for r in 1 2 3 4; do
  if [ $((r % 2)) = 1 ]; then o="0 1"; else o="1 0"; fi
  for h in $o; do bin/one.sh speedA-h$h-r$r "AIEN_Q38_FP8_HEAD=$h" $FULL $M 64 $G report.json; done
done
for s in golden license prose codelong code arch; do
  for h in 0 1; do bin/one.sh dumpA-$s-h$h "AIEN_Q38_FP8_HEAD=$h" $AB run $M $S/$s.json out; done
done
for s in prose codelong; do
  for h in 0 1; do bin/one.sh genA-$s-h$h "AIEN_Q38_FP8_HEAD=$h" $AB run $M $S/${s}_prompt.json out --gen 256; done
done
echo DONE > runs/chain-a.done
