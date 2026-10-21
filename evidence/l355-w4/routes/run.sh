#!/bin/bash
# L355 NV16 go/no-go (a): router flips. Forced 512 with each step's top-8 per layer, FP8 vs w4e (MXFP4 BEST store), gen-p4 + chat.
set -u
B=$HOME/workspace/targets/l355/sc-merged/release/q36_w4_drift; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; O=$HOME/workspace/sc-l355/evidence/l355-w4/routes
export AIEN_L355_ROUTES=1 AIEN_Q36_W4_STORE=default
echo "$(date -u +%FT%TZ) start bin $(sha256sum $B | cut -c1-16)" >> $O/run.out
for p in p4 chat-max-24; do for arm in fp8 w4e; do
  quiet-build nice -n 10 env -u AIEN_COORD_KEY $B forced $M $R/drift-$p.json 512 $arm $O/forced-$arm-$p.json > $O/forced-$arm-$p.log.txt 2>&1 &
done; done
wait
echo "$(date -u +%FT%TZ) end" >> $O/run.out
