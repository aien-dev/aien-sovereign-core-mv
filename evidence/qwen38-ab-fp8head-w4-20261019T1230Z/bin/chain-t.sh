#!/bin/bash
# timelines: F, W, WW whole model on golden (47 tokens after the first), GPU stamps per launch kind
cd ~/workspace/ab-w4-work; AB=$HOME/workspace/targets/ab-w4/sc-t/release/qwen38_ab; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
export LIB=$HOME/workspace/targets/ab-w4/lib-b/libomega_gpu.a
bin/one.sh tl-F "AIEN_Q38_FP8_HEAD=1" $AB timeline $M $HOME/workspace/ab-w4-work/seq/golden.json tl.json
bin/one.sh tl-W "AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1" $AB timeline $M $HOME/workspace/ab-w4-work/seq/golden.json tl.json
bin/one.sh tl-WW "AIEN_Q38_W4=1 AIEN_Q38_W4_HEAD=1" $AB timeline $M $HOME/workspace/ab-w4-work/seq/golden.json tl.json
echo DONE > runs/chain-t.done
