#!/bin/bash
# long.sh: drift at long context: the first 32768 tokens of a long document, teacher-forced, 4-bit weights + FP8 head,
# bf16 vs 8-bit KV. Every step's NLL is kept; logits only for the last 512 steps.
set -u; J=$HOME/workspace/speed7-work/bin/job.sh; S=$HOME/workspace/speed7-work/seq; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
AB=$HOME/workspace/targets/speed7/release/qwen38_ab; W="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1"
$J long-W "$W" $AB run $M $S/long32k.json long --tail 512
$J long-W8 "$W AIEN_Q38_KV8=1" $AB run $M $S/long32k.json long --tail 512
