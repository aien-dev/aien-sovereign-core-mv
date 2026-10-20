#!/bin/bash
# s2.sh: whole-model decode speed, 4-bit weights + FP8 head, bf16 KV vs 8-bit V only (KV8=2), at 8k/32k/64k,
# three A/B pairs in rotated order (box load ~80 gave 2-5x kernel jitter). 128k is s3.sh (needs ~68 GiB).
set -u; J=$HOME/workspace/speed7-work/bin/job.sh; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
AB=$HOME/workspace/targets/speed7/release/qwen38_ab; W="AIEN_Q38_W4=1 AIEN_Q38_FP8_HEAD=1"
$J speed-W-r1 "$W" $AB speed $M 8192,32768,65536 512 7 speed.json
$J speed-WV-r1 "$W AIEN_Q38_KV8=2" $AB speed $M 8192,32768,65536 512 7 speed.json
$J speed-WV-r2 "$W AIEN_Q38_KV8=2" $AB speed $M 65536,32768,8192 512 7 speed.json
$J speed-W-r2 "$W" $AB speed $M 65536,32768,8192 512 7 speed.json
$J speed-W-r3 "$W" $AB speed $M 32768,65536,8192 512 7 speed.json
$J speed-WV-r3 "$W AIEN_Q38_KV8=2" $AB speed $M 32768,65536,8192 512 7 speed.json
