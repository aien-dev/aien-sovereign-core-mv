#!/bin/bash
# m2.sh: whole model with the split cache (omega 3efab8c1, sc 3fc1132e). Plain engine weights.
# Drift on the 6 texts for V-only (KV8=2) and K-only (KV8=3) against drift-P (bf16, 10:05Z);
# golden for each form; gold-P8 again (its 10:03Z run hit NV_ERR_NO_MEMORY); then drift-P again with
# the new build as the null control (same bf16 engine twice: expect identical, and a refactor check).
set -u; J=$HOME/workspace/speed7-work/bin/job.sh; S=$HOME/workspace/speed7-work/seq; M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
R=$HOME/workspace/targets/speed7/release; FULL=$R/qwen38_omega_full; AB=$R/qwen38_ab
T=$S/golden.json,$S/license.json,$S/prose.json,$S/codelong.json,$S/code.json,$S/arch.json
O=golden,license,prose,codelong,code,arch
$J gold-P2 "" $FULL $M 64 $S/golden-full.json report.json
$J drift-PV "AIEN_Q38_KV8=2" $AB run $M $T $O
$J drift-PK "AIEN_Q38_KV8=3" $AB run $M $T $O
$J gold-PV "AIEN_Q38_KV8=2" $FULL $M 64 $S/golden-full.json report.json
$J gold-PK "AIEN_Q38_KV8=3" $FULL $M 64 $S/golden-full.json report.json
$J gold-P8b "AIEN_Q38_KV8=1" $FULL $M 64 $S/golden-full.json report.json
$J drift-P2 "" $AB run $M $T $O
