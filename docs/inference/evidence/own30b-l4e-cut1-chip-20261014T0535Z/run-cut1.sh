#!/usr/bin/env bash
# W-L4e cut 1 small-weight chip check: numerics mutants (attn-bf16 vs residual-bf16), 3 real layers.
set -uo pipefail
INV=$(cd "$(dirname "$0")" && pwd)
SC=$HOME/workspace/sc-own30b-specwp
BIN=/tmp/l4e/target/release
CKPT=$HOME/models/qwen3-30b-a3b-instruct-2507-fp8
G=$SC/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
P5=$SC/docs/inference/evidence/own30b-l4c-cut2-chip-20261014T0227Z/p5-code-json.txt
out=$INV/${OUT:-cut1}; mkdir -p "$out"
{ echo "sc $(git -C $SC rev-parse HEAD) dirty=$(git -C $SC status --porcelain -uno | wc -l)"; echo "omega archive /tmp/l4e/libomega_gpu.a $(sha256sum /tmp/l4e/libomega_gpu.a | cut -d' ' -f1)";
  sha256sum $BIN/own30b_exact_probe $BIN/own30b_spec_check; free -g | head -2; date -u +%FT%TZ; } > $out/ground.txt
export AIEN_OMEGA_SPIN_US=2000
flock /tmp/aien-gb10.lock $BIN/own30b_exact_probe $CKPT 3 $G $out/probe --paths fast,mutant,mutant-res,mutant-fp8 --max-m 5 --reps 2 > $out/probe.raw 2>&1; echo "probe rc=$?" >> $out/ground.txt
for mk in attn-bf16 residual-bf16 act-fp8; do
  flock /tmp/aien-gb10.lock $BIN/own30b_spec_check $CKPT $G $out/check-$mk --layers 3 --drafter suffix --suffix-global --runs 1 --new 64 \
    --extra-prompt p5-code-json $P5 --new-extra 128 --verify exact --numerics-mutant $mk > $out/check-$mk.raw 2>&1; echo "check $mk rc=$?" >> $out/ground.txt
done
( cd $out && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
