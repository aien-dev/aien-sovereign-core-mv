#!/usr/bin/env bash
# W-L4e cut 2 small-weight chip check: break-even back-off on vs off, 3 real layers, exact verify.
set -uo pipefail
INV=$(cd "$(dirname "$0")" && pwd)
SC=$HOME/workspace/sc-own30b-specwp
BIN=/tmp/l4e/target/release
CKPT=$HOME/models/qwen3-30b-a3b-instruct-2507-fp8
G=$SC/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
P5=$SC/docs/inference/evidence/own30b-l4c-cut2-chip-20261014T0227Z/p5-code-json.txt
out=$INV/${OUT:-cut2}; mkdir -p "$out"
{ echo "sc $(git -C $SC rev-parse HEAD) dirty=$(git -C $SC status --porcelain -uno | wc -l)"; echo "omega archive /tmp/l4e/libomega_gpu.a $(sha256sum /tmp/l4e/libomega_gpu.a | cut -d' ' -f1)";
  sha256sum $BIN/own30b_spec_check; free -g | head -2; date -u +%FT%TZ; } > $out/ground.txt
export AIEN_OMEGA_SPIN_US=2000
for be in 1 0; do
  flock /tmp/aien-gb10.lock env AIEN_OWN30B_SPEC_BREAK_EVEN=$be $BIN/own30b_spec_check $CKPT $G $out/be$be --layers 3 --drafter suffix --suffix-global --runs 2 --new 64 \
    --extra-prompt p5-code-json $P5 --new-extra 256 --verify exact > $out/be$be.raw 2>&1; echo "be=$be rc=$?" >> $out/ground.txt
done
( cd $out && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
