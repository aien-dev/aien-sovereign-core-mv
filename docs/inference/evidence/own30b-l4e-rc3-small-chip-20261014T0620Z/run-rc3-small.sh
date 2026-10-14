#!/usr/bin/env bash
# RC3 small-weight chip check: spec + one-list decode composition, 3 real layers, exact verify, server switches.
set -uo pipefail
INV=$(cd "$(dirname "$0")" && pwd)
SC=$HOME/workspace/sc-own30b-rc3
BIN=/tmp/l4e/rc3-target/release
CKPT=$HOME/models/qwen3-30b-a3b-instruct-2507-fp8
G=$SC/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
P5=$SC/docs/inference/evidence/own30b-l4c-cut2-chip-20261014T0227Z/p5-code-json.txt
out=$INV/${OUT:-rc3-small}; mkdir -p "$out"
{ echo "sc $(git -C $SC rev-parse HEAD) dirty=$(git -C $SC status --porcelain -uno | wc -l) omega $(git -C $HOME/workspace/omega-own30b-rc3 rev-parse HEAD)";
  sha256sum $BIN/own30b_spec_check; free -g | head -2; date -u +%FT%TZ; } > $out/ground.txt
export AIEN_OMEGA_SPIN_US=2000
run() { local name=$1; shift
  flock /tmp/aien-gb10.lock $BIN/own30b_spec_check $CKPT $G $out/$name --layers 3 --drafter suffix --suffix-global --runs 2 --new 64 \
    --extra-prompt p5-code-json $P5 --new-extra 256 --verify exact --wp-device-host "$@" > $out/$name.raw 2>&1; echo "$name rc=$?" >> $out/ground.txt; }
run graph-off-strict --decode-graph off
run graph-on --decode-graph on --resident on
AIEN_OWN30B_SPEC_BREAK_EVEN=0 run graph-on-nobe --decode-graph on --resident on
( cd $out && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
