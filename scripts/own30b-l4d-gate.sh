#!/usr/bin/env bash
# W-L4d full-model gate (PREPARED, run by the orchestrator in a pause window: it loads the 30 GB model,
# so the live server on :18091 must be stopped first; the binary refuses with under 70 GB available).
#
#   scripts/own30b-l4d-gate.sh [omega-archive]
#
# One own30b_spec_check run: 4 golden prompts at 64 tokens + the code/JSON prompt at 256, suffix drafter,
# --verify decode,exact (old exact verify and the new fast exact verify side by side), 2 runs.
# PASS needs: spec ON tokens == spec OFF tokens bit for bit for both verify paths on every prompt and run;
# the extra-accept mutant caught on each path; the numerics mutant (bf16-truncated attention inside the
# exact verify) caught on at least one prompt. tok/s off vs on per prompt is in the log.
# Builds the binary against the omega archive (default: omega own30b/spec-exact build), then runs it
# through scripts/own30b-chip-run.sh-style stamping under flock /tmp/aien-gb10.lock.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
lib=${1:-$HOME/workspace/omega-own30b-spec/build/libomega_gpu.a}
ckpt=${OWN30B_CHECKPOINT:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
golden=$repo/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
p5=$repo/docs/inference/evidence/own30b-l4c-cut2-chip-20261014T0227Z/p5-code-json.txt
ev=$repo/docs/inference/evidence/own30b-l4d-gate-chip-$(date -u +%Y%m%dT%H%MZ)
[ -f "$lib" ] || { echo "no omega archive at $lib"; exit 2; }
nm "$lib" | grep " T omega_gpu_decode_attn_rows$" >/dev/null || { echo "$lib lacks omega_gpu_decode_attn_rows (needs omega own30b/spec-exact)"; exit 2; }
stage=$(mktemp -d /tmp/l4d-gate.XXXX); cp "$lib" "$stage/libomega_gpu.a"
export AIEN_OMEGA_GPU_LIB=$stage/libomega_gpu.a AIEN_OMEGA_GPU_LIB_SHA256=$(sha256sum "$lib" | cut -d' ' -f1) CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/l4d/target}
( cd "$repo" && cargo build --release -p aien-inference-abi --bin own30b_spec_check ) || { echo "build failed"; exit 2; }
bin=$CARGO_TARGET_DIR/release/own30b_spec_check
mkdir -p "$ev"
name=exact
{
	echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "sc commit $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$repo" status --porcelain -uno | wc -l)"
	echo "binary $bin sha256 $(sha256sum "$bin" | cut -d' ' -f1)"
	echo "omega archive $lib sha256 $AIEN_OMEGA_GPU_LIB_SHA256"
	echo "cmd flock /tmp/aien-gb10.lock env AIEN_OMEGA_SPIN_US=2000 $bin $ckpt $golden $ev/$name --drafter suffix --suffix-global --runs 2 --new 64 --extra-prompt p5-code-json $p5 --new-extra 256 --verify decode,exact"
	free -g | head -2
} > "$ev/$name.start"
crumb whisper odysseus-9e "$repo" - "W-L4d gate: full 30B model load, spec exact verify, ~15 min" >/dev/null 2>&1 || true
flock /tmp/aien-gb10.lock env AIEN_OMEGA_SPIN_US=2000 "$bin" "$ckpt" "$golden" "$ev/$name" --drafter suffix --suffix-global --runs 2 --new 64 \
	--extra-prompt p5-code-json "$p5" --new-extra 256 --verify decode,exact > "$ev/$name.raw" 2> "$ev/$name.err"
rc=$?
echo "$rc" > "$ev/$name.rc"
{ echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g | head -2; } > "$ev/$name.end"
( cd "$ev" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
rm -rf "$stage"
echo "gate rc=$rc evidence $ev"
grep -E "ON verify|negative control|numerics control|RESULT|FAIL" "$ev/$name.raw" | head -60
exit "$rc"
