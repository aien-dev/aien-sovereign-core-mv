#!/usr/bin/env bash
# W-L1H full-model gate: W8A8 (e4m3 activations, the shipped whole-prompt path) vs W8A16 (bf16 activations,
# same FP8 weights). PREPARED, NOT RUN by W-L1H. It loads the 30 GB model six times (one load per tool per
# mode), so the live server on :18091 must be stopped by the orchestrator first; the script refuses while
# anything listens there, and every binary refuses with under 70 GB available.
#
#   scripts/own30b-l1h-gate.sh [omega-archive]
#
# omega-archive: default = built here from ~/workspace/omega-own30b-a16 (must be at the W-L1H commit,
# physics from ~/workspace/physics-own30b-k as omega's physics.lock pins).
#
# Per mode (a8 then a16), each run sealed under flock /tmp/aien-gb10.lock:
#   check-<mode>  own30b_check chip --whole-prompt --prefill-device-host --prefill-acts <mode> --tokens-only
#                 --new 16 --prompts 1,2,3,4: greedy tokens vs the golden file (golden lists are 16 tokens;
#                 --new 16 keeps the compare to the golden length).
#   gate-<mode>   own30b_wp_gate --lens 1,512,2048 --devhost --mutants --acts <mode>: per-layer drift of the
#                 whole-prompt path vs plain decode (OFF, f32 activations); the "DRIFT" line per length.
#                 (With a16 the tool's own PASS rule, ON vs the W8A8 control, is not the question; read DRIFT.)
#   ttft-<mode>   own30b_ttft --lens 512,2048 --reps 2 --new-tokens 32 --devhost --acts <mode>: TTFT OFF vs ON
#                 and ON-vs-OFF token agreement.
#
# Pass rules for W8A16 (fixed before any full-model number exists):
#   R1 golden: check-a16 p1..p4 IDENTICAL, or diverging only where the CPU top-2 margin is below 1e-3
#      (own30b_check's own rule). p1 must be IDENTICAL (the RC p1 flip).
#   R2 drift: at every length, DRIFT a16 max_l relerr(ON,OFF) < DRIFT a8, and top1 AGREE.
#   R3 speed: TTFT a16 <= 1.10 x TTFT a8 at 512 and at 2048 (median of reps, ON with the device router).
#   R4 controls: gate-a16 mutant runs still differ (routing / gate|up scale mutants change the numbers).
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
omega=${OMEGA_DIR:-$HOME/workspace/omega-own30b-a16}
physics=${PHYSICS_DIR:-$HOME/workspace/physics-own30b-k}
ckpt=${OWN30B_CHECKPOINT:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
golden=$repo/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
ev=$repo/docs/inference/evidence/own30b-l1h-gate-chip-$(date -u +%Y%m%dT%H%MZ)
stage=$(mktemp -d /tmp/l1h-gate.XXXX)
trap 'rm -rf "$stage"' EXIT

if ss -ltn 'sport = :18091' 2>/dev/null | grep -q LISTEN; then
	echo "refused: something listens on :18091 (the live server); the orchestrator must stop it first"; exit 2
fi
if [ -n "${1:-}" ]; then
	lib=$1
else
	lib=$stage/omega/libomega_gpu.a
	make -C "$omega" OUT_DIR="$stage/omega" PHYSICS_DIR="$physics" "$lib" >"$stage/omega-build.log" 2>&1 \
		|| { echo "omega build failed (log $stage/omega-build.log)"; tail -5 "$stage/omega-build.log"; exit 2; }
fi
[ -f "$lib" ] || { echo "no omega archive at $lib"; exit 2; }
for s in omega_gpu_pf_hgemm omega_gpu_pf_hconv omega_gpu_pf_hswiglu omega_gpu_pf_qgemm omega_gpu_pf_router_logits; do
	nm "$lib" | grep -q " T $s$" || { echo "$lib lacks $s (needs omega own30b/a16)"; exit 2; }
done
cp "$lib" "$stage/libomega_gpu.a"
export AIEN_OMEGA_GPU_LIB=$stage/libomega_gpu.a AIEN_OMEGA_GPU_LIB_SHA256=$(sha256sum "$lib" | cut -d' ' -f1)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/l1h-gate/target}
( cd "$repo" && cargo build --release -p aien-inference-abi --bin own30b_check --bin own30b_wp_gate --bin own30b_ttft ) \
	|| { echo "build failed"; exit 2; }
bindir=$CARGO_TARGET_DIR/release
mkdir -p "$ev"
{
	echo "W-L1H gate $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "sc commit $(git -C "$repo" rev-parse HEAD) branch $(git -C "$repo" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$repo" status --porcelain -uno | wc -l)"
	echo "omega $(git -C "$omega" rev-parse HEAD 2>/dev/null) archive sha256 $AIEN_OMEGA_GPU_LIB_SHA256"
	for b in own30b_check own30b_wp_gate own30b_ttft; do echo "binary $b sha256 $(sha256sum "$bindir/$b" | cut -d' ' -f1)"; done
	echo "golden sha256 $(sha256sum "$golden" | cut -d' ' -f1)"
} > "$ev/MANIFEST"

# run <name> <binary> <args...>: one sealed load under the lock.
run() {
	local name=$1 bin=$2; shift 2
	{
		echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "cmd flock /tmp/aien-gb10.lock env AIEN_OMEGA_SPIN_US=2000 $bin $*"
		free -g | head -2
	} > "$ev/$name.start"
	crumb whisper odysseus-9e "$repo" - "W-L1H gate: chip run $name (full 30B load)" >/dev/null 2>&1 || true
	flock /tmp/aien-gb10.lock env AIEN_OMEGA_SPIN_US=2000 "$bin" "$@" > "$ev/$name.raw" 2> "$ev/$name.err"
	local rc=$?
	echo "$rc" > "$ev/$name.rc"
	{ echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"; free -g | head -2; } > "$ev/$name.end"
	echo "$name rc=$rc"
}

for acts in a8 a16; do
	run "check-$acts" "$bindir/own30b_check" chip "$ckpt" "$golden" "$ev/check-$acts" \
		--whole-prompt --prefill-device-host --prefill-acts "$acts" --tokens-only --new 16 --prompts 1,2,3,4
	run "gate-$acts" "$bindir/own30b_wp_gate" "$ckpt" "$ev/gate-$acts" --lens 1,512,2048 --devhost --mutants --acts "$acts"
	run "ttft-$acts" "$bindir/own30b_ttft" "$ckpt" "$ev/ttft-$acts" --lens 512,2048 --reps 2 --new-tokens 32 --devhost --acts "$acts"
done

{
	echo "== R1 golden (TOKENS lines)"
	for acts in a8 a16; do grep -h "^TOKENS" "$ev/check-$acts.raw" | sed -E "s/^TOKENS ([^:]+):.*; (IDENTICAL|DIVERGES.*)\$/$acts \1 \2/"; done
	echo "== R2 drift (DRIFT lines)"
	grep -h "DRIFT" "$ev"/gate-a8.raw "$ev"/gate-a16.raw
	echo "== R3 TTFT (raw lines with ttft)"
	for acts in a8 a16; do grep -h -i "ttft" "$ev/ttft-$acts.raw" | sed "s/^/$acts /" | head -12; done
	echo "== summaries"
	grep -h "SUMMARY\|VERDICT\|RULE" "$ev"/*.raw
} > "$ev/SUMMARY.txt" 2>&1
( cd "$ev" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "evidence $ev"
cat "$ev/SUMMARY.txt"
