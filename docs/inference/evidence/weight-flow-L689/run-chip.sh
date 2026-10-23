#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/workspace/hermes-desk/sc-weight-flow-L689"
R="$HOME/handoffs/weight-flow-L689"
window=''
finish(){ rc=$?; printf '%s\n' "$rc" > "$R/chip-sweep.rc"; if [ -n "$window" ]; then spark-window close "$window" >> "$R/window-close.log" 2>&1; fi; exit "$rc"; }
trap finish EXIT
export CARGO_BUILD_JOBS=3 CARGO_TARGET_DIR="$R/target-native"
export AIEN_REQUIRE_NATIVE_LIBS=1 AIEN_OMEGA_DIR="$HOME/workspace/hermes-desk/omega-weight-flow-L689" AIEN_PHYSICS_DIR="$HOME/workspace/hermes-desk/physics-weight-flow-L689"
quiet-build cargo build --release -p aien-inference-abi --bin own30b_check > "$R/native-final-build.log" 2>&1
B="$CARGO_TARGET_DIR/release/own30b_check"
{ date -u; git rev-parse HEAD; uname -a; uptime; free -m; nvidia-smi --query-gpu=name,driver_version,power.draw,utilization.gpu --format=csv; sha256sum "$B" "$R/policy" native/weight-flow/policy.om crates/aien-heartbeat/src/weight_flow.rs crates/aien-heartbeat/src/weight_policy.rs crates/aien-inference-abi/src/qwen3_moe_omega/ribbon.rs crates/aien-inference-abi/src/bin/own30b_check.rs; ldd "$B"; } > "$R/chip-identity.txt"
# Correctness window only: an unrelated courier holds the shared quiet descriptor.
# Never stop it or mislabel shared-machine samples as isolated speed qualification.
spark-window open weight-flow-L689 --ledger L689 --lease 15 --gb10 shared --quiet none > "$R/window-open.log" 2>&1
window=$(spark-window id)
printf '%s\n' "$window" > "$R/window-id.txt"
ledger note L689 "Shared-machine native GPU correctness sweep starts, one legacy OWN30B model load slots8, two prompts and four modes; no live35B changes. Quiet descriptor held by unrelated courier so latency samples explicitly unqualified for speedup, no process interference."
spark-window run spark-receipt weight-flow-L689 --idle 0 -- timeout 600 env AIEN_OMEGA_SPIN_US=2000 "$B" chip \
 "$HOME/models/qwen3-30b-a3b-instruct-2507-fp8" \
 "$HOME/workspace/sc-408/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json" \
 "$R/chip-sweep" --tokens-only --prompts 1,2 --new 16 --max-context 256 --weights fp8 --attn fused --router device --lm-head gemv --prefill auto --paged-experts 8 --manifest "$R/model-manifest.json" --weight-flow sweep --flow-policy "$R/policy" --flow-limit 8 \
 > "$R/chip-sweep.stdout" 2> "$R/chip-sweep.stderr"
