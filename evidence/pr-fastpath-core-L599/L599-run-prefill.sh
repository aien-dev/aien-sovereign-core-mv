#!/bin/bash
set -u
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
ROOT="$HOME/workspace/hermes-desk/core-qualification-L599"
OUT="$HOME/handoffs/pr-failure-triage-20261022"
actual=$(git -C "$ROOT" rev-parse HEAD)
[ "$actual" = a844edd6955a2958d1395aa79536c70b6b81f35b ] || exit 90
[ -z "$(git -C "$ROOT" status --porcelain)" ] || exit 91
[ -f "$HOME/models/TinyLlama-1.1B-Chat-v1.0/model.safetensors" ] || exit 92
cd "$ROOT" || exit 93
set +e
timeout --kill-after=20s 1800 quiet-build nice -n 19 env CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR="$HOME/workspace/targets/core-qualification-L599" AIEN_E2E_CHECKPOINT="$HOME/models/TinyLlama-1.1B-Chat-v1.0" AIEN_E2E2_RECEIPT="$OUT/L599-prefill-behavior-receipt.json" cargo test --release -p aien-runtime --test prefill_e2e2 prefill_e2e2_real_checkpoint_append_gate -- --ignored --exact --nocapture > "$OUT/L599-prefill-behavior.raw.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/L599-prefill-behavior.exit"
python3 -c 'from pathlib import Path;import json;h=Path.home();o=h/"handoffs/pr-failure-triage-20261022";print("CPU_PREFILL_BEHAVIOR_EXIT",(o/"L599-prefill-behavior.exit").read_text());print((o/"L599-prefill-behavior.raw.log").read_text()[-10000:]);p=o/"L599-prefill-behavior-receipt.json";print("RECEIPT_EXISTS",p.exists());print(p.read_text()[-6000:] if p.exists() else "No behavior receipt produced")'
exit "$rc"
