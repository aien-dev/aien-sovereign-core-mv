#!/usr/bin/env bash
# Paired benchmark (sc#587 item 8): the board bridge against the pre-board
# flow (agent-mail + crumb whisper/sniff + scripted orchestrator relay), same
# task, model, budget and poll limits. The rule is evidence/swarm-bridge/bench/RULE.md,
# committed before any run. Arms alternate order per pair.
# Usage: tools/swarm-bridge/bench.sh [pairs]   (default 3)
set -euo pipefail
. "$(dirname "$0")/common.sh"

pairs=${1:-3}
[ -x "$SB_BIN" ] || { log "build swarm-bridge first ($SB_BIN missing)"; exit 2; }
stamp=$(date -u +%Y%m%dT%H%M%SZ)
res="$SB_REPO/evidence/swarm-bridge/bench/results/$stamp"
mkdir -p "$res"
sb_seed
for p in $(seq 1 "$pairs"); do
  if [ $((p % 2)) -eq 1 ]; then order="mail board"; else order="board mail"; fi
  for arm in $order; do
    run="bench-$stamp-p$p-$arm"
    sb_run_arm "$arm" "$run" || log "run $run failed to complete"
    sb_evidence "$SB_DEMO/$run" "$res/$run"
  done
done
sb_summary "$res" >"$res/SUMMARY.json"
log "summary: $res/SUMMARY.json"
jq . "$res/SUMMARY.json"
