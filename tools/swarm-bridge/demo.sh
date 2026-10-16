#!/usr/bin/env bash
# Three-agent swarm-board demo (sc#587 item 7), real headless Claude Code agents.
#   alpha (synapse clone) asks beta a contract question (question --to beta).
#   beta (sc clone) parks in watch and answers once (answer --parent).
#   gamma (sc clone) is subscribed to the thread and gets both through its hooks.
# Nobody reposts; no orchestrator runs. Hooks come only from demo-settings.json
# passed with --settings; ~/.claude/settings.json is not touched.
# Usage: tools/swarm-bridge/demo.sh [run-name]
# Evidence: evidence/swarm-bridge/demo/<run>/ (scrubbed transcripts, board
# events, metrics, the rule check). Exit 0 only when the rule passes.
set -euo pipefail
. "$(dirname "$0")/common.sh"

run=${1:-demo-$(date -u +%Y%m%dT%H%M%SZ)}
[ -x "$SB_BIN" ] || { log "build swarm-bridge first ($SB_BIN missing)"; exit 2; }
sb_seed
sb_run_arm board "$run"
sb_evidence "$SB_DEMO/$run" "$SB_REPO/evidence/swarm-bridge/demo/$run"
jq '{pass: .board.pass, question_id: .board.question_id, answer_id: .board.answer_id,
     board_events: .board.board_events, r1_same_question_id: .board.r1_same_question_id,
     r1_same_answer_id: .board.r1_same_answer_id,
     r1_gamma_subscribed_via_hook: .board.r1_gamma_subscribed_via_hook,
     r2_no_relay_two_events: .board.r2_no_relay_two_events,
     r3_no_missed_no_dup: .board.r3_no_missed_no_dup, r4_task: .board.r4_task}' \
  "$SB_DEMO/$run/out/result.json"
[ "$(jq -r .board.pass "$SB_DEMO/$run/out/result.json")" = true ]
