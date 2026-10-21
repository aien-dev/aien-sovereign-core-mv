#!/usr/bin/env bash
# Stroke gate at N = 8 heads (L312, coordinator 02:22Z "8 conversations at once"), red before green, then
# the same command at N = 1, 2, 4 for the ms-per-beat ladder. The tool takes the chip lock itself and waits
# for memory. START=n resumes at step n. Logs: ~/handoffs/motor-stroke8-gate.log, evidence in $E.
set -u
B=$HOME/workspace/targets/motor/sc/release/qwen36_stroke_gate
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-motor/evidence/q36-host-20261020
E=$HOME/workspace/sc-motor/evidence/motor-chip-20261021
LOG=$HOME/handoffs/motor-stroke8-gate.log
START=${START:-1}
P8="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json $G/gen-chat-code.json $G/gen-chat-prose.json $G/gen-chat-chat.json $G/gen-chat-review.json"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
# run NAME GOLDENS [ENV=VAL ...]: one tool run; retries when the box is crowded (exit 2 + "refusing"). Sets rc.
run() {
  local name=$1 goldens=$2; shift 2
  local try
  for try in 1 2 3 4 5 6; do
    env "$@" nice -n 5 "$B" "$M" "$E/$name.json" $goldens > "$E/$name.log" 2>&1; rc=$?
    if [ $rc -eq 2 ] && grep -q "refusing" "$E/$name.log"; then
      say "$name attempt $try: memory guard refused ($(grep -m1 'refusing' "$E/$name.log" | cut -c1-120)); retrying"
      sleep 60; continue
    fi
    break
  done
  say "$name exit $rc: $(grep -m1 '^QWEN36_STROKE_GATE' "$E/$name.log" | cut -c1-230)"
}
for g in $P8; do [ -s "$g" ] || { say "STOP: golden $g missing"; exit 4; }; done
say "stroke8 gate start bin sha $(sha256sum "$B" | cut -c1-16) START=$START"
if [ "$START" -le 1 ]; then
  run stroke8-mutant4 "$P8" AIEN_Q36_TEST_MUTANT=4
  [ "$rc" -eq 1 ] || { say "STOP: mutant 4 did not go red (exit $rc)"; exit 4; }
fi
if [ "$START" -le 2 ]; then
  run stroke8-swap-control "$P8" AIEN_STROKE_SWAP=control
  [ "$rc" -eq 1 ] || { say "STOP: swap control did not go red (exit $rc)"; exit 4; }
fi
if [ "$START" -le 3 ]; then
  run stroke8-real "$P8"
  [ "$rc" -eq 0 ] || { say "STOP: the real N=8 run did not pass (exit $rc)"; exit 4; }
fi
if [ "$START" -le 4 ]; then
  run stroke8-swap-feature "$P8" AIEN_STROKE_SWAP=feature
  [ "$rc" -eq 0 ] || { say "STOP: swap feature did not pass (exit $rc)"; exit 4; }
fi
if [ "$START" -le 5 ]; then
  run stroke1-real "$G/gen-p1.json"
  run stroke2-real "$G/gen-p1.json $G/gen-p2.json"
  run stroke4-real "$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json"
fi
say "stroke8 gate done"
