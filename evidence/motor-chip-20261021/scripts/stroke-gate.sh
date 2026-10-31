#!/usr/bin/env bash
# L312 stroke v1 gate: 4 conversations per stroke on the chip, red before green.
# The tool waits for memory (no lock held) and takes /tmp/aien-gb10.lock around the chip part.
# START=n resumes at step n (the earlier steps' logs for the SAME binary sha stand as evidence).
set -u
B=${B:-$HOME/workspace/targets/motor/sc/release/qwen36_stroke_gate}
M=${M:-$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0}
G=$HOME/workspace/sc-motor/evidence/q36-host-20261020
E=$HOME/workspace/sc-motor/evidence/motor-chip-20261021
LOG=$HOME/handoffs/motor-stroke-gate.log
P="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json"
START=${START:-1}
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
# run NAME [ENV=VAL ...]: runs the tool, retrying (up to 6 times) when its memory guard refused
# (exit 2 + "refusing": the box is crowded, not a verdict). Sets rc.
run() {
  local name=$1; shift
  for attempt in 1 2 3 4 5 6; do
    env "$@" nice -n 5 "$B" "$M" "$E/stroke-$name.json" $P > "$E/stroke-$name.log" 2>&1; rc=$?
    if [ $rc -eq 2 ] && grep -q "refusing" "$E/stroke-$name.log"; then
      say "$name attempt $attempt: memory guard refused ($(grep refusing "$E/stroke-$name.log" | tail -1 | cut -c1-120)); retrying"
      sleep 60; continue
    fi
    break
  done
  say "$name exit $rc: $(grep '^QWEN36_STROKE_GATE' "$E/stroke-$name.log" | tail -1 | cut -c1-200)"
}
say "stroke gate start bin sha $(sha256sum "$B" | cut -c1-16) START=$START"
if [ "$START" -le 1 ]; then
  # 1. negative control: mutant 4 (shared expert dropped) must FAIL (exit 1 = judged; 2 = error = STOP)
  run mutant4 AIEN_Q36_TEST_MUTANT=4
  [ $rc -eq 1 ] || { say "STOP: mutant 4 did not go red (exit $rc)"; exit 4; }
fi
if [ "$START" -le 2 ]; then
  # 2. negative control: head states swapped, feeds not: must FAIL
  run swap-control AIEN_STROKE_SWAP=control
  [ $rc -eq 1 ] || { say "STOP: swap control did not go red (exit $rc)"; exit 4; }
fi
if [ "$START" -le 3 ]; then
  # 3. the real thing: p1..p4 interleaved, one stroke per step
  run p1234
fi
# 4. head swap as a feature: states and feeds move together, must PASS
run swap-feature AIEN_STROKE_SWAP=feature
say "stroke gate done"
