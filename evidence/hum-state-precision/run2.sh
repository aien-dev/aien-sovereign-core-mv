#!/bin/bash
# L313 second queue (advisor review): fp8 with the scale per v column, and a report-only score on
# the README passage (ppl 17.5, not saturated like the AGPL text). Waits for run.sh to finish.
# Binary: a pinned copy built from the commit that added Fp8E4m3Col (bin-col below).
set -u
B=$HOME/workspace/targets/hum-state-precision/bin-col/q36_state_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-state-precision/evidence/q36-host-20261020
O=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision
N=${N:-512}
until grep -q "ALL DONE" "$O/run.out"; do sleep 60; done
run() { # set name args...
  local set=$1 name=$2; shift 2
  [ -s "$O/$name.json" ] && ! grep -q '"complete": false' "$O/$name.json" && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) start $name MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
  AIEN_L313_SET=$set quiet-build nice -n 10 "$B" "$@" "$O/$name.json" > "$O/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) end $name exit $?"
}
run all score-readme score "$M" "$G/passage-readme.txt" 400
run col score-agpl-col score "$M" "$G/passage-agpl.txt" 400
for p in p1 p2 p3 p4 chat-max-24; do run col drift-col-$p drift "$M" "$G/gen-$p.json" "$N"; done
echo "$(date -u +%FT%TZ) ALL DONE"
