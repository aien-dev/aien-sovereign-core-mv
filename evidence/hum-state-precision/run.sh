#!/bin/bash
# L313 host runs, one process at a time, each under its own quiet-build so chip measurements
# can get in between. Binary: q36_state_drift from sc hum/state-precision (built by
# ~/handoffs/hum-state-precision/build.sh). Model rev 95a723d0 (sha-checked by L306).
set -u
B=$HOME/workspace/targets/hum-state-precision/sc/release/q36_state_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
G=$HOME/workspace/sc-hum-state-precision/evidence/q36-host-20261020
O=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision
N=${N:-512}
run() { # name args...
  local name=$1; shift
  [ -s "$O/$name.json" ] && ! grep -q '"complete": false' "$O/$name.json" && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) start $name MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
  quiet-build nice -n 10 "$B" "$@" "$O/$name.json" > "$O/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) end $name exit $?"
}
run score-agpl score "$M" "$G/passage-agpl.txt" 400
for p in p1 p2 p3 p4 chat-max-24; do run drift-$p drift "$M" "$G/gen-$p.json" "$N"; done
for p in p1 p2 p3 p4 chat-max-24; do run chunk-$p chunk "$M" "$O/drift-$p.json" "$N"; done
echo "$(date -u +%FT%TZ) ALL DONE"
