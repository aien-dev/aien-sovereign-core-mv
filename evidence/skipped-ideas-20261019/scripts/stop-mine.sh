#!/bin/bash
# stop-mine.sh : stop this lab's own running jobs (chain, lab runs, rows waiter) by exact pid, leave notes.
W=$HOME/workspace/skipped-work
for p in $(pgrep -f "^bash bin/chain.sh") $(pgrep -f "^bash bin/mtp.sh") $(pgrep -f "^bash bin/rows.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "targets/skipped/release/qwen38_lab"); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in bash*) continue;; esac
  kill $p
done
sleep 3
for d in $W/runs/quant-story-20261019T1723Z $W/runs/mtp-code-20261019T1726Z $W/runs/rows-L8-20261019T1659Z; do
  [ -f $d/.final ] && continue
  echo "STOPPED by the lab agent at $(date -u +%FT%TZ) (its own process) after Drake's quiet-lock rule change: long CPU lab runs now run WITHOUT the quiet lock at nice 19 ionice -c3, timing runs through quiet-measure. No numbers from this run are used." > $d/STOPPED.txt
  touch $d/.final
done
pgrep -af qwen38_lab | grep -v pgrep | cut -c1-120
