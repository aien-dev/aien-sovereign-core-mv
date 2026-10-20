#!/bin/bash
# usage: stress-loop.sh <runs> <out> [test filter]
# Starts 40 busy loops (own pids only), runs the heartbeat tests N times, stops the loops.
runs=$1; out=$2; filter=$3
pids=()
for i in $(seq 40); do yes > /dev/null & pids+=($!); done
echo "stress pids started: ${#pids[@]}; load: $(cut -d' ' -f1-3 /proc/loadavg)" > "$out"
pass=0; fail=0
for n in $(seq "$runs"); do
  if cargo test -q -p aien-heartbeat $filter -- --test-threads=1 > /tmp/fixhb-run.txt 2>&1; then
    pass=$((pass+1)); echo "run $n PASS load=$(cut -d' ' -f1 /proc/loadavg)" >> "$out"
  else
    fail=$((fail+1)); echo "run $n FAIL load=$(cut -d' ' -f1 /proc/loadavg)" >> "$out"; grep -E "panicked|assert|FAILED|failed" /tmp/fixhb-run.txt | head -5 >> "$out"
  fi
done
kill "${pids[@]}" 2>/dev/null
echo "SUMMARY pass=$pass fail=$fail" >> "$out"
