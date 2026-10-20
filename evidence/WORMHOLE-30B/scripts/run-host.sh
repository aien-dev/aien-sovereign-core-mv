#!/bin/sh
# L190: wait for memory, one-read timing in a spark-window (no chip), then W3 forward acceptance.
cd /home/drakestapleton/workspace/sc-wormhole-30b || exit 1
B=/home/drakestapleton/workspace/targets/sc-wormhole-30b/release/wormhole30
M=/home/drakestapleton/models
E=evidence/WORMHOLE-30B
I=evidence/WORMHOLE-ACCEPT-69825c92/inputs
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
until [ "$(avail)" -ge 48 ]; do sleep 30; done
echo "$(date -u +%H:%MZ) mem $(avail) GiB free, opening window"
ledger note L190 "$(date -u +%H:%MZ) measure: in line for a window (one-read load timing, --gb10 none)"
spark-window open L190-oneread --ledger L190 --gb10 none --lease 45 --wait || exit 2
echo "$(date -u +%H:%MZ) window open, mem $(avail) GiB"
spark-window run /usr/bin/time -v $B oneread $M/qwen3-30b-a3b-instruct-2507-fp8 $E/02-one-read.json 2 > /tmp/l190/oneread.log 2>&1
echo "oneread exit=$?"
spark-window close
grep -E "PASS|FAIL|Elapsed|Maximum resident" /tmp/l190/oneread.log
until [ "$(avail)" -ge 48 ]; do sleep 30; done
quiet-build $B accept $M/qwen3-30b-a3b-instruct-2507-fp8 $M/qwen3.8-27b-fp8-017b9c7a/tokenizer.json $M/qwen3-30b-a3b-instruct-2507-fp8/tokenizer.json $E/04-accept-forward.json 1,2,4,8 $I/q38-code-golden.json $I/q38-codelong.json $I/q38-prose.json > /tmp/l190/accept.log 2>&1
echo "accept exit=$?"
grep -E "PASS|FAIL" /tmp/l190/accept.log
