#!/bin/sh
# L190 chain 2 (runs as user unit l190-chain, detached from any chat): rebuild wormhole30 (cache count fix),
# ring check again, one-read timing in a spark-window, then the endpoint receipt (leaves the pair serving).
cd /home/drakestapleton/workspace/sc-wormhole-30b || exit 1
export PATH=/home/drakestapleton/.local/bin:/home/drakestapleton/.cargo/bin:/usr/local/bin:/usr/bin:/bin
B=/home/drakestapleton/workspace/targets/sc-wormhole-30b/release/wormhole30
M=/home/drakestapleton/models
E=evidence/WORMHOLE-30B
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
say() { echo "$(date -u +%H:%MZ) $*"; }
rm -f /tmp/l190/serve-*.log
CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/sc-wormhole-30b quiet-build cargo build --release -p aien-wormhole --bin wormhole30 2>&1 | grep -E "^error|Finished"
until [ "$(avail)" -ge 48 ]; do sleep 30; done
say "ringcheck start, mem $(avail) GiB"
quiet-build $B ringcheck $M/qwen3-30b-a3b-instruct-2507-fp8 $E/06-ringcheck.json 64 512 evidence/WORMHOLE-ACCEPT-69825c92/inputs/q38-codelong.json > /tmp/l190/ringcheck.log 2>&1
say "ringcheck exit=$?"; grep -E "PASS|FAIL" /tmp/l190/ringcheck.log
until [ "$(avail)" -ge 48 ]; do sleep 30; done
ledger note L190 "$(date -u +%H:%MZ) measure: in line for a window (one-read load timing, --gb10 none)"
spark-window open L190-oneread --ledger L190 --gb10 none --lease 45 --wait
say "window open, mem $(avail) GiB"
spark-window run /usr/bin/time -v $B oneread $M/qwen3-30b-a3b-instruct-2507-fp8 $E/02-one-read.json 2 > /tmp/l190/oneread.log 2>&1
say "oneread exit=$?"
spark-window close
grep -E "PASS|FAIL" /tmp/l190/oneread.log
/home/drakestapleton/handoffs/l190/endpoint.sh
say "endpoint exit=$?"
