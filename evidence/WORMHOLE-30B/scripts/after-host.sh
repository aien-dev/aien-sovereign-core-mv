#!/bin/sh
# L190: after run-host.sh (oneread + accept) finishes: rebuild wormhole30, ring exactness check, endpoint receipt.
cd /home/drakestapleton/workspace/sc-wormhole-30b || exit 1
until grep -q "accept exit=" /tmp/l190/run-host.log; do sleep 30; done
until grep -q "Finished\|error" /tmp/l190/build-serve.log 2>/dev/null; do sleep 30; done
CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/sc-wormhole-30b quiet-build cargo build --release -p aien-wormhole --bin wormhole30 2>&1 | grep -E "^error|Finished"
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 48 ]; do sleep 30; done
quiet-build /home/drakestapleton/workspace/targets/sc-wormhole-30b/release/wormhole30 ringcheck /home/drakestapleton/models/qwen3-30b-a3b-instruct-2507-fp8 evidence/WORMHOLE-30B/06-ringcheck.json 64 512 evidence/WORMHOLE-ACCEPT-69825c92/inputs/q38-codelong.json > /tmp/l190/ringcheck.log 2>&1
echo "ringcheck exit=$?"; grep -E "PASS|FAIL" /tmp/l190/ringcheck.log
/home/drakestapleton/handoffs/l190/endpoint.sh
echo "endpoint exit=$?"
