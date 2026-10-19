#!/bin/bash
cd ~/workspace/sc-batch
echo "start $(date -u +%FT%TZ) load $(cat /proc/loadavg) head $(git rev-parse --short HEAD)"
AIEN_FORCE_CPU_STUB=1 CARGO_TARGET_DIR=~/workspace/targets/q38-batch/target-host nice -n 10 cargo test --locked -j 8 -p aien-serve-http -p aien-inference-abi -p aien-omega-q38
echo "exit $? end $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
