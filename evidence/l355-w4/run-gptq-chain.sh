#!/bin/bash
# L355: after the GPTQ store is written, judge it: goldens (run-arm) and held-out (after the fp8 held-out reference exists).
set -u
D=$HOME/workspace/sc-l355/evidence/l355-w4; cd $D; S=$(cat gptq-store-path.txt)
while pgrep -f "q36_w4_store gptq" > /dev/null; do sleep 30; done
if [ ! -s runs/gptq-v1.json ] || [ ! -f "$S" ]; then echo "$(date -u +%FT%TZ) [gptq chain] store missing, stop" >> runs/run.out; exit 1; fi
echo "$(date -u +%FT%TZ) [gptq chain] store ready: $(cat runs/gptq-v1.json)" >> runs/run.out
STORE=$S ./run-arm.sh w4g w4e "" &
while ! grep -q "\[heldout w4e\] end" runs/run.out; do sleep 30; done
./heldout.sh w4g $S
wait
./verdict-gptq.sh w4g w4e > verdict-w4g.json 2> verdict-w4g.err.txt
echo "$(date -u +%FT%TZ) [gptq chain] verdict $(jq -r .verdict verdict-w4g.json)" >> runs/run.out
