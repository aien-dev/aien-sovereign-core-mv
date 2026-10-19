#!/bin/bash
# mtp.sh SEQ [CONFIGS] : MTP head trace over a sequence, then the online drafter learning replay.
W=$HOME/workspace/skipped-work; B=$HOME/workspace/targets/skipped/release/qwen38_lab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
seq=$1; CF=${2:-}
TS=$(date -u +%Y%m%dT%H%MZ); O=$W/runs/mtp-$seq-$TS; mkdir -p $O
unset AIEN_LAB_QUIET_LOCK
{ date -u; uptime; grep MemAvailable /proc/meminfo; echo "branch head $(git -C ~/workspace/sc-skipped rev-parse HEAD)"; sha256sum $B $W/seq/$seq.json; } > $O/ground.txt
echo "nice -n 19 ionice -c3 spark-receipt skipped-mtp-$seq-$TS -- $B mtp $M $W/seq/$seq.json $O/trace" > $O/command.txt
nice -n 19 ionice -c3 ~/.local/bin/spark-receipt skipped-mtp-$seq-$TS -- $B mtp $M $W/seq/$seq.json $O/trace > $O/run.log 2>&1
echo "rc=$?" >> $O/run.log
cp $O/trace/trace.json $O/trace.json 2>/dev/null
if [ -n "$CF" ]; then LA="--configs $CF"; else LA=""; fi
echo "nice -n 19 ionice -c3 spark-receipt skipped-learn-$seq-$TS -- $B learn $M $O/trace $O/learn.json $LA" >> $O/command.txt
nice -n 19 ionice -c3 ~/.local/bin/spark-receipt skipped-learn-$seq-$TS -- $B learn $M $O/trace $O/learn.json $LA > $O/learn.log 2>&1
echo "rc=$?" >> $O/learn.log
{ uptime; grep MemAvailable /proc/meminfo; } >> $O/ground.txt
grep -v '^ "' $O/run.log | grep -v '^layer' > $O/summary.txt; grep -v '^ "' $O/learn.log | grep -v '^tick' >> $O/summary.txt
touch $O/.final
bash $W/bin/publish.sh "skipped-ideas lab evidence: MTP trace + online drafter learning on $seq ($TS)" >> $W/chain.log 2>&1
