#!/bin/bash
# rows.sh LAYERS REPS : chip timing of the exact multi-row path (idea 5), exclusive quiet lock + GB10 lock.
W=$HOME/workspace/skipped-work; B=$HOME/workspace/targets/skipped/release/qwen38_lab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
L=${1:-8}; R=${2:-7}
TS=$(date -u +%Y%m%dT%H%MZ); O=$W/runs/rows-L$L-$TS; mkdir -p $O
{ date -u; uptime; grep MemAvailable /proc/meminfo; echo "branch head $(git -C ~/workspace/sc-skipped rev-parse HEAD)"; sha256sum $B $HOME/workspace/targets/q38-f7-omega/libomega_gpu.a; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv; } > $O/ground.txt
echo "flock -x /tmp/aien-quiet.lock flock /tmp/aien-gb10.lock nice -n 10 spark-receipt skipped-rows-L$L-$TS -- $B rows $M $L $R $O/out.json" > $O/command.txt
flock -x /tmp/aien-quiet.lock flock /tmp/aien-gb10.lock nice -n 10 ~/.local/bin/spark-receipt skipped-rows-L$L-$TS -- $B rows $M $L $R $O/out.json > $O/run.log 2>&1
echo "rc=$?" >> $O/run.log
{ uptime; grep MemAvailable /proc/meminfo; } >> $O/ground.txt
grep -v '^ "' $O/run.log | head -40 > $O/summary.txt
touch $O/.final
bash $W/bin/publish.sh "skipped-ideas lab evidence: rows timing L$L ($TS)" >> $W/chain.log 2>&1
