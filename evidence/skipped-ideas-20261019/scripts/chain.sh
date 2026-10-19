#!/bin/bash
# chain.sh EXP:SEQ ... : run lab experiments one after another (CPU, nice 19, quiet lock shared),
# each with a receipt, and publish (commit + push) after every run.
W=$HOME/workspace/skipped-work; B=$HOME/workspace/targets/skipped/release/qwen38_lab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
for job in "$@"; do
  exp=${job%%:*}; seq=${job#*:}
  TS=$(date -u +%Y%m%dT%H%MZ); O=$W/runs/$exp-$seq-$TS; mkdir -p $O
  extra=""
  if [ "$exp" = skip ]; then
    bj=$(ls -d $W/runs/base-$seq-*/ | tail -1)out.json
    extra="--base $bj"
  fi
  { date -u; uptime; grep MemAvailable /proc/meminfo; echo "branch head $(git -C ~/workspace/sc-skipped rev-parse HEAD)"; git -C ~/workspace/sc-skipped status --porcelain | grep -v '^??' ; sha256sum $B $W/seq/$seq.json; } > $O/ground.txt
  echo "flock -s /tmp/aien-quiet.lock nice -n 19 spark-receipt skipped-$exp-$seq-$TS -- $B run $M $W/seq/$seq.json $exp $O/out.json $extra" > $O/command.txt
  flock -s /tmp/aien-quiet.lock nice -n 19 ~/.local/bin/spark-receipt skipped-$exp-$seq-$TS -- $B run $M $W/seq/$seq.json $exp $O/out.json $extra > $O/run.log 2>&1
  echo "rc=$?" >> $O/run.log
  grep -v '^layer' $O/run.log | grep -v '^ "' | head -60 > $O/summary.txt
  touch $O/.final
  bash $W/bin/publish.sh "skipped-ideas lab evidence: $exp on $seq ($TS)" >> $W/chain.log 2>&1
done
