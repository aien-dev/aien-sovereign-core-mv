#!/bin/bash
# Make sequence files and run the smoke test. Usage: smoke.sh
W=$HOME/workspace/skipped-work; B=$HOME/workspace/targets/skipped/release/qwen38_lab
M=$HOME/models/qwen3.8-27b-fp8-017b9c7a
S=$HOME/workspace/ab-w4-work/seq; R=$HOME/workspace/ab-w4-work/runs
mkdir -p $W/seq $W/runs
$B mkseq $S/prose_prompt.json $R/genA-prose-h0-20261019T135944Z/out.json $W/seq/story.json
$B mkseq $S/codelong_prompt.json $R/genA-codelong-h0-20261019T140338Z/out.json $W/seq/code.json
# golden: prompt + 24 generated
G=$HOME/handoffs/qwen38-ref/gen-chat-max-24.json
$B mkseq <(sed 's/"prompt_ids"/"ids"/' $G) $G $W/seq/golden.json
sha256sum $W/seq/*.json $B > $W/seq/SHA256SUMS
cat $W/seq/SHA256SUMS
TS=$(date -u +%Y%m%dT%H%MZ); O=$W/runs/smoke-golden-$TS; mkdir -p $O
{ date -u; uptime; grep MemAvailable /proc/meminfo; git -C ~/workspace/sc-skipped rev-parse HEAD; git -C ~/workspace/sc-skipped status --porcelain; } > $O/ground.txt
echo "nice -n 19 $B run $M $W/seq/golden.json smoke $O/out.json" > $O/command.txt
nice -n 19 ~/.local/bin/spark-receipt skipped-smoke-golden -- $B run $M $W/seq/golden.json smoke $O/out.json > $O/run.log 2>&1; echo "rc=$?" >> $O/run.log
tail -12 $O/run.log
