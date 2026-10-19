#!/bin/sh
# Waits for the suites, then publishes the summary and the READY line (mechanical part only).
E=evidence/qwen38-cascade-suite-20261019T2130Z
cd $HOME/workspace/sc-cascade || exit 1
while ! grep -q ALLDONE $E/progress.txt 2>/dev/null; do sleep 120; done
evidence/cascade-summary.sh evidence/qwen38-cascade-chain-20261019T1650Z $E > $E/summary.csv
$HOME/workspace/targets/sc-cascade/publish.sh
sha=$(git rev-parse HEAD)
echo "READY 2026-10-19 aien-sovereign-core cascade/mtp-chain @ $sha (L014): MTP chain up to 8, one verify boundary, n-gram source, draft tree; measurement suites finished, results in $E/summary.csv, exactness flags in each JSON (mutants must FAIL). Handoff ~/handoffs/2026-10-19-cascade-mtp-chain.md" >> $HOME/handoffs/2026-10-18-pr-desk.md
ledger note L014 "suites finished; summary + READY line written at $sha; review the JSON pass flags before merging"
