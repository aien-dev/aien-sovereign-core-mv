#!/usr/bin/env bash
# The first ab stage was stopped after round 1 (host contaminated, before the
# per-sample hold existed). Move its samples out of samples.jsonl into a
# separate raw file so the summary only uses the full held rerun.
set -euo pipefail
OUT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a/benchmarks/data/headstart/patched-20261015-4c83e38
S="$OUT/samples.jsonl"
X="$OUT/samples-ab-run1-interrupted-unheld.jsonl"
[ -e "$X" ] && { echo "already split: $X"; exit 1; }
jq -c 'select(.kind == "ab")' "$S" > "$X"
jq -c 'select(.kind != "ab")' "$S" > "$S.tmp"
mv "$S.tmp" "$S"
echo "moved $(wc -l < "$X") ab samples; kept $(wc -l < "$S")"
