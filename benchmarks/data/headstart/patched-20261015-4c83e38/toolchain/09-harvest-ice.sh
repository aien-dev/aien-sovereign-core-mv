#!/usr/bin/env bash
# Move the ICE reports left by the first (pre-harvest) abincr run out of the
# measured worktree: keep each report's head (up to the backtrace) in the
# committed data, delete the files so the worktree root is clean again.
set -euo pipefail
S=~/workspace/headstart-exp/aien-work/src
OUT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a/benchmarks/data/headstart/patched-20261015-4c83e38
dst="$OUT/logs/abincr-run1-ice-heads.txt"
: > "$dst"
n=0
for f in "$S"/rustc-ice-*.txt; do
  [ -e "$f" ] || continue
  n=$((n + 1))
  { printf '== %s ==\n' "${f##*/}"
    awk '/^stack backtrace:/ { exit } NR > 12 { exit } { print }' "$f" | LC_ALL=C tr -cd '\11\12\15\40-\176'
  } >> "$dst"
  rm -f "$f"
done
echo "harvested $n ICE reports into $dst"
echo "distinct panic sites:"
grep -h 'panicked at' "$dst" | sort | uniq -c
echo "distinct assertion messages:"
grep -h -A1 'panicked at' "$dst" | grep -v 'panicked at' | grep -v '^--' | sort | uniq -c
