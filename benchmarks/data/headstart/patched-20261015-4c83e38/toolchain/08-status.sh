#!/usr/bin/env bash
# Report whether the measured AIEN source tree is clean and who holds the quiet flag.
S=~/workspace/headstart-exp/aien-work/src
cd "$S" || exit 1
echo "head: $(git log --oneline -1)"
echo "dirty files: $(git status --short | wc -l)"
git status --short | head -5
quietlock check; echo "quietlock check rc=$?"
ls -la ~/workspace/.spark-quiet* 2>/dev/null | head -3
