#!/bin/bash
set -eu
H=$HOME
SOURCE=$H/workspace/pipeline/clones/aien-architecture
WT=$H/workspace/hermes-desk/architecture-planner-L700
OUT=$H/handoffs/L700-parent-replay
HEAD=b159d1d513ef63f322a6a36bf3ade50f1a202fce
BASE=1a3594340c70969c245554b07062efd8ead8a026
mkdir -p "$OUT"
git -C "$SOURCE" rev-parse --show-toplevel
test ! -e "$WT"
git -C "$SOURCE" -c core.hooksPath=/dev/null worktree add --detach "$WT" "$HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
set +e
(cd "$WT" && "$H/workspace/pipeline/target-sc/release/aien-test" plan --base "$BASE" --repo github.com/aien-dev/aien-architecture --evidence-dir "$H/.local/state/aien-local-ci/evidence") > "$OUT/plan.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/plan.exit"
git -C "$WT" diff --name-status "$BASE...$HEAD" > "$OUT/changed-paths.txt"
git -C "$WT" status --short
printf 'Planner exit=%s; evidence=%s\n' "$RC" "$OUT"
