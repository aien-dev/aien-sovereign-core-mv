#!/usr/bin/env bash
# One untimed diagnostic of the Headstart-ON incremental failure, run as
# `quietlock run -- 13-diag-once.sh inner` (refused, exit 75, while another
# job holds the quiet flag). Steps: prime an ON incremental check on a fresh
# target dir, apply the same body edit as the measured series (step 1), run
# ONE incremental ON check with -j 8 capturing full stderr, revert the edit.
set -euo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
S=$E/aien-work/src
WT=~/aien-sovereign-core/.claude/worktrees/agent-adc52ad674d7b621a
D=$WT/benchmarks/data/headstart/patched-20261015-4c83e38/diag-once
if [ "${1:-}" != inner ]; then
  exec quietlock run -- "$0" inner
fi
mkdir -p "$D"
CARGO=$H/cargo/target/release/cargo
export RUSTC=$H/rustc/build/host/stage1/bin/rustc
export CARGO_UNSTABLE_HEADSTART=true
export CARGO_TARGET_DIR=$E/aien-work/target-diag-once-ON
F=crates/cortex-rs/src/lib.rs
ANCHOR='    pub fn in_memory() -> Result<Self, String> {'
cd "$S"
trap 'git checkout -- "$F"' EXIT
git diff --quiet || { echo "worktree not clean" >&2; exit 1; }
rm -rf "$CARGO_TARGET_DIR"
"$CARGO" check --workspace --all-targets -j 8 > "$D/prime.stderr.log" 2>&1
echo "prime rc=$?" > "$D/result.txt"
n=$(grep -nF -- "$ANCHOR" "$F" | cut -d: -f1)
sed -i "${n}a\\    let _headstart_m0_probe = std::hint::black_box(1u64);" "$F"
set +e
"$CARGO" check --workspace --all-targets -j 8 > "$D/incr-on.stderr.log" 2>&1
rc=$?
set -e
echo "incremental ON check rc=$rc" >> "$D/result.txt"
git checkout -- "$F"
for f in rustc-ice-*.txt; do
  [ -e "$f" ] || continue
  awk '/^stack backtrace:/ { exit } NR > 12 { exit } { print }' "$f" > "$D/${f%.txt}.head.txt"
  rm -f "$f"
done
rm -rf "$CARGO_TARGET_DIR"
cat "$D/result.txt"
