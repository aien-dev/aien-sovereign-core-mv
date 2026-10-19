#!/usr/bin/env bash
# tests/wt_mirror_lock_test.sh: two pollers syncing one mirror at once must both succeed (sc#688 SC-LINKED-COMPOSE).
#
# Seen on the Spark 2026-10-19 23:32Z judging sc 0f8d073d: the sc poller ran `wt.sh sync aienos` for the ../aienos
# sibling while another poller synced the same mirror; one fetch lost the ref lock ("cannot lock ref
# 'refs/remotes/origin/feat/real-sky': is at 38c036ee but expected 9997853e"), the sibling was not provided and
# SC-LINKED-COMPOSE failed on "../aienos is missing". Every `wt.sh` change to a mirror now holds one lock per
# mirror root.
#
# Leg 1 (old wt.sh from git, no lock): parallel syncs while the clone's refs move. Reports how often a sync failed;
#        it is a race, so a run that does not hit it says so and is not a failure.
# Leg 2 (this wt.sh): the same load must give zero failed syncs.
# Needs only git. Uses AIEN_WT_BASE so nothing under /var/lib/aien-run is touched; sync never calls sudo.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(mktemp -d); trap 'rm -rf "$R"' EXIT
pass=0; fail=0
ok()  { pass=$((pass+1)); printf 'PASS %s\n' "$1"; }
bad() { fail=$((fail+1)); printf 'FAIL %s: %s\n' "$1" "$2"; }

git init -q "$R/origin"; git -C "$R/origin" commit -q --allow-empty -m 0
for i in $(seq 1 40); do git -C "$R/origin" branch "b$i"; done
git clone -q "$R/origin" "$R/clone"

load() {  # load <wt.sh> <base> : 6 parallel sync loops while 40 remote-tracking refs keep moving; prints failures
  local tool=$1 base=$2 n
  mkdir -p "$base/mirror"; rm -f "$R/err"
  ( for r in $(seq 1 25); do
      git -C "$R/origin" commit -q --allow-empty -m "$r"
      for i in $(seq 1 40); do git -C "$R/origin" branch -f "b$i" HEAD; done
      git -C "$R/clone" fetch -q origin 2>/dev/null
    done ) &
  local mover=$!
  for p in $(seq 1 6); do
    ( while kill -0 $mover 2>/dev/null; do
        AIEN_WT_BASE=$base bash "$tool" sync probe "$R/clone" >/dev/null 2>>"$R/err.$p" || echo x >> "$R/err"
      done ) &
  done
  wait
  n=$(cat "$R/err" 2>/dev/null | wc -l); echo "$n"
}

old="$R/wt-old.sh"
if git -C "$here/../../.." show origin/main:tools/aien-local-ci/wt.sh > "$old" 2>/dev/null && ! grep -q 'mirror_lock' "$old"; then
  sed -i 's|^BASE=/var/lib/aien-run$|BASE=${AIEN_WT_BASE:-/var/lib/aien-run}|' "$old"
  n=$(load "$old" "$R/b-old")
  if (( n > 0 )); then ok "no lock: $n of the parallel syncs failed (the race is real)"
  else printf 'NOTE no lock: the race did not show this run (it is timing dependent)\n'; fi
else
  printf 'NOTE leg 1 skipped: no unlocked wt.sh on origin/main to compare with\n'
fi
n=$(load "$here/../wt.sh" "$R/b-new")
(( n == 0 )) && ok "mirror lock: every parallel sync succeeded" || bad "mirror lock" "$n syncs failed: $(cat "$R"/err.* | sort | uniq -c | head -3)"
AIEN_WT_BASE="$R/b-new" bash "$here/../wt.sh" sync probe "$R/clone" >/dev/null
heads=$(git -C "$R/b-new/mirror/probe.git" rev-parse refs/remotes/origin/b40 2>/dev/null)
[[ $heads == "$(git -C "$R/clone" rev-parse refs/remotes/origin/b40)" ]] && ok "mirror ends at the clone's refs" \
  || bad "final refs" "mirror b40=$heads"
printf '%d passed, %d failed\n' "$pass" "$fail"
(( fail == 0 ))
