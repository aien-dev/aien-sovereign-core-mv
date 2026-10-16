#!/usr/bin/env bash
# cold-warm-proof.sh: the periodic proof that a warm pull request target dir gives the same answers as a cold one.
#
#   cold-warm-proof.sh <repo> <pr-head-sha40>
#
# Order (so the warm run is warm from priming, not from the cold run):
#   1. empty target/<repo>/pr, prime it from the trusted clone's origin/main, run the head (warm), --no-cache
#   2. empty target/<repo>/pr, run the head (cold), --no-cache
#   3. compare: per gate, the verdict and the digest of the receipt's `core` minus stdout_sha256/stderr_sha256 (cargo's
#      "Compiling" lines differ between cold and warm by construction; `volatile`, the timings, is never compared); the warm bundle must say target.warm=true, the cold one target.cold=true. Exit 0 only if equal.
#   4. prime again from main, so the live worker finds the dir warm.
#
# It shares /var/lib/aien-run/target/<repo>/pr with the live worker (the exec helper allows no other dir), so it
# REFUSES unless ~/.local/state/aien-local-ci/PAUSE exists and the live worker is not running:
#   touch ~/.local/state/aien-local-ci/PAUSE && aien-local-ci stop
# Its bundles go to a separate state dir (never published, no --github). Results: <proof dir>/result.txt.
set -euo pipefail
repo=${1:?repo}; sha=${2:?pr-head-sha40}
[[ $repo =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*$ && $sha =~ ^[0-9a-f]{40}$ ]] || { echo "bad repo or sha" >&2; exit 2; }
BIN=${AIEN_TEST_BIN:-$HOME/workspace/pipeline/target-sc/release/aien-test}
CLONES=${AIEN_CLONES:-$HOME/workspace/pipeline/clones}
LIVE=${AIEN_STATE:-$HOME/.local/state/aien-local-ci}
[[ -e $LIVE/PAUSE ]] || { echo "refused: $LIVE/PAUSE must exist (the proof empties the shared pr target dir)" >&2; exit 2; }
if [[ -f $LIVE/worker.pid ]] && kill -0 "$(cat "$LIVE/worker.pid")" 2>/dev/null; then
  echo "refused: the live worker (pid $(cat "$LIVE/worker.pid")) is running; aien-local-ci stop first" >&2; exit 2
fi
P=${AIEN_PROOF_DIR:-$HOME/.local/state/aien-local-ci-proof/$repo-${sha:0:12}-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$P/warm" "$P/cold"
git -C "$CLONES/$repo" fetch -q --all
main=$(git -C "$CLONES/$repo" rev-parse --verify 'refs/remotes/origin/main^{commit}')
w() { "$BIN" worker --clones "$CLONES" --sandbox user "$@"; }
# The live warm record must never outlive what this script does to the shared dir.
rm -f "$LIVE/warm/$repo.json"
t0=$(date +%s)
w --state-dir "$P/warm" --once "$repo" "$main" --prime-only --cold | tee "$P/warm/prime.txt"
t1=$(date +%s)
w --state-dir "$P/warm" --once "$repo" "$sha" --no-cache > "$P/warm/run.txt" || true
t2=$(date +%s)
w --state-dir "$P/cold" --once "$repo" "$sha" --no-cache --cold > "$P/cold/run.txt" || true
t3=$(date +%s)
# Re-prime for the live worker, recorded in the live state dir.
w --state-dir "$LIVE" --once "$repo" "$main" --prime-only | tee "$P/warm/reprime.txt" || true
gates() { # <state> -> "<gate> <verdict> <core-digest>" per receipt, sorted
  for f in "$1/evidence/$repo/$sha"/*.json; do
    jq -r '[.core.gate, .core.derived_verdict, (.core | del(.stdout_sha256, .stderr_sha256) | tostring)] | @tsv' "$f" \
      | while IFS=$'\t' read -r g v core; do printf '%s %s %s\n' "$g" "$v" "$(printf '%s' "$core" | jq -cS . | sha256sum | cut -c1-16)"; done
  done | sort
}
gates "$P/warm" > "$P/warm/gates.txt"; gates "$P/cold" > "$P/cold/gates.txt"
wb=$P/warm/bundles/$repo/$sha/bundle.json cb=$P/cold/bundles/$repo/$sha/bundle.json
rc=0
{
  echo "repo=$repo head=$sha main=$main prime_s=$((t1 - t0)) warm_run_s=$((t2 - t1)) cold_run_s=$((t3 - t2))"
  echo "warm target: $(jq -c .target "$wb" 2>/dev/null)"
  echo "cold target: $(jq -c .target "$cb" 2>/dev/null)"
  jq -e '.target.warm == true and .target.primed_from_main == "'"$main"'"' "$wb" >/dev/null 2>&1 || { echo "FAIL warm bundle does not say warm from $main"; rc=1; }
  jq -e '.target.cold == true and .target.warm == false' "$cb" >/dev/null 2>&1 || { echo "FAIL cold bundle does not say cold"; rc=1; }
  [[ -s $P/warm/gates.txt ]] || { echo "FAIL no receipts"; rc=1; }
  if diff "$P/cold/gates.txt" "$P/warm/gates.txt" > "$P/diff.txt"; then echo "EQUAL $(wc -l < "$P/warm/gates.txt") gates (verdict + receipt core digest)"
  else echo "DIFFERENT (see $P/diff.txt)"; rc=1; fi
} | tee "$P/result.txt"
grep -q '^FAIL\|^DIFFERENT' "$P/result.txt" && exit 1
exit $rc
