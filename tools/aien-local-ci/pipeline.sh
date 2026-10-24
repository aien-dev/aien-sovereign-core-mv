#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pipeline.sh: Spark side of the local pipeline (ADR 0033, amended 2026-10-24: Spark qualification is the Alpha
# stage, not a merge gate).
#
#   pipeline.sh publish|merge|alpha|round [--repos a,b]      (round = publish, then merge, then alpha)
#
# publish  copy every signed bundle the worker wrote ($STATE/bundles/<repo>/<sha>/bundle.json) that verifies
#          (verify-bundle exit 0 PASS or 1 FAIL with a key from origin/main:.aien/verify.pub) byte-exact to the
#          receipts clone, one commit per round, pull --rebase and push (3 attempts, never forced).
# merge    merge open pull requests labeled exactly ready-to-merge that are not drafts and merge cleanly. No Spark
#          bundle, no status check and no up-to-date requirement: the Spark never blocks a merge (Drake 2026-10-24).
# alpha    tag the newest commit on main whose published bundle is a full-scope PASS (every gate, GPU included,
#          `aien-test verify-bundle --require-full`) as alpha/YYYYMMDD.N, an annotated tag naming the bundle digest,
#          and push the tag. Never moves a branch. Run `aien-alpha REPO` to qualify main's head on the Spark.
#
# It never reads the signing key and never signs anything. One instance at a time (flock; a second one exits 0).
# Every action and every refusal is one JSON line in $STATE/pipeline.jsonl.
# Exit: 0 round done (refusals included), 2 usage or configuration error.
set -euo pipefail

STATE=${AIEN_STATE:-$HOME/.local/state/aien-local-ci}
CLONES=${AIEN_CLONES:-$HOME/workspace/pipeline/clones}
RECEIPTS=${AIEN_RECEIPTS:-$HOME/workspace/pipeline/aien-receipts}
RECEIPTS_BRANCH=${AIEN_RECEIPTS_BRANCH:-main}
BIN=${AIEN_TEST_BIN:-$HOME/workspace/pipeline/target-sc/release/aien-test}
OWNER=${AIEN_GH_OWNER:-aien-dev}
REPOS=${AIEN_REPOS:-aien-sovereign-core,omega,aienos,physics,aien-protocols,aegis-runtime,spark-rsi,aien-architecture,benchmarks,synapse,interplane}
GH=${GH:-gh}
GIT=${GIT:-git}
MERGE_LABEL="ready-to-merge"
ALPHA_PREFIX="alpha/"
RECEIPTS_URL=${AIEN_RECEIPTS_URL:-https://github.com/$OWNER/aien-receipts/blob/$RECEIPTS_BRANCH}

usage() { echo "usage: pipeline.sh publish|merge|alpha|round [--repos a,b]" >&2; exit 2; }
die() { echo "pipeline.sh: $*" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
cmd=$1; shift
case "$cmd" in publish|merge|alpha|round) ;; *) usage ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repos) [[ $# -ge 2 ]] || usage; REPOS=$2; shift 2 ;;
    *) usage ;;
  esac
done
IFS=, read -r -a repo_list <<< "$REPOS"
[[ ${#repo_list[@]} -gt 0 ]] || die "no repositories named"
for r in "${repo_list[@]}"; do
  [[ $r =~ ^[A-Za-z0-9._-]+$ && $r != . && $r != .. ]] || die "bad repository name '$r'"
done

command -v jq >/dev/null || die "jq not found"
command -v flock >/dev/null || die "flock not found"
command -v sha256sum >/dev/null || die "sha256sum not found"
command -v "$GIT" >/dev/null || die "git ($GIT) not found"
[[ -d $CLONES ]] || die "clones directory $CLONES missing"
[[ -d $RECEIPTS/.git ]] || die "receipts clone $RECEIPTS is not a git clone"
[[ -x $BIN ]] || die "aien-test binary $BIN missing or not executable"
if [[ $cmd == merge || $cmd == round ]]; then command -v "$GH" >/dev/null || die "gh ($GH) not found"; fi

mkdir -p "$STATE"
JOURNAL=$STATE/pipeline.jsonl
exec 9>>"$STATE/pipeline.lock"
# A second instance exits 0 at once; AIEN_LOCK_WAIT=SECS (aien-alpha) waits that long for the running one instead.
if [[ ${AIEN_LOCK_WAIT:-0} =~ ^[1-9][0-9]*$ ]]; then flock -w "$AIEN_LOCK_WAIT" 9 || exit 0; else flock -n 9 || exit 0; fi
WORK=$(mktemp -d "$STATE/.pipeline.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
GH_DEFER=$WORK/gh-deferred
# ghc ARGS...: every GitHub call goes through here. ~/.local/bin/gh (the shared rate-limit guard) exits 98 when it
# refuses (allowance reserved or rate-limit marker set): then this run makes no further GitHub call, and every
# journal line from then on carries deferred="gh guard exit 98: try next round". Never a retry loop, never a FAIL.
ghc() {
  local rc=0
  [[ -e $GH_DEFER ]] && return 98
  "$GH" "$@" || rc=$?
  [[ $rc != 98 ]] || : > "$GH_DEFER"
  return "$rc"
}

# journal key=value ...: one JSON line in the journal (built by jq), a short line on stdout.
journal() {
  local args=(--arg utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)") kv
  for kv in "$@"; do args+=(--arg "${kv%%=*}" "${kv#*=}"); done
  [[ ! -e $GH_DEFER ]] || args+=(--arg deferred "gh guard exit 98: try next round")
  jq -nc "${args[@]}" '$ARGS.named' >> "$JOURNAL"
  echo "$*"
}

digest() { sha256sum "$1" | cut -d' ' -f1; }

# keys CLONE REF: the trusted public keys at REF, one per line (same rule as aien-test: '#' comments, 64 hex).
keys() {
  "$GIT" -C "$1" show "$2:.aien/verify.pub" 2>/dev/null | sed 's/#.*//' | tr -d ' \t\r' | tr 'A-F' 'a-f' \
    | grep -E '^[0-9a-f]{64}$' || true
}

# verify FILE REPO SHA KEYS [REQUIRE] [full]: verify-bundle with every key; prints the best exit code (0, 1, 2, 3,
# other) and the list of codes. --repo and --commit come from the caller (the path), never from the bundle. A sixth
# argument "full" adds --require-full: only a full-scope (Alpha) qualification can pass.
verify() {
  local file=$1 repo=$2 sha=$3 keylist=$4 require=${5:-} scope=${6:-} key rc codes="" best=""
  local extra=()
  [[ -n $require ]] && extra+=(--require "$require")
  [[ $scope == full ]] && extra+=(--require-full)
  while IFS= read -r key; do
    [[ -n $key ]] || continue
    rc=0
    "$BIN" verify-bundle "$file" --repo "github.com/$OWNER/$repo" --commit "$sha" --pubkey "$key" "${extra[@]}" \
      >/dev/null 2>&1 || rc=$?
    codes+="${codes:+,}$rc"
  done <<< "$keylist"
  for rc in 0 1 2 3; do [[ ,$codes, == *",$rc,"* ]] && { best=$rc; break; }; done
  echo "${best:-${codes%%,*}} ${codes:-none}"
}

verdict_name() { case "$1" in 0) echo PASS ;; 1) echo FAIL ;; 2) echo NOT_RUN ;; 3) echo REJECTED ;; *) echo "EXIT_$1" ;; esac; }

declare -A SLUGS=()
SLUG_DIR=$STATE/slugs
SLUG_TTL=${AIEN_SLUG_TTL:-86400}
# slug REPO: canonical owner/name on GitHub (aien-dev/<repo> may redirect, e.g. to aien-dev/<repo>-mv), resolved once
# per run. Cached on disk in $STATE/slugs/<repo>: used while younger than AIEN_SLUG_TTL seconds (24 h); refreshed
# with one `gh api` call when missing or older; an older cached name is still used when the refresh fails (an
# exhausted API limit must not refuse every pull request). Every name, cached or fresh, passes the same regex.
slug() {
  local s="" f=$SLUG_DIR/$1 age="" fresh
  if [[ -z ${SLUGS[$1]:-} ]]; then
    [[ -f $f ]] && { s=$(head -n 1 "$f"); age=$(( $(date +%s) - $(stat -c %Y "$f") )); }
    if [[ -n $s && $age -lt $SLUG_TTL && $s =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
      journal action=slug repo="$1" result=cached slug="$s" >/dev/null
    elif fresh=$(ghc api "repos/$OWNER/$1" --jq .full_name 2>/dev/null) && [[ $fresh =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
      s=$fresh
      mkdir -p "$SLUG_DIR" && printf '%s\n' "$s" > "$f.tmp" && mv "$f.tmp" "$f"
      journal action=slug repo="$1" result=refreshed slug="$s" >/dev/null
    elif [[ -n $s && $s =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
      journal action=slug repo="$1" result=stale-cache slug="$s" reason="refresh failed, cache is ${age}s old" >/dev/null
    else
      return 1
    fi
    SLUGS[$1]=$s
  fi
  echo "${SLUGS[$1]}"
}

fetch_receipts() { "$GIT" -C "$RECEIPTS" fetch -q origin; }

do_publish() {
  local branch dirty repo clone keylist f sha tmp ldig dest old v best codes listed=() records=() n
  branch=$("$GIT" -C "$RECEIPTS" symbolic-ref --short HEAD 2>/dev/null || true)
  if [[ $branch != "$RECEIPTS_BRANCH" ]]; then
    journal action=publish result=refused reason="receipts clone is on '$branch', not $RECEIPTS_BRANCH"; return 0
  fi
  dirty=$("$GIT" -C "$RECEIPTS" status --porcelain)
  if [[ -n $dirty ]]; then
    journal action=publish result=refused reason="receipts clone has uncommitted changes"; return 0
  fi
  if ! "$GIT" -C "$RECEIPTS" pull -q --rebase origin "$RECEIPTS_BRANCH"; then
    journal action=publish result=refused reason="receipts pull failed"; return 0
  fi
  for repo in "${repo_list[@]}"; do
    [[ -d $STATE/bundles/$repo ]] || continue
    clone=$CLONES/$repo
    if [[ ! -d $clone ]]; then journal action=publish repo="$repo" result=skipped reason="no trusted clone"; continue; fi
    if ! "$GIT" -C "$clone" fetch -q origin; then
      journal action=publish repo="$repo" result=skipped reason="fetch of trusted clone failed"; continue
    fi
    keylist=$(keys "$clone" origin/main)
    if [[ -z $keylist ]]; then
      journal action=publish repo="$repo" result=skipped reason="not enrolled (no key in origin/main:.aien/verify.pub)"; continue
    fi
    for f in "$STATE/bundles/$repo"/*/bundle.json; do
      [[ -f $f ]] || continue
      sha=$(basename "$(dirname "$f")")
      [[ $sha =~ ^[0-9a-f]{40}$ ]] || continue
      tmp=$WORK/candidate.json
      cp -- "$f" "$tmp"
      ldig=$(digest "$tmp")
      dest=$RECEIPTS/$repo/$sha/bundle.json
      old=""
      if [[ -f $dest ]]; then old=$(digest "$dest"); [[ $old == "$ldig" ]] && continue; fi
      if ! jq -e '.plan | has("blocked")' "$tmp" >/dev/null 2>&1; then
        journal action=publish repo="$repo" sha="$sha" result=refused reason="bundle unreadable or has no plan.blocked"; continue
      fi
      if ! jq -e '.plan.blocked == null' "$tmp" >/dev/null; then
        journal action=publish repo="$repo" sha="$sha" result=skipped reason="plan.blocked is not null"; continue
      fi
      v=$(verify "$tmp" "$repo" "$sha" "$keylist"); best=${v%% *}; codes=${v#* }
      if [[ $best != 0 && $best != 1 ]]; then
        journal action=publish repo="$repo" sha="$sha" result=refused \
          reason="verify-bundle $(verdict_name "$best") (exit codes per key: $codes)"; continue
      fi
      mkdir -p "$(dirname "$dest")"
      cp -- "$tmp" "$dest"
      "$GIT" -C "$RECEIPTS" add -- "$repo/$sha/bundle.json"
      listed+=("$repo/$sha $(verdict_name "$best") sha256:$ldig${old:+ replaces sha256:$old}")
      records+=("$repo $sha $ldig $(verdict_name "$best")")
    done
  done
  n=${#listed[@]}
  [[ $n -gt 0 ]] || return 0
  if ! "$GIT" -C "$RECEIPTS" commit -q -m "publish $n bundle(s)" -m "$(printf '%s\n' "${listed[@]}")"; then
    "$GIT" -C "$RECEIPTS" reset -q --hard "origin/$RECEIPTS_BRANCH"
    journal action=publish result=failed reason="receipts commit failed"; return 0
  fi
  local attempt pushed=0
  for attempt in 1 2 3; do
    if "$GIT" -C "$RECEIPTS" push -q origin "HEAD:$RECEIPTS_BRANCH"; then pushed=1; break; fi
    if ! "$GIT" -C "$RECEIPTS" pull -q --rebase origin "$RECEIPTS_BRANCH"; then
      "$GIT" -C "$RECEIPTS" rebase --abort 2>/dev/null || true
      break
    fi
  done
  local rec
  if [[ $pushed == 1 ]]; then
    for rec in "${records[@]}"; do
      read -r repo sha ldig v <<< "$rec"
      journal action=publish repo="$repo" sha="$sha" result=published verdict="$v" digest="$ldig" reason="pushed"
    done
  else
    # Only this round's own commit is dropped (the tree was clean before); the next round recomputes it.
    "$GIT" -C "$RECEIPTS" fetch -q origin 2>/dev/null || true
    "$GIT" -C "$RECEIPTS" reset -q --hard "origin/$RECEIPTS_BRANCH"
    for rec in "${records[@]}"; do
      read -r repo sha ldig v <<< "$rec"
      journal action=publish repo="$repo" sha="$sha" result=failed digest="$ldig" reason="push failed after 3 attempts"
    done
  fi
}

# merge_one REPO SLUG NUM SHA: the PR is still at SHA, not a draft, and merges without conflicts. Nothing else:
# no bundle, no status check, no up-to-date rule. Journals the outcome.
merge_one() {
  local repo=$1 s=$2 num=$3 sha=$4 pr state head draft
  merge_refuse() { journal action=merge repo="$repo" pr="$num" sha="$sha" result=refused reason="$1"; }
  if ! pr=$(ghc api "repos/$s/pulls/$num" --jq '"\(.mergeable_state // "unknown") \(.head.sha) \(.draft)"'); then
    merge_refuse "cannot read pull request"; return 0
  fi
  read -r state head draft <<< "$pr"
  [[ $head == "$sha" ]] || { merge_refuse "head moved to $head"; return 0; }
  [[ $draft == false ]] || { merge_refuse "draft"; return 0; }
  case $state in
    clean|unstable|behind|has_hooks) ;;
    dirty) merge_refuse "conflicts with the base: merge the base in and push"; return 0 ;;
    unknown)
      journal action=merge repo="$repo" pr="$num" sha="$sha" result=wait reason="GitHub has not computed mergeability yet"
      return 0 ;;
    *) merge_refuse "mergeable state $state"; return 0 ;;
  esac
  if ghc pr merge "$num" -R "$s" --merge --match-head-commit "$sha"; then
    journal action=merge repo="$repo" pr="$num" sha="$sha" result=merged reason="labeled, not a draft, merges cleanly ($state)"
  else
    journal action=merge repo="$repo" pr="$num" sha="$sha" result=failed reason="gh pr merge failed"
  fi
}

do_merge() {
  local repo s prs num sha draft labeled
  # Drake 2026-10-18: "stop all agents from trying to merge". Merge-only stop; publish and alpha keep going.
  if [[ -e $STATE/MERGE-PAUSE ]]; then journal action=merge result=paused reason="MERGE-PAUSE present"; return 0; fi
  for repo in "${repo_list[@]}"; do
    if ! s=$(slug "$repo"); then journal action=merge repo="$repo" result=refused reason="cannot resolve GitHub slug"; continue; fi
    if ! prs=$(ghc api "repos/$s/pulls?state=open&per_page=100" --jq \
      ".[] | [.number, .head.sha, .draft, ([.labels[].name] | index(\"$MERGE_LABEL\") != null)] | @tsv"); then
      journal action=merge repo="$repo" result=refused reason="cannot list pull requests"; continue
    fi
    while IFS=$'\t' read -r num sha draft labeled; do
      [[ -n $num ]] || continue
      # Unlabeled pull requests are not the pipeline's to merge: skipped without a journal line (every round).
      [[ $labeled == true ]] || continue
      if [[ ! $sha =~ ^[0-9a-f]{40}$ ]]; then
        journal action=merge repo="$repo" pr="$num" result=refused reason="bad head sha"; continue
      fi
      if [[ $draft != false ]]; then
        journal action=merge repo="$repo" pr="$num" sha="$sha" result=refused reason="draft"; continue
      fi
      merge_one "$repo" "$s" "$num" "$sha"
    done <<< "$prs"
  done
}

# alpha_name CLONE: the next free alpha/YYYYMMDD.N name (UTC date, N from 1).
alpha_name() {
  local day n=1
  day=$(date -u +%Y%m%d)
  while "$GIT" -C "$1" rev-parse -q --verify "refs/tags/$ALPHA_PREFIX$day.$n" >/dev/null; do n=$((n + 1)); done
  echo "$ALPHA_PREFIX$day.$n"
}

# do_alpha: per repository, the newest commit on origin/main with a published full-scope PASS bundle becomes the next
# Alpha, unless it already carries an alpha tag or the current Alpha already contains it.
do_alpha() {
  local repo clone keylist require cands sha tmp v best best_sha best_time t latest name dig msg
  if ! fetch_receipts; then journal action=alpha result=refused reason="receipts fetch failed"; return 0; fi
  for repo in "${repo_list[@]}"; do
    clone=$CLONES/$repo
    [[ -d $clone ]] || continue
    # One cheap scan per repository: only bundles whose signed plan says scope full are looked at.
    cands=$("$GIT" -C "$RECEIPTS" grep -lE '"scope"[[:space:]]*:[[:space:]]*"full"' "origin/$RECEIPTS_BRANCH" -- "$repo/" 2>/dev/null \
      | sed -n "s|^origin/$RECEIPTS_BRANCH:$repo/\([0-9a-f]\{40\}\)/bundle.json\$|\1|p" || true)
    [[ -n $cands ]] || continue
    if ! "$GIT" -C "$clone" fetch -q origin main '+refs/tags/alpha/*:refs/tags/alpha/*'; then
      journal action=alpha repo="$repo" result=refused reason="fetch of trusted clone failed"; continue
    fi
    keylist=$(keys "$clone" origin/main)
    if [[ -z $keylist ]]; then
      journal action=alpha repo="$repo" result=skipped reason="not enrolled (no key in origin/main:.aien/verify.pub)"; continue
    fi
    require=$("$GIT" -C "$clone" show "origin/main:.aien/always" 2>/dev/null | grep -v '^#' | grep -v '^$' | paste -sd, - || true)
    best_sha="" best_time=0
    while IFS= read -r sha; do
      "$GIT" -C "$clone" merge-base --is-ancestor "$sha" origin/main 2>/dev/null || continue
      t=$("$GIT" -C "$clone" log -1 --format=%ct "$sha")
      (( t > best_time )) || continue
      tmp=$WORK/alpha.json
      "$GIT" -C "$RECEIPTS" show "origin/$RECEIPTS_BRANCH:$repo/$sha/bundle.json" > "$tmp" 2>/dev/null || continue
      v=$(verify "$tmp" "$repo" "$sha" "$keylist" "$require" full); best=${v%% *}
      if [[ $best != 0 ]]; then
        journal action=alpha repo="$repo" sha="$sha" result=skipped reason="verify-bundle --require-full $(verdict_name "$best")"
        continue
      fi
      best_sha=$sha best_time=$t
    done <<< "$cands"
    [[ -n $best_sha ]] || continue
    [[ -z $("$GIT" -C "$clone" tag -l "${ALPHA_PREFIX}*" --points-at "$best_sha") ]] || continue
    latest=$("$GIT" -C "$clone" for-each-ref --sort=-creatordate --count=1 --format='%(*objectname)' "refs/tags/${ALPHA_PREFIX%/}/")
    if [[ -n $latest ]] && "$GIT" -C "$clone" merge-base --is-ancestor "$best_sha" "$latest" 2>/dev/null; then continue; fi
    name=$(alpha_name "$clone")
    "$GIT" -C "$RECEIPTS" show "origin/$RECEIPTS_BRANCH:$repo/$best_sha/bundle.json" > "$WORK/alpha.json"
    dig=$(digest "$WORK/alpha.json")
    msg=$(printf 'Alpha: %s %s\n\nEvery gate passed on the Spark: full-scope qualification, GPU gates included.\nbundle: sha256:%s\nreceipt: %s/%s/%s/bundle.json\nchecked: aien-test verify-bundle --require-full, exit 0 (PASS)\n' \
      "$repo" "$best_sha" "$dig" "$RECEIPTS_URL" "$repo" "$best_sha")
    if ! GIT_COMMITTER_NAME=${AIEN_TAGGER_NAME:-AIEN Spark pipeline} GIT_COMMITTER_EMAIL=${AIEN_TAGGER_EMAIL:-aien-spark@users.noreply.github.com} \
      "$GIT" -C "$clone" tag -a "$name" "$best_sha" -m "$msg"; then
      journal action=alpha repo="$repo" sha="$best_sha" result=failed reason="git tag $name failed"; continue
    fi
    if "$GIT" -C "$clone" push -q origin "refs/tags/$name"; then
      journal action=alpha repo="$repo" sha="$best_sha" tag="$name" digest="$dig" result=tagged reason="full-scope PASS on main"
    else
      "$GIT" -C "$clone" tag -d "$name" >/dev/null
      journal action=alpha repo="$repo" sha="$best_sha" tag="$name" result=failed reason="tag push failed"
    fi
  done
}

case "$cmd" in
  publish) do_publish ;;
  merge) do_merge ;;
  alpha) do_alpha ;;
  round) do_publish; do_merge; do_alpha ;;
esac
exit 0
