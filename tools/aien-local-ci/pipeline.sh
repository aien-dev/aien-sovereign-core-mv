#!/usr/bin/env bash
# pipeline.sh: Spark side of the local verification pipeline, ADR 0033 phase 3 (lane PIPELINE).
#
#   pipeline.sh publish|rerun|merge|round [--repos a,b]      (round = publish, then rerun, then merge)
#
# publish  copy every signed bundle the worker wrote ($STATE/bundles/<repo>/<sha>/bundle.json) that verifies
#          (verify-bundle exit 0 PASS or 1 FAIL with a key from origin/main:.aien/verify.pub) byte-exact to the
#          receipts clone, one commit per round, pull --rebase and push (3 attempts, never forced).
# rerun    for each open pull request whose head has a published bundle, rerun the latest completed
#          "AIEN Verification" workflow run once per bundle digest when its latest attempt started before publication.
# merge    merge open pull requests labeled exactly ready-to-merge, only when every guard holds (see README).
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
WORKFLOW_NAME="AIEN Verification"
MERGE_LABEL="ready-to-merge"

usage() { echo "usage: pipeline.sh publish|rerun|merge|round [--repos a,b]" >&2; exit 2; }
die() { echo "pipeline.sh: $*" >&2; exit 2; }

[[ $# -ge 1 ]] || usage
cmd=$1; shift
case "$cmd" in publish|rerun|merge|round) ;; *) usage ;; esac
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
if [[ $cmd != rerun ]]; then [[ -x $BIN ]] || die "aien-test binary $BIN missing or not executable"; fi
if [[ $cmd != publish ]]; then command -v "$GH" >/dev/null || die "gh ($GH) not found"; fi

mkdir -p "$STATE"
JOURNAL=$STATE/pipeline.jsonl
RERUNS=$STATE/pipeline-reruns
RERUN_FAILS=$STATE/pipeline-rerun-fails
PUBLISHED=$STATE/pipeline-published
SLUG_CACHE=$STATE/pipeline-slugs
SEEN=$STATE/pipeline-seen
ETAGS=$STATE/pipeline-etag
exec 9>>"$STATE/pipeline.lock"
flock -n 9 || exit 0
WORK=$(mktemp -d "$STATE/.pipeline.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# journal key=value ...: one JSON line in the journal (built by jq), a short line on stdout.
journal() {
  local args=(--arg utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)") kv
  for kv in "$@"; do args+=(--arg "${kv%%=*}" "${kv#*=}"); done
  jq -nc "${args[@]}" '$ARGS.named' >> "$JOURNAL"
  echo "$*"
}

digest() { sha256sum "$1" | cut -d' ' -f1; }

# keys CLONE REF: the trusted public keys at REF, one per line (same rule as aien-test: '#' comments, 64 hex).
keys() {
  "$GIT" -C "$1" show "$2:.aien/verify.pub" 2>/dev/null | sed 's/#.*//' | tr -d ' \t\r' | tr 'A-F' 'a-f' \
    | grep -E '^[0-9a-f]{64}$' || true
}

# verify FILE REPO SHA KEYS [REQUIRE]: verify-bundle with every key; prints the best exit code (0, 1, 2, 3, other)
# and the list of codes. --repo and --commit come from the caller (the path), never from the bundle.
verify() {
  local file=$1 repo=$2 sha=$3 keylist=$4 require=${5:-} key rc codes="" best=""
  local extra=()
  [[ -n $require ]] && extra=(--require "$require")
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

# published REPO SHA: digest of the receipts copy at origin/<branch> (empty when absent). Caller fetches first.
published() {
  local tmp=$WORK/published
  if "$GIT" -C "$RECEIPTS" show "origin/$RECEIPTS_BRANCH:$1/$2/bundle.json" > "$tmp" 2>/dev/null; then digest "$tmp"
  else echo ""; fi
}

# publish_time REPO SHA DIGEST: epoch seconds this digest was published (our push record, else the receipts commit).
publish_time() {
  local t
  t=$(awk -v r="$1" -v s="$2" -v d="$3" '$1==r && $2==s && $3==d {t=$4} END {print t}' "$PUBLISHED" 2>/dev/null || true)
  [[ -n $t ]] || t=$("$GIT" -C "$RECEIPTS" log -1 --format=%ct "origin/$RECEIPTS_BRANCH" -- "$1/$2/bundle.json")
  echo "$t"
}

declare -A SLUGS=()
SLUG_TTL=${AIEN_SLUG_TTL:-86400}
# slug REPO: canonical owner/name on GitHub (aien-dev/<repo> may redirect). A repository is renamed or moved
# rarely, so the answer is kept in $STATE/pipeline-slugs ("repo slug epoch") for SLUG_TTL seconds instead of asking
# GitHub on every 5-minute tick (L337, G3-ci-judge-6).
slug() {
  local s t now
  if [[ -z ${SLUGS[$1]:-} ]]; then
    now=$(date -u +%s)
    read -r s t < <(awk -v r="$1" '$1 == r {s = $2; t = $3} END {print s, t}' "$SLUG_CACHE" 2>/dev/null || true) || true
    if [[ ${s:-} =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && ${t:-} =~ ^[0-9]+$ ]] && (( now - t < SLUG_TTL )); then
      SLUGS[$1]=$s
    else
      s=$("$GH" api "repos/$OWNER/$1" --jq .full_name 2>/dev/null) || return 1
      [[ $s =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || return 1
      SLUGS[$1]=$s
      { grep -v "^$1 " "$SLUG_CACHE" 2>/dev/null || true; echo "$1 $s $now"; } > "$SLUG_CACHE.$$" \
        && mv -f "$SLUG_CACHE.$$" "$SLUG_CACHE"
    fi
  fi
  echo "${SLUGS[$1]}"
}

# Bundles already judged unpublishable ("repo sha digest keyhash binstamp" per line in $SEEN). The verdict is a function
# of the bundle bytes, the trusted keys and the verify-bundle binary, so each of those is in the key: a new key, a rebuilt
# binary or a rewritten bundle is looked at again, anything else is not (L337, G3-ci-judge-4).
declare -A SEEN_SET=()
BSTAMP=$(stat -c '%s-%Y' "$BIN" 2>/dev/null || echo none)
seen() { echo "$1" >> "$SEEN"; SEEN_SET["$1"]=1; }
load_seen() {
  local line
  [[ -f $SEEN ]] || return 0
  if grep -qv " $BSTAMP\$" "$SEEN"; then
    grep " $BSTAMP\$" "$SEEN" > "$SEEN.$$" || true
    mv -f "$SEEN.$$" "$SEEN"
  fi
  while IFS= read -r line; do SEEN_SET["$line"]=1; done < "$SEEN"
}

# ghget PATH JQ: GET PATH and print JQ applied to the answer, like `gh api PATH --jq JQ`, but conditional: the answer
# and its ETag are kept under $STATE/pipeline-etag, and the next call sends If-None-Match. GitHub replies 304 when
# nothing changed, which costs no allowance, so a PR list that did not change in 5 minutes is free, and the rerun and
# merge steps of one round share one list (L337, G3-ci-judge-7). Non-zero when GitHub gives neither 200 nor 304.
ghget() {
  local path=$1 expr=$2 key f raw rc=0 head body status etag cond=()
  mkdir -p "$ETAGS"
  key=$(printf '%s' "$path" | sha256sum | cut -c1-32); f=$ETAGS/$key
  [[ -s $f.etag && -f $f.body ]] && cond=(-H "If-None-Match: $(cat "$f.etag")")
  raw=$("$GH" api -i "${cond[@]}" "$path" 2>/dev/null) || rc=$?
  head=${raw%%$'\r\n\r\n'*}; body=${raw#*$'\r\n\r\n'}
  status=$(printf '%s' "$head" | head -n 1 | awk '{print $2}')
  if [[ $status == 304 && ${#cond[@]} -gt 0 ]]; then
    body=$(cat "$f.body")
  elif [[ $status == 200 ]]; then
    etag=$(printf '%s\n' "$head" | tr -d '\r' | awk 'tolower($0) ~ /^etag:/ {sub(/^[^:]*: */, ""); print; exit}')
    if [[ -n $etag ]]; then
      printf '%s' "$body" > "$f.body.$$" && printf '%s' "$etag" > "$f.etag.$$" \
        && mv -f "$f.body.$$" "$f.body" && mv -f "$f.etag.$$" "$f.etag"
    else
      rm -f "$f.etag" "$f.body"
    fi
  else
    rm -f "$f.etag" "$f.body"; return 1
  fi
  jq -r "$expr" <<< "$body"
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
  load_seen
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
    # One sha256sum call per directory instead of a copy and a digest per bundle: most bundles are already
    # published or already judged unpublishable, and a round must not re-check what cannot change (L337, G3-ci-judge-4).
    khash=$(printf '%s' "$keylist" | sha256sum | cut -d' ' -f1)
    local lfiles=("$STATE/bundles/$repo"/*/bundle.json) rfiles=("$RECEIPTS/$repo"/*/bundle.json) d p
    local -A LDIG=() RDIG=()
    [[ -e ${lfiles[0]} ]] || continue
    while read -r d p; do p=${p%/bundle.json}; LDIG[${p##*/}]=$d; done < <(sha256sum -- "${lfiles[@]}" 2>/dev/null || true)
    if [[ -e ${rfiles[0]} ]]; then
      while read -r d p; do p=${p%/bundle.json}; RDIG[${p##*/}]=$d; done < <(sha256sum -- "${rfiles[@]}" 2>/dev/null || true)
    fi
    for f in "${lfiles[@]}"; do
      [[ -f $f ]] || continue
      sha=${f%/bundle.json}; sha=${sha##*/}
      [[ $sha =~ ^[0-9a-f]{40}$ ]] || continue
      ldig=${LDIG[$sha]:-}
      [[ -n $ldig && ${RDIG[$sha]:-} == "$ldig" ]] && continue
      [[ -n $ldig && -n ${SEEN_SET["$repo $sha $ldig $khash $BSTAMP"]:-} ]] && continue
      tmp=$WORK/candidate.json
      cp -- "$f" "$tmp"
      ldig=$(digest "$tmp")
      dest=$RECEIPTS/$repo/$sha/bundle.json
      old=""
      if [[ -f $dest ]]; then old=$(digest "$dest"); [[ $old == "$ldig" ]] && continue; fi
      if ! jq -e '.plan | has("blocked")' "$tmp" >/dev/null 2>&1; then
        journal action=publish repo="$repo" sha="$sha" result=refused reason="bundle unreadable or has no plan.blocked"
        seen "$repo $sha $ldig $khash $BSTAMP"; continue
      fi
      if ! jq -e '.plan.blocked == null' "$tmp" >/dev/null; then
        journal action=publish repo="$repo" sha="$sha" result=skipped reason="plan.blocked is not null"
        seen "$repo $sha $ldig $khash $BSTAMP"; continue
      fi
      v=$(verify "$tmp" "$repo" "$sha" "$keylist"); best=${v%% *}; codes=${v#* }
      if [[ $best != 0 && $best != 1 ]]; then
        journal action=publish repo="$repo" sha="$sha" result=refused \
          reason="verify-bundle $(verdict_name "$best") (exit codes per key: $codes)"
        # NOT_RUN and REJECTED are verdicts on these bytes with these keys and this binary; any other exit
        # code (a crash, a missing binary) may pass next time, so it is looked at again.
        [[ $best == 2 || $best == 3 ]] && seen "$repo $sha $ldig $khash $BSTAMP"
        continue
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
  local rec now
  if [[ $pushed == 1 ]]; then
    now=$(date +%s)
    for rec in "${records[@]}"; do
      read -r repo sha ldig v <<< "$rec"
      echo "$repo $sha $ldig $now" >> "$PUBLISHED"
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

do_rerun() {
  local repo s prs num sha pdig run id status created cepoch ptime
  if ! fetch_receipts; then journal action=rerun result=refused reason="receipts fetch failed"; return 0; fi
  for repo in "${repo_list[@]}"; do
    [[ -d $CLONES/$repo ]] || continue
    if ! s=$(slug "$repo"); then journal action=rerun repo="$repo" result=refused reason="cannot resolve GitHub slug"; continue; fi
    if ! prs=$(ghget "repos/$s/pulls?state=open&per_page=100" '.[] | "\(.number) \(.head.sha)"'); then
      journal action=rerun repo="$repo" result=refused reason="cannot list pull requests"; continue
    fi
    while read -r num sha; do
      [[ $sha =~ ^[0-9a-f]{40}$ ]] || continue
      pdig=$(published "$repo" "$sha")
      [[ -n $pdig ]] || continue
      grep -qxF "$repo $sha $pdig" "$RERUNS" 2>/dev/null && continue
      # A refused rerun is not retried every tick: each refusal trips the gh guard and stalls every
      # chat on this machine. Back off one hour after a failure, give up after three.
      if [[ -f $RERUN_FAILS ]]; then
        local nfail lastfail
        nfail=$(grep -c "^$repo $sha $pdig " "$RERUN_FAILS" || true)
        lastfail=$(grep "^$repo $sha $pdig " "$RERUN_FAILS" | tail -1 | cut -d" " -f4 || true)
        (( nfail >= 3 )) && continue
        [[ -n $lastfail ]] && (( $(date -u +%s) - lastfail < 3600 )) && continue
      fi
      if ! run=$("$GH" api "repos/$s/actions/runs?head_sha=$sha&per_page=20" --jq \
        "[.workflow_runs[] | select(.name == \"$WORKFLOW_NAME\")] | sort_by(.created_at) | last
         | if . == null then \"\" else \"\(.id) \(.status) \(.run_started_at // .created_at)\" end"); then
        journal action=rerun repo="$repo" pr="$num" sha="$sha" result=refused reason="cannot list workflow runs"; continue
      fi
      if [[ -z $run ]]; then
        journal action=rerun repo="$repo" pr="$num" sha="$sha" result=wait reason="no $WORKFLOW_NAME run for this head yet"; continue
      fi
      read -r id status created <<< "$run"
      if [[ $status != completed ]]; then
        journal action=rerun repo="$repo" pr="$num" sha="$sha" result=wait reason="run $id is $status"; continue
      fi
      if ! cepoch=$(date -u -d "$created" +%s 2>/dev/null); then
        journal action=rerun repo="$repo" pr="$num" sha="$sha" result=refused reason="unreadable run time '$created'"; continue
      fi
      ptime=$(publish_time "$repo" "$sha" "$pdig")
      [[ -n $ptime && $cepoch -lt $ptime ]] || continue
      if "$GH" api -X POST "repos/$s/actions/runs/$id/rerun" >/dev/null; then
        echo "$repo $sha $pdig" >> "$RERUNS"
        journal action=rerun repo="$repo" pr="$num" sha="$sha" digest="$pdig" result=rerun reason="run $id predates publication"
      else
        echo "$repo $sha $pdig $(date -u +%s)" >> "$RERUN_FAILS"
        journal action=rerun repo="$repo" pr="$num" sha="$sha" result=failed reason="rerun request for run $id failed (backing off 1h)"
      fi
    done <<< "$prs"
  done
}

# merge_one REPO SLUG NUM SHA BASE: all guards, then the merge. Prints nothing; journals the outcome.
merge_one() {
  local repo=$1 s=$2 num=$3 sha=$4 base=$5 clone=$CLONES/$1 pr state head lfile tmp ldig pdig keylist require v best
  local ctxs checks statuses ctx found bad active
  merge_refuse() { journal action=merge repo="$repo" pr="$num" sha="$sha" result=refused reason="$1"; }
  # Up to date with the base. GitHub does not require it here (strict=false), and two PRs merged
  # back to back each pass alone but leave main with a stale root crumb (Drake 2026-10-18: "don't
  # make them all collapse"). This is a local check, so it runs before the first GitHub call: a PR that is
  # behind costs no request (L337, G3-ci-judge-5).
  if ! "$GIT" -C "$clone" cat-file -e "$sha^{commit}" 2>/dev/null; then
    "$GIT" -C "$clone" fetch -q origin "pull/$num/head" 2>/dev/null || true
  fi
  if ! "$GIT" -C "$clone" merge-base --is-ancestor "origin/$base" "$sha" 2>/dev/null; then
    merge_refuse "behind origin/$base: merge the base in, recompile crumbs, let checks rerun"; return 0
  fi
  if ! pr=$("$GH" api "repos/$s/pulls/$num" --jq '"\(.mergeable_state // "unknown") \(.head.sha) \(.draft)"'); then
    merge_refuse "cannot read pull request"; return 0
  fi
  read -r state head _ <<< "$pr"
  [[ $head == "$sha" ]] || { merge_refuse "head moved to $head"; return 0; }
  [[ ${pr##* } == false ]] || { merge_refuse "draft"; return 0; }
  [[ $state != dirty ]] || { merge_refuse "mergeable state dirty (conflicts)"; return 0; }
  lfile=$STATE/bundles/$repo/$sha/bundle.json
  [[ -f $lfile ]] || { merge_refuse "no local bundle for head"; return 0; }
  tmp=$WORK/merge.json
  cp -- "$lfile" "$tmp"
  ldig=$(digest "$tmp")
  pdig=$(published "$repo" "$sha")
  [[ -n $pdig ]] || { merge_refuse "bundle not published"; return 0; }
  [[ $pdig == "$ldig" ]] || { merge_refuse "digest mismatch: local sha256:$ldig, published sha256:$pdig"; return 0; }
  keylist=$(keys "$clone" "origin/$base")
  [[ -n $keylist ]] || { merge_refuse "no key in origin/$base:.aien/verify.pub"; return 0; }
  require=$("$GIT" -C "$clone" show "origin/$base:.aien/always" 2>/dev/null | grep -v '^#' | grep -v '^$' | paste -sd, - || true)
  v=$(verify "$tmp" "$repo" "$sha" "$keylist" "$require"); best=${v%% *}
  [[ $best == 0 ]] || { merge_refuse "verify-bundle $(verdict_name "$best") (exit codes per key: ${v#* })"; return 0; }
  if ! ctxs=$("$GH" api "repos/$s/branches/$base/protection/required_status_checks" --jq '.contexts[]'); then
    merge_refuse "cannot read required status checks of $base"; return 0
  fi
  [[ -n $ctxs ]] || { merge_refuse "$base has no required status checks"; return 0; }
  if ! checks=$("$GH" api "repos/$s/commits/$sha/check-runs?per_page=100" --jq '.check_runs[] | [.name, (.conclusion // .status)] | @tsv'); then
    merge_refuse "cannot read check runs"; return 0
  fi
  if ! statuses=$("$GH" api "repos/$s/commits/$sha/status?per_page=100" --jq '.statuses[] | [.context, .state] | @tsv'); then
    merge_refuse "cannot read commit statuses"; return 0
  fi
  while IFS= read -r ctx; do
    [[ -n $ctx ]] || continue
    found=$(printf '%s\n%s\n' "$checks" "$statuses" | C="$ctx" awk -F'\t' '$1 == ENVIRON["C"] {print $2}')
    [[ -n $found ]] || { merge_refuse "required context '$ctx' missing"; return 0; }
    bad=$(grep -vx success <<< "$found" | head -n 1 || true)
    [[ -z $bad ]] || { merge_refuse "required context '$ctx' is $bad"; return 0; }
  done <<< "$ctxs"
  if ! active=$("$GH" api "repos/$s/actions/runs?head_sha=$sha&per_page=20" --jq \
    "[.workflow_runs[] | select(.name == \"$WORKFLOW_NAME\" and .status != \"completed\")] | length"); then
    merge_refuse "cannot list workflow runs"; return 0
  fi
  [[ $active == 0 ]] || { merge_refuse "$WORKFLOW_NAME run queued or in progress"; return 0; }
  if "$GH" pr merge "$num" -R "$s" --merge --match-head-commit "$sha"; then
    journal action=merge repo="$repo" pr="$num" sha="$sha" digest="$ldig" result=merged reason="all guards green"
    MERGED_REPO=$repo
  else
    journal action=merge repo="$repo" pr="$num" sha="$sha" result=failed reason="gh pr merge failed"
  fi
}

do_merge() {
  local repo s prs num sha base draft labeled
  MERGED_REPO=""
  # Drake 2026-10-18: "stop all agents from trying to merge". Merge-only stop; publish and rerun keep going.
  if [[ -e $STATE/MERGE-PAUSE ]]; then journal action=merge result=paused reason="MERGE-PAUSE present"; return 0; fi
  if ! fetch_receipts; then journal action=merge result=refused reason="receipts fetch failed"; return 0; fi
  for repo in "${repo_list[@]}"; do
    [[ -d $CLONES/$repo ]] || continue
    if ! s=$(slug "$repo"); then journal action=merge repo="$repo" result=refused reason="cannot resolve GitHub slug"; continue; fi
    if ! "$GIT" -C "$CLONES/$repo" fetch -q origin; then
      journal action=merge repo="$repo" result=refused reason="fetch of trusted clone failed"; continue
    fi
    if ! prs=$(ghget "repos/$s/pulls?state=open&per_page=100" \
      ".[] | [.number, .head.sha, .base.ref, .draft, ([.labels[].name] | index(\"$MERGE_LABEL\") != null)] | @tsv"); then
      journal action=merge repo="$repo" result=refused reason="cannot list pull requests"; continue
    fi
    while IFS=$'\t' read -r num sha base draft labeled; do
      [[ -n $num ]] || continue
      # One merge per repository per round: the next one must first catch up with the new base.
      [[ $MERGED_REPO != "$repo" ]] || break
      if [[ ! $sha =~ ^[0-9a-f]{40}$ ]]; then
        journal action=merge repo="$repo" pr="$num" result=refused reason="bad head sha"; continue
      fi
      if [[ ! $base =~ ^[A-Za-z0-9._/-]+$ || $base == *..* ]]; then
        journal action=merge repo="$repo" pr="$num" sha="$sha" result=refused reason="bad base ref"; continue
      fi
      # Unlabeled pull requests are not the pipeline's to merge: skipped without a journal line (every round).
      [[ $labeled == true ]] || continue
      if [[ $draft != false ]]; then
        journal action=merge repo="$repo" pr="$num" sha="$sha" result=refused reason="draft"; continue
      fi
      merge_one "$repo" "$s" "$num" "$sha" "$base"
    done <<< "$prs"
  done
}

case "$cmd" in
  publish) do_publish ;;
  rerun) do_rerun ;;
  merge) do_merge ;;
  round) do_publish; do_rerun; do_merge ;;
esac
exit 0
