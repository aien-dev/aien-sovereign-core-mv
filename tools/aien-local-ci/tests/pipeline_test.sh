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

# pipeline_test.sh: offline proofs for tools/aien-local-ci/pipeline.sh (no network, no real gh, no real key).
# Fakes: gh (answers from JSON files, applies the real --jq filter with jq), aien-test (exit code chosen by the
# test bundle's "fake" member, REJECTED for a wrong key, repo or commit). Throwaway git repositories stand in for
# the trusted clone, a bare repository for the receipts remote.
#   bash tools/aien-local-ci/tests/pipeline_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIPE=$HERE/../pipeline.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

export GIT_CONFIG_GLOBAL=$T/gitconfig GIT_CONFIG_NOSYSTEM=1
: > "$GIT_CONFIG_GLOBAL"
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.invalid GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.invalid
export FAKE_LOG=$T/calls.log FAKE_GH=$T/gh
export AIEN_STATE=$T/state AIEN_CLONES=$T/clones AIEN_RECEIPTS=$T/receipts AIEN_TEST_BIN=$T/bin/aien-test
export GH=$T/bin/gh AIEN_REPOS=demo,plain
mkdir -p "$T/bin" "$FAKE_GH" "$AIEN_STATE" "$AIEN_CLONES"
: > "$FAKE_LOG"

cat > "$T/bin/gh" <<'EOF'
#!/usr/bin/env bash
echo "gh $*" >> "$FAKE_LOG"
[[ -z ${FAKE_GH98:-} ]] || exit 98
if [[ $1 == pr && $2 == merge ]]; then exit 0; fi
[[ $1 == api ]] || exit 9
shift
method=GET expr="" path=""
while [[ $# -gt 0 ]]; do
  case $1 in -X) method=$2; shift 2 ;; --jq) expr=$2; shift 2 ;; *) path=$1; shift ;; esac
done
[[ $method == GET ]] || exit 0
f=$FAKE_GH/$(printf '%s' "$path" | tr -c 'A-Za-z0-9._-' '_').json
[[ -f $f ]] || { echo "gh: HTTP 404 $path" >&2; exit 1; }
if [[ -n $expr ]]; then jq -r "$expr" "$f"; else cat "$f"; fi
EOF
cat > "$T/bin/aien-test" <<'EOF'
#!/usr/bin/env bash
echo "aien-test $*" >> "$FAKE_LOG"
[[ $1 == verify-bundle ]] || exit 9
file=$2; shift 2
full=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo) repo=$2; shift 2 ;; --commit) commit=$2; shift 2 ;; --pubkey) key=$2; shift 2 ;; --require) shift 2 ;;
    --require-full) full=1; shift ;;
    *) exit 2 ;;
  esac
done
[[ $(jq -r .repository "$file") == "$repo" && $(jq -r .commit "$file") == "$commit" ]] || exit 3
[[ $(jq -r .fake.key "$file") == "$key" ]] || exit 3
[[ -z $full || $(jq -r '.plan.scope // "changed"' "$file") == full ]] || exit 3
exit "$(jq -r .fake.rc "$file")"
EOF
chmod +x "$T/bin/gh" "$T/bin/aien-test"

ghset() { printf '%s\n' "$2" > "$FAKE_GH/$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_').json"; }
sha() { printf '%040x' "$1"; }
K1=$(printf '1%.0s' {1..64}) K2=$(printf '2%.0s' {1..64}) K3=$(printf '3%.0s' {1..64})
S=aien-spark/demo

# Trusted clone of "demo": verify.pub holds two keys (old K1 first, then K2) and a comment; .aien/always two gates.
git init -q --bare -b main "$T/origin-demo.git"
git clone -q "$T/origin-demo.git" "$AIEN_CLONES/demo" 2>/dev/null
mkdir -p "$AIEN_CLONES/demo/.aien"
printf '# trusted keys, oldest first\n%s\n%s  # current\n' "$K1" "$K2" > "$AIEN_CLONES/demo/.aien/verify.pub"
printf '# always\ng1\n\ng2\n' > "$AIEN_CLONES/demo/.aien/always"
git -C "$AIEN_CLONES/demo" add -A && git -C "$AIEN_CLONES/demo" commit -q -m init && git -C "$AIEN_CLONES/demo" push -q origin main
# "plain": a clone with no verify.pub (not enrolled).
git init -q --bare -b main "$T/origin-plain.git"
git clone -q "$T/origin-plain.git" "$AIEN_CLONES/plain" 2>/dev/null
echo x > "$AIEN_CLONES/plain/README" && git -C "$AIEN_CLONES/plain" add -A && git -C "$AIEN_CLONES/plain" commit -q -m init && git -C "$AIEN_CLONES/plain" push -q origin main
# Receipts remote and the pipeline's clone of it, plus a second clone used to race a push.
git init -q --bare -b main "$T/receipts.git"
git clone -q "$T/receipts.git" "$T/seed" 2>/dev/null
echo shelf > "$T/seed/README.md" && git -C "$T/seed" add -A && git -C "$T/seed" commit -q -m shelf && git -C "$T/seed" push -q origin main
git clone -q "$T/receipts.git" "$AIEN_RECEIPTS"

# bundle REPO SHA KEY RC [BLOCKED] [COMMIT_IN_BODY] [SCOPE]   (no SCOPE: the key is absent, as in old bundles)
bundle() {
  local d=$AIEN_STATE/bundles/$1/$2
  mkdir -p "$d"
  jq -n --arg repo "github.com/aien-dev/$1" --arg c "${6:-$2}" --arg k "$3" --argjson rc "$4" --argjson b "${5:-null}" \
    --arg sc "${7:-}" \
    '{repository:$repo, commit:$c, plan:({blocked:$b, verdict:"PASS"} + (if $sc == "" then {} else {scope:$sc} end)),
      signature:{pubkey:$k}, fake:{key:$k, rc:$rc}}' > "$d/bundle.json"
}
remote_has() { git --git-dir="$T/receipts.git" cat-file -e "main:$1/$2/bundle.json" 2>/dev/null; }
remote_commits() { git --git-dir="$T/receipts.git" rev-list --count main; }
jl() { jq -c "select($1)" "$AIEN_STATE/pipeline.jsonl" | wc -l; }

A=$(sha 1) B=$(sha 2) C=$(sha 3) D=$(sha 4) E=$(sha 5) F=$(sha 6) W=$(sha 7) P=$(sha 8)
bundle demo "$A" "$K2" 0            # PASS, current key
bundle demo "$B" "$K2" 1            # FAIL, still published
bundle demo "$C" "$K3" 0            # key not in verify.pub: REJECTED (3) with both keys
bundle demo "$D" "$K2" 2            # NOT_RUN
bundle demo "$E" "$K2" 0 '"gpu"'    # plan.blocked set
bundle demo "$F" "$K1" 0            # older key (first line of verify.pub)
bundle demo "$W" "$K2" 0 null "$A"  # body names another commit: verify uses the path, so REJECTED
bundle plain "$P" "$K2" 0           # repository not enrolled

echo "== usage and configuration"
rc=0; bash "$PIPE" bogus >/dev/null 2>&1 || rc=$?; check "unknown command exits 2" [ "$rc" = 2 ]
rc=0; bash "$PIPE" publish --repos 'bad/name' >/dev/null 2>&1 || rc=$?; check "bad --repos exits 2" [ "$rc" = 2 ]
rc=0; AIEN_RECEIPTS=$T/nowhere bash "$PIPE" publish >/dev/null 2>&1 || rc=$?; check "missing receipts clone exits 2" [ "$rc" = 2 ]

echo "== publish"
before=$(remote_commits)
bash "$PIPE" publish > "$T/out1" 2>&1
check "one commit for the round" [ "$(remote_commits)" = $((before + 1)) ]
check "PASS bundle published" remote_has demo "$A"
check "FAIL bundle published" remote_has demo "$B"
check "multi-key verify.pub: bundle under the older key published" remote_has demo "$F"
check "PASS copy is byte-exact" cmp -s <(git --git-dir="$T/receipts.git" show "main:demo/$A/bundle.json") "$AIEN_STATE/bundles/demo/$A/bundle.json"
check "REJECTED (3) not published" bash -c "! git --git-dir='$T/receipts.git' cat-file -e 'main:demo/$C/bundle.json' 2>/dev/null"
check "NOT_RUN (2) not published" bash -c "! git --git-dir='$T/receipts.git' cat-file -e 'main:demo/$D/bundle.json' 2>/dev/null"
check "plan.blocked bundle not published" bash -c "! git --git-dir='$T/receipts.git' cat-file -e 'main:demo/$E/bundle.json' 2>/dev/null"
check "commit taken from the path, not the body (REJECTED)" bash -c "! git --git-dir='$T/receipts.git' cat-file -e 'main:demo/$W/bundle.json' 2>/dev/null"
check "unenrolled repository not published" bash -c "! git --git-dir='$T/receipts.git' cat-file -e 'main:plain/$P/bundle.json' 2>/dev/null"
check "journal: REJECTED refusal" [ "$(jl ".sha==\"$C\" and .result==\"refused\" and (.reason|test(\"REJECTED\"))")" = 1 ]
check "journal: NOT_RUN refusal" [ "$(jl ".sha==\"$D\" and .result==\"refused\" and (.reason|test(\"NOT_RUN\"))")" = 1 ]
check "journal: blocked skip" [ "$(jl ".sha==\"$E\" and .result==\"skipped\" and (.reason|test(\"blocked\"))")" = 1 ]
check "journal: not enrolled" [ "$(jl ".repo==\"plain\" and (.reason|test(\"not enrolled\"))")" = 1 ]
check "journal: published lines carry digest" [ "$(jl ".sha==\"$A\" and .result==\"published\" and (.digest|length)==64")" = 1 ]
check "commit message lists the shas" bash -c "git --git-dir='$T/receipts.git' log -1 --format=%B main | grep -q 'demo/$A PASS sha256:'"
check "verify-bundle got --repo from the path" grep -q -- "--repo github.com/aien-dev/demo --commit $A" "$FAKE_LOG"
check "publish never passes --require" bash -c "! grep -q 'aien-test.*--require' '$FAKE_LOG'"
check "publish made no GitHub call" bash -c "! grep -q '^gh ' '$FAKE_LOG'"

before=$(remote_commits)
bash "$PIPE" publish > "$T/out2" 2>&1
check "identical bundles are not re-committed" [ "$(remote_commits)" = "$before" ]

echo "== publish: replacement by digest, with a racing push"
digA0=$(sha256sum "$AIEN_STATE/bundles/demo/$A/bundle.json" | cut -d' ' -f1)
jq '.extra = 1' "$AIEN_STATE/bundles/demo/$A/bundle.json" > "$T/a" && mv "$T/a" "$AIEN_STATE/bundles/demo/$A/bundle.json"
digA1=$(sha256sum "$AIEN_STATE/bundles/demo/$A/bundle.json" | cut -d' ' -f1)
cat > "$T/bin/racegit" <<EOF
#!/usr/bin/env bash
if [[ " \$* " == *" push "* && ! -e "$T/raced" ]]; then
  touch "$T/raced"
  git -C "$T/seed" pull -q --rebase origin main && echo other >> "$T/seed/OTHER" && git -C "$T/seed" add -A \
    && git -C "$T/seed" commit -q -m other && git -C "$T/seed" push -q origin main
fi
exec git "\$@"
EOF
chmod +x "$T/bin/racegit"
before=$(remote_commits)
GIT=$T/bin/racegit bash "$PIPE" publish > "$T/out3" 2>&1
check "race happened (someone else pushed first)" [ -e "$T/raced" ]
check "replaced bundle pushed after pull --rebase" [ "$(git --git-dir="$T/receipts.git" show "main:demo/$A/bundle.json" | sha256sum | cut -d' ' -f1)" = "$digA1" ]
check "the other publisher's commit kept (no force push)" git --git-dir="$T/receipts.git" cat-file -e main:OTHER
check "two new commits: theirs and ours" [ "$(remote_commits)" = $((before + 2)) ]
check "commit message names the replaced digest" bash -c "git --git-dir='$T/receipts.git' log -1 --format=%B main | grep -q 'replaces sha256:$digA0'"

echo "== merge: no Spark gate"
ghset "repos/aien-dev/demo" '{"full_name":"aien-spark/demo"}'
ghset "repos/aien-dev/plain" '{"full_name":"aien-spark/plain"}'
ghset "repos/aien-spark/plain/pulls?state=open&per_page=100" '[]'
pulls() { # NUM:SHA:DRAFT:LABEL ...
  local out="[" sep="" n s d l
  for spec in "$@"; do
    IFS=: read -r n s d l <<< "$spec"
    out+="$sep{\"number\":$n,\"draft\":$d,\"head\":{\"sha\":\"$s\"},\"base\":{\"ref\":\"main\"},\"labels\":[{\"name\":\"other\"}${l:+,{\"name\":\"$l\"\}}]}"
    sep=,
  done
  ghset "repos/$S/pulls?state=open&per_page=100" "$out]"
}
pr() { ghset "repos/$S/pulls/$1" "{\"number\":$1,\"draft\":${4:-false},\"mergeable_state\":\"$3\",\"head\":{\"sha\":\"$2\"}}"; }
L=ready-to-merge
for i in 41 42 43 44 45 46 47 48 49 50 51 52; do printf -v "N$i" '%s' "$(sha "$i")"; done
# None of these heads has a bundle, a status or a check run.
pulls "41:$N41:false:$L" "42:$N42:false:" "43:$N43:true:$L" "44:$N44:false:$L" "45:$N45:false:$L" \
      "46:$N46:false:$L" "47:$N47:false:$L" "48:$N48:false:$L" "49:$N49:false:ready-to-merge-later" "50:$N50:false:$L" \
      "51:$N51:false:$L" "52:$N52:false:$L"
pr 41 "$N41" clean; pr 44 "$N44" dirty; pr 45 "$N45" behind; pr 46 "$N46" unstable
pr 47 "$(sha 470)" clean; pr 48 "$N48" unknown; pr 50 "$N50" blocked; pr 51 "$N51" unstable; pr 52 "$N52" unstable
# "unstable" = a check failed or is running and no branch rule requires it: the head's check runs decide.
runs() { ghset "repos/$S/commits/$1/check-runs?per_page=100" "{\"check_runs\":[$2]}"; }
runs "$N46" '{"name":"AIEN Checks","status":"completed","conclusion":"failure"},{"name":"CodeQL","status":"completed","conclusion":"success"}'
runs "$N51" '{"name":"AIEN Checks","status":"in_progress","conclusion":null},{"name":"CodeQL","status":"completed","conclusion":"success"}'
runs "$N52" '{"name":"AIEN Checks","status":"completed","conclusion":"success"},{"name":"AIEN Alpha","status":"completed","conclusion":"skipped"},{"name":"CodeQL","status":"completed","conclusion":"neutral"}'
c0=$(wc -l < "$FAKE_LOG")
bash "$PIPE" merge > "$T/out7" 2>&1
merges() { grep -c "^gh pr merge $1 " "$FAKE_LOG" || true; }
refused() { [ "$(jl ".action==\"merge\" and .pr==\"$1\" and .result==\"refused\" and (.reason|test(\"$2\"))")" = 1 ]; }
check "clean, no bundle: merged with --match-head-commit on the canonical slug" grep -qx "gh pr merge 41 -R $S --merge --match-head-commit $N41" "$FAKE_LOG"
check "behind the base: merged (no up-to-date rule)" [ "$(merges 45)" = 1 ]
check "unstable with a failed hosted check: refused, naming the check" refused 46 "hosted checks failed on this head: AIEN Checks"
check "unstable with a check still running: wait, not merged" [ "$(jl ".action==\"merge\" and .pr==\"51\" and .result==\"wait\"")$(merges 51)" = 10 ]
check "unstable with every check completed and none failed (skipped/neutral ok): merged" [ "$(merges 52)" = 1 ]
check "every mergeable PR merges in the same round" [ "$(grep -c '^gh pr merge' "$FAKE_LOG")" = 3 ]
check "merge never calls aien-test" bash -c "! tail -n +$((c0 + 1)) '$FAKE_LOG' | grep -q '^aien-test'"
check "merge reads check runs only for unstable heads, never statuses, protection or workflow runs" \
  bash -c "! tail -n +$((c0 + 1)) '$FAKE_LOG' | grep -qE '/status|protection|actions/runs' && [ \"\$(tail -n +$((c0 + 1)) '$FAKE_LOG' | grep -c check-runs)\" = 3 ]"
check "never --admin" bash -c "! grep -q -- '--admin' '$FAKE_LOG'"
check "label missing: skipped, no journal line" [ "$(jl ".action==\"merge\" and .pr==\"42\"")" = 0 ]
check "label not exact (ready-to-merge-later): skipped, no journal line" [ "$(jl ".action==\"merge\" and .pr==\"49\"")" = 0 ]
check "draft: refused" refused 43 "draft"
check "conflicts: refused" refused 44 "conflicts"
check "head moved: refused" refused 47 "head moved"
check "mergeability unknown: wait, not merged" [ "$(jl ".action==\"merge\" and .pr==\"48\" and .result==\"wait\"")$(merges 48)" = 10 ]
check "blocked by branch rules: refused" refused 50 "mergeable state blocked"

echo "== alpha"
DAY=$(date -u +%Y%m%d)
main_commit() { # NAME DATE: a commit on demo's main with that committer date, pushed; prints its sha
  echo "$1" > "$AIEN_CLONES/demo/$1" && git -C "$AIEN_CLONES/demo" add "$1" \
    && GIT_COMMITTER_DATE=$2 git -C "$AIEN_CLONES/demo" commit -q -m "$1" && git -C "$AIEN_CLONES/demo" push -q origin main \
    && git -C "$AIEN_CLONES/demo" rev-parse HEAD
}
origin_tag() { git --git-dir="$T/origin-demo.git" rev-parse -q --verify "refs/tags/$1^{commit}" 2>/dev/null; }
alpha_tags() { git --git-dir="$T/origin-demo.git" tag -l 'alpha/*' | wc -l; }
X1=$(main_commit x1 2026-01-01T00:00:00Z) X2=$(main_commit x2 2026-01-02T00:00:00Z)
X3=$(main_commit x3 2026-01-03T00:00:00Z) X4=$(main_commit x4 2026-01-04T00:00:00Z)
XB=$(git -C "$AIEN_CLONES/demo" commit-tree -p "$X4" -m "branch only" "$X4^{tree}")
bundle demo "$X1" "$K2" 0 null "" full   # full PASS, older
bundle demo "$X2" "$K2" 0 null "" full   # full PASS: the expected Alpha
bundle demo "$X3" "$K2" 1 null "" full   # full FAIL, newer
bundle demo "$X4" "$K2" 0                # newest on main, but a changed-scope (merge-era) PASS
bundle demo "$XB" "$K2" 0 null "" full   # full PASS, not on main
bash "$PIPE" publish > "$T/out8" 2>&1
c0=$(wc -l < "$FAKE_LOG")
bash "$PIPE" alpha > "$T/out8a" 2>&1
check "newest full-scope PASS on main is tagged alpha/$DAY.1" [ "$(origin_tag "alpha/$DAY.1")" = "$X2" ]
check "exactly one alpha tag" [ "$(alpha_tags)" = 1 ]
check "the tag is annotated" [ "$(git --git-dir="$T/origin-demo.git" cat-file -t "refs/tags/alpha/$DAY.1")" = tag ]
check "tag message names the bundle digest" bash -c "git --git-dir='$T/origin-demo.git' cat-file -p 'refs/tags/alpha/$DAY.1' \
  | grep -qx 'bundle: sha256:$(sha256sum "$AIEN_STATE/bundles/demo/$X2/bundle.json" | cut -d' ' -f1)'"
check "alpha verifies with --require-full and the base .aien/always" \
  bash -c "tail -n +$((c0 + 1)) '$FAKE_LOG' | grep -q -- '--commit $X2 --pubkey $K2 --require g1,g2 --require-full'"
check "full FAIL is skipped with its verdict" [ "$(jl ".action==\"alpha\" and .sha==\"$X3\" and (.reason|test(\"FAIL\"))")" = 1 ]
check "changed-scope bundle is never a candidate" bash -c "! tail -n +$((c0 + 1)) '$FAKE_LOG' | grep -q -- '--commit $X4 '"
check "a head not on main is never tagged" [ -z "$(git --git-dir="$T/origin-demo.git" tag -l 'alpha/*' --points-at "$XB")" ]
check "alpha never moves main" [ "$(git --git-dir="$T/origin-demo.git" rev-parse main)" = "$X4" ]
check "journal: tagged line carries tag and digest" [ "$(jl ".action==\"alpha\" and .result==\"tagged\" and .tag==\"alpha/$DAY.1\" and (.digest|length)==64")" = 1 ]
bash "$PIPE" alpha > "$T/out8b" 2>&1
check "second round: no second tag for the same Alpha" [ "$(alpha_tags)" = 1 ]
X5=$(main_commit x5 2026-01-05T00:00:00Z)
bundle demo "$X5" "$K2" 0 null "" full
bash "$PIPE" publish > "$T/out8c" 2>&1
bash "$PIPE" alpha > "$T/out8d" 2>&1
check "a newer full PASS becomes alpha/$DAY.2" [ "$(origin_tag "alpha/$DAY.2")" = "$X5" ]
check "the older full PASS (x1) is never tagged" [ "$(alpha_tags)" = 2 ]

echo "== single instance"
bundle demo "$(sha 99)" "$K2" 0
exec 8>>"$AIEN_STATE/pipeline.lock"
flock 8
before=$(remote_commits); lines=$(wc -l < "$AIEN_STATE/pipeline.jsonl")
rc=0; bash "$PIPE" round > "$T/out9" 2>&1 || rc=$?
flock -u 8; exec 8>&-
check "second instance exits 0" [ "$rc" = 0 ]
check "second instance did nothing" [ "$(remote_commits) $(wc -l < "$AIEN_STATE/pipeline.jsonl")" = "$before $lines" ]
bash "$PIPE" publish > "$T/out10" 2>&1
check "after the lock is released the bundle publishes" remote_has demo "$(sha 99)"

echo "== slug cache"
check "resolved slug cached on disk" [ "$(cat "$AIEN_STATE/slugs/demo" 2>/dev/null)" = "$S" ]
check "a refresh is journaled" [ "$(jl '.action == "slug" and .repo == "demo" and .result == "refreshed"')" -ge 1 ]
rm -f "$FAKE_GH/repos_aien-dev_demo.json"
calls=$(grep -c "gh api repos/aien-dev/demo " "$FAKE_LOG" || true)
refused() { jl '.repo == "demo" and .reason == "cannot resolve GitHub slug"'; }
r0=$(refused)
bash "$PIPE" merge > "$T/out11" 2>&1
check "fresh cache used without an API call" [ "$(grep -c "gh api repos/aien-dev/demo " "$FAKE_LOG" || true)" = "$calls" ]
check "cache use is journaled" [ "$(jl '.action == "slug" and .repo == "demo" and .result == "cached"')" -ge 1 ]
touch -d '2 days ago' "$AIEN_STATE/slugs/demo"
bash "$PIPE" merge > "$T/out12" 2>&1
check "old cache tried a refresh" [ "$(grep -c "gh api repos/aien-dev/demo " "$FAKE_LOG" || true)" -gt "$calls" ]
check "refresh failed: old cache used, journaled stale-cache" [ "$(jl '.action == "slug" and .repo == "demo" and .result == "stale-cache"')" -ge 1 ]
check "no slug refusal while a cached name exists" [ "$(refused)" = "$r0" ]
rm -f "$AIEN_STATE/slugs/demo"
bash "$PIPE" merge > "$T/out13" 2>&1
check "no cache and no API: refused as before" [ "$(refused)" -gt "$r0" ]

echo "== GitHub guard (exit 98)"
ghset "repos/aien-dev/demo" '{"full_name":"aien-spark/demo"}'
rm -f "$AIEN_STATE/slugs/demo" "$AIEN_STATE/slugs/plain"
c0=$(grep -c '^gh ' "$FAKE_LOG" || true); l0=$(wc -l < "$AIEN_STATE/pipeline.jsonl")
rc=0; FAKE_GH98=1 bash "$PIPE" round > "$T/out14" 2>&1 || rc=$?
check "guard refusal: round still exits 0" [ "$rc" = 0 ]
check "guard refusal: one GitHub call, then none (no retry loop)" [ "$(( $(grep -c '^gh ' "$FAKE_LOG") - c0 ))" = 1 ]
check "guard refusal: later journal lines say try next round" \
  [ "$(tail -n +$((l0 + 1)) "$AIEN_STATE/pipeline.jsonl" | jq -c 'select(.deferred == "gh guard exit 98: try next round")' | wc -l)" -ge 1 ]
check "guard refusal: nothing merged" [ "$(tail -n +$((l0 + 1)) "$AIEN_STATE/pipeline.jsonl" | jq -c 'select(.result == "merged")' | wc -l)" = 0 ]

echo "== merge pause"
: > "$AIEN_STATE/MERGE-PAUSE"
m0=$(grep -c '^gh pr merge' "$FAKE_LOG" || true); l0=$(wc -l < "$AIEN_STATE/pipeline.jsonl")
bash "$PIPE" merge > "$T/out15" 2>&1
check "MERGE-PAUSE: merge step journals paused" [ "$(tail -n +$((l0 + 1)) "$AIEN_STATE/pipeline.jsonl" | jq -c 'select(.action == "merge" and .result == "paused")' | wc -l)" = 1 ]
check "MERGE-PAUSE: no gh pr merge" [ "$(grep -c '^gh pr merge' "$FAKE_LOG" || true)" = "$m0" ]
check "MERGE-PAUSE: the file is left in place" [ -e "$AIEN_STATE/MERGE-PAUSE" ]
check "rebase-train.sh never adds ready-to-merge or enables auto-merge" \
  bash -c "! grep -nE 'labels\\[\\]=(\"?\\\$MERGE_LABEL|ready-to-merge)|--auto|auto-merge\"' '$HERE/../rebase-train.sh'"
rm -f "$AIEN_STATE/MERGE-PAUSE"

echo "== secrets"
check "pipeline.sh never names the signing key file" bash -c "! grep -q 'verify\\.key' '$PIPE'"

echo
echo "pipeline_test: $pass passed, $fail failed"
[[ $fail == 0 ]]
