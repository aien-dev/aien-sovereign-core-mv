#!/usr/bin/env bash
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
while [[ $# -gt 0 ]]; do
  case $1 in --repo) repo=$2 ;; --commit) commit=$2 ;; --pubkey) key=$2 ;; --require) ;; *) exit 2 ;; esac; shift 2
done
[[ $(jq -r .repository "$file") == "$repo" && $(jq -r .commit "$file") == "$commit" ]] || exit 3
[[ $(jq -r .fake.key "$file") == "$key" ]] || exit 3
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

# bundle REPO SHA KEY RC [BLOCKED] [COMMIT_IN_BODY]
bundle() {
  local d=$AIEN_STATE/bundles/$1/$2
  mkdir -p "$d"
  jq -n --arg repo "github.com/aien-dev/$1" --arg c "${6:-$2}" --arg k "$3" --argjson rc "$4" --argjson b "${5:-null}" \
    '{repository:$repo, commit:$c, plan:{blocked:$b, verdict:"PASS"}, signature:{pubkey:$k}, fake:{key:$k, rc:$rc}}' > "$d/bundle.json"
}
remote_has() { git --git-dir="$T/receipts.git" cat-file -e "main:$1/$2/bundle.json" 2>/dev/null; }
remote_commits() { git --git-dir="$T/receipts.git" rev-list --count main; }
jl() { jq -c "select($1)" "$AIEN_STATE/pipeline.jsonl" | wc -l; }

A=$(sha 1) B=$(sha 2) C=$(sha 3) D=$(sha 4) E=$(sha 5) F=$(sha 6) W=$(sha 7) P=$(sha 8)
M1=$(sha 21) M2=$(sha 22) M3=$(sha 23) M4=$(sha 24) M5=$(sha 25) M6=$(sha 26) M7=$(sha 27) M8=$(sha 28) M9=$(sha 29)
bundle demo "$A" "$K2" 0            # PASS, current key
bundle demo "$B" "$K2" 1            # FAIL, still published
bundle demo "$C" "$K3" 0            # key not in verify.pub: REJECTED (3) with both keys
bundle demo "$D" "$K2" 2            # NOT_RUN
bundle demo "$E" "$K2" 0 '"gpu"'    # plan.blocked set
bundle demo "$F" "$K1" 0            # older key (first line of verify.pub)
bundle demo "$W" "$K2" 0 null "$A"  # body names another commit: verify uses the path, so REJECTED
bundle plain "$P" "$K2" 0           # repository not enrolled
for m in "$M1" "$M2" "$M3" "$M4" "$M5" "$M6" "$M7" "$M8" "$M9"; do bundle demo "$m" "$K2" 0; done

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

# A clone whose origin is github.com/aien-spark/<repo> (synapse): the expected repository follows the origin URL,
# the way the worker names it in the bundle. A wrapper git reports that URL; everything else is real git.
cat > "$T/bin/sparkgit" <<EOF
#!/bin/bash
if [[ "\$*" == "-C $AIEN_CLONES/demo remote get-url origin" ]]; then echo "git@github.com:aien-spark/demo.git"; exit 0; fi
exec git "\$@"
EOF
chmod +x "$T/bin/sparkgit"
GIT=$T/bin/sparkgit bash "$PIPE" publish > "$T/out1b" 2>&1
check "origin under aien-spark: verify-bundle expects github.com/aien-spark/demo" grep -q -- "--repo github.com/aien-spark/demo --commit" "$FAKE_LOG"

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

echo "== rerun"
ghset "repos/aien-dev/demo" '{"full_name":"aien-spark/demo"}'
ghset "repos/aien-spark/plain" '{"full_name":"aien-spark/plain"}'   # only under aien-spark, no aien-dev redirect (like synapse)
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
runs() { ghset "repos/$S/actions/runs?head_sha=$1&per_page=20" "{\"workflow_runs\":[$2]}"; }
run() { echo "{\"id\":$1,\"name\":\"$2\",\"status\":\"$3\",\"created_at\":\"$4\",\"run_started_at\":\"$4\"}"; }
pulls "1:$A:false:" "2:$B:false:" "3:$F:false:" "4:$C:false:"
runs "$A" "$(run 10 'AIEN Verification' completed 2000-01-01T00:00:00Z),$(run 11 'AIEN Verification' completed 2000-01-02T00:00:00Z),$(run 12 'Sovereign CI' completed 2000-01-03T00:00:00Z)"
runs "$B" "$(run 20 'AIEN Verification' in_progress 2000-01-01T00:00:00Z)"
runs "$F" "$(run 30 'AIEN Verification' completed 2099-01-01T00:00:00Z)"
bash "$PIPE" rerun > "$T/out4" 2>&1
posts() { grep -c "^gh api -X POST repos/$S/actions/runs/$1/rerun$" "$FAKE_LOG" || true; }
check "latest AIEN Verification run predating publication is rerun (canonical slug)" [ "$(posts 11)" = 1 ]
check "older run and other workflows untouched" [ "$(posts 10)$(posts 12)" = 00 ]
check "in-progress run is not rerun" [ "$(posts 20)" = 0 ]
check "run started after publication is not rerun" [ "$(posts 30)" = 0 ]
check "head without a published bundle is not looked up" bash -c "! grep -q 'head_sha=$C' '$FAKE_LOG'"
check "repo only under aien-spark resolves (no aien-dev redirect)" [ "$(jl ".repo==\"plain\" and (.reason|test(\"cannot resolve\"))")" = 0 ]
check "journal: in-progress wait" [ "$(jl ".action==\"rerun\" and .sha==\"$B\" and .result==\"wait\"")" = 1 ]
bash "$PIPE" rerun > "$T/out5" 2>&1
check "second round: no second rerun for the same digest" [ "$(posts 11)" = 1 ]
jq '.extra = 2' "$AIEN_STATE/bundles/demo/$A/bundle.json" > "$T/a" && mv "$T/a" "$AIEN_STATE/bundles/demo/$A/bundle.json"
bash "$PIPE" round > "$T/out6" 2>&1
check "new digest for the same head: rerun once more" [ "$(posts 11)" = 2 ]
check "round with no labeled PR merges nothing" bash -c "! grep -q '^gh pr merge' '$FAKE_LOG'"
check "journal: unlabeled PR skipped silently" [ "$(jl ".action==\"merge\" and .pr==\"1\"")" = 0 ]

echo "== merge"
CTX1="Workspace Verification & Invariant Audit" CTX2="AIEN Verification (Spark)"
ghset "repos/$S/branches/main/protection/required_status_checks" "{\"contexts\":[\"$CTX1\",\"$CTX2\"]}"
L=ready-to-merge
pulls "41:$M1:false:$L" "42:$M2:false:" "43:$M3:true:$L" "44:$M4:false:$L" "45:$M5:false:$L" \
      "46:$M6:false:$L" "47:$M7:false:$L" "48:$M8:false:$L" "49:$M9:false:ready-to-merge-later"
for i in 1 2 3 4 5 6 7 8 9; do
  m=$(sha $((20 + i)))
  state=clean; [[ $i == 8 ]] && state=dirty
  draft=false; [[ $i == 3 ]] && draft=true
  ghset "repos/$S/pulls/4$i" "{\"number\":4$i,\"draft\":$draft,\"mergeable_state\":\"$state\",\"head\":{\"sha\":\"$m\"}}"
  ghset "repos/$S/commits/$m/check-runs?per_page=100" "{\"check_runs\":[{\"name\":\"$CTX1\",\"status\":\"completed\",\"conclusion\":\"success\"},{\"name\":\"other\",\"status\":\"completed\",\"conclusion\":\"failure\"}]}"
  ghset "repos/$S/commits/$m/status?per_page=100" "{\"statuses\":[{\"context\":\"$CTX2\",\"state\":\"success\"}]}"
  runs "$m" "$(run $((100 + i)) 'AIEN Verification' completed 2099-01-01T00:00:00Z)"
done
ghset "repos/$S/commits/$M5/status?per_page=100" '{"statuses":[]}'
ghset "repos/$S/commits/$M6/check-runs?per_page=100" "{\"check_runs\":[{\"name\":\"$CTX1\",\"status\":\"completed\",\"conclusion\":\"failure\"}]}"
runs "$M7" "$(run 107 'AIEN Verification' completed 2099-01-01T00:00:00Z),$(run 117 'AIEN Verification' queued 2099-01-02T00:00:00Z)"
jq '.tampered = true' "$AIEN_STATE/bundles/demo/$M4/bundle.json" > "$T/m" && mv "$T/m" "$AIEN_STATE/bundles/demo/$M4/bundle.json"
bash "$PIPE" merge > "$T/out7" 2>&1
merges() { grep -c "^gh pr merge $1 " "$FAKE_LOG" || true; }
refused() { [ "$(jl ".action==\"merge\" and .pr==\"$1\" and .result==\"refused\" and (.reason|test(\"$2\"))")" = 1 ]; }
check "all green: merged with --match-head-commit on the canonical slug" grep -qx "gh pr merge 41 -R $S --merge --match-head-commit $M1" "$FAKE_LOG"
check "exactly one merge call" [ "$(grep -c '^gh pr merge' "$FAKE_LOG")" = 1 ]
check "never --admin" bash -c "! grep -q -- '--admin' '$FAKE_LOG'"
check "label missing: skipped, no journal line" [ "$(jl ".action==\"merge\" and .pr==\"42\"")" = 0 ]
check "label not exact (ready-to-merge-later): skipped, no journal line" [ "$(jl ".action==\"merge\" and .pr==\"49\"")" = 0 ]
check "draft: refused" refused 43 "draft"
check "digest mismatch: refused" refused 44 "digest mismatch"
check "required context missing: refused" refused 45 "AIEN Verification \\\\(Spark\\\\).* missing"
check "required context failed: refused" refused 46 "Invariant Audit.* is failure"
check "run queued: refused" refused 47 "queued or in progress"
check "dirty: refused" refused 48 "dirty"
check "merge verify passes --require from base .aien/always" grep -q -- "--commit $M1 --pubkey $K2 --require g1,g2" "$FAKE_LOG"

rm "$AIEN_STATE/bundles/demo/$M1/bundle.json"
bash "$PIPE" merge > "$T/out8" 2>&1
check "no local bundle: refused" refused 41 "no local bundle"
check "still exactly one merge call" [ "$(grep -c '^gh pr merge' "$FAKE_LOG")" = 1 ]

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

echo "== secrets"
check "pipeline.sh never names the signing key file" bash -c "! grep -q 'verify\\.key' '$PIPE'"

echo
echo "pipeline_test: $pass passed, $fail failed"
[[ $fail == 0 ]]
