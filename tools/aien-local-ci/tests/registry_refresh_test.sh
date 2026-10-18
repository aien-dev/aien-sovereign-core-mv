#!/usr/bin/env bash
# registry_refresh_test.sh: offline proofs for registry-refresh.sh on throwaway directories (no sudo, no network).
#   bash tools/aien-local-ci/tests/registry_refresh_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$HERE/../registry-refresh.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

IDX=index.crates.io-1949cf8c6b5b557f
J=$T/judge O=$T/oper
mkdir -p "$J/cache/$IDX" "$J/index/$IDX/.cache" "$O/cache/$IDX" "$O/index/$IDX/.cache/ru/st" "$O/index/$IDX/.cache/2" "$O/index/$IDX/.cache/3/a"
export AIEN_JUDGE_REGISTRY=$J AIEN_OPERATOR_REGISTRY=$O AIEN_JOURNAL=$T/journal.jsonl AIEN_INSTALL="install"

mk() { printf '%s' "$2" > "$O/cache/$IDX/$1.crate"; sha256sum "$O/cache/$IDX/$1.crate" | cut -d' ' -f1; }
sha_rustls=$(mk rustls-0.23.46 'rustls bytes')
sha_old=$(mk rustls-0.23.45 'old rustls')
sha_ab=$(mk ab-1.0.0 'two letter name')
sha_abc=$(mk abc-2.0.0 'three letter name')
_=$(mk tampered-1.0.0 'these bytes do not match the lockfile')
echo 'index entry rustls' > "$O/index/$IDX/.cache/ru/st/rustls"
echo 'index entry ab' > "$O/index/$IDX/.cache/2/ab"
echo 'index entry abc' > "$O/index/$IDX/.cache/3/a/abc"
# The judge already has the old rustls, with different bytes on purpose: it must never be replaced.
printf 'judge copy, keep me' > "$J/cache/$IDX/rustls-0.23.45.crate"

pkg() { printf '[[package]]\nname = "%s"\nversion = "%s"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "%s"\n\n' "$@"; }
{
  printf 'version = 4\n\n'
  pkg rustls 0.23.46 "$sha_rustls"
  pkg rustls 0.23.45 "$sha_old"
  pkg ab 1.0.0 "$sha_ab"
  pkg abc 2.0.0 "$sha_abc"
  pkg notcached 0.1.0 0000000000000000000000000000000000000000000000000000000000000000
  printf '[[package]]\nname = "aien-local"\nversion = "0.1.0"\n\n'
  printf '[[package]]\nname = "gitdep"\nversion = "0.1.0"\nsource = "git+https://example.invalid/x#abc"\n\n'
} > "$T/Cargo.lock"

# 1 dry run adds nothing and names what it would add
out=$(bash "$R" "$T/Cargo.lock" --dry-run 2>&1); rc=$?
check "dry run exits 0" test $rc -eq 0
check "dry run names rustls 0.23.46" grep -q 'would add rustls-0.23.46.crate' <<<"$out"
check "dry run copies nothing" test ! -e "$J/cache/$IDX/rustls-0.23.46.crate"
check "dry run writes no journal" test ! -e "$AIEN_JOURNAL"

# 2 real run: adds the missing verified crates with their index entries, keeps the existing one, reports the uncached one
out=$(bash "$R" "$T/Cargo.lock" 2>&1); rc=$?
check "real run exits 0" test $rc -eq 0
check "rustls 0.23.46 added" cmp -s "$O/cache/$IDX/rustls-0.23.46.crate" "$J/cache/$IDX/rustls-0.23.46.crate"
check "added crate is read-only" test "$(stat -c %a "$J/cache/$IDX/rustls-0.23.46.crate" 2>/dev/null || stat -f %Lp "$J/cache/$IDX/rustls-0.23.46.crate")" = 444
check "index entry copied (ru/st/rustls)" cmp -s "$O/index/$IDX/.cache/ru/st/rustls" "$J/index/$IDX/.cache/ru/st/rustls"
check "two-letter name index path (2/ab)" test -f "$J/index/$IDX/.cache/2/ab"
check "three-letter name index path (3/a/abc)" test -f "$J/index/$IDX/.cache/3/a/abc"
check "existing judge crate never replaced" test "$(cat "$J/cache/$IDX/rustls-0.23.45.crate")" = 'judge copy, keep me'
check "uncached crate reported, not fatal" grep -q 'cannot add notcached-0.1.0.crate' <<<"$out"
check "summary counts" grep -q '5 crates.io packages, 3 added, 1 not in operator cache, 0 refused' <<<"$out"
check "journal line written" test "$(wc -l < "$AIEN_JOURNAL")" -eq 1
check "journal names the lockfile digest" test "$(jq -r .lock_sha256 "$AIEN_JOURNAL")" = "$(sha256sum "$T/Cargo.lock" | cut -d' ' -f1)"
check "journal lists the added crates" test "$(jq -r '.added|join(",")' "$AIEN_JOURNAL")" = 'rustls-0.23.46.crate,ab-1.0.0.crate,abc-2.0.0.crate'

# 3 second run is a no-op
out=$(bash "$R" "$T/Cargo.lock" 2>&1); rc=$?
check "second run exits 0 with 0 added" grep -q ', 0 added,' <<<"$out"

# 4 a checksum mismatch is refused: nothing copied, exit 1, named in the journal
pkg tampered 1.0.0 1111111111111111111111111111111111111111111111111111111111111111 >> "$T/Cargo.lock"
out=$(bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "mismatch exits 1" test $rc -eq 1
check "mismatch is named REFUSED" grep -q 'REFUSED tampered-1.0.0.crate' <<<"$out"
check "mismatched crate not copied" test ! -e "$J/cache/$IDX/tampered-1.0.0.crate"
check "journal records the refusal" test "$(jq -r '.refused|length' <(tail -1 "$AIEN_JOURNAL"))" = 1

# 5 index directory mismatch is fatal before any copy
mkdir -p "$T/judge2/cache/index.crates.io-ffffffffffffffff"
out=$(AIEN_JUDGE_REGISTRY=$T/judge2 bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "index dir mismatch refused" test $rc -eq 1
check "index dir mismatch message" grep -q 'index directory mismatch' <<<"$out"

echo "registry_refresh_test: $pass passed, $fail failed"
(( fail == 0 ))
