#!/usr/bin/env bash
# registry_refresh_test.sh: offline proofs for registry-refresh.sh on throwaway directories (no sudo, no network).
#   bash tools/aien-local-ci/tests/registry_refresh_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$HERE/../registry-refresh.sh
T=$(mktemp -d)
trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
mode() { stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1"; }

IDX=index.crates.io-1949cf8c6b5b557f
J=$T/judge O=$T/oper S=$T/judge/src/$IDX
mkdir -p "$J/cache/$IDX" "$J/index/$IDX/.cache" "$S" "$O/cache/$IDX" "$O/index/$IDX/.cache/ru/st" "$O/index/$IDX/.cache/2" "$O/index/$IDX/.cache/3/a"
export AIEN_JUDGE_REGISTRY=$J AIEN_OPERATOR_REGISTRY=$O AIEN_JOURNAL=$T/journal.jsonl AIEN_INSTALL="install" AIEN_AS_ROOT=

# A real .crate: a gzip tar holding one <name>-<version>/ directory, like crates.io serves.
# mk <dest dir> <name> <version>: writes the .crate, prints its sha256.
mk() {
  local d=$1 top=$2-$3 w; w=$(mktemp -d "$T/w.XXXXXX")
  mkdir -p "$w/$top/src"
  printf '[package]\nname = "%s"\nversion = "%s"\n' "$2" "$3" > "$w/$top/Cargo.toml"
  printf 'pub fn f() {}\n' > "$w/$top/src/lib.rs"
  tar -czf "$d/$top.crate" -C "$w" "$top"
  sha256sum "$d/$top.crate" | cut -d' ' -f1
}
sha_rustls=$(mk "$O/cache/$IDX" rustls 0.23.46)
sha_old=$(mk "$O/cache/$IDX" rustls 0.23.45)
sha_ab=$(mk "$O/cache/$IDX" ab 1.0.0)
sha_abc=$(mk "$O/cache/$IDX" abc 2.0.0)
_=$(mk "$O/cache/$IDX" tampered 1.0.0)
echo 'index entry rustls' > "$O/index/$IDX/.cache/ru/st/rustls"
echo 'index entry ab' > "$O/index/$IDX/.cache/2/ab"
echo 'index entry abc' > "$O/index/$IDX/.cache/3/a/abc"
# The judge already has the old rustls, unpacked, with different bytes on purpose: it must never be replaced.
printf 'judge copy, keep me' > "$J/cache/$IDX/rustls-0.23.45.crate"
mkdir -p "$S/rustls-0.23.45"; printf '{"v":1}' > "$S/rustls-0.23.45/.cargo-ok"
# The state an add-only refresh left behind: the .crate is in the judge cache but its source was never
# unpacked, so the read-only judge cargo fails "failed to unpack package". It must be repaired.
sha_half=$(mk "$J/cache/$IDX" half 1.0.0)

pkg() { printf '[[package]]\nname = "%s"\nversion = "%s"\nsource = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "%s"\n\n' "$@"; }
{
  printf 'version = 4\n\n'
  pkg rustls 0.23.46 "$sha_rustls"
  pkg rustls 0.23.45 "$sha_old"
  pkg ab 1.0.0 "$sha_ab"
  pkg abc 2.0.0 "$sha_abc"
  pkg half 1.0.0 "$sha_half"
  pkg notcached 0.1.0 0000000000000000000000000000000000000000000000000000000000000000
  printf '[[package]]\nname = "aien-local"\nversion = "0.1.0"\n\n'
  printf '[[package]]\nname = "gitdep"\nversion = "0.1.0"\nsource = "git+https://example.invalid/x#abc"\n\n'
} > "$T/Cargo.lock"

# 1 dry run adds nothing and names what it would add or unpack
out=$(bash "$R" "$T/Cargo.lock" --dry-run 2>&1); rc=$?
check "dry run exits 0" test $rc -eq 0
check "dry run names rustls 0.23.46" grep -q 'would add rustls-0.23.46.crate' <<<"$out"
check "dry run names the unpack-only repair" grep -q 'would unpack half-1.0.0' <<<"$out"
check "dry run copies nothing" test ! -e "$J/cache/$IDX/rustls-0.23.46.crate"
check "dry run unpacks nothing" test ! -e "$S/rustls-0.23.46" -a ! -e "$S/half-1.0.0"
check "dry run writes no journal" test ! -e "$AIEN_JOURNAL"

# 2 real run: adds the missing verified crates unpacked with their index entries, repairs the unpacked-less one,
#   keeps the existing one, reports the uncached one
out=$(bash "$R" "$T/Cargo.lock" 2>&1); rc=$?
check "real run exits 0" test $rc -eq 0
check "rustls 0.23.46 added" cmp -s "$O/cache/$IDX/rustls-0.23.46.crate" "$J/cache/$IDX/rustls-0.23.46.crate"
check "added crate is read-only" test "$(mode "$J/cache/$IDX/rustls-0.23.46.crate")" = 444
check "added crate is unpacked into registry/src" grep -q 'version = "0.23.46"' "$S/rustls-0.23.46/Cargo.toml"
check "unpacked source carries cargo's .cargo-ok marker" test "$(cat "$S/rustls-0.23.46/.cargo-ok")" = '{"v":1}'
check "unpacked source file is read-only" test "$(mode "$S/rustls-0.23.46/src/lib.rs")" = 444
check "unpacked source dir is read-only" test "$(mode "$S/rustls-0.23.46")" = 555
check "no staging leftovers in registry/src" test -z "$(ls -A "$S" | grep '^\.' || true)"
check "index entry copied (ru/st/rustls)" cmp -s "$O/index/$IDX/.cache/ru/st/rustls" "$J/index/$IDX/.cache/ru/st/rustls"
check "two-letter name index path (2/ab)" test -f "$J/index/$IDX/.cache/2/ab"
check "three-letter name index path (3/a/abc)" test -f "$J/index/$IDX/.cache/3/a/abc"
check "two- and three-letter crates unpacked" test -f "$S/ab-1.0.0/.cargo-ok" -a -f "$S/abc-2.0.0/.cargo-ok"
check "crate without source gets unpacked from the judge copy" test -f "$S/half-1.0.0/.cargo-ok" -a -f "$S/half-1.0.0/src/lib.rs"
check "existing judge crate never replaced" test "$(cat "$J/cache/$IDX/rustls-0.23.45.crate")" = 'judge copy, keep me'
check "uncached crate reported, not fatal" grep -q 'cannot add notcached-0.1.0.crate' <<<"$out"
check "summary counts" grep -q '6 crates.io packages, 3 added, 1 unpacked, 1 not in operator cache, 0 refused' <<<"$out"
check "journal line written" test "$(wc -l < "$AIEN_JOURNAL")" -eq 1
check "journal names the lockfile digest" test "$(jq -r .lock_sha256 "$AIEN_JOURNAL")" = "$(sha256sum "$T/Cargo.lock" | cut -d' ' -f1)"
check "journal lists the added crates" test "$(jq -r '.added|join(",")' "$AIEN_JOURNAL")" = 'rustls-0.23.46.crate,ab-1.0.0.crate,abc-2.0.0.crate'
check "journal lists the unpacked crate" test "$(jq -r '.unpacked|join(",")' "$AIEN_JOURNAL")" = 'half-1.0.0.crate'

# 3 second run is a no-op
out=$(bash "$R" "$T/Cargo.lock" 2>&1); rc=$?
check "second run exits 0 with 0 added, 0 unpacked" grep -q ', 0 added, 0 unpacked,' <<<"$out"

# 4 a checksum mismatch is refused: nothing copied or unpacked, exit 1, named in the journal
pkg tampered 1.0.0 1111111111111111111111111111111111111111111111111111111111111111 >> "$T/Cargo.lock"
out=$(bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "mismatch exits 1" test $rc -eq 1
check "mismatch is named REFUSED" grep -q 'REFUSED tampered-1.0.0.crate' <<<"$out"
check "mismatched crate not copied" test ! -e "$J/cache/$IDX/tampered-1.0.0.crate"
check "mismatched crate not unpacked" test ! -e "$S/tampered-1.0.0"
check "journal records the refusal" test "$(jq -r '.refused|length' <(tail -1 "$AIEN_JOURNAL"))" = 1

# 5 an archive that is not exactly one <name>-<version>/ directory is refused before anything is written
w=$(mktemp -d "$T/w.XXXXXX"); mkdir -p "$w/evil-1.0.0" "$w/other"; echo x > "$w/other/planted"
tar -czf "$O/cache/$IDX/evil-1.0.0.crate" -C "$w" evil-1.0.0 other
pkg evil 1.0.0 "$(sha256sum "$O/cache/$IDX/evil-1.0.0.crate" | cut -d' ' -f1)" >> "$T/Cargo.lock"
out=$(bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "bad archive exits 1" test $rc -eq 1
check "bad archive is named REFUSED" grep -q 'REFUSED evil-1.0.0.crate: does not unpack' <<<"$out"
check "bad archive: no crate, no source, no stray dir" test ! -e "$J/cache/$IDX/evil-1.0.0.crate" -a ! -e "$S/evil-1.0.0" -a ! -e "$S/other"

# 6 a judge .crate whose bytes do not match the lockfile is refused, not replaced and not unpacked
printf 'judge bytes are wrong' > "$J/cache/$IDX/wrongjudge-1.0.0.crate"
pkg wrongjudge 1.0.0 2222222222222222222222222222222222222222222222222222222222222222 >> "$T/Cargo.lock"
out=$(bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "wrong judge crate is named REFUSED" grep -q 'REFUSED wrongjudge-1.0.0.crate: sha256' <<<"$out"
check "wrong judge crate kept as is, not unpacked" test "$(cat "$J/cache/$IDX/wrongjudge-1.0.0.crate")" = 'judge bytes are wrong' -a ! -e "$S/wrongjudge-1.0.0"

# 7 a half-unpacked source directory (no .cargo-ok) is refused, never replaced
sha_part=$(mk "$O/cache/$IDX" part 1.0.0)
mkdir -p "$S/part-1.0.0"; echo 'half written' > "$S/part-1.0.0/marker"
pkg part 1.0.0 "$sha_part" >> "$T/Cargo.lock"
out=$(bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "half-unpacked dir is named REFUSED" grep -q 'REFUSED part-1.0.0.crate: .* exists without .cargo-ok' <<<"$out"
check "half-unpacked dir left as is" test "$(cat "$S/part-1.0.0/marker")" = 'half written' -a ! -e "$S/part-1.0.0/.cargo-ok"

# 8 index directory mismatch is fatal before any copy
mkdir -p "$T/judge2/cache/index.crates.io-ffffffffffffffff"
out=$(AIEN_JUDGE_REGISTRY=$T/judge2 bash "$R" "$T/Cargo.lock" 2>&1) && rc=0 || rc=$?
check "index dir mismatch refused" test $rc -eq 1
check "index dir mismatch message" grep -q 'index directory mismatch' <<<"$out"

echo "registry_refresh_test: $pass passed, $fail failed"
(( fail == 0 ))
