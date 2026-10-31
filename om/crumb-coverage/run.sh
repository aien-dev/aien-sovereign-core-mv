#!/usr/bin/env bash
# om/crumb-coverage/run.sh: .crumb coverage of this repository (DRA-95).
# Builds om/crumb-coverage/crumb-coverage.om with the pinned seeded omc and feeds it `git ls-files -z`
# from the repository root. It prints tracked_dirs, crumb_dirs and enforced_dirs (crumb directories with
# at least one invariant whose "Check: ..." resolves to an existing test function or script), plus every
# invariant whose check does not resolve.
#   run.sh               report for this repository (exit 0 every check resolves, 1 some do not);
#   run.sh --self-test   fixture repositories with positive cases and negative controls (exit 0 PASS, 1 FAIL).
# The shell only sequences; every count and judgment is made by the .om program.
# Compiler: AIEN_OMC (OMC and OMEGA_OMC are accepted too) must hash to omega.pin, else NOT_RUN (exit 77).
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
not_run() { echo "crumb-coverage: NOT_RUN $1: $2" >&2; printf 'AIEN-OBS not_run_reason "%s"\n' "$1"; exit 77; }

omc=${AIEN_OMC:-${OMC:-${OMEGA_OMC:-}}}
[ -n "$omc" ] || not_run omc_missing "set AIEN_OMC to omega-mv seed/out/omc at the commit in om/crumb-coverage/omega.pin"
[ -x "$omc" ] || not_run omc_missing "$omc is not executable"
pin=$(sed -n 's/^omc_sha256: //p' "$here/omega.pin")
got=$(sha256sum < "$omc" | cut -d' ' -f1)
[ "$got" = "$pin" ] || not_run omc_unpinned "omc sha256 $got, pinned $pin"

b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
{ printf 'om2\n'; cat "$root/om/vendor/json.om" "$here/crumb-coverage.om"; } | "$omc" > "$b/cc" 2> "$b/cc.err" \
  || { echo "crumb-coverage: compile failed: $(cat "$b/cc.err")" >&2; exit 1; }
chmod 700 "$b/cc"
echo "crumb-coverage: omc_sha256 $got"
echo "crumb-coverage: binary_sha256 $(sha256sum < "$b/cc" | cut -d' ' -f1)"

if [ "${1:-}" != "--self-test" ]; then
  [ $# -eq 0 ] || { echo "usage: om/crumb-coverage/run.sh [--self-test]" >&2; exit 2; }
  echo "crumb-coverage: tree $(git -C "$root" rev-parse HEAD) (git ls-files: the index, staged changes included)"
  git -C "$root" ls-files -z > "$b/list" || exit 3
  cd "$root" && exec "$b/cc" < "$b/list"
fi

# ---- self-test: each case is a fresh git repository ----
fail=0
newrepo() { rm -rf "$b/r" && mkdir -p "$b/r" && git -C "$b/r" init -q; }
put() { mkdir -p "$(dirname "$b/r/$1")" && printf '%s\n' "$2" > "$b/r/$1"; }
runcase() { # name want_rc grep_pattern...
  name=$1 want=$2; shift 2
  git -C "$b/r" add -A
  git -C "$b/r" ls-files -z > "$b/list"
  out=$(cd "$b/r" && "$b/cc" < "$b/list" 2>&1); rc=$?
  printf '%s\n' "$out" | sed "s/^/  $name| /"
  ok=1
  [ "$rc" = "$want" ] || ok=0
  for pat in "$@"; do printf '%s\n' "$out" | grep -qxF -- "$pat" || { ok=0; echo "  $name| missing line: $pat"; }; done
  if [ $ok = 1 ]; then echo "ok   $name: exit $rc"; else echo "FAIL $name: exit $rc, want $want"; fail=1; fi
}
fixture() {
  newrepo
  put crates/x/tests/t.rs '#[test]
fn good_test() {}'
  put crates/x/src/m.rs '#[cfg(test)]
mod tests {
    #[test]
    fn good_lib() {}
}'
  put crates/x/src/lib.rs 'mod m;
#[test]
fn good_root() {}'
  put scripts/check.sh 'exit 0'
  put a/b/c.txt c
  put a/d.txt d
}

fixture
put crates/x/.crumb '{"invariants": [
 "T holds. Check: cargo test -p x --test t -- --exact good_test",
 "M holds. Check: cargo test -p x --lib -- --exact m::tests::good_lib",
 "L holds. Check: cargo test -p x --lib -- --exact good_root",
 "S holds. Check: sh scripts/check.sh"], "name": "x"}'
put .crumb '{"name": "root"}'
runcase resolved 0 'crumb-coverage tracked_dirs 8' 'crumb-coverage crumb_dirs 2' \
  'crumb-coverage enforced_dirs 1' 'crumb-coverage invariants 4 resolved 4 unresolved 0' \
  'crumb crates/x invariants 4 resolved 4' 'crumb . invariants 0 resolved 0'

fixture
put crates/x/.crumb '{"invariants": [
 "T holds. Check: cargo test -p x --test t -- --exact no_such_test",
 "M holds. Check: cargo test -p x --lib -- --exact other::tests::good_lib",
 "N holds and names no check",
 "S holds. Check: sh scripts/missing.sh",
 "W holds. Check: cargo test -p y --test t -- --exact good_test",
 "A holds. Check: sh /etc/passwd"]}'
runcase unresolved-control 1 'crumb-coverage enforced_dirs 0' \
  'crumb-coverage invariants 6 resolved 0 unresolved 6' \
  'unresolved crates/x: T holds. Check: cargo test -p x --test t -- --exact no_such_test' \
  'unresolved crates/x: N holds and names no check'

fixture
ln -s t.rs "$b/r/crates/x/tests/u.rs"
put crates/x/.crumb '{"invariants": ["U holds. Check: cargo test -p x --test u -- --exact good_test"]}'
runcase symlink-control 1 'crumb-coverage invariants 1 resolved 0 unresolved 1'

fixture
put crates/x/.crumb '{"invariants": ["T holds. Check: cargo test -p x --test t -- --exact good_test",]}'
runcase bad-json-control 4 'crumb-coverage refused 4 crumb-json'

fixture
put crates/x/.crumb '{"invariants": "T holds. Check: sh scripts/check.sh"}'
runcase not-array-control 4 'crumb-coverage refused 4 invariants-not-array-of-strings'

newrepo
printf 'b\0a\0' > "$b/list"
out=$("$b/cc" < "$b/list" 2>&1); rc=$?
if [ $rc = 4 ] && [ "$out" = "crumb-coverage refused 4 paths-not-in-git-order" ]; then echo "ok   unsorted-control: exit 4"
else echo "FAIL unsorted-control: exit $rc: $out"; fail=1; fi

if [ $fail = 0 ]; then echo "crumb-coverage self-test: PASS"; else echo "crumb-coverage self-test: FAIL"; fi
exit $fail
