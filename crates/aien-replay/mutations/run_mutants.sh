#!/bin/sh
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

# Prove the KV ownership checks catch realistic faults.
#
# usage: run_mutants.sh native   apply each kv.mutants line, run the proptest,
#                                expect a FAILED test (KILLED)
#        run_mutants.sh miri     apply each miri.mutants line, expect native
#                                tests to still pass and `cargo miri test` to fail
#        run_mutants.sh trn1     switch off each TRN1 check (trn1.mutants), expect
#                                the shared corpus test to fail; needs TRN1_VECTORS
#                                or the sibling ../aien-protocols checkout
#
# Works on a scratch copy; the real tree is never modified. Exit 0 only if
# every mutant is KILLED. A mutant that does not compile counts as BROKEN
# (a bad mutant, not a kill) and fails the run.
set -eu

mode=${1:-native}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
case "$mode" in
native) table="$here/kv.mutants" ;;
miri) table="$here/miri.mutants" ;;
trn1) table="$here/trn1.mutants" ;;
*) echo "usage: $0 native|miri|trn1" >&2; exit 2 ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM
mkdir -p "$work/crates"
for c in aien-kv-cache aien-platform aien-platform-linux aien-replay; do
	mkdir -p "$work/crates/$c"
	(cd "$root/crates/$c" && tar cf - --exclude=./target .) | (cd "$work/crates/$c" && tar xf -)
done
# Minimal workspace root: same [workspace.package] values, only the three
# crates the test needs, so the copy builds without the aien-protocols sibling.
awk '/^members = \[/ { print "members = [\"crates/aien-kv-cache\", \"crates/aien-platform\", \"crates/aien-platform-linux\"]"; skip = 1; next }
     skip && /^\]/ { skip = 0; next }
     !skip { print }' "$root/Cargo.toml" >"$work/Cargo.toml"

src="$work/crates/aien-kv-cache/src/lib.rs"
test=kv_ownership_props
if [ "$mode" = trn1 ]; then
	src="$work/crates/aien-replay/src/lib.rs"
	test=trn1_conformance
	vectors=$(cd "${TRN1_VECTORS:-$root/../aien-protocols/specs/execution-transcript/vectors}" && pwd)
	export TRN1_VECTORS="$vectors" TRN1_REQUIRE=1
fi
cp "$src" "$work/lib.rs.orig"
manifest="$work/crates/aien-replay/Cargo.toml"
export CARGO_TARGET_DIR="${MUTANT_TARGET_DIR:-$work/target}"
cargo=${CARGO:-cargo}

run_native() { $cargo test --offline --manifest-path "$manifest" --test "$test" >"$1" 2>&1; }
run_miri() { $cargo +nightly miri test --manifest-path "$manifest" --test "$test" >"$1" 2>&1; }

fail=0
killed=0
total=0
tab=$(printf '\t')
while IFS="$tab" read -r id expr what; do
	case "$id" in '' | '#'*) continue ;; esac
	total=$((total + 1))
	cp "$work/lib.rs.orig" "$src"
	sed -i "$expr" "$src"
	if cmp -s "$src" "$work/lib.rs.orig"; then
		echo "BROKEN   $id: sed expression matched nothing"
		fail=1
		continue
	fi
	log="$work/$id.log"
	if [ "$mode" != miri ]; then
		if run_native "$log"; then
			echo "SURVIVED $id ($what)"
			fail=1
		elif grep -q "test result: FAILED" "$log"; then
			echo "KILLED   $id ($what)"
			killed=$((killed + 1))
		else
			echo "BROKEN   $id: did not compile"
			tail -n 20 "$log"
			fail=1
		fi
	else
		if ! run_native "$log.native"; then
			echo "NOTE     $id: native tests also fail, so this is not a Miri-only catch"
		fi
		if run_miri "$log"; then
			echo "SURVIVED $id under Miri ($what)"
			fail=1
		elif grep -q "Undefined Behavior" "$log"; then
			echo "KILLED   $id under Miri ($what)"
			killed=$((killed + 1))
		else
			echo "BROKEN   $id: Miri run failed without reporting UB"
			tail -n 20 "$log"
			fail=1
		fi
	fi
done <"$table"

echo "mode=$mode killed=$killed total=$total"
exit "$fail"
