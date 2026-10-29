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

# Prove the aien-test unit tests catch realistic faults.
#
# usage: tools/aien-test/mutations/run_mutants.sh
#
# Applies each line of aien-test.mutants (id TAB file TAB sed expression TAB
# what it models) to a scratch copy of the crate, runs `cargo test`, and
# expects a FAILED test (KILLED). Exit 0 only if every mutant is KILLED. A
# mutant whose sed changes nothing, or that does not compile, is BROKEN and
# fails the run. The real tree is never modified.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
crate=$(cd "$here/.." && pwd)
root=$(cd "$here/../../.." && pwd)
table="$here/aien-test.mutants"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM
mkdir -p "$work/tools/aien-test"
(cd "$crate" && tar cf - --exclude=./target --exclude=./mutations .) | (cd "$work/tools/aien-test" && tar xf -)
cp "$root/Cargo.lock" "$work/Cargo.lock"
# Minimal workspace root: same [workspace.package] values, only this crate.
awk '/^members = \[/ { print "members = [\"tools/aien-test\"]"; skip = 1; next }
     skip && /^\]/ { skip = 0; next }
     !skip { print }' "$root/Cargo.toml" >"$work/Cargo.toml"

export CARGO_TARGET_DIR="${MUTANT_TARGET_DIR:-$work/target}"
cargo=${CARGO:-cargo}
manifest="$work/tools/aien-test/Cargo.toml"

# Sanity: the unmutated copy must pass, or every "kill" would be meaningless.
if ! $cargo test --offline --manifest-path "$manifest" >"$work/baseline.log" 2>&1; then
	echo "BASELINE FAILED: the unmutated crate does not pass its own tests"
	tail -n 30 "$work/baseline.log"
	exit 1
fi

fail=0
killed=0
total=0
tab=$(printf '\t')
while IFS="$tab" read -r id file expr what; do
	case "$id" in '' | '#'*) continue ;; esac
	total=$((total + 1))
	src="$work/tools/aien-test/src/$file"
	cp "$crate/src/$file" "$src"
	sed -i "$expr" "$src"
	if cmp -s "$src" "$crate/src/$file"; then
		echo "BROKEN   $id: sed expression matched nothing"
		fail=1
		continue
	fi
	log="$work/$id.log"
	if $cargo test --offline --manifest-path "$manifest" >"$log" 2>&1; then
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
	cp "$crate/src/$file" "$src"
done <"$table"

echo "killed=$killed total=$total"
exit "$fail"
