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

# Release gate against the REAL candidate manifest (release/candidate.toml, omega.lock, Cargo.lock), CPU only, offline.
# scripts/test-release-checks.sh uses an invented candidate; this one proves the gate refuses a package or tree that
# is not CAND-4, using the committed manifest. No signing key is read or generated; the binaries are throwaway
# scripts, never real builds. Nothing is published.
#
# How the gate binds a package to the candidate revision (scripts/check-release-candidate.sh):
#   tree:    omega.lock == omega-commit, Cargo.lock holds exactly the [pins] revisions.
#   package: release.toml candidate + [model] table match, and with --native the sha256 of bin/aien must equal
#            [executables] aien-cli-native-release. The binary digest is what ties a package to a source revision:
#            a build from any other sovereign-core source gives another digest and is refused.
#   NOT bound: the sovereign-core-commit field in candidate.toml is informational (candidate.toml itself changed after
#            that commit); the executable digests are the binding. A package without --native is refused (no digest).
#
# Mutation note: the negative "post-candidate binary refused" case passes wrongly if the line
#   [[ "$want_aien" == "$exe" ]] || die ...
# in check-release-candidate.sh is removed or weakened (for example `true ||`). The mutation section below applies
# that change to a copy of the gate and requires the negative check to turn red, so it cannot pass vacuously.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-release-real.XXXXXX")"
trap 'rm -rf "$W"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "ok: $*"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

# fixture: the real manifest and locks, the real gate and packager, throwaway binaries
T="$W/tree"; mkdir -p "$T/scripts" "$T/release" "$W/bins"
cp "$ROOT/scripts/check-release-candidate.sh" "$ROOT/scripts/package-release.sh" "$T/scripts/"
cp "$ROOT/release/candidate.toml" "$T/release/"; cp "$ROOT/Cargo.lock" "$ROOT/cudapp.lock" "$T/"
# The fixture tree is the candidate's tree: omega.lock is the candidate's omega-commit, not this revision's. Main may
# move ahead of the candidate; whether THIS revision is releasable is reported (not enforced) by release-dry-run.yml,
# and enforced at tag time by release.yml.
sed -n 's/^omega-commit = "\([0-9a-f]\{40\}\)"$/\1/p' "$T/release/candidate.toml" > "$T/omega.lock"
[[ -s "$T/omega.lock" ]] || fail "release/candidate.toml has no full omega-commit"
echo licence > "$T/LICENSE"; echo notice > "$T/NOTICE"; echo third > "$T/THIRD_PARTY.md"; echo constitution > "$T/CONSTITUTION.md"; echo readme > "$T/README.md"; echo '#!/bin/sh' > "$T/install.sh"
for b in aien-cli spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do
    printf '#!/bin/sh\necho throwaway %s\n' "$b" > "$W/bins/$b"; chmod 755 "$W/bins/$b"
done
gate() { (cd "$T" && bash scripts/check-release-candidate.sh "$@" > "$W/log" 2>&1); }
must_pass() { gate "$@" || { cat "$W/log" >&2; fail "gate refused: $*"; }; }
must_fail() { local want="$1"; shift; if gate "$@"; then fail "gate accepted ($want)"; fi
    grep -q -- "$want" "$W/log" || { cat "$W/log" >&2; fail "refused for the wrong reason, wanted: $want"; }; }

CAND="$(sed -n 's/^candidate = "\(.*\)"/\1/p' "$T/release/candidate.toml")"
NATIVE_DIGEST="$(sed -n 's/^aien-cli-native-release = "\([0-9a-f]\{64\}\)".*/\1/p' "$T/release/candidate.toml")"
[[ "$CAND" == CAND-4 && -n "$NATIVE_DIGEST" ]] || fail "release/candidate.toml is not CAND-4 with a native digest"
grep -Eq '^sovereign-core-commit = "[0-9a-f]{40}"' "$T/release/candidate.toml" || fail "sovereign-core-commit is not a full hash"

# positive control, tree: a tree on the candidate's pins passes
must_pass; grep -q "candidate=CAND-4" "$W/log"; pass "candidate tree passes the tree gate as CAND-4"

# negative, tree: a tree on other pins than the candidate's (what a later main would look like) is refused
cp "$T/omega.lock" "$W/omega.ok"; cp "$T/Cargo.lock" "$W/cargo.ok"
printf '%s\n' "0000000000000000000000000000000000000000" > "$T/omega.lock"; must_fail "omega.lock is"; cp "$W/omega.ok" "$T/omega.lock"
pass "omega.lock after the candidate refused"
want_pr="$(sed -n 's/^aien-protocols = "\(.*\)"/\1/p' "$T/release/candidate.toml")"
sed -i "s/$want_pr/1111111111111111111111111111111111111111/g" "$T/Cargo.lock"; must_fail "consumes aien-protocols"; cp "$W/cargo.ok" "$T/Cargo.lock"
pass "Cargo.lock aien-protocols after the candidate refused"
must_pass; pass "tree restored, accepted again"

# packages built by the real packager from the real tree, with throwaway binaries
(cd "$T" && AIEN_RELEASE_BIN_DIR="$W/bins" bash scripts/package-release.sh "$W/post.tar.gz" >"$W/plog" 2>&1) || { cat "$W/plog" >&2; fail "packaging failed"; }
# negative, package: a binary that is not the candidate's (any post-candidate build) is refused on the native target
native_negative() { # gate-script
    (cd "$T" && bash "$1" --package "$W/post.tar.gz" --native > "$W/log" 2>&1) && return 1
    grep -q "is not CAND-4's aien-cli-native-release" "$W/log"
}
native_negative scripts/check-release-candidate.sh || { cat "$W/log" >&2; fail "post-candidate binary not refused for the right reason"; }
grep -q "$NATIVE_DIGEST" "$W/log" || fail "refusal does not name the candidate digest"
pass "package from a non-candidate binary refused (--native): digest differs from CAND-4 aien-cli-native-release"
# no --native: the manifest has no digest for that package kind, so it is refused (never passed)
must_fail "no candidate digest for this package kind" --package "$W/post.tar.gz"; pass "package without --native refused: no candidate digest for the kind"
# a package that names another candidate is refused
rm -rf "$W/u"; mkdir "$W/u"; tar -xzf "$W/post.tar.gz" -C "$W/u"; sed -i 's/^candidate = .*/candidate = "CAND-3"/' "$W/u/release.toml"
tar -czf "$W/cand3.tar.gz" -C "$W/u" .
must_fail "names candidate 'CAND-3', not CAND-4" --package "$W/cand3.tar.gz"; pass "package naming CAND-3 refused"

# positive control, package: manifest whose native digest is this package's binary
GOOD="$(sha "$W/bins/aien-cli")"
sed -i "s/^aien-cli-native-release = \"[0-9a-f]\{64\}\"/aien-cli-native-release = \"$GOOD\"/" "$T/release/candidate.toml"
must_fail "is not CAND-4's sc-" --package "$W/post.tar.gz" --native; pass "aien-cli matches but helper digests do not: refused"
for h in spark-cockpit-rs spark-inquisitor cortex-encoder-rs cortex-rs spark-supervisor spark-debugger; do
    sed -i "s/^sc-$h = \"[0-9a-f]\{64\}\"/sc-$h = \"$(sha "$W/bins/$h")\"/" "$T/release/candidate.toml"
done
must_pass --package "$W/post.tar.gz" --native; pass "positive control: every shipped executable matches its manifest digest, package passes"
cp "$ROOT/release/candidate.toml" "$T/release/candidate.toml"

# mutation: weaken the digest comparison in a copy of the gate; the negative must turn red
sed 's/\[\[ "\$want_aien" == "\$exe" \]\] ||/true ||/' "$T/scripts/check-release-candidate.sh" > "$T/scripts/mutant.sh"
cmp -s "$T/scripts/mutant.sh" "$T/scripts/check-release-candidate.sh" && fail "mutation did not apply (gate text changed; update this test)"
if native_negative scripts/mutant.sh; then fail "negative test stayed green against a gate that skips the digest comparison"; fi
pass "mutation (digest comparison removed) is caught"

bash "$ROOT/scripts/test-dry-run-workflow.sh"
echo "PASS: release gate refuses non-candidate trees and packages against the real CAND-4 manifest"
