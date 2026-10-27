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

# PREFILL-GATE counterexample check (shell only).
#
# Copies the two new main-compatible test files onto a clean checkout of the
# base ref (default origin/main) and runs them there. The defect is present on
# main, so each test must FAIL with the PREFILL_GATE_VIOLATION marker.
#
# Exit 0 ONLY when every test fails on the base ref with that marker. A test
# that passes on main, or fails for another reason (build error), exits 1.
#
# The temporary worktree is created as a sibling of this checkout (kept out of
# the repository tree), and is removed on exit.
#
# Usage: bash scripts/prefill-gate/fails-on-main.sh [base-ref]
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${1:-origin/main}"
TMP="$(dirname "$ROOT")/.prefill-gate-main-check-$$"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/prefill-gate-main}"

cleanup() {
    git -C "$ROOT" worktree remove --force "$TMP" >/dev/null 2>&1
    git -C "$ROOT" worktree prune >/dev/null 2>&1
}
trap cleanup EXIT

git -C "$ROOT" worktree add --detach "$TMP" "$BASE" >/dev/null 2>&1 || {
    echo "cannot create worktree of $BASE at $TMP"
    exit 2
}

mkdir -p "$TMP/crates/aien-scheduler/tests" "$TMP/crates/aien-runtime/tests"
cp "$ROOT/crates/aien-scheduler/tests/prefill_gate_test.rs" "$TMP/crates/aien-scheduler/tests/"
cp "$ROOT/crates/aien-runtime/tests/swarm_prefill_gate_test.rs" "$TMP/crates/aien-runtime/tests/"

ALL_OK=1
check() {
    local pkg="$1" test="$2" log rc
    log="$(mktemp -t "prefill-main-${test}-XXXXXX.log")"
    (cd "$TMP" && cargo test -p "$pkg" --test "$test") >"$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] && grep -q "PREFILL_GATE_VIOLATION" "$log"; then
        echo "FAILS ON $BASE (expected): $pkg --test $test"
    else
        echo "UNEXPECTED on $BASE: $pkg --test $test exit $rc without PREFILL_GATE_VIOLATION. Log: $log"
        tail -n 30 "$log"
        ALL_OK=0
    fi
}

check aien-scheduler prefill_gate_test
check aien-runtime swarm_prefill_gate_test

if [ "$ALL_OK" -eq 1 ]; then
    echo "PREFILL-GATE counterexample confirmed: new tests fail on $BASE"
    exit 0
fi
exit 1
