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

# Mandatory gate for the tests that need the REAL omega composition library.
#
# The normal `cargo test --workspace` CI step builds the stub (no omega checkout), where every
# compose-linked test is `#[ignore]`, so a failure in them never shows (sovereign-core #306: c25 failed
# on main for days behind the stub). This script builds the library from an omega checkout at
# omega.lock and RUNS those tests, ignored ones included. It fails if the library is not really linked
# or if no linked test ran, so it cannot pass vacuously.
#
# Usage: scripts/test-linked-compose.sh OMEGA_DIR PHYSICS_DIR AIENOS_REPO
#   OMEGA_DIR    omega checkout whose HEAD equals omega.lock
#   PHYSICS_DIR  physics checkout at omega's physics.lock
#   AIENOS_REPO  aienos clone holding omega's aienos.lock commit
# Env: CARGO_TARGET_DIR (optional), LINKED_REPEAT (default 3: how many times the approved_attacks suite runs).
# CPU only: never sets AIEN_OMEGA_DIR (that would link the GPU engine) and unsets the stub/fallback switches.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMW=$(cd "${1:?OMEGA_DIR}" && pwd); PHW=$(cd "${2:?PHYSICS_DIR}" && pwd); AO=$(cd "${3:?AIENOS_REPO}" && pwd)
die() { echo "test-linked-compose: $*" >&2; exit 3; }
pin=$(tr -d '[:space:]' <"$ROOT/omega.lock")
[ "$(git -C "$OMW" rev-parse HEAD)" = "$pin" ] || die "omega HEAD != omega.lock $pin"
[ "$(git -C "$PHW" rev-parse HEAD)" = "$(tr -d '[:space:]' <"$OMW/physics.lock")" ] || die "physics HEAD != omega physics.lock"
[ "$(git -C "$AO" rev-parse HEAD)" = "$(tr -d '[:space:]' <"$OMW/aienos.lock")" ] || die "aienos HEAD != omega aienos.lock"
REPEAT=${LINKED_REPEAT:-3}
export AIEN_OMEGA_COMPOSE_DIR="$OMW" AIEN_PHYSICS_DIR="$PHW" AIEN_AIENOS_LOCK_REPO="$AO"
unset AIEN_OMEGA_DIR AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_SHA AIEN_FORCE_CPU_STUB AIEN_DEV_FALLBACK
cd "$ROOT"
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT
run() {  # run test targets with --include-ignored; the tests are serial (one process-wide compose home)
    cargo test "$@" -- --include-ignored --test-threads=1 2>&1 | tee -a "$LOG"
}
count() {  # how many tests the targets hold, ignored ones included
    cargo test "$@" -- --list --include-ignored 2>/dev/null | grep -c ": test$" || true
}
# Every aien-runtime test target, discovered from the directory so a new one is gated
# automatically (sovereign-core #323: minted_grant_test, refusal_table_test and the rest
# were linked-only and outside this gate). Left out, each needing what CI does not have:
#   prefill_e2e0/1/2     a real TinyLlama checkpoint (AIEN_E2E_CHECKPOINT).
# aien-cli: compose_ordinary_flow_test and allen_concurrency_test (sc#325, real daemon + CLI,
# no model) are gated. Outside this gate, in other crates: aien-cli recovery_matrix_test and
# provenance_link_live_test (a real SmolLM2 on disk, minutes of CPU per case; run on the
# Spark, receipts on the PR), and the GB10/real-checkpoint inference tests (heavy queue).
# The aien-runtime library (unit) tests run too (`--lib`, sc#364): six of them open a compose home and take the
# home_guard slot (sc#363); run() is serial (--test-threads=1) regardless.
RT=()  # 46 targets on 2026-10-08; the floor below catches a lost glob or a deleted target
for f in "$ROOT"/crates/aien-runtime/tests/*.rs; do
    t=$(basename "$f" .rs)
    case "$t" in prefill_e2e*) continue ;; esac
    RT+=(--test "$t")
done
[ "${#RT[@]}" -ge 90 ] || die "found only $((${#RT[@]} / 2)) aien-runtime test targets"
# aien-omega-compose: the compose suite, callback safety (abi-qual) and the ABI manifest + native identity.
OC=(--test compose --test callbacks --test abi_manifest)
CLI=(--test compose_ordinary_flow_test --test allen_concurrency_test)
ATT=$(count -p aien-runtime --test approved_attacks_test)
ALL=$(( $(count -p aien-omega-compose "${OC[@]}") + $(count -p aien-runtime --lib) + $(count -p aien-runtime "${RT[@]}") + $(count -p aien-cli "${CLI[@]}") ))
[ "$ATT" -gt 0 ] && [ "$ALL" -gt "$ATT" ] || die "could not list the linked tests"
: >"$LOG"
run -p aien-omega-compose "${OC[@]}"
run -p aien-runtime --lib
run -p aien-runtime "${RT[@]}"
run -p aien-cli "${CLI[@]}"
# The approved_attacks suite again (it once hid a race, #306): REPEAT runs in all.
for i in $(seq 2 "$REPEAT"); do
    echo "== approved_attacks run $i/$REPEAT"
    run -p aien-runtime --test approved_attacks_test
done
# Not vacuous: the library was linked (no stub warning) and the tests really ran.
if grep -q "building the stub" "$LOG"; then die "a stub was built: the composition library is not linked"; fi
want=$(( ALL + ATT * (REPEAT - 1) ))
ran=$(grep -E '^test result: ok\. [0-9]+ passed' "$LOG" | awk '{s += $4} END {print s + 0}')
[ "$ran" -ge "$want" ] || die "only $ran linked tests passed (expected $want: $ALL linked tests + approved_attacks $ATT x $((REPEAT - 1)) more)"
if grep -E '^test result:.* [1-9][0-9]* ignored' "$LOG" >/dev/null; then die "tests were still ignored"; fi
echo "test-linked-compose: PASS ($ran linked tests passed: $ALL across compose + callbacks + abi_manifest, aien-runtime library tests, every aien-runtime test target but prefill_e2e*, aien-cli ordinary flow + ALLEN concurrency; approved_attacks x$REPEAT)"
