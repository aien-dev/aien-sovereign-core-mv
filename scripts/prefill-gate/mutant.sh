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

# PREFILL-GATE mutation check (shell only).
#
# Runs the prefill-gate tests with one mutant cargo feature switched on. The
# mutant features are off by default and exist only for this check:
#   no_check            aien-scheduler/prefill_mutant_no_check
#                       (M1: scheduler readiness check removed)
#   ready_at_alloc      aien-kv-cache/prefill_mutant_ready_at_alloc
#                       (M2: block tables marked ready at allocation, no fence)
#   ready_at_alloc_e2e  M2 again, caught end to end through aien-runtime
#
# Exit 0 ONLY when the mutant is caught: cargo test fails AND at least one test
# whose name contains "prefill_gate" is reported FAILED. A build error is not a
# catch and exits 1. A mutant that survives (all tests pass) exits 1.
#
# Usage: bash scripts/prefill-gate/mutant.sh no_check|ready_at_alloc|ready_at_alloc_e2e
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-}"
case "$MODE" in
    no_check)
        ARGS=(-p aien-scheduler --features prefill_mutant_no_check) ;;
    ready_at_alloc)
        ARGS=(-p aien-kv-cache --features prefill_mutant_ready_at_alloc) ;;
    ready_at_alloc_e2e)
        ARGS=(-p aien-runtime --features aien-kv-cache/prefill_mutant_ready_at_alloc) ;;
    *)
        echo "usage: $0 no_check|ready_at_alloc|ready_at_alloc_e2e" >&2
        exit 2 ;;
esac

LOG="$(mktemp -t "prefill-mutant-${MODE}-XXXXXX.log")"
cd "$ROOT" || exit 2
cargo test "${ARGS[@]}" --no-fail-fast >"$LOG" 2>&1
RC=$?

CAUGHT_RE='^test .*prefill_gate.* \.\.\. FAILED$'
if [ "$RC" -eq 0 ]; then
    echo "MUTANT SURVIVED ($MODE): every test passed with the mutant on. Log: $LOG"
    exit 1
fi
if grep -Eq "$CAUGHT_RE" "$LOG"; then
    echo "MUTANT CAUGHT ($MODE): cargo test exit $RC"
    grep -E "$CAUGHT_RE" "$LOG"
    exit 0
fi
echo "MUTANT NOT CAUGHT BY A PREFILL GATE TEST ($MODE): cargo exit $RC with no prefill_gate test failure (build error?). Log: $LOG"
tail -n 30 "$LOG"
exit 1
