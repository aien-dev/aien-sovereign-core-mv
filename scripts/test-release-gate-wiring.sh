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

# Static guard for where the release gate blocks (merge-owner decision on sovereign-core#268, 2026-10-07):
#   release.yml (tag push): every step that runs scripts/check-release-candidate.sh is fail-closed. The only
#     continue-on-error allowed anywhere in the file is the manual dry run:
#       continue-on-error: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run }}
#   release-dry-run.yml (every pull request): only the step that runs check-release-candidate.sh on this revision
#     may be non-blocking (main may move ahead of the candidate); the publish guard and the refusal suite
#     (test-release-real-tree.sh) block.
#   scripts/test-release-gate-wiring.sh [release.yml] [release-dry-run.yml]
# Mutation note: the self-checks below inject `continue-on-error: true` into the tag gate step (plain, and hidden
# behind a comment holding the allowed text), weaken the allowed expression, drop the tag gate, put an `if:` on the gate
# step and on its job, and make the refusal suite non-blocking or conditional; each must turn this guard red.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL="${1:-$ROOT/.github/workflows/release.yml}"
DRY="${2:-$ROOT/.github/workflows/release-dry-run.yml}"
ALLOWED="continue-on-error: \${{ github.event_name == 'workflow_dispatch' && inputs.dry_run }}"

# One line per step: "<continue-on-error value or NONE>\t<IF or NOIF>\t<step text, joined>". A step starts at any
# "- key:" list item and ends at the next step, a job key (two-space indent) or a top-level key. IF means the step
# carries its own `if:` condition.
steps() {
    awk '
        function flush() { if (inside) printf "%s\t%s\t%s\n", (coe == "" ? "NONE" : coe), (cond ? "IF" : "NOIF"), run
            inside = 0; coe = ""; run = ""; cond = 0 }
        /^[[:space:]]*- [a-zA-Z_-]+:/ { flush(); inside = 1 }
        /^([a-z]|  [a-z])[a-zA-Z_-]*:/ { flush() }
        inside && /^[[:space:]]*(- )?if:/ { cond = 1 }
        inside && /^[[:space:]]*continue-on-error:/ { sub(/^[[:space:]]*/, ""); coe = $0; next }
        inside { run = run " " $0 }
        END { flush() }
    ' "$1"
}

# Jobs (two-space key under jobs:) that carry a job-level `if:` (four-space indent) and run the release gate.
conditional_gate_jobs() {
    awk '
        function flush() { if (job != "" && cond && gate) print job; job = ""; cond = 0; gate = 0 }
        /^[a-z]/ { flush(); injobs = ($0 ~ /^jobs:/); next }
        injobs && /^  [a-zA-Z_-]+:/ { flush(); job = $1; next }
        injobs && /^    if:/ { cond = 1 }
        injobs && /check-release-candidate\.sh/ { gate = 1 }
        END { flush() }
    ' "$1"
}

check_release() {
    local f="$1" gates bad line
    gates="$(steps "$f" | grep -c 'check-release-candidate\.sh' || true)"
    [[ "$gates" -ge 2 ]] || { echo "FAIL: $f runs check-release-candidate.sh in $gates step(s), expected the tree and package gates" >&2; return 1; }
    # every continue-on-error must be exactly the manual dry-run expression: no other value, no trailing comment
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
        [[ "$line" == "$ALLOWED" ]] || { echo "FAIL: $f has a continue-on-error other than the manual dry run: $line" >&2; return 1; }
    done < <(grep -E '^[[:space:]]*continue-on-error:' "$f" || true)
    # a gate step, or the job around it, must not carry a condition that can skip it
    bad="$(steps "$f" | awk -F'\t' '$2 == "IF" && $3 ~ /check-release-candidate\.sh/' || true)"
    [[ -z "$bad" ]] || { echo "FAIL: $f has an if: on a release gate step: $bad" >&2; return 1; }
    bad="$(conditional_gate_jobs "$f")"
    [[ -z "$bad" ]] || { echo "FAIL: $f has a job-level if: on a job that runs the release gate: $bad" >&2; return 1; }
    grep -qE '^[[:space:]]+tags:' "$f" || { echo "FAIL: $f no longer runs on tag push" >&2; return 1; }
}

check_dry() {
    local f="$1" coe cond run
    local saw_tree=0 saw_suite=0 saw_guard=0
    while IFS=$'\t' read -r coe cond run; do
        if [[ "$run" == *test-release-real-tree.sh* ]]; then
            saw_suite=1; [[ "$coe" == NONE && "$cond" == NOIF ]] || { echo "FAIL: $f refusal suite is not blocking ($coe, $cond)" >&2; return 1; }
        elif [[ "$run" == *test-dry-run-workflow.sh* || "$run" == *test-release-gate-wiring.sh* ]]; then
            saw_guard=1; [[ "$coe" == NONE && "$cond" == NOIF ]] || { echo "FAIL: $f static guard is not blocking ($coe, $cond)" >&2; return 1; }
        elif [[ "$run" == *check-release-candidate.sh* ]]; then
            saw_tree=1
        elif [[ "$coe" != NONE ]]; then
            echo "FAIL: $f has a non-blocking step other than the tree report: $run" >&2; return 1
        fi
    done < <(steps "$f")
    [[ $saw_tree == 1 && $saw_suite == 1 && $saw_guard == 1 ]] || { echo "FAIL: $f lacks the tree report, refusal suite or static guard" >&2; return 1; }
}

check_release "$REL"
check_dry "$DRY"

# self-checks: each mutation of a copy must apply (differ from the original) and be refused
M="$(mktemp -d)"; trap 'rm -rf "$M"' EXIT
n=0
must_refuse() { local what="$1" kind="$2" src="$3" mut="$4"
    cmp -s "$src" "$mut" && { echo "FAIL: mutation did not apply ($what); update this guard" >&2; exit 1; }
    if "$kind" "$mut" 2>/dev/null; then echo "FAIL: guard does not catch: $what" >&2; exit 1; fi
    n=$((n + 1)); }
sed '/^[[:space:]]*id: gate$/a\        continue-on-error: true' "$REL" > "$M/r1"
must_refuse "continue-on-error: true on the tag gate" check_release "$REL" "$M/r1"
sed "s/github.event_name == 'workflow_dispatch' \&\& inputs.dry_run/inputs.dry_run || true/" "$REL" > "$M/r2"
must_refuse "weakened dry-run expression" check_release "$REL" "$M/r2"
grep -v 'run: scripts/check-release-candidate.sh$' "$REL" > "$M/r3"
must_refuse "tree gate removed" check_release "$REL" "$M/r3"
sed "/^[[:space:]]*id: gate\$/a\\        if: github.event_name != 'push'" "$REL" > "$M/r4"
must_refuse "if: on the tag gate step" check_release "$REL" "$M/r4"
awk -v allowed="$ALLOWED" 'f == 0 && (i = index($0, allowed)) { $0 = substr($0, 1, i - 1) "continue-on-error: true  # " allowed; f = 1 } 1' "$REL" > "$M/r5"
must_refuse "continue-on-error: true hidden behind a comment holding the allowed text" check_release "$REL" "$M/r5"
sed "/^  build-release:\$/a\\    if: github.event_name != 'push'" "$REL" > "$M/r6"
must_refuse "job-level if: on the job that runs the gate" check_release "$REL" "$M/r6"
sed '/run: bash scripts\/test-release-real-tree.sh/i\        continue-on-error: true' "$DRY" > "$M/d1"
must_refuse "refusal suite made non-blocking" check_dry "$DRY" "$M/d1"
sed '/run: bash scripts\/test-release-real-tree.sh/i\        if: false' "$DRY" > "$M/d2"
must_refuse "refusal suite skipped by if:" check_dry "$DRY" "$M/d2"
echo "ok: tag release gate is fail-closed (only the manual dry run may continue, no condition on a gate step or its job); dry run blocks on the publish guard and the refusal suite; $n mutants refused"
