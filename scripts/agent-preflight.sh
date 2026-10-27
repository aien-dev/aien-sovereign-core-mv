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

set -eo pipefail

echo "============================================================"
echo "    AIEN Sovereign Agent Preflight Verification Harness     "
echo "============================================================"

FAILED=0

# Step 1: Branch Guard (Zero direct commits to main)
echo -n "[1/5] Verifying Branch Isolation... "
if bash "$(dirname "$0")/preflight-branch-guard.sh"; then :; else FAILED=1; fi

# Step 2: Zero Disk Secrets Audit
echo -n "[2/5] Auditing Zero Disk Secrets... "
SECRET_FILES=$(find . -maxdepth 3 -name ".env*" -o -name "*.key" -o -name "*secret*" -o -name "*.pem" 2>/dev/null | grep -v ".git" | grep -v "target" || true)
if [ -n "$SECRET_FILES" ]; then
    echo "FAILED"
    echo "Error: Stray plaintext secret files detected on disk:"
    echo "$SECRET_FILES"
    FAILED=1
else
    echo "PASSED (Zero disk secrets detected)"
fi

# Step 3: Unslop and Sovereign Voice Audit on the branch (committed + working changes)
echo -n "[3/5] Verifying Unslop & Anti-Slop Discipline... "
# Diff against the merge base with main so committed branch work is audited,
# not only uncommitted edits. Falls back to HEAD when no main ref exists.
DIFF_BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo HEAD)
DASH_VIOLATIONS=$(git diff "$DIFF_BASE" --':(exclude)*.lock' ':(exclude)*.svg' ':(exclude)*.bin' 2>/dev/null | grep -E '^\+[^+]' | grep -E '(—|–)' | grep -v "replace" | grep -v "contains" | grep -v "ZERO EM DASHES" | grep -v "forbidden" | grep -v "assert\!" || true)
if [ -n "$DASH_VIOLATIONS" ]; then
    echo "FAILED"
    echo "Error: Prohibited em dash or en dash characters introduced in git diff:"
    echo "$DASH_VIOLATIONS"
    FAILED=1
else
    echo "PASSED (Zero em/en dashes in diff)"
fi

# Step 4: Test Suite Verification
# Runs one stamped job per crate through the shared aien-proof board: crates
# whose code and local dependencies are unchanged replay their stamp, identical
# runs from other agents are joined, and real runs pass the CPU, memory, and GPU
# gates. Set AIEN_PROOF_OFF=1 to fall back to a plain workspace test.
echo "[4/5] Running Workspace Test Suite..."
if [ "${AIEN_PROOF_OFF:-0}" = "1" ]; then
    TEST_CMD=(cargo test --workspace)
elif command -v aien-proof >/dev/null 2>&1; then
    TEST_CMD=(aien-proof crates)
else
    TEST_CMD=(cargo run -q -p aien-proof -- crates)
fi
if "${TEST_CMD[@]}"; then
    echo "PASSED (100% test pass)"
else
    echo "FAILED (Test suite failed)"
    FAILED=1
fi

# Step 5: Vulnerability & Advisory Audit
echo -n "[5/5] Checking RustSec Vulnerability Advisories... "
if ! cargo audit --version >/dev/null 2>&1; then
    echo "SKIPPED (cargo-audit not installed)"
elif cargo audit; then
    echo "PASSED (No vulnerabilities; review any warnings above)"
else
    echo "FAILED (Vulnerabilities reported above)"
    FAILED=1
fi

echo "============================================================"
if [ "$FAILED" -eq 0 ]; then
    echo " ✓ PREFLIGHT VERIFICATION COMPLETE: Ready for PR Creation"
    echo "============================================================"
    exit 0
else
    echo " ✗ PREFLIGHT FAILED: Resolve defects before opening PR"
    echo "============================================================"
    exit 1
fi
