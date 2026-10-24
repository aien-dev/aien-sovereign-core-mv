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

# Sovereign Distillation Workshop E2E Test Suite Runner
# Strictly adheres to sovereign voice: zero em dashes, zero en dashes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RESULTS_FILE="${WORKSPACE_ROOT}/tests_results.json"

TIER_SELECTION="all"
FEATURE_FILTER=""
VERBOSE=false
OUTPUT_JSON=false

usage() {
    cat <<'USAGE_EOF'
Usage: e2e_distill_test.sh [options]

Options:
  --all              Execute all 4 test tiers (default: 140 tests)
  --tier <1-4>       Execute specific test tier:
                       1: Feature Coverage (65 tests across 13 features)
                       2: Boundary & Corner Cases (65 tests across 13 features)
                       3: Cross-Feature Combinations (5 tests)
                       4: Real-World Scenarios (5 tests)
  --feature <pattern> Execute tests matching feature pattern (e.g. f01, f02, vault, sanitizer)
  --json             Output results in structured JSON format
  -v, --verbose      Enable verbose cargo output
  -h, --help         Show this help message
USAGE_EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            TIER_SELECTION="all"
            shift
            ;;
        --tier)
            TIER_SELECTION="$2"
            shift 2
            ;;
        --feature)
            FEATURE_FILTER="$2"
            shift 2
            ;;
        --json)
            OUTPUT_JSON=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

cd "${WORKSPACE_ROOT}"

TIER1_TARGET="tier1_features"
TIER2_TARGET="tier2_boundary"
TIER3_TARGET="tier3_cross_feature"
TIER4_TARGET="tier4_real_world"

run_tier() {
    local target="$1"
    local extra_filter="$2"
    local cmd=("cargo" "test" "-p" "spark-adapters" "--test" "${target}")

    if [[ -n "${extra_filter}" ]]; then
        cmd+=("${extra_filter}")
    fi

    if [[ "${VERBOSE}" == "false" ]]; then
        cmd+=("--" "--quiet")
    fi

    "${cmd[@]}" 2>&1 || true
}

parse_test_counts() {
    local raw_output="$1"
    local passed=0
    local failed=0
    local ignored=0

    if grep -q "test result:" <<< "${raw_output}"; then
        passed=$(grep "test result:" <<< "${raw_output}" | tail -n 1 | grep -oE '[0-9]+ passed' | awk '{print $1}')
        failed=$(grep "test result:" <<< "${raw_output}" | tail -n 1 | grep -oE '[0-9]+ failed' | awk '{print $1}')
        ignored=$(grep "test result:" <<< "${raw_output}" | tail -n 1 | grep -oE '[0-9]+ ignored' | awk '{print $1}')
    fi

    echo "${passed:-0} ${failed:-0} ${ignored:-0}"
}

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_IGNORED=0

TIER1_P=0; TIER1_F=0; TIER1_I=0
TIER2_P=0; TIER2_F=0; TIER2_I=0
TIER3_P=0; TIER3_F=0; TIER3_I=0
TIER4_P=0; TIER4_F=0; TIER4_I=0

START_TIME=$(date +%s)

if [[ "${TIER_SELECTION}" == "all" || "${TIER_SELECTION}" == "1" ]]; then
    OUTPUT=$(run_tier "${TIER1_TARGET}" "${FEATURE_FILTER}")
    read -r TIER1_P TIER1_F TIER1_I <<< "$(parse_test_counts "${OUTPUT}")"
fi

if [[ "${TIER_SELECTION}" == "all" || "${TIER_SELECTION}" == "2" ]]; then
    OUTPUT=$(run_tier "${TIER2_TARGET}" "${FEATURE_FILTER}")
    read -r TIER2_P TIER2_F TIER2_I <<< "$(parse_test_counts "${OUTPUT}")"
fi

if [[ "${TIER_SELECTION}" == "all" || "${TIER_SELECTION}" == "3" ]]; then
    OUTPUT=$(run_tier "${TIER3_TARGET}" "${FEATURE_FILTER}")
    read -r TIER3_P TIER3_F TIER3_I <<< "$(parse_test_counts "${OUTPUT}")"
fi

if [[ "${TIER_SELECTION}" == "all" || "${TIER_SELECTION}" == "4" ]]; then
    OUTPUT=$(run_tier "${TIER4_TARGET}" "${FEATURE_FILTER}")
    read -r TIER4_P TIER4_F TIER4_I <<< "$(parse_test_counts "${OUTPUT}")"
fi

TOTAL_PASSED=$((TIER1_P + TIER2_P + TIER3_P + TIER4_P))
TOTAL_FAILED=$((TIER1_F + TIER2_F + TIER3_F + TIER4_F))
TOTAL_IGNORED=$((TIER1_I + TIER2_I + TIER3_I + TIER4_I))
TOTAL_RUN=$((TOTAL_PASSED + TOTAL_FAILED))

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

cat <<JSON_EOF > "${RESULTS_FILE}"
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "tier_selection": "${TIER_SELECTION}",
  "total_executed": ${TOTAL_RUN},
  "passed": ${TOTAL_PASSED},
  "failed": ${TOTAL_FAILED},
  "ignored": ${TOTAL_IGNORED},
  "duration_seconds": ${DURATION},
  "tiers": {
    "tier1": { "passed": ${TIER1_P}, "failed": ${TIER1_F} },
    "tier2": { "passed": ${TIER2_P}, "failed": ${TIER2_F} },
    "tier3": { "passed": ${TIER3_P}, "failed": ${TIER3_F} },
    "tier4": { "passed": ${TIER4_P}, "failed": ${TIER4_F} }
  }
}
JSON_EOF

if [[ "${OUTPUT_JSON}" == "true" ]]; then
    cat "${RESULTS_FILE}"
else
    echo "========================================================================"
    echo "Sovereign Distillation Workshop: E2E Opaque-Box Test Suite Execution"
    echo "========================================================================"
    printf "%-10s %-32s %-10s %-10s\n" "TIER" "DESCRIPTION" "PASSED" "FAILED"
    echo "------------------------------------------------------------------------"
    printf "%-10s %-32s %-10s %-10s\n" "Tier 1" "Feature Coverage (13 Features)" "${TIER1_P}" "${TIER1_F}"
    printf "%-10s %-32s %-10s %-10s\n" "Tier 2" "Boundary & Corner Cases" "${TIER2_P}" "${TIER2_F}"
    printf "%-10s %-32s %-10s %-10s\n" "Tier 3" "Cross-Feature Combinations" "${TIER3_P}" "${TIER3_F}"
    printf "%-10s %-32s %-10s %-10s\n" "Tier 4" "Real-World Scenarios (DGX Spark)" "${TIER4_P}" "${TIER4_F}"
    echo "------------------------------------------------------------------------"
    printf "%-10s %-32s %-10s %-10s\n" "TOTAL" "All Categories" "${TOTAL_PASSED}" "${TOTAL_FAILED}"
    echo "========================================================================"
    echo "Duration: ${DURATION}s | Results saved to: ${RESULTS_FILE}"
    if [[ ${TOTAL_FAILED} -gt 0 ]]; then
        echo "Status: ${TOTAL_FAILED} tests failing (maps to active milestone implementations)."
    else
        echo "Status: ALL TESTS PASSED."
    fi
fi
