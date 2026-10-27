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

set -euo pipefail

# AIEN Sovereign Core E2E Test Suite Runner
# Executes all 4 tiers: Feature Coverage, Boundaries, Interactions, Real-World Workflows

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "AIEN Sovereign Core: TinyLlama Real-Model Execution E2E Test Suite"
echo "================================================================="
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Manifest:  ${SCRIPT_DIR}/Cargo.toml"
echo "Target:    ${WORKSPACE_ROOT}/target"
echo ""

echo ">>> Phase 1: Checking Build & Dependency Compilation..."
cargo check --manifest-path "${SCRIPT_DIR}/Cargo.toml" --target-dir "${WORKSPACE_ROOT}/target"

echo ""
echo ">>> Phase 2: Executing Tier 1 (Feature Coverage: 90 tests)..."
cargo test --manifest-path "${SCRIPT_DIR}/Cargo.toml" --target-dir "${WORKSPACE_ROOT}/target" --test test_tier1_feature_coverage -- --nocapture

echo ""
echo ">>> Phase 3: Executing Tier 2 (Boundary & Corner Cases: 90 tests)..."
cargo test --manifest-path "${SCRIPT_DIR}/Cargo.toml" --target-dir "${WORKSPACE_ROOT}/target" --test test_tier2_boundaries -- --nocapture

echo ""
echo ">>> Phase 4: Executing Tier 3 (Cross-Feature Interactions: 18 tests)..."
cargo test --manifest-path "${SCRIPT_DIR}/Cargo.toml" --target-dir "${WORKSPACE_ROOT}/target" --test test_tier3_cross_feature -- --nocapture

echo ""
echo ">>> Phase 5: Executing Tier 4 (Real-World Application Scenarios: 9 tests)..."
cargo test --manifest-path "${SCRIPT_DIR}/Cargo.toml" --target-dir "${WORKSPACE_ROOT}/target" --test test_tier4_real_world -- --nocapture

echo ""
echo "================================================================="
echo "All 207 E2E tests passed successfully across all 4 tiers."
echo "Status: TEST_READY certified."
echo "================================================================="
