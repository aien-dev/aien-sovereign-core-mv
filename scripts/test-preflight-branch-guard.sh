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

# Verify the branch guard in both contexts using throwaway git repositories.
set -euo pipefail
GUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preflight-branch-guard.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aien-guard-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
git init -q -b main . && git -c user.name=t -c user.email=t@t commit -q --allow-empty -m init

expect() { # want(0|1) description env...
    local want="$1" desc="$2"; shift 2
    local rc=0
    env -u AIEN_PREFLIGHT_CONTEXT -u GITHUB_ACTIONS -u GITHUB_REF "$@" bash "$GUARD" >/dev/null 2>&1 || rc=$?
    [ "$rc" -eq "$want" ] || { echo "FAIL: $desc (exit $rc, wanted $want)"; exit 1; }
    echo "ok: $desc"
}

expect 1 "developer on main is refused"            AIEN_PREFLIGHT_CONTEXT=developer
expect 1 "default context on main is refused"      AIEN_PREFLIGHT_NONE=1
expect 0 "merged-tree on CI runner for main is accepted" AIEN_PREFLIGHT_CONTEXT=merged-tree GITHUB_ACTIONS=true GITHUB_REF=refs/heads/main
expect 1 "merged-tree from a local shell is refused"     AIEN_PREFLIGHT_CONTEXT=merged-tree
expect 1 "merged-tree on CI for a PR ref is refused"     AIEN_PREFLIGHT_CONTEXT=merged-tree GITHUB_ACTIONS=true GITHUB_REF=refs/pull/1/merge
git checkout -q -b feat/x
expect 0 "developer on feature branch is accepted" AIEN_PREFLIGHT_CONTEXT=developer
expect 1 "unknown context is refused on a feature branch" AIEN_PREFLIGHT_CONTEXT=mereged-tree
git checkout -q --detach
expect 0 "developer on detached PR merge ref is accepted" AIEN_PREFLIGHT_CONTEXT=developer
echo "PREFLIGHT_BRANCH_GUARD_TESTS_PASS"
