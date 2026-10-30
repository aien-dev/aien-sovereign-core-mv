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

# Static guard for the offline-signing handoff (Drake ruling 1, sc#385): no private release key in CI, ever.
#   scripts/test-release-signing-workflow.sh [release.yml] [release-publish.yml]
# release.yml: no AIEN_RELEASE_SIGNING_KEY and no secret other than GITHUB_TOKEN; `ssh-keygen -Y sign` only inside the
#   dry-run-sign job (a throwaway key made in that job), which runs only on a manual dry run; every `gh release create`
#   makes a --draft; release.yml never publishes, edits or uploads to a release.
# release-publish.yml: manual trigger only; no secret other than GITHUB_TOKEN; never signs; the publish gate
#   (scripts/release-publish-gate.sh) runs before the only `gh release edit`; nothing in it can be skipped or made
#   non-blocking (no if:, no continue-on-error); the tag input reaches shell only through env.
# Mutation note: the self-checks below inject each forbidden pattern into a copy; each must turn this guard red.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL="${1:-$ROOT/.github/workflows/release.yml}"
PUB="${2:-$ROOT/.github/workflows/release-publish.yml}"
err() { echo "FAIL: $*" >&2; return 1; }
code() { grep -nvE '^[[:space:]]*#' "$1" || true; }   # numbered lines that are not comments
# job_lines <file> <pattern>: "<job>\t<line>" for each non-comment line matching pattern, inside jobs:
job_lines() {
    awk -v pat="$2" '
        /^[a-z]/ { injobs = ($0 ~ /^jobs:/); job = ""; next }
        injobs && /^  [a-zA-Z_-]+:[[:space:]]*$/ { job = $1; sub(/:$/, "", job); next }
        $0 ~ /^[[:space:]]*#/ { next }
        $0 ~ pat { printf "%s\t%s\n", job, $0 }' "$1"
}
check_rel() {
    local f="$1" bad
    ! grep -q 'AIEN_RELEASE_SIGNING_KEY' "$f" || { err "$f names AIEN_RELEASE_SIGNING_KEY"; return 1; }
    bad="$(code "$f" | grep -E 'secrets\.' | grep -vE 'secrets\.GITHUB_TOKEN' || true)"
    [[ -z "$bad" ]] || { err "$f uses a secret other than GITHUB_TOKEN: $bad"; return 1; }
    bad="$(job_lines "$f" 'ssh-keygen[[:space:]].*-Y[[:space:]]+sign' | awk -F'\t' '$1 != "dry-run-sign"' || true)"
    [[ -z "$bad" ]] || { err "$f signs outside the dry-run-sign job: $bad"; return 1; }
    job_lines "$f" '^    if:' | grep -E '^dry-run-sign	' | grep -q "github.event_name == 'workflow_dispatch' && inputs.dry_run$" \
        || { err "$f dry-run-sign job is not limited to a manual dry run"; return 1; }
    bad="$(code "$f" | grep -E 'gh release create' | grep -vE -- '--draft( |$)' || true)"
    [[ -z "$bad" ]] || { err "$f creates a release that is not a draft: $bad"; return 1; }
    bad="$(code "$f" | grep -E -- '--draft=false|gh release (edit|upload)|gh api .*releases|action-gh-release|upload-release' || true)"
    [[ -z "$bad" ]] || { err "$f publishes, edits or uploads to a release: $bad"; return 1; }
}
check_pub() {
    local f="$1" bad gate edit
    awk '/^on:/ { on = 1; next } on && /^[a-z]/ { on = 0 } on && /^  [a-z_]+:/ { print $1 }' "$f" | grep -vxq 'workflow_dispatch:' \
        && { err "$f has a trigger other than workflow_dispatch"; return 1; }
    bad="$(code "$f" | grep -E 'secrets\.' | grep -vE 'secrets\.GITHUB_TOKEN' || true)"
    [[ -z "$bad" ]] || { err "$f uses a secret other than GITHUB_TOKEN: $bad"; return 1; }
    ! code "$f" | grep -qE 'ssh-keygen[[:space:]].*-Y[[:space:]]+sign' || { err "$f signs"; return 1; }
    ! code "$f" | grep -qE '^[0-9]+:[[:space:]]*(- )?(if|continue-on-error):' || { err "$f has an if: or continue-on-error"; return 1; }
    gate="$(code "$f" | grep -E 'run: bash scripts/release-publish-gate\.sh release-assets docs/release/allowed_signers$' | head -1 | cut -d: -f1)"
    edit="$(code "$f" | grep -E 'gh release edit' | cut -d: -f1)"
    [[ -n "$gate" ]] || { err "$f does not run the publish gate against docs/release/allowed_signers"; return 1; }
    [[ "$(printf '%s\n' "$edit" | grep -c .)" -eq 1 && "$edit" -gt "$gate" ]] || { err "$f publishes before (or without) the gate"; return 1; }
    bad="$(code "$f" | grep -E '\$\{\{ *inputs\.' | grep -vE '^[0-9]+:[[:space:]]+(TAG|ref):' || true)"
    [[ -z "$bad" ]] || { err "$f interpolates an input outside env/ref: $bad"; return 1; }
}
check_rel "$REL"
check_pub "$PUB"

M="$(mktemp -d)"; trap 'rm -rf "$M"' EXIT
n=0
must_refuse() { local what="$1" kind="$2" src="$3" mut="$4"
    cmp -s "$src" "$mut" && { echo "FAIL: mutation did not apply ($what); update this guard" >&2; exit 1; }
    if "$kind" "$mut" 2>/dev/null; then echo "FAIL: guard does not catch: $what" >&2; exit 1; fi
    n=$((n + 1)); }
sed '/^      - name: Generate Checksums$/i\      - name: sign\n        env:\n          K: ${{ secrets.AIEN_RELEASE_SIGNING_KEY }}\n        run: echo' "$REL" > "$M/r1"
must_refuse "release key secret back in release.yml" check_rel "$REL" "$M/r1"
sed '/^      - name: Generate Checksums$/i\      - name: sign\n        env:\n          K: ${{ secrets.SOME_KEY }}\n        run: echo' "$REL" > "$M/r2"
must_refuse "another secret in release.yml" check_rel "$REL" "$M/r2"
sed 's/^          sha256sum sovereign-\* aien-cli-digests.txt > SHA256SUMS.txt$/&\n          ssh-keygen -Y sign -f k -n aien-release SHA256SUMS.txt/' "$REL" > "$M/r3"
must_refuse "signing in the draft-release job" check_rel "$REL" "$M/r3"
sed 's/gh release create "$TAG" --draft --verify-tag/gh release create "$TAG" --verify-tag/' "$REL" > "$M/r4"
must_refuse "non-draft gh release create" check_rel "$REL" "$M/r4"
sed 's/^            --notes-file ..\/release-notes-with-digests.md$/&\n          gh release edit "$TAG" --draft=false/' "$REL" > "$M/r5"
must_refuse "release.yml publishing the draft" check_rel "$REL" "$M/r5"
sed "s/^    if: github.event_name == 'workflow_dispatch' \&\& inputs.dry_run\$/    if: always()/" "$REL" > "$M/r6"
must_refuse "dry-run-sign job runs outside a manual dry run" check_rel "$REL" "$M/r6"
sed 's/^  workflow_dispatch:$/  push:\n    tags: ["v*"]\n&/' "$PUB" > "$M/p1"
must_refuse "publish workflow on push" check_pub "$PUB" "$M/p1"
sed 's/^        run: bash scripts\/release-publish-gate.sh release-assets docs\/release\/allowed_signers$/        continue-on-error: true\n&/' "$PUB" > "$M/p2"
must_refuse "publish gate made non-blocking" check_pub "$PUB" "$M/p2"
grep -v 'run: bash scripts/release-publish-gate.sh' "$PUB" > "$M/p3"
must_refuse "publish gate removed" check_pub "$PUB" "$M/p3"
sed 's/^      - name: Check the tag name$/      - name: early\n        run: gh release edit "$TAG" --draft=false\n&/' "$PUB" > "$M/p4"
must_refuse "publish before the gate" check_pub "$PUB" "$M/p4"
sed 's/^      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}$/&\n      K: ${{ secrets.AIEN_RELEASE_SIGNING_KEY }}/' "$PUB" > "$M/p5"
must_refuse "release key secret in the publish workflow" check_pub "$PUB" "$M/p5"
sed 's/^        run: gh release edit "$TAG" --draft=false$/        run: echo ${{ inputs.tag }}; gh release edit "$TAG" --draft=false/' "$PUB" > "$M/p6"
must_refuse "input interpolated into a run line" check_pub "$PUB" "$M/p6"
sed 's/^        run: bash scripts\/release-publish-gate.sh release-assets docs\/release\/allowed_signers$/        run: bash scripts\/release-publish-gate.sh release-assets release-assets\/dry-run-signers/' "$PUB" > "$M/p7"
must_refuse "publish gate pointed at a signer file other than allowed_signers" check_pub "$PUB" "$M/p7"
echo "ok: no release key in CI; release.yml signs only in the manual dry run and only drafts; release-publish.yml is manual, keyless and publishes only after the blocking publish gate; $n mutants refused"
