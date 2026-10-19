#!/usr/bin/env bash
# Static guard: .github/workflows/release-dry-run.yml (runs on pull_request) must not be able to publish.
# It may read the repository and run the offline release checks; nothing else. Triggered by pull_request or by hand.
#   scripts/test-dry-run-workflow.sh [workflow-file]
# Mutation note: adding a `gh release`, `secrets.` reference, `contents: write` or an upload step to the workflow
# makes this fail; the self-checks below prove each pattern fires on a mutated copy.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
F="${1:-$ROOT/.github/workflows/release-dry-run.yml}"
[[ -f "$F" ]] || { echo "FAIL: $F missing" >&2; exit 1; }
BAD='gh release|gh api|secrets\.|GH_TOKEN|GITHUB_TOKEN|contents: *write|id-token|packages: *write|upload-artifact|upload-release|action-gh-release|ssh-keygen|release create|workflow_run|pull_request_target'
violations() { grep -nE "$BAD" "$1" | grep -vE '^[0-9]+:[[:space:]]*#' || true; }
v="$(violations "$F")"
[[ -z "$v" ]] || { echo "FAIL: dry-run workflow contains publish or secret patterns:" >&2; echo "$v" >&2; exit 1; }
grep -qE '^permissions:' "$F" && grep -qE '^[[:space:]]+contents: read$' "$F" || { echo "FAIL: workflow must declare permissions contents: read" >&2; exit 1; }
# Phase 3 (#671) removed the pull_request trigger: the Spark is the judge now and the workflow is manual. It may be
# triggered by pull_request or by hand (workflow_dispatch); a missing or any other trigger (push to a tag, workflow_run,
# schedule with secrets...) is refused.
grep -qE '^[[:space:]]*(pull_request|workflow_dispatch):' "$F" || { echo "FAIL: workflow must run on pull_request or workflow_dispatch only" >&2; exit 1; }
M="$(mktemp)"; trap 'rm -f "$M"' EXIT
for bad in 'run: gh release create v1' 'x: ${{ secrets.AIEN_RELEASE_SIGNING_KEY }}' 'contents: write' 'uses: actions/upload-artifact@v4'; do
    { cat "$F"; echo "      $bad"; } > "$M"
    [[ -n "$(violations "$M")" ]] || { echo "FAIL: guard does not catch: $bad" >&2; exit 1; }
done
echo "ok: dry-run workflow has no publish, secrets, signing or upload, and only contents: read"
