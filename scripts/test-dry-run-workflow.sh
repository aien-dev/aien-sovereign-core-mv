#!/usr/bin/env bash
# Static guard: .github/workflows/release-dry-run.yml (pull_request or manual workflow_dispatch) must not be able to publish.
# It may read the repository and run the offline release checks; nothing else.
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
# Triggers: since Phase 3 (#671) the hosted workflow is manual (workflow_dispatch); the Spark judge runs this guard.
# The workflow may be triggered only by pull_request or workflow_dispatch, never by push, schedule or anything that
# runs with write power; it must declare at least one of the two.
grep -qE '^[[:space:]]*(pull_request|workflow_dispatch):' "$F" || { echo "FAIL: workflow must run on pull_request or workflow_dispatch" >&2; exit 1; }
trig="$(awk '/^on:/{f=1;next} f&&/^[^[:space:]#]/{f=0} f&&/^  [A-Za-z_]+:/{sub(/:.*/,"");gsub(/ /,"");print}' "$F")"
for t in $trig; do
    case "$t" in pull_request|workflow_dispatch) ;; *) echo "FAIL: workflow trigger not allowed: $t" >&2; exit 1 ;; esac
done
M="$(mktemp)"; trap 'rm -f "$M"' EXIT
for bad in 'run: gh release create v1' 'x: ${{ secrets.AIEN_RELEASE_SIGNING_KEY }}' 'contents: write' 'uses: actions/upload-artifact@v4'; do
    { cat "$F"; echo "      $bad"; } > "$M"
    [[ -n "$(violations "$M")" ]] || { echo "FAIL: guard does not catch: $bad" >&2; exit 1; }
done
for trigger in push: schedule: workflow_run:; do
    awk -v t="$trigger" '{print} /^on:/{print "  " t}' "$F" > "$M"
    bash "${BASH_SOURCE[0]}" "$M" >/dev/null 2>&1 && { echo "FAIL: guard does not catch trigger: $trigger" >&2; exit 1; }
done
echo "ok: dry-run workflow has no publish, secrets, signing or upload, and only contents: read"
