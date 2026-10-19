#!/usr/bin/env bash
# protected-paths.sh: pre-edit hook. Blocks edits to pipeline and doctrine
# state: .crumb, crumb.lock, .aien/, tests/gates/, .github/workflows/,
# .forgejo/ and tools/aien-test/, unless AIEN_PIPELINE_EDIT=1 is set.
# Rule A4 (ADR 0033, amendment 2026-10-11): a change to the pipeline's own
# configuration is BLOCKED_OPERATOR; the human operator runs it by hand.
# .crumb.local is temporary coordination state and stays editable.
#
# Canonical copy: aien-architecture tools/agent-hooks/ (ADR 0038). Every repo in
# the Aienix tree carries an identical copy; edit the canonical copy and recopy.
# The duplication ends when the aienix tree import happens.
#
# Input: the tool call JSON on stdin (Claude Code format, .tool_input.file_path,
# or .tool_input.notebook_path for notebooks).
# Exit 0 = allow, exit 2 = block with a one-line reason on stderr.
set -u
[ "${AIEN_PIPELINE_EDIT:-}" = "1" ] && exit 0
input=$(cat)
case "$input" in *crumb*|*.aien*|*gates*|*.github*|*.forgejo*|*aien-test*) ;; *) exit 0 ;; esac
python3 -c '
import json, os, sys

try:
    ti = json.loads(sys.argv[1]).get("tool_input", {}) or {}
except Exception:
    sys.exit(0)
path = ti.get("file_path") or ti.get("notebook_path") or ""
if not path:
    sys.exit(0)
path = os.path.abspath(path)

# Make the path relative to its repository root (nearest ancestor holding .git).
d, rel = os.path.dirname(path), path
while True:
    if os.path.exists(os.path.join(d, ".git")):
        rel = os.path.relpath(path, d)
        break
    up = os.path.dirname(d)
    if up == d:
        break
    d = up

parts = [p for p in rel.split(os.sep) if p]
pairs = list(zip(parts, parts[1:]))
hit = (
    (parts and parts[-1] in (".crumb", "crumb.lock"))
    or ".aien" in parts[:-1]
    or ".forgejo" in parts[:-1]
    or ("tests", "gates") in pairs
    or (".github", "workflows") in pairs
    or ("tools", "aien-test") in pairs
)
if hit:
    print("BLOCKED (rule A4, ADR 0033): %s is pipeline or doctrine state; changes are BLOCKED_OPERATOR. Leave it to the human operator, who sets AIEN_PIPELINE_EDIT=1 for a deliberate pipeline change." % rel, file=sys.stderr)
    sys.exit(2)
' "$input"
