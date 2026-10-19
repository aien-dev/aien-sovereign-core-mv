#!/usr/bin/env bash
# no-admin-merge.sh: pre-command hook. Blocks `gh pr merge ... --admin`.
#
# Canonical copy: aien-architecture tools/agent-hooks/ (ADR 0038). Every repo in
# the Aienix tree carries an identical copy; edit the canonical copy and recopy.
# The duplication ends when the aienix tree import happens.
#
# Input: the tool call JSON on stdin (Claude Code format, .tool_input.command).
# Exit 0 = allow, exit 2 = block with a one-line reason on stderr.
# Guards against mistakes, not against a determined adversary.
set -u
input=$(cat)
case "$input" in *gh*admin*) ;; *) exit 0 ;; esac
python3 -c '
import json, os, shlex, sys

def segments(cmd):
    lx = shlex.shlex(cmd, posix=True, punctuation_chars=";&|()\n")
    lx.whitespace = " \t\r"
    lx.wordchars += "+:@%^,"
    try:
        toks = list(lx)
    except ValueError:
        toks = cmd.split()
    seg, depth = [], 0
    for t in toks:
        if t == "(":
            depth += 1
        elif t == ")":
            depth = max(0, depth - 1)
        if depth == 0 and t and all(c in ";&|\n" for c in t):
            if seg:
                yield seg
            seg = []
        else:
            seg.append(t)
    if seg:
        yield seg

def bad(cmd, level=0):
    for seg in segments(cmd):
        for i, t in enumerate(seg):
            if level < 3 and " " in t and i > 0 and seg[i - 1] in ("-c", "eval"):
                if bad(t, level + 1):
                    return True
            if os.path.basename(t) == "gh" and seg[i + 1:i + 3] == ["pr", "merge"]:
                if any(a == "--admin" or a.startswith("--admin=") for a in seg[i + 3:]):
                    return True
    return False

try:
    cmd = json.loads(sys.argv[1]).get("tool_input", {}).get("command", "") or ""
except Exception:
    sys.exit(0)
if bad(cmd):
    print("BLOCKED: gh pr merge --admin bypasses branch protection. Merge only after the required checks pass; never use --admin.", file=sys.stderr)
    sys.exit(2)
' "$input"
