#!/usr/bin/env bash
# no-force-push.sh: pre-command hook. Blocks `git push` with --force, -f (also
# inside a short-flag cluster such as -uf) or a +refspec, unless
# --force-with-lease is present.
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
case "$input" in *git*push*) ;; *) exit 0 ;; esac
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

# git options that take a separate argument before the subcommand
GIT_ARG_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}

def push_args(seg, i):
    j = i + 1
    while j < len(seg) and seg[j].startswith("-"):
        j += 2 if seg[j] in GIT_ARG_OPTS else 1
    if j < len(seg) and seg[j] == "push":
        return seg[j + 1:]
    return None

def is_force(args):
    if any(a.startswith("--force-with-lease") for a in args):
        return False
    for a in args:
        if a == "--force" or a.startswith("--force="):
            return True
        if a.startswith("-") and not a.startswith("--") and "f" in a[1:]:
            return True
        if a.startswith("+") and len(a) > 1:
            return True
    return False

def bad(cmd, level=0):
    for seg in segments(cmd):
        for i, t in enumerate(seg):
            if level < 3 and " " in t and i > 0 and seg[i - 1] in ("-c", "eval"):
                if bad(t, level + 1):
                    return True
            if os.path.basename(t) == "git":
                args = push_args(seg, i)
                if args is not None and is_force(args):
                    return True
    return False

try:
    cmd = json.loads(sys.argv[1]).get("tool_input", {}).get("command", "") or ""
except Exception:
    sys.exit(0)
if bad(cmd):
    print("BLOCKED: force push rewrites shared history. Push normally, or use --force-with-lease on your own branch.", file=sys.stderr)
    sys.exit(2)
' "$input"
