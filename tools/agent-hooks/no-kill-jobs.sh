#!/usr/bin/env bash
# no-kill-jobs.sh: pre-command hook. Blocks kill, pkill and killall aimed at
# running jobs: aien-test, aien-serve-http, aien-local-ci and `claude -p`
# workers (standing rule: never kill running jobs). A bare `claude` target for
# pkill/killall is blocked too, since it would hit the `claude -p` workers.
# For `kill PID` the PID's command line is read from /proc.
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
case "$input" in *kill*) ;; *) exit 0 ;; esac
python3 -c '
import json, os, re, shlex, sys

PROTECTED = re.compile(r"aien-test|aien-serve-http|aien-local-ci|claude\s+(\S+\s+)*-p\b")
KILLERS = {"kill", "pkill", "killall"}

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

def pid_cmdline(pid):
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""

def bad(cmd, level=0):
    for seg in segments(cmd):
        for i, t in enumerate(seg):
            if level < 3 and " " in t and i > 0 and seg[i - 1] in ("-c", "eval"):
                if bad(t, level + 1):
                    return True
            name = os.path.basename(t)
            if name not in KILLERS:
                continue
            rest = seg[i + 1:]
            if PROTECTED.search(" ".join(rest)):
                return True
            for a in rest:
                if name != "kill" and a == "claude":
                    return True
                if name == "kill" and re.fullmatch(r"-?\d+", a) and not (a.startswith("-") and len(a) <= 3):
                    if PROTECTED.search(pid_cmdline(a.lstrip("-"))):
                        return True
    return False

try:
    cmd = json.loads(sys.argv[1]).get("tool_input", {}).get("command", "") or ""
except Exception:
    sys.exit(0)
if bad(cmd):
    print("BLOCKED: never kill running jobs (aien-test, aien-serve-http, aien-local-ci, claude -p workers). Wait for them or ask the human owner.", file=sys.stderr)
    sys.exit(2)
' "$input"
