#!/usr/bin/env bash
# skill-context.sh: pre-edit hook. Points the agent at the skill that governs
# the file it is about to edit. Walks up from the file to the repo root looking
# for skills/*/SKILL.md whose optional frontmatter `paths:` glob matches the
# file (nearest wins); otherwise uses the component skill (the first skill in
# agents.toml with no `paths:`, else the first skill found with no `paths:`).
# Prints the reminder once per directory per session (state under $TMPDIR).
# Never blocks: always exits 0.
#
# Canonical copy: aien-architecture tools/agent-hooks/ (ADR 0038). Every repo in
# the Aienix tree carries an identical copy; edit the canonical copy and recopy.
# The duplication ends when the aienix tree import happens.
#
# Input: the tool call JSON on stdin (Claude Code format, .tool_input.file_path,
# or .tool_input.notebook_path for notebooks; .session_id keys the state file).
set -u
input=$(cat)
python3 -c '
import fnmatch, glob, json, os, re, sys

try:
    ev = json.loads(sys.argv[1])
    ti = ev.get("tool_input", {}) or {}
except Exception:
    sys.exit(0)
path = ti.get("file_path") or ti.get("notebook_path") or ""
if not path:
    sys.exit(0)
path = os.path.abspath(path)
fdir = os.path.dirname(path)

root = None
d = fdir
while True:
    if os.path.exists(os.path.join(d, ".git")):
        root = d
        break
    up = os.path.dirname(d)
    if up == d:
        break
    d = up
if root is None:
    sys.exit(0)

def front_paths(skill_md):
    """Return the list of `paths:` globs from YAML frontmatter, [] if none."""
    try:
        with open(skill_md, encoding="utf-8") as f:
            text = f.read(8192)
    except OSError:
        return []
    m = re.match(r"---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return []
    out, inlist = [], False
    for line in m.group(1).splitlines():
        k = re.match(r"paths:\s*(.*)$", line)
        if k:
            v = k.group(1).strip()
            if v.startswith("["):
                return [x.strip().strip("\"\x27") for x in v.strip("[]").split(",") if x.strip()]
            if v:
                return [v.strip("\"\x27")]
            inlist = True
            continue
        if inlist:
            item = re.match(r"\s*-\s*(.+)$", line)
            if item:
                out.append(item.group(1).strip().strip("\"\x27"))
            elif line.strip():
                inlist = False
    return out

def matches(rel, pat):
    pat = pat[2:] if pat.startswith("./") else pat
    if fnmatch.fnmatch(rel, pat):
        return True
    if pat.endswith("/**") and (rel == pat[:-3] or rel.startswith(pat[:-2])):
        return True
    return "**/" in pat and fnmatch.fnmatch(rel, pat.replace("**/", ""))

found, fallback = None, None
d = fdir
while True:
    for skill_md in sorted(glob.glob(os.path.join(d, "skills", "*", "SKILL.md"))):
        pats = front_paths(skill_md)
        rel = os.path.relpath(path, d)
        if pats and any(matches(rel, p) for p in pats):
            found = skill_md
            break
        if not pats and fallback is None:
            fallback = skill_md
    if found or d == root:
        break
    d = os.path.dirname(d)

if not found:
    try:
        import tomllib
        with open(os.path.join(root, "agents.toml"), "rb") as f:
            for s in tomllib.load(f).get("skills", []):
                cand = os.path.join(root, s.get("path", ""), "SKILL.md")
                if os.path.isfile(cand) and not front_paths(cand):
                    found = cand
                    break
    except Exception:
        pass
found = found or fallback
if not found:
    sys.exit(0)

sid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(ev.get("session_id") or os.getppid()))
state = os.path.join(os.environ.get("TMPDIR") or "/tmp", "aien-skill-context-%s.seen" % sid)
try:
    with open(state, encoding="utf-8") as f:
        if fdir in f.read().splitlines():
            sys.exit(0)
except OSError:
    pass
try:
    with open(state, "a", encoding="utf-8") as f:
        f.write(fdir + "\n")
except OSError:
    pass
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
      "additionalContext": "Read %s before editing here" % os.path.relpath(found, root)}}))
' "$input"
exit 0
