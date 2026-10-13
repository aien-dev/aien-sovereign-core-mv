#!/bin/sh
# calc-guard.sh -- Claude Code PreToolUse hook (Write, Edit, MultiEdit). Aien calculator, cut 8.
#
# Derived numbers in decision documents must carry a calculator receipt. This hook looks at what
# is about to be written to a numbers-bearing document (docs/adr/, docs/**/evidence/, FINDINGS.md,
# README.md inside an evidence dir, handoffs/*.md, specs/). If the new text has 3 or more numeric
# claims and neither a `calc:<hex>` id nor a `CALC-EXEMPT: <reason>` line, it complains.
#
# Modes (env CALC_GUARD_MODE):
#   warn  (default) never blocks; tells the model what to do (additionalContext) and prints a line on stderr.
#   deny  blocks the write with a message that says exactly what to do.
#   off   does nothing.
# Env CALC_GUARD_LOG=<file> appends one line per flagged write (for measuring false positives).
#
# A courtesy gate, like manual-guard: it is easy to get around on purpose. Light by design: only
# those paths, only 3+ claims, one id passes the whole document, the exempt line is greppable.
# Chat text and Bash commands are deliberately not checked (it would be routed around within a day).
# Needs: POSIX sh, jq, GNU grep (-P).

mode=${CALC_GUARD_MODE:-warn}
[ "$mode" = off ] && exit 0

inp=$(cat)
tool=$(printf '%s' "$inp" | jq -r '.tool_name // ""' 2>/dev/null) || exit 0
case "$tool" in Write | Edit | MultiEdit) ;; *) exit 0 ;; esac
path=$(printf '%s' "$inp" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || exit 0
[ -n "$path" ] || exit 0

case "/$path" in
  */docs/adr/* | */docs/*/evidence/* | */docs/evidence/* | */evidence/README.md | */evidence/*/README.md | \
    */FINDINGS.md | */handoffs/*.md | */specs/*) ;;
  *) exit 0 ;;
esac

text=$(printf '%s' "$inp" | jq -r '(.tool_input.content // .tool_input.new_string // ([.tool_input.edits[]?.new_string] | join("\n")) // "")' 2>/dev/null) || exit 0
[ -n "$text" ] || exit 0

id_re='calc:[0-9a-f]{12,}'
exempt_re='CALC-EXEMPT:[[:space:]]*[^[:space:]]'
covered() { printf '%s\n' "$1" | grep -Eq "$id_re|$exempt_re"; }

# One id or one exempt line covers the whole document, including later edits to it.
covered "$text" && exit 0
if [ "$tool" != Write ] && [ -f "$path" ] && grep -Eq "$id_re|$exempt_re" "$path" 2>/dev/null; then exit 0; fi

# A number followed by one of the spec units, or a power of two. Not glued to a name or a longer word.
# Bare `B` (bytes) needs a space and no leading `-`: `Qwen3-30B`, `1B` and `7B` are model sizes in billions.
claim_re='(?<![A-Za-z0-9_])(?:[0-9][0-9.,_]*[ \t]?(?:GB/s|tok/s|GiB|MiB|GB|MB|KB|ms|us|ns|%|FLOP|param|launch)(?![A-Za-z0-9_])|(?<![-A-Za-z0-9_])[0-9][0-9.,_]*[ \t]B(?![A-Za-z0-9_])|2\^[-(0-9])'
n=$(printf '%s\n' "$text" | grep -oP "$claim_re" | wc -l | tr -d ' ')
[ "${n:-0}" -ge 3 ] || exit 0

if [ -n "$CALC_GUARD_LOG" ]; then
  printf '%s mode=%s claims=%s path=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$n" "$path" >>"$CALC_GUARD_LOG" 2>/dev/null
fi

q="'"
bt='`'
core="This doc states derived numbers (${n} numeric claims, no calc: id). Run ${bt}aien-calc eval ${q}<expr>${q}${bt} for each and paste the ${bt}calc:${bt} id after the number, or add ${bt}CALC-EXEMPT: <why>${bt}."

if [ "$mode" = deny ]; then
  jq -cn --arg m "$core" '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$m}}'
  exit 0
fi
printf 'calc-guard: WARN %s (%s)\n' "$core" "$path" >&2
jq -cn --arg m "WARN (calc-guard, warn-only): $core" '{hookSpecificOutput:{hookEventName:"PreToolUse",additionalContext:$m}}'
exit 0
