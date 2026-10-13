#!/bin/sh
# calc-receipts-gate.sh -- CI gate for calculator receipts. Aien calculator, cut 8.
#
# For each changed numbers-bearing document (docs/adr/, docs/**/evidence/, FINDINGS.md, README.md in
# an evidence dir, handoffs/*.md, specs/) every `calc:<64 hex>` it cites must
#   1. resolve to calc-receipts/<hex>.json next to the document, or docs/calc-receipts/<hex>.json;
#   2. hash: the BLAKE3 of that file equals the id (b3sum, or `aien-calc b3`);
#   3. verify (mode full): `aien-calc verify <file>` says MATCH (a full recompute).
# Mode hash does 1 and 2 only (no Rust toolchain needed); full is the default when aien-calc is available.
#
# Usage: calc-receipts-gate.sh [--mode full|hash|auto] [file ...]
#   With no files it checks files changed since $CALC_GATE_BASE (default origin/main) via git.
# Env:   CALC_GATE_ENFORCE=1  exit 1 on any problem (default: warn-only, always exit 0, prints WARN)
#        AIEN_CALC_BIN        path of aien-calc (default: from PATH)
#        CALC_GATE_ROOT       repo root (default: git toplevel, else the current directory)
# Warn-only first, then flip: set CALC_GATE_ENFORCE=1 in the workflow (a one-line change).

mode=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) mode=$2; shift 2 ;;
    --mode=*) mode=${1#--mode=}; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
case "$mode" in full | hash | auto) ;; *) echo "calc-receipts-gate: unknown mode '$mode'" >&2; exit 2 ;; esac

root=${CALC_GATE_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
calc=${AIEN_CALC_BIN:-$(command -v aien-calc 2>/dev/null)}
[ -n "$calc" ] && [ ! -x "$calc" ] && calc=
[ "$mode" = auto ] && { if [ -n "$calc" ]; then mode=full; else mode=hash; fi; }
if [ "$mode" = full ] && [ -z "$calc" ]; then
  echo "calc-receipts-gate: mode full needs aien-calc (set AIEN_CALC_BIN or put it on PATH)" >&2
  exit 2
fi

b3() {
  if command -v b3sum >/dev/null 2>&1; then b3sum "$1" | cut -d' ' -f1
  elif [ -n "$calc" ]; then "$calc" b3 "$1" | cut -d' ' -f1
  else return 1
  fi
}

if [ $# -gt 0 ]; then
  files=$(printf '%s\n' "$@")
else
  base=${CALC_GATE_BASE:-origin/main}
  files=$(git -C "$root" diff --name-only --diff-filter=ACMR "$base...HEAD" 2>/dev/null || git -C "$root" diff --name-only --diff-filter=ACMR HEAD~1 2>/dev/null)
fi

problems=0
checked=0
nfiles=0
problem() { problems=$((problems + 1)); printf 'calc-receipts-gate: WARN %s\n' "$1" >&2; }

for f in $files; do
  case "/$f" in
    */docs/adr/* | */docs/*/evidence/* | */docs/evidence/* | */evidence/README.md | */evidence/*/README.md | \
      */FINDINGS.md | */handoffs/*.md | */specs/*) ;;
    *) continue ;;
  esac
  [ -f "$f" ] || [ -f "$root/$f" ] || continue
  [ -f "$f" ] || f="$root/$f"
  case "$f" in */calc-receipts/*.json) continue ;; esac
  nfiles=$((nfiles + 1))
  dir=$(dirname "$f")
  for id in $(grep -oE 'calc:[0-9a-f]{12,}' "$f" | sort -u); do
    hex=${id#calc:}
    if [ "${#hex}" -lt 64 ]; then problem "$f: short id $id (cite the full 64-hex id so it can be resolved)"; continue; fi
    if [ "${#hex}" -gt 64 ]; then problem "$f: malformed id $id (longer than 64 hex digits)"; continue; fi
    checked=$((checked + 1))
    rec=
    for cand in "$dir/calc-receipts/$hex.json" "$root/docs/calc-receipts/$hex.json"; do
      if [ -f "$cand" ]; then rec=$cand; break; fi
    done
    if [ -z "$rec" ]; then problem "$f: receipt missing: $id (expected $dir/calc-receipts/$hex.json or docs/calc-receipts/$hex.json)"; continue; fi
    got=$(b3 "$rec") || { problem "$f: cannot check hash of $id: no b3sum or aien-calc available"; continue; }
    if [ "$got" != "$hex" ]; then problem "$f: hash wrong: $rec hashes to calc:$got, the document cites $id"; continue; fi
    if [ "$mode" = full ]; then
      out=$("$calc" verify "$rec" 2>&1)
      case "$out" in
        MATCH*) ;;
        *) problem "$f: verify mismatch: $id $(printf '%s' "$out" | tr '\n' ' ')" ;;
      esac
    fi
  done
done

if [ "${CALC_GATE_ENFORCE:-0}" = 1 ]; then policy=enforce; else policy=warn-only; fi
printf 'calc-receipts-gate: %s receipt(s) checked in %s file(s), %s problem(s) [mode %s, %s]\n' "$checked" "$nfiles" "$problems" "$mode" "$policy"
if [ "$problems" -gt 0 ] && [ "$policy" = enforce ]; then exit 1; fi
exit 0
