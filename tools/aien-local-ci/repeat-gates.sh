#!/usr/bin/env bash
# repeat-gates.sh [SINCE_UTC] [STATE]: how much gate time went into running the same gate on the same head again.
# Reads every receipt under $STATE/evidence (core.commit, core.gate, core.stderr_sha256, volatile.duration_ms and
# started_utc). The first run of each (head, gate) is the work; every later run is a repeat. "equal" repeats gave
# byte-identical stderr, so they were pure waste; the rest changed (a flake, a retry after a harness fault, or a real change).
# Prints: day, total gate seconds, repeat seconds and share, equal-stderr repeat seconds. SINCE_UTC (default all) keeps
# only receipts that started at or after it, e.g. 2026-10-21T00:00 to check the rounds after a harness fix (L337, G3-ci-judge-3).
set -euo pipefail
SINCE=${1:-}
STATE=${2:-${AIEN_STATE:-$HOME/.local/state/aien-local-ci}}
find "$STATE/evidence" -name '*.json' -print0 | xargs -0 jq -r \
  '[.core.commit, .core.gate, (.core.stderr_sha256 // ""), (.volatile.duration_ms // 0), (.volatile.started_utc // "")] | @tsv' \
  | sort -t$'\t' -k1,1 -k2,2 -k5,5 \
  | awk -F'\t' -v since="$SINCE" '
      { k = $1 "\t" $2; take = (since == "" || $5 >= since)
        if (take) { tot += $4 }
        if (k == pk && take) { rep += $4; n++; if ($3 == ps) { eq += $4 } }
        pk = k; ps = $3 }
      END { if (tot == 0) { print "no receipts in the window"; exit }
            printf "total_s=%.0f repeat_s=%.0f (%.1f%%) repeat_runs=%d equal_stderr_s=%.0f\n", tot/1000, rep/1000, 100*rep/tot, n, eq/1000 }'
