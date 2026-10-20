#!/bin/bash
# table.sh <out-dir>...  : one line per text per variant, from the per-text JSON files
for d in "$@"; do
  for t in golden license prose codelong code arch; do
    f=$d/$t.json; [ -f $f ] || continue
    printf '%-28s %-9s ' "$d" "$t"
    grep -E '"(positions|agree|agree_pct|kl_mean|largest_ref_margin_at_a_flip|nll_ref_mean|nll_coded_mean)"' $f | tr -d ' ,\n' | sed 's/"//g'; echo
  done
done
