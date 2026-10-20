#!/bin/bash
# t2.sh dir... : compact table: same-pick percent and NLL delta per text
for d in "$@"; do
  printf '%-34s' "$d"
  for t in golden license prose codelong code arch; do
    f=$d/$t.json; [ -f $f ] || { printf ' %-22s' -; continue; }
    ap=$(grep '"agree_pct"' $f | tr -dc '0-9.'); nc=$(grep '"nll_coded_mean"' $f | tr -dc '0-9.'); nr=$(grep '"nll_ref_mean"' $f | tr -dc '0-9.')
    printf ' %5.1f/%+.3f         ' $ap $(echo "$nc - $nr" | bc -l)
  done; echo
done
