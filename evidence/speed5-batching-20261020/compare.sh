#!/usr/bin/env bash
# compare.sh <ref-dir> <ref-call> <dir>...: every stream's token ids (good attempt of each call in good.txt) against
# prompt i's reference answer = stream i of <ref-call> in <ref-dir> (for prompts 0..7 a call n1-p<i> in <ref-dir>, when
# present, wins: a stream alone). One line per stream: SAME, or DIFF with the first different generated position.
# Negative control: each stream against prompt (i+1)%32's reference, which must differ. Totals at the end.
set -euo pipefail
R=$1; RC=$2; shift 2
ids() { jq -c '.usage.aien_token_ids // []' "$1"; }
att() { awk -v l="$2" '$1==l && $2 !~ /FAILED/ {print $2}' "$1/good.txt"; }
ref() { local a; a=$(att "$R" "n1-p$1"); if [ -n "$a" ] && [ -f "$R/$a-$1.json" ]; then ids "$R/$a-$1.json"; else ids "$R/$(att "$R" "$RC")-$1.json"; fi; }
first_diff() { jq -n --argjson a "$1" --argjson b "$2" '[range(0; ([$a|length, $b|length]|max)) | select($a[.] != $b[.])][0] // -1'; }
declare -A REF
for i in $(seq 0 31); do REF[$i]=$(ref "$i" 2>/dev/null || echo '[]'); done
tot_same=0; tot_diff=0; neg_same=0; neg_diff=0
for D in "$@"; do
  for L in $(awk '$2 !~ /FAILED/ {print $1}' "$D/good.txt"); do
    A=$(att "$D" "$L")
    for f in "$D/$A"-*.json; do
      case $f in *.req.json) continue;; esac
      i=${f##*-}; i=${i%.json}
      got=$(ids "$f"); n=$(jq length <<<"$got"); r=${REF[$i]}
      if [ "$n" -gt 0 ] && [ "$got" = "$r" ]; then res=SAME; tot_same=$((tot_same + 1));
      else res="DIFF at $(first_diff "$got" "$r") ($n vs $(jq length <<<"$r") tokens)"; tot_diff=$((tot_diff + 1)); fi
      j=$(( (i + 1) % 32 ))
      if [ "$got" = "${REF[$j]}" ]; then neg_same=$((neg_same + 1)); else neg_diff=$((neg_diff + 1)); fi
      echo "$(basename "$D") $L prompt $i tokens $n $res"
    done
  done
done
echo "TOTAL vs reference: $tot_same same, $tot_diff different; negative control: $neg_diff differ, $neg_same same (must be 0)"
