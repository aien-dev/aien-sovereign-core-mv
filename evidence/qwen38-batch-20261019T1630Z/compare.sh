#!/usr/bin/env bash
# compare.sh <single-dir> <batch-dir>: every batch stream's token ids against its prompt's
# single-stream greedy answer (one-at-a-time door, single-dir) and against the batch door's own
# n=1 answer in the same pass (c-n1-pI for c-* labels, n1-pI otherwise). Uses the attempt named in
# good.txt. Prints one line per stream, then totals. A negative control compares prompt i's stream
# with prompt (i+1)%8's reference, which must differ.
set -euo pipefail
S=$1; B=$2
ids() { jq -c '.usage.aien_token_ids' "$1"; }
good() { awk -v l="$2" '$1==l{print $2}' "$1/good.txt"; }
same=0; diff=0; same1=0; diff1=0; negsame=0; negdiff=0
for lab in $(awk '$2 !~ /FAILED/ {print $1}' "$B/good.txt"); do
  att=$(good "$B" "$lab")
  case $lab in c-*) pre=c-;; *) pre=;; esac
  for f in "$B/$att"-*.json; do
    case $f in *.req.json) continue;; esac
    i=${f##*-}; i=${i%.json}
    ref=$(ids "$S/$(good "$S" "single-p$i")-$i.json")
    got=$(ids "$f")
    n=$(jq 'length' <<<"$got")
    one=$(ids "$B/$(good "$B" "${pre}n1-p$i")-$i.json")
    j=$(( (i + 1) % 8 ))
    neg=$(ids "$S/$(good "$S" "single-p$j")-$j.json")
    if [ "$got" = "$ref" ] && [ "$n" -gt 0 ]; then r=SAME; same=$((same+1)); else r=DIFF; diff=$((diff+1)); fi
    if [ "$got" = "$one" ]; then r1=SAME; same1=$((same1+1)); else r1=DIFF; diff1=$((diff1+1)); fi
    if [ "$got" = "$neg" ]; then negsame=$((negsame+1)); else negdiff=$((negdiff+1)); fi
    echo "$lab prompt $i tokens $n vs-single-door $r vs-batch-n1 $r1"
  done
done
echo "TOTAL vs single door: $same same, $diff different; vs batch n=1: $same1 same, $diff1 different"
echo "NEGATIVE CONTROL (stream vs the next prompt's reference, must differ): $negdiff differ, $negsame same"
