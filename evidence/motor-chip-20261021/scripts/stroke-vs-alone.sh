#!/usr/bin/env bash
# Every head of a stroke run against the same conversation run alone on the chip: chip tokens equal, position by position.
# usage: stroke-vs-alone.sh <stroke.json> <alone.json>...   (alone = qwen36_stroke_gate N=1 output, or qwen38_omega_full gate output)
set -eu
S=$1; shift
i=0; all=1
for a in "$@"; do
  name=$(jq -r ".conversations[$i].name" "$S")
  chip=$(jq -c ".conversations[$i].chip_ids" "$S")
  alone=$(jq -c 'if .conversations then .conversations[0].chip_ids else .chip_ids end' "$a")
  r=$(jq -n --argjson a "$chip" --argjson b "$alone" '{n: ($a|length), m: ($b|length), equal: ($a == $b), first_diff: ([range(0; ([($a|length),($b|length)]|min)) | select($a[.] != $b[.])] | .[0])}' -c)
  echo "head $i $name vs $(basename "$a"): $r"
  echo "$r" | grep -q '"equal":true' || all=0
  i=$((i+1))
done
[ $all -eq 1 ] && echo "ALL HEADS EQUAL THEIR RUN ALONE" || { echo "A HEAD DIFFERS FROM ITS RUN ALONE"; exit 1; }
