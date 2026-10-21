#!/bin/bash
# table.sh: drift table (markdown) from drift-*.json, drift-col-*.json, score-*.json, chunk-*.json here.
# A divergence at an f32 top1-top2 margin under 0.1 logits (L306 gate-1 margin_min) is tagged near-tie.
cd "$(dirname "$0")"
echo "| prompt | state | steps | first divergence | f32 margin there | agree | cos min | flipped expert rows | state rel err @64 / @256 / last | worst layer max |"
echo "|---|---|---|---|---|---|---|---|---|---|"
for f in drift-*.json; do
  jq -r --arg f "${f%.json}" '
    . as $r | .states[1:][] |
    (.first_divergence) as $d |
    ( if $d == null then "none" else ($d|tostring) end ) as $ds |
    ( if $d == null then "" else ($r.f32_margin[$d] | . * 1000 | round / 1000 | tostring) + (if $r.f32_margin[$d] < 0.1 then " near-tie" else "" end) end ) as $m |
    (.per_row.state_rel) as $s |
    "| \($f | sub("^drift-";"")) | \(.name) | \($r.steps)\(if $r.complete then "" else " (running)" end) | \($ds) | \($m) | \(.agree)/\(.rows) | \(.cos_min*1e6|round/1e6) | \(.flips_total) | \(($s[63] // 0)*1e4|round/1e4) / \(($s[255] // 0)*1e4|round/1e4) / \(($s[-1] // 0)*1e4|round/1e4) | \(.state_rel_worst_layer_max*1e4|round/1e4) |"' "$f"
done
echo
echo "| passage | state | perplexity | mean NLL | NLL minus f32 | argmax agree | cos min | flipped expert rows |"
echo "|---|---|---|---|---|---|---|---|"
for f in score-*.json; do
  jq -r --arg f "${f%.json}" '
    (.states[0].extra.mean_nll) as $b | .states[] |
    "| \($f | sub("^score-";"")) | \(.name) | \(.extra.perplexity*1e4|round/1e4) | \(.extra.mean_nll*1e5|round/1e5) | \((.extra.mean_nll - $b)*1e5|round/1e5) | \(.agree)/\(.rows) | \(.cos_min*1e6|round/1e6) | \(.flips_total) |"' "$f"
done
if ls chunk-*.json >/dev/null 2>&1; then
  echo
  echo "| prompt | state | first divergence | agree | cos min | flipped expert rows | state rel err last |"
  echo "|---|---|---|---|---|---|---|"
  for f in chunk-*.json; do
    jq -r --arg f "${f%.json}" '.states[1:][] |
      "| \($f | sub("^chunk-";"")) | \(.name) | \(.first_divergence // "none") | \(.agree)/\(.rows) | \(.cos_min*1e6|round/1e6) | \(.flips_total) | \((.state_rel_last // 0)|.*1e6|round/1e6) |"' "$f"
  done
fi
