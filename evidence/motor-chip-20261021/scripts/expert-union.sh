#!/usr/bin/env bash
# Expert union per layer across N host goldens (qwen38_check generate: routes[step][layer][k expert ids]).
# Step 0 of a golden is the prefill (every prompt token's k ids flattened); the union uses the decode steps 1.. only,
# aligned by decode index. For every such step where all N goldens have a route and every layer: |union of the N rows' experts|.
# Prints per-N mean union (over steps x layers), the independent-draw expectation 256*(1-(1-8/256)^N), and per-layer means.
# usage: expert-union.sh gen-a.json gen-b.json ...
set -eu
jq -s '
  (map(.routes | length) | min) as $steps
  | (.[0].routes[0] | length) as $layers
  | length as $n
  | [range(1; $steps) as $s | range(0; $layers) as $l
      | {s: $s, l: $l, u: ([.[] | .routes[$s][$l][]] | unique | length)}] as $cells
  | {n: $n, decode_steps: ($steps - 1), layers: $layers,
     mean_union: (($cells | map(.u) | add) / ($cells | length)),
     min_union: ($cells | map(.u) | min), max_union: ($cells | map(.u) | max),
     independent_draws: (256 * (1 - pow(1 - 8/256; $n))),
     per_layer_mean: [range(0; $layers) as $l | ($cells | map(select(.l == $l) | .u) | add / length)]}
' "$@"
