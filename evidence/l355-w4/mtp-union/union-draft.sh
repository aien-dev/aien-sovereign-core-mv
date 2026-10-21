#!/usr/bin/env bash
# L355 (Fable final review point 3): expert union per layer when each of N conversations adds one draft row (MTP k=1).
# Same inputs and alignment as motor/serve evidence/motor-chip-20261021/scripts/expert-union.sh (decode steps 1.. of
# qwen38_check goldens, routes[step][layer][k ids]); the draft row of conversation i at step s routes like its step s+1.
# Prints the mean union of the N rows alone (base) and of the N rows plus their N drafts (steps s and s+1 of every golden).
set -eu
jq -s '
  . as $all
  | (map(.routes | length) | min) as $steps | (.[0].routes[0] | length) as $layers | length as $n
  | [range(1; $steps - 1) as $s | range(0; $layers) as $l
      | {b: ([$all[] | .routes[$s][$l][]] | unique | length),
         d: ([$all[] | (.routes[$s][$l][], .routes[$s + 1][$l][])] | unique | length)}] as $c
  | {n: $n, decode_steps_used: ($steps - 2), layers: $layers,
     base_union: (($c | map(.b) | add) / ($c | length)),
     draft_union: (($c | map(.d) | add) / ($c | length)),
     draft_max: ($c | map(.d) | max)}
' "$@"
