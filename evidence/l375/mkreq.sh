#!/bin/bash
# L375: a Hermes-shaped chat request (no real dump on the Spark yet): 40 tools (~11k tokens, tool
# descriptions cut from this repo's docs), a ~6k-token system text whose last lines are a date line
# and a memory block (as Hermes renders them), then one user turn. Deterministic.
# mkreq.sh <out.json> <user message> [changed-tool-index] [ntools]
set -euo pipefail
out=$1; user=$2; changed=${3:--1}; ntools=${4:-40}
cd "$(git rev-parse --show-toplevel)"
mapfile -t docs < <(find docs -name '*.md' -size +3k | sort | head -70)
tools='[]'
for i in $(seq 0 $((ntools - 1))); do
  d=$(head -c 1500 "${docs[$i]}" | tr -s '\n' ' ' | cut -c1-1050)
  [ "$i" = "$changed" ] && d="CHANGED: $d"
  tools=$(jq -c --arg n "$(printf 'tool_%02d' $i)" --arg d "$d" '. + [{type:"function",function:{name:$n,description:$d,
    parameters:{type:"object",properties:{path:{type:"string",description:"Absolute path the tool works on"},
    limit:{type:"integer",description:"Most items to return"}},required:["path"]}}}]' <<<"$tools")
done
sys=""
for i in $(seq 40 69); do sys+=$(head -c 900 "${docs[$i]}" | tr -s '\n' ' ')$'\n\n'; done
sys+=$'Current date: 2026-10-21 (Tuesday)\n\nMEMORY:\n- Drake prefers plain English.\n- The Spark serves the 35B on port 18091.\n'
jq -n --argjson tools "$tools" --arg sys "$sys" --arg user "$user" \
  '{model:"qwen3.6-35b-a3b-fp8", stream:true, temperature:0, max_tokens:48, aien_return_token_ids:true, tools:$tools,
    messages:[{role:"system",content:$sys},{role:"user",content:$user}]}' > "$out"
echo "$out: $(wc -c < "$out") bytes"
