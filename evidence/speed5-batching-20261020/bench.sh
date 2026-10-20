#!/usr/bin/env bash
# N concurrent greedy requests against a test aien-serve-http (never the live 18091).
# usage: bench.sh <url> <outdir> <label> <max_tokens> <prompt index>...   (prompts from prompts.txt, line i+1)
# Writes <outdir>/<label>-<i>.json per request, <label>.times (index, start, end epoch ns), <label>.summary.
set -euo pipefail
URL=$1; OUT=$2; LABEL=$3; MAXT=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
: > "$OUT/$LABEL.times"; : > "$OUT/$LABEL.summary"
pids=()
for i in "$@"; do
  P=$(sed -n "$((i + 1))p" "$HERE/prompts.txt")
  jq -cn --arg p "$P" --argjson m "$MAXT" '{model:"qwen3.8-27b-fp8",messages:[{role:"user",content:$p}],temperature:0,max_tokens:$m,aien_return_token_ids:true,chat_template_kwargs:{enable_thinking:false}}' > "$OUT/$LABEL-$i.req.json"
  (
    t0=$(date +%s%N)
    curl -s -m 3600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/$LABEL-$i.req.json" > "$OUT/$LABEL-$i.json"
    t1=$(date +%s%N)
    echo "$i $t0 $t1" >> "$OUT/$LABEL.times"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
TOT=0
for i in "$@"; do
  c=$(jq '.usage.completion_tokens // 0' "$OUT/$LABEL-$i.json" 2>/dev/null || echo 0)
  f=$(jq -r '.choices[0].finish_reason // "none"' "$OUT/$LABEL-$i.json" 2>/dev/null || echo none)
  echo "$LABEL req $i completion_tokens $c finish $f" >> "$OUT/$LABEL.summary"
  TOT=$((TOT + c))
done
T0=$(sort -k2 -n "$OUT/$LABEL.times" | head -1 | awk '{print $2}')
T1=$(sort -k3 -n "$OUT/$LABEL.times" | tail -1 | awk '{print $3}')
echo "$LABEL n=$# total_completion_tokens=$TOT client_wall_ms=$(( (T1 - T0) / 1000000 ))" >> "$OUT/$LABEL.summary"
