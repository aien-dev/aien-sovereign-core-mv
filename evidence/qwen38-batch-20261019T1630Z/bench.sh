#!/usr/bin/env bash
# Continuous-batching benchmark against a test aien-serve-http (never the live 18091).
# usage: bench.sh <url> <outdir> <label> <n_concurrent> <max_tokens> [prompt indexes...]
# Writes <outdir>/<label>-<i>.json per request, <outdir>/<label>.times (start/end epoch ns per request)
# and <outdir>/<label>.summary.
set -euo pipefail
URL=$1; OUT=$2; LABEL=$3; N=$4; MAXT=$5; shift 5
mkdir -p "$OUT"
PROMPTS=(
  "Explain in plain English why the sky is blue, in about 150 words."
  "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring."
  "List five practical tips for keeping a small vegetable garden healthy in late summer."
  "Summarise the plot of Romeo and Juliet in one paragraph."
  "What is the difference between TCP and UDP? Give three concrete differences."
  "Write a short, calm bedtime story about a lighthouse keeper and a seagull."
  "Describe how a bicycle's gears change the effort needed to pedal uphill."
  "Give a recipe for a simple tomato soup for two people, with steps."
)
IDX=("$@")
if [ ${#IDX[@]} -eq 0 ]; then IDX=($(seq 0 $((N-1)))); fi
: > "$OUT/$LABEL.times"
pids=()
for i in "${IDX[@]}"; do
  P=${PROMPTS[$i]}
  jq -cn --arg p "$P" --argjson m "$MAXT" '{model:"qwen3.8-27b-fp8",messages:[{role:"user",content:$p}],temperature:0,max_tokens:$m,aien_return_token_ids:true,chat_template_kwargs:{enable_thinking:false}}' > "$OUT/$LABEL-$i.req.json"
  (
    t0=$(date +%s%N)
    curl -s -m 1800 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/$LABEL-$i.req.json" > "$OUT/$LABEL-$i.json"
    t1=$(date +%s%N)
    echo "$i $t0 $t1" >> "$OUT/$LABEL.times"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
# summary: total completion tokens, wall from first start to last end, aggregate tok/s
T0=$(sort -k2 -n "$OUT/$LABEL.times" | head -1 | awk '{print $2}')
T1=$(sort -k3 -n "$OUT/$LABEL.times" | tail -1 | awk '{print $3}')
TOT=0
for i in "${IDX[@]}"; do
  c=$(jq '.usage.completion_tokens // 0' "$OUT/$LABEL-$i.json")
  f=$(jq -r '.choices[0].finish_reason // "none"' "$OUT/$LABEL-$i.json")
  echo "$LABEL req $i completion_tokens $c finish $f" >> "$OUT/$LABEL.summary"
  TOT=$((TOT + c))
done
WALL_MS=$(( (T1 - T0) / 1000000 ))
echo "$LABEL n=$N max_tokens=$MAXT total_completion_tokens=$TOT wall_ms=$WALL_MS aggregate_tok_s=$(awk -v t=$TOT -v w=$WALL_MS 'BEGIN{printf "%.3f", t/(w/1000)}')" | tee -a "$OUT/$LABEL.summary"
