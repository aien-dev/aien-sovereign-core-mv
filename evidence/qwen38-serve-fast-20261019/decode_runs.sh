#!/bin/bash
# decode_runs.sh <base-url> <server-log> <out-dir> <tag> <n>: n greedy 256-token decodes of the serve
# agent's short prompt (qwen38_serve_speed.sh SHORT), token ids kept, figures from the server's log line.
set -uo pipefail
URL=$1; LOG=$2; OUT=$3; TAG=$4; N=$5
mkdir -p "$OUT"
SHORT='[{"role":"user","content":"Write a detailed essay about the history of lighthouses, from ancient times to today."}]'
for i in $(seq 1 "$N"); do
  f="$OUT/$TAG-$i"
  jq -n --argjson m "$SHORT" '{messages:$m, temperature:0, max_tokens:256, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}' > "$f.req.json"
  for try in $(seq 1 240); do
    w=$(curl -s -m 1800 -o "$f.json" -w '%{time_total}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @"$f.req.json")
    jq -e '.error | tostring | test("busy"; "i")' "$f.json" >/dev/null 2>&1 || break
    sleep 1
  done
  id=$(jq -r '.id' "$f.json")
  line=$(grep -B1 "$id:" "$LOG" | grep 'qwen38:' | head -1 | sed 's/aien-serve-http: qwen38: //')
  printf '%s\t%s\tclient_wall_s=%s\tcompletion=%s\tids_sha=%s\t%s\n' "$(date -u +%FT%TZ)" "$TAG-$i" "$w" \
    "$(jq '.usage.completion_tokens' "$f.json")" "$(jq -c '.usage.aien_token_ids' "$f.json" | sha256sum | cut -c1-12)" "$line" | tee -a "$OUT/runs.tsv"
done
