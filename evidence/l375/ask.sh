#!/bin/bash
# ask.sh <port> <req.json> <tag> <outdir>: streams one chat request; writes <tag>.ids (generated token
# ids from usage.aien_token_ids) and appends "<tag> ttft_ms=<first content or tool delta> total_ms=..." to <outdir>/times.txt
set -uo pipefail
port=$1; req=$2; tag=$3; od=$4
t0=$(date +%s%N); first=""
curl -sN --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$req" "http://127.0.0.1:$port/v1/chat/completions" > "$od/$tag.sse" &
cpid=$!
# The first delta with text: poll the growing file (1 ms grain is enough against seconds).
while kill -0 $cpid 2>/dev/null; do
  if [ -z "$first" ] && grep -q '"content":"[^"]\|"reasoning_content":"[^"]\|"tool_calls"' "$od/$tag.sse" 2>/dev/null; then first=$(date +%s%N); fi
  sleep 0.001
done
wait $cpid
t1=$(date +%s%N)
[ -z "$first" ] && first=$t1
grep -o '"aien_token_ids":\[[0-9,]*\]' "$od/$tag.sse" | tail -1 | grep -o '[0-9][0-9,]*' > "$od/$tag.ids"
echo "$tag ttft_ms=$(( (first - t0) / 1000000 )) total_ms=$(( (t1 - t0) / 1000000 )) ids=$(tr ',' '\n' < "$od/$tag.ids" | grep -c .)" | tee -a "$od/times.txt"
