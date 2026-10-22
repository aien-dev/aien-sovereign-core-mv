#!/bin/bash
# L371: one request. Usage: ask.sh PORT MODEL DOC(design|log|code|big) OUT.json [max_tokens]
# Body: system + (document, blank line, the questions); greedy; non-streaming. Wall time from curl.
set -euo pipefail
P=$1; M=$2; D=$3; OUT=$4; MAX=${5:-700}
H=$(dirname "$0"); DOCS=${L371_DOCS:-$HOME/handoffs/l371-docs}
F=$(jq -r ".$D.file" "$H/questions.json"); Q=$(jq -r ".$D.question" "$H/questions.json")
jq -n --arg m "$M" --rawfile doc "$DOCS/$F" --arg q "$Q" --argjson max "$MAX" \
  '{model:$m, temperature:0, max_tokens:$max, messages:[{role:"system",content:"You are a careful assistant. Answer only from the material given."},{role:"user",content:($doc + "\n\n" + $q)}]}' > "$OUT.req"
T0=$(date +%s.%N)
code=$(curl -s -o "$OUT" -w '%{http_code}' --max-time ${ASK_MAX_TIME:-3600} -H 'Content-Type: application/json' --data-binary @"$OUT.req" "http://127.0.0.1:$P/v1/chat/completions")
T1=$(date +%s.%N)
rm -f "$OUT.req"
echo "{\"doc\":\"$D\",\"model\":\"$M\",\"http\":$code,\"t0\":$T0,\"t1\":$T1}" > "$OUT.time"
echo "$D $M http $code wall $(echo "$T1 - $T0" | bc) s, prompt $(jq -r '.usage.prompt_tokens // .error.message' "$OUT" | head -c 200)"
