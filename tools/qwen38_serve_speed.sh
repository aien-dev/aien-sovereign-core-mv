#!/bin/bash
# Quick speed test through the live Qwen3.8-27B server (sc #642): decode on a short prompt (3 runs,
# 256 tokens, greedy), cold TTFT at about 1.7k and 30k prompt tokens, and next-turn TTFT on a
# follow-up that extends the 30k conversation. Client wall from curl; prefill / decode / chip
# figures from the server's own log line for each reply id.
# Usage: qwen38_serve_speed.sh <base-url> <corpus.txt> <out-dir> [server-log]
set -uo pipefail
URL=$1; CORPUS=$2; OUT=$3; LOG=${4:-$HOME/.local/state/aien-q38/serve.log}
mkdir -p "$OUT"
# run <name> <messages-json> <max_tokens>
run() {
  jq -n --argjson m "$2" --argjson n "$3" \
    '{messages:$m, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}}' > "$OUT/$1.req.json"
  for try in $(seq 1 240); do
    w=$(curl -s -m 1800 -o "$OUT/$1.json" -w '%{time_total}' "$URL/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @"$OUT/$1.req.json")
    # Retry only "busy" answers (server_busy, chip lock); any other error is the result.
    jq -e '.error | tostring | test("busy"; "i")' "$OUT/$1.json" >/dev/null 2>&1 || break
    sleep 5
  done
  id=$(jq -r '.id' "$OUT/$1.json")
  line=$(grep -B1 "$id:" "$LOG" | grep 'qwen38:' | head -1 | sed 's/aien-serve-http: qwen38: //')
  printf '%s\tclient_wall_s=%s\tprompt=%s\tcompletion=%s\t%s\n' "$1" "$w" \
    "$(jq '.usage.prompt_tokens' "$OUT/$1.json")" "$(jq '.usage.completion_tokens' "$OUT/$1.json")" "$line" | tee -a "$OUT/runs.tsv"
}
text() { head -c "$1" "$CORPUS" | iconv -c -f utf-8 -t utf-8; }
: > "$OUT/runs.tsv"
SHORT='[{"role":"user","content":"Write a detailed essay about the history of lighthouses, from ancient times to today."}]'
for i in 1 2 3; do run "decode$i" "$SHORT" 256; done
T17=$(text 7100); T30=$(text 100000)
run ttft_1p7k "$(jq -n --arg t "$T17" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)}]')" 1
run ttft_30k "$(jq -n --arg t "$T30" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)}]')" 1
A=$(jq -r '.choices[0].message.content' "$OUT/ttft_30k.json")
run next_turn_30k "$(jq -n --arg t "$T30" --arg a "$A" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)},{role:"assistant",content:$a},{role:"user",content:"Now name the three most important documents it mentions."}]')" 1
