#!/bin/bash
# Quick speed test through the live Qwen3.8-27B server (sc #642): decode on a short prompt (3 runs,
# 256 tokens, greedy), cold TTFT at about 1.7k and 30k prompt tokens, and next-turn TTFT on a
# follow-up that extends the 30k conversation. Client wall from curl; prefill / decode / chip
# figures from the server's own log line for each reply id. Token ids are requested for every
# reply; the three decodes are compared with the shipped reference ids (MATCH or MISMATCH).
# Usage: qwen38_serve_speed.sh [base-url] [corpus.txt] <out-dir> [server-log]
#   base-url    default http://127.0.0.1:18091
#   corpus.txt  default: the shipped corpus crates/aien-inference-abi/fixtures/qwen38/speed-corpus.txt
#               ("" also means default); any UTF-8 text of at least 100 kB works
#   server-log  default $AIEN_SERVE_LOG, else the log tools/serve-27b.sh writes:
#               ${XDG_STATE_HOME:-$HOME/.local/state}/aien/serve-27b.log
# Writes <out-dir>/runs.tsv and <out-dir>/receipt.json (build, binary sha256, model checksum list,
# machine state, per-run figures, median chip ms per token, tokens per second, share of the
# 10.15 tokens per second ceiling in docs/qwen38/CEILING.md, MATCH or MISMATCH). See
# tools/qwen38_receipt.sh for how the server is found. Needs curl, jq, iconv, sha256sum and awk.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FIX="$REPO/crates/aien-inference-abi/fixtures/qwen38"
USAGE="usage: qwen38_serve_speed.sh [base-url] [corpus.txt] <out-dir> [server-log]"
URL=${1:-http://127.0.0.1:18091}
CORPUS=${2:-$FIX/speed-corpus.txt}
OUT=${3:?$USAGE}
LOG=${4:-${AIEN_SERVE_LOG:-${XDG_STATE_HOME:-$HOME/.local/state}/aien/serve-27b.log}}
REF="$FIX/decode-lighthouses-256.json"
for t in curl jq iconv sha256sum awk; do command -v "$t" >/dev/null || { echo "qwen38_serve_speed.sh: needs $t" >&2; exit 2; }; done
[ -f "$CORPUS" ] || { echo "qwen38_serve_speed.sh: corpus not found: $CORPUS" >&2; exit 2; }
[ -f "$REF" ] || { echo "qwen38_serve_speed.sh: reference ids not found: $REF" >&2; exit 2; }
[ -f "$LOG" ] || { echo "qwen38_serve_speed.sh: server log not found: $LOG (pass it as the 4th argument or set AIEN_SERVE_LOG)" >&2; exit 2; }
mkdir -p "$OUT"
# shellcheck source=tools/qwen38_receipt.sh
. "$REPO/tools/qwen38_receipt.sh"
q38_server_facts "$URL"
Q38_MACHINE_START=$(q38_machine_json)
: > "$OUT/runs.jsonl"
# run <name> <messages-json> <max_tokens>
run() {
  local w=""
  jq -n --argjson m "$2" --argjson n "$3" \
    '{messages:$m, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}' > "$OUT/$1.req.json"
  for try in $(seq 1 240); do
    w=$(curl -s -m 1800 -o "$OUT/$1.json" -w '%{time_total}' "$URL/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @"$OUT/$1.req.json")
    # Retry only "busy" answers (server_busy, chip lock); any other error is the result.
    jq -e '.error | tostring | test("busy"; "i")' "$OUT/$1.json" >/dev/null 2>&1 || break
    sleep 5
  done
  q38_run_json "$1" "$OUT/$1.json" "$w" "$LOG" >> "$OUT/runs.jsonl"
  printf '%s\tclient_wall_s=%s\tprompt=%s\tcompletion=%s\t%s\n' "$1" "$w" \
    "$(jq '.usage.prompt_tokens' "$OUT/$1.json")" "$(jq '.usage.completion_tokens' "$OUT/$1.json")" \
    "$(q38_log_line "$LOG" "$(jq -r '.id' "$OUT/$1.json")")" | tee -a "$OUT/runs.tsv"
}
text() { head -c "$1" "$CORPUS" | iconv -c -f utf-8 -t utf-8; }
: > "$OUT/runs.tsv"
# Same request as the reference (evidence/qwen38-serve-fast-20261019/decode_runs.sh).
SHORT=$(jq -c '.request.messages' "$REF")
for i in 1 2 3; do run "decode$i" "$SHORT" 256; done
T17=$(text 7100); T30=$(text 100000)
run ttft_1p7k "$(jq -n --arg t "$T17" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)}]')" 1
run ttft_30k "$(jq -n --arg t "$T30" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)}]')" 1
A=$(jq -r '.choices[0].message.content' "$OUT/ttft_30k.json")
run next_turn_30k "$(jq -n --arg t "$T30" --arg a "$A" '[{role:"user",content:("Summarize this text in one sentence.\n\n" + $t)},{role:"assistant",content:$a},{role:"user",content:"Now name the three most important documents it mentions."}]')" 1

# Decode ids against the shipped reference, and the speed summary.
WANT=$(jq -c '.ids' "$REF")
MATCH=MATCH
for i in 1 2 3; do
  [ "$(jq -c '.usage.aien_token_ids' "$OUT/decode$i.json")" = "$WANT" ] || MATCH=MISMATCH
done
MED=$(jq -s '[.[] | select(.name | startswith("decode")) | .chip_ms_per_token | select(. != null)] | sort
  | if length == 0 then null elif length % 2 == 1 then .[length / 2 | floor] else (.[length / 2 - 1] + .[length / 2]) / 2 end' "$OUT/runs.jsonl")
EXTRA=$(jq -n --arg ref "crates/aien-inference-abi/fixtures/qwen38/decode-lighthouses-256.json" \
  --arg refsha "$(sha256sum "$REF" | cut -d' ' -f1)" --arg corpus "$CORPUS" \
  --arg csha "$(sha256sum "$CORPUS" | cut -d' ' -f1)" --arg m "$MATCH" --argjson med "$MED" \
  --argjson ceil "$Q38_CEILING_TOK_S" \
  '{reference:{file:$ref, sha256:$refsha}, corpus:{file:$corpus, sha256:$csha},
    decode_ids:$m,
    decode_median_chip_ms_per_token:$med,
    decode_tokens_per_s:(if $med then 1000 / $med else null end),
    ceiling_tokens_per_s:$ceil,
    share_of_ceiling:(if $med then (1000 / $med) / $ceil else null end)}')
q38_write_receipt "$OUT/receipt.json" speed "$URL" "$OUT/runs.jsonl" "$EXTRA"
echo "server binary sha256 $Q38_BIN_SHA256, build $Q38_BUILD, model dir $Q38_MODEL_DIR"
echo "decode ids against the shipped reference: $MATCH"
echo "median chip ms per token (3 decodes): $MED"
echo "receipt: $OUT/receipt.json"
