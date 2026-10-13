#!/usr/bin/env bash
# T2 metric: time to first token of the FINAL chat request for a ~2.3k-token prompt,
# without and with ahead-of-Enter prefill. Needs only bash, curl, awk (no Python).
# Usage: own30b-t2-ttft.sh [base-url, default http://127.0.0.1:18091] [runs, default 3]
# The server must be an aien-serve-http built from this branch with the Omega engine.
# Chip rule: run the SERVER under `flock /tmp/aien-gb10.lock` is NOT needed here (the server
# takes the lock per generation itself); but a chip window must be agreed first.
set -euo pipefail
BASE="${1:-http://127.0.0.1:18091}"
RUNS="${2:-3}"
SENT="The assistant follows the operator rules, uses tools carefully, and answers briefly. "
SYS=""
for _ in $(seq 1 28); do SYS+="$SENT"; done   # about 2.3k tokens with the Qwen tokenizer

esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# first-byte time of a streamed 1-token chat request, in ms
ttft_ms() { # $1=session or "" $2=user text $3=salt
  local sess="" body
  [ -n "$1" ] && sess=",\"session\":\"$1\""
  body="{\"messages\":[{\"role\":\"system\",\"content\":\"[$3] $(esc "$SYS")\"},{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":1,\"temperature\":0,\"stream\":true$sess}"
  curl -s -o /dev/null -w '%{time_starttransfer}' -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body" | awk '{printf "%.0f", $1*1000}'
}
wait_idle() {
  for _ in $(seq 1 6000); do
    curl -s "$BASE/v1/aien/sessions" | grep -q '"pending_session":null' && return 0
    sleep 0.05
  done
  echo "prefill did not finish" >&2; return 1
}
prefill() { # $1=session $2=draft $3=salt
  curl -s -X POST "$BASE/v1/aien/prefill" -H 'Content-Type: application/json' \
    -d "{\"session\":\"$1\",\"messages\":[{\"role\":\"system\",\"content\":\"[$3] $(esc "$SYS")\"},{\"role\":\"user\",\"content\":\"$2\"}]}"
}

FINAL="What is the capital of France? Answer in one word."
DRAFT="What is the capital of France? Answer in one"
for i in $(seq 1 "$RUNS"); do
  salt="cold-$RANDOM-$i"
  # a different salt each run so no earlier KV can match (the salt is the first thing in the prompt)
  cold=$(ttft_ms "" "$FINAL" "$salt")
  salt2="warm-$RANDOM-$i"
  pf=$(prefill "t2-$i" "$DRAFT" "$salt2")
  wait_idle
  warm=$(ttft_ms "t2-$i" "$FINAL" "$salt2")
  echo "run $i: without prefill-ahead ${cold} ms | with prefill-ahead ${warm} ms | prefill reply: $pf"
done
