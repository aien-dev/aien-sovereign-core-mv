#!/bin/bash
# L375 on the merge train: the 35B door's shared prompt cache, against the train's server.
#   prefix.sh <port> <out> <server_log>
# One server, one request at a time, Hermes-shaped requests (evidence/l375/mkreq.sh with 21 tools and a long
# system text, then one user turn; 16,830 prompt tokens with 20 tools, measured by the 35B tokenizer):
#   1 a         chat A, cache on (fills the cache)
#   2 b-pool    chat B: same head, another user message, cache on: restores the head
#   3 b-cold    chat B with aien_test.prefix_pool false (read cold): ids must equal 2
#   4 c-pool    chat C: one tool description changed (it moves to the end of the tools), cache on
#   5 c-cold    chat C read cold: ids must equal 4
#   6 d-mutant  chat D with aien_test.prefix_mutant (the restore lands one position early)
#   7 d-cold    chat D read cold: ids must DIFFER from 6 (the red control)
# PASS (exit 0): 2 == 3, 4 == 5, 6 != 7, and the server log shows a prompt cache restore for 2, 4 and 6.
# FAIL: exit 1. No verdict: exit 2 (switch not honored, no ids, no echo, prompt over the context: the case
# needs MERGE_TRAIN_CTX >= 18432). Receipts (time to first word, tokens re-read) in <out>/RESULTS.txt.
set -u
P=${1:?port}; O=${2:?out}; LOG=${3:?server log}
D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$D/../../.." && pwd)
mkdir -p "$O"; : > "$O/times.txt"
nov() { echo "l375 prefix: NO VERDICT: $*" | tee "$O/verdict.txt"; exit 2; }
curl -s -m 10 "http://127.0.0.1:$P/health" > "$O/health.json" || nov "no answer from port $P"
for s in prefix_pool prefix_mutant; do
  jq -e --arg s "$s" '.aien_test.honored | index($s)' "$O/health.json" > /dev/null || nov "switch $s not honored (lever not in this build)"
done
M=$(curl -s -m 10 "http://127.0.0.1:$P/v1/models" | jq -r '.data[0].id // empty'); [ -n "$M" ] || nov "no model id"

req() { # <tag> <user message> <changed tool index or -1> <aien_test json or null>
  (cd "$REPO" && bash evidence/l375/mkreq.sh "$O/$1.base.json" "$2" "$3" 21 > /dev/null) || nov "mkreq failed for $1"
  jq --arg m "$M" --argjson t "$4" '.model = $m | if $t == null then . else .aien_test = $t end' \
    "$O/$1.base.json" > "$O/$1.json"
}
ask() { # <tag>: one streamed request; ids, time to first word, the server log lines it wrote
  local n0 k; n0=$(wc -l < "$LOG")
  bash "$REPO/evidence/l375/ask.sh" "$P" "$O/$1.json" "$1" "$O" > /dev/null
  tail -n +"$((n0 + 1))" "$LOG" | grep -E 'qwen36-batch: (join|leave|prompt cache)|tool (order|marks)' > "$O/$1.log"
  grep -q 'context_length_exceeded' "$O/$1.sse" && nov "$1: prompt over the server's context (needs MERGE_TRAIN_CTX >= 18432)"
  if grep -qE '^(data: )?\{"error":' "$O/$1.sse" || [ ! -s "$O/$1.ids" ]; then
    echo "l375 prefix: FAIL: $1: the server failed the request (see $O/$1.sse)" | tee "$O/verdict.txt"; exit 1
  fi
  for k in $(jq -r '.aien_test // {} | keys[]' "$O/$1.json"); do
    grep -q "\"applied\":\[[^]]*\"$k\"" "$O/$1.sse" || nov "$1: the reply does not echo $k in usage.aien_test.applied"
  done
}
restored() { grep -q 'join: .*(prompt cache, row' "$O/$1.log"; }

req a "Summarise what the batch door does in two sentences." -1 null;            ask a
req b-pool "List three risks of a shared prompt cache." -1 null;                 ask b-pool
req b-cold "List three risks of a shared prompt cache." -1 '{"prefix_pool":false}'; ask b-cold
req c-pool "Which tool would you use to read a file?" 10 null;                   ask c-pool
req c-cold "Which tool would you use to read a file?" 10 '{"prefix_pool":false}'; ask c-cold
req d-mutant "Name one thing the merge train checks." 10 '{"prefix_mutant":true}'; ask d-mutant
req d-cold "Name one thing the merge train checks." 10 '{"prefix_pool":false}';  ask d-cold

fail=()
cmp -s "$O/b-pool.ids" "$O/b-cold.ids" || fail+=("chat B: cache words != cold words")
cmp -s "$O/c-pool.ids" "$O/c-cold.ids" || fail+=("chat C (tool changed): cache words != cold words")
cmp -s "$O/d-mutant.ids" "$O/d-cold.ids" && fail+=("RED-FAIL: the one-position-off restore gave the cold words")
for t in b-pool c-pool d-mutant; do restored "$t" || fail+=("$t: no prompt cache restore in the server log"); done
for t in b-cold c-cold d-cold; do grep -q "join: prompt [0-9]* tok, reuse 0 tok (cold" "$O/$t.log" || fail+=("$t: not a cold read in the server log"); done
grep -q "POSITION MUTANT" "$O/d-mutant.log" || fail+=("d-mutant: no POSITION MUTANT restore in the server log")

{
  echo "L375 prompt cache on the merge train, port $P, model $M, $(date -u +%FT%H:%MZ)"
  echo "time to first word (ms) and server join lines:"
  for t in a b-pool b-cold c-pool c-cold d-mutant d-cold; do
    echo "  $(grep "^$t " "$O/times.txt")"
    grep -o 'join: prompt [0-9]* tok, reuse [0-9]* tok ([a-z ]*' "$O/$t.log" | sed 's/^/      /'
  done
  echo "tool order: $(grep -ho 'tool order: .*' "$O"/c-pool.log | head -1)"
  echo "cache saves: $(cat "$O"/*.log | grep -c 'prompt cache save:'), restores: $(cat "$O"/*.log | grep -c 'prompt cache restore:')"
  grep -h 'prompt cache restore:' "$O"/*.log | sed 's/^.*prompt cache restore: /  restore: /' | head -5
} > "$O/RESULTS.txt"

if [ ${#fail[@]} -eq 0 ]; then
  echo "l375 prefix: PASS (B and C identical cold vs cache, mutant red, restores seen)" | tee "$O/verdict.txt"; exit 0
fi
printf 'l375 prefix: FAIL: %s\n' "${fail[@]}" | tee "$O/verdict.txt"; exit 1
