#!/bin/bash
# L369 door gate: the same requests through the batch door with the chip prefill off (AIEN_Q36_PREFILL_CHUNK=0, prompts
# token by token = today's path) and on (chunks of 128) must give the same token ids, request by request.
#   door-gate.sh run <port> <label>   send every request to a running server, replies in <label>/
#   door-gate.sh cmp <labelA> <labelB> compare: PASS = every reply's ids identical. Exit 0 PASS, 1 FAIL, 2 no verdict.
# Requests, all at once (rows join mid-spell, chunks and single steps share beats): the 32 short prompts of L354's gate
# (greedy, 8..48 tokens), three long prompts (~256, ~2k, ~8k tokens of sc docs + a question; greedy, 24 tokens), one
# SAMPLED request on the ~2k prompt (temperature 0.8, top_p 0.95, seed 7) beside them, then a second turn on the ~2k
# conversation (the door continues its row from the copy at the last turn header). Prompt 0 alone must equal its golden.
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
G=$HOME/workspace/sc-q36-batch/evidence/q36-host-20261020/gen-chat-max-24.json
body() { # <model> <max_tokens> <messages json> [sampled]
  if [ "${4:-}" = sampled ]; then
    jq -cn --arg m "$1" --argjson n "$2" --argjson msg "$3" '{model:$m, messages:$msg, temperature:0.8, top_p:0.95, seed:7, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}'
  else
    jq -cn --arg m "$1" --argjson n "$2" --argjson msg "$3" '{model:$m, messages:$msg, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}'
  fi; }
ask() { # <port> <model> <max_tokens> <messages json> <out> [sampled]
  local t0=$(date +%s.%N) b busy=0; b=$(body "$2" "$3" "$4" "${6:-}")
  while curl -s -m 1800 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' -d "$b" > "$5.raw"; grep -q '"server_busy"' "$5.raw" && [ $busy -lt 2400 ]; do busy=$((busy+1)); sleep 0.5; done
  jq --arg t0 "$t0" --arg t1 "$(date +%s.%N)" '{ids:.usage.aien_token_ids, text:.choices[0].message.content, finish:.choices[0].finish_reason, prompt_tokens:.usage.prompt_tokens, t0:$t0, t1:$t1}' "$5.raw" > "$5" 2>/dev/null || echo '{"ids":null}' > "$5"; }
ntok() { if [ "$1" = 0 ]; then echo 24; else echo $(( 8 + ($1 * 7) % 41 )); fi; }
user() { jq -cn --rawfile c "$1" '[{role:"user", content:$c}]'; }
userline() { jq -cn --arg c "$1" '[{role:"user", content:$c}]'; }
case ${1:?run|cmp} in
run)
  P=${2:?port}; L=${3:?label}; mkdir -p "$L"
  M=$(curl -s -m 10 http://127.0.0.1:$P/v1/models | jq -r '.data[0].id // empty'); [ -n "$M" ] || { echo "door-gate: no answer from $P"; exit 2; }
  mapfile -t S < short.txt
  T0=$(date +%s.%N)
  for n in 8192 2048 256; do ask $P "$M" 24 "$(user long-$n.txt)" "$L/long-$n.json" & sleep 0.3; done
  ask $P "$M" 24 "$(user long-2048.txt)" "$L/sampled-2048.json" sampled &
  for ((i=0; i<${#S[@]}; i++)); do ( sleep $(echo "scale=3; ($i % 8) * 0.25" | bc); ask $P "$M" $(ntok $i) "$(userline "${S[$i]}")" "$L/short-$i.json" ) & done
  wait
  T1=$(date +%s.%N)
  R=$(jq -r .text "$L/long-2048.json")
  jq -cn --rawfile a long-2048.txt --arg r "$R" '[{role:"user",content:$a},{role:"assistant",content:$r},{role:"user",content:"Now say it again in five words."}]' > "$L/turn2.messages.json"
  ask $P "$M" 16 "$(cat "$L/turn2.messages.json")" "$L/turn2.json"
  echo "door-gate run $L: $(ls "$L"/*.json | grep -vc messages) replies in $(echo "$T1 - $T0" | bc) s; prompt tokens long: $(for n in 256 2048 8192; do jq -r .prompt_tokens "$L/long-$n.json"; done | tr '\n' ' ')"
  ;;
cmp)
  A=${2:?labelA}; B=${3:?labelB}; same=0; diff=0; bad=""
  for f in "$A"/*.json; do
    n=$(basename "$f"); case $n in *messages*) continue;; esac
    a=$(jq -c .ids "$f"); b=$(jq -c .ids "$B/$n" 2>/dev/null)
    if [ "$a" != null ] && [ -n "$a" ] && [ "$a" = "$b" ]; then same=$((same+1)); else diff=$((diff+1)); bad="$bad $n"; fi
  done
  gold=$( [ "$(jq -c .ids "$B/short-0.json")" = "$(jq -c .generated_ids $G)" ] && echo true || echo false)
  V=FAIL; [ $diff = 0 ] && [ $same -gt 0 ] && [ $gold = true ] && V=PASS
  jq -n --arg a "$A" --arg b "$B" --argjson same $same --argjson diff $diff --arg bad "$bad" --argjson gold $gold --arg v $V --arg at "$(date -u +%FT%TZ)" \
    '{at:$at, a:$a, b:$b, identical:$same, differ:$diff, differ_names:$bad, prompt0_eq_golden:$gold, verdict:$v}' > "cmp-$A-vs-$B.json"
  echo "door-gate cmp $A vs $B: $same identical, differ [$bad ], golden $gold -> $V"
  [ $V = PASS ]
  ;;
esac
