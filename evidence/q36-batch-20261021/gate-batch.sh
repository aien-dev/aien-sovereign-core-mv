#!/bin/bash
# L354 gate: many conversations at once through the 35B batch door must give each request exactly the
# greedy token ids it gets alone.
#   gate-batch.sh <alone_port> <batch_port|-> <label> [stagger_ms]   (batch_port - = only collect $AL/)
# alone: every prompt of prompts.txt, one at a time, to <alone_port> (the live single door 18091 = the gated served
#   path), kept in $AL/ and reused when complete. together: all prompts to <batch_port> at once, request i
#   started (i % 8) * stagger_ms later (rows join and leave mid-spell), max_tokens 8..48 (prompt 0: 24, its host
#   golden gen-chat-max-24). turn2: a second turn on prompt 5's conversation (the batch door continues its row).
# PASS = every together reply's ids == alone, turn2 == alone, prompt 0 alone == golden. Exit 0 PASS, 1 FAIL, 2 no verdict.
# Never run this under flock: the servers take /tmp/aien-gb10.lock themselves.
set -u
AL=${ALONE:-alone}   # the alone replies folder (ALONE=dir keeps several references)
A=${1:?alone port}; B=${2:?batch port}; L=${3:?label}; ST=${4:-250}
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
G=$D/../q36-host-20261020/gen-chat-max-24.json
mapfile -t P < prompts.txt
N=${#P[@]}
model() { curl -s -m 10 http://127.0.0.1:$1/v1/models | jq -r '.data[0].id // empty'; }
body() { # <model> <max_tokens> <messages json>
  jq -cn --arg m "$1" --argjson n "$2" --argjson msg "$3" '{model:$m, messages:$msg, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}'; }
ask() { # <port> <model> <max_tokens> <messages json> <out>
  local t0=$(date +%s.%N) b; b=$(body "$2" "$3" "$4"); local busy=0
  # A real client's retry on 503 server_busy (every slot taken): wait half a second, ask again.
  while curl -s -m 900 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' -d "$b" > "$5.raw"; grep -q '"server_busy"' "$5.raw" && [ $busy -lt 2400 ]; do busy=$((busy+1)); sleep 0.5; done
  jq --arg t0 "$t0" --arg t1 "$(date +%s.%N)" '{ids:.usage.aien_token_ids, text:.choices[0].message.content, finish:.choices[0].finish_reason, prompt_tokens:.usage.prompt_tokens, t0:$t0, t1:$t1, busy_retries:($busy|tonumber)}' --arg busy $busy "$5.raw" > "$5" 2>/dev/null || echo '{"ids":null}' > "$5"; }
ntok() { if [ "$1" = 0 ]; then echo 24; else echo $(( 8 + ($1 * 7) % 41 )); fi; }
user() { jq -cn --arg c "$1" '[{role:"user", content:$c}]'; }
MB=$( [ "$B" = - ] && echo - || model $B)
# The alone replies are kept ($AL/done): then 18091 need not serve (it may be stopped for tests).
if [ -f $AL/done ] && [ "$B" != - ]; then MA=$MB; else MA=$(model $A); fi
[ -n "$MA" ] && [ -n "$MB" ] || { echo "gate-batch $L: no answer from port $A or $B -> ERROR (not a verdict)"; exit 2; }
mkdir -p "$AL" "$L"
if [ ! -f $AL/done ]; then
  for ((i=0; i<N; i++)); do ask $A "$MA" $(ntok $i) "$(user "${P[$i]}")" $AL/$i.json; done
  T5=$(jq -r .text $AL/5.json)
  M2=$(jq -cn --arg a "${P[5]}" --arg r "$T5" '[{role:"user",content:$a},{role:"assistant",content:$r},{role:"user",content:"Now say it again in one short sentence."}]')
  echo "$M2" > $AL/turn2.messages.json
  ask $A "$MA" 16 "$M2" $AL/turn2.json
  date -u +%FT%TZ > $AL/done
fi
[ "$B" = - ] && { echo "gate-batch: alone replies kept in $AL/"; exit 0; }
T0=$(date +%s.%N)
for ((i=0; i<N; i++)); do
  ( sleep $(echo "scale=3; ($i % 8) * $ST / 1000" | bc); ask $B "$MB" $(ntok $i) "$(user "${P[$i]}")" "$L/$i.json" ) &
done
wait
T1=$(date +%s.%N)
ask $B "$MB" 16 "$(cat $AL/turn2.messages.json)" "$L/turn2.json"
same=0; diff=0; bad=""
for ((i=0; i<N; i++)); do
  if [ "$(jq -c .ids $AL/$i.json)" != null ] && [ "$(jq -c .ids $AL/$i.json)" = "$(jq -c .ids $L/$i.json)" ]; then same=$((same+1)); else diff=$((diff+1)); bad="$bad $i"; fi
done
t2=$( [ "$(jq -c .ids $AL/turn2.json)" != null ] && [ "$(jq -c .ids $AL/turn2.json)" = "$(jq -c .ids $L/turn2.json)" ] && echo true || echo false)
gold=$( [ "$(jq -c .ids $AL/0.json)" = "$(jq -c .generated_ids $G)" ] && echo true || echo false)
toks=$(cat $L/[0-9]*.json | jq -s '[.[].ids | length] | add')
V=FAIL; [ $diff = 0 ] && [ $t2 = true ] && [ $gold = true ] && V=PASS
jq -n --arg at "$(date -u +%FT%TZ)" --arg l "$L" --arg a "$A" --arg b "$B" --argjson n $N --argjson same $same --argjson diff $diff --arg bad "$bad" \
  --argjson t2 $t2 --argjson gold $gold --argjson toks "${toks:-0}" --arg wall "$(echo "$T1 - $T0" | bc)" --arg st $ST --arg v $V \
  '{at:$at, label:$l, alone_port:$a, batch_port:$b, prompts:$n, identical:$same, differ:$diff, differ_idx:$bad, turn2_identical:$t2, prompt0_alone_eq_golden:$gold, together_tokens:$toks, together_wall_s:$wall, stagger_ms:$st, verdict:$v}' > "$L/verdict.json"
echo "gate-batch $L: $same/$N identical, differ [$bad ], turn2 $t2, golden $gold, $toks tokens in $(echo "$T1 - $T0" | bc) s -> $V"
[ $V = PASS ]
