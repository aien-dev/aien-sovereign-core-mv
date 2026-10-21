#!/bin/bash
# L355 copy of L312 motor-scripts/gate2-http.sh: golden = host W4 chat golden (identical ids to the FP8 golden), evidence here.
# L312 motor gate 2 through the real HTTP door: greedy token ids of the chat prompt from the served 35B must equal the
# host reference golden (evidence/q36-host-20261020/gen-chat-max-24.json, thinking off, 24 tokens).
#   gate2-http.sh <port> <label>    -> writes $E/gate2-<label>.json, prints PASS/FAIL, exit 0/1
# A mutant server (AIEN_Q36_TEST_MUTANT=4) run through the same script must print FAIL (negative control).
set -u
PORT=${1:?port}; LABEL=${2:?label}
G=${G:-$HOME/workspace/sc-l355/evidence/l355-w4/gold-w4e-chat-max-24.json}
E=$HOME/workspace/sc-l355/evidence/l355-w4
mkdir -p $E
MODEL=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | jq -r '.data[0].id // empty' 2>/dev/null)
if [ -z "$MODEL" ]; then echo "gate2 $LABEL port $PORT: no answer from /v1/models -> ERROR (not a verdict)"; exit 2; fi
BODY=$(jq -cn --arg m "$MODEL" '{model:$m, messages:[{role:"user", content:"Write a C function that returns the larger of two ints."}], temperature:0, max_tokens:24, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}')
T0=$(date +%s.%N)
RESP=$(curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY")
T1=$(date +%s.%N)
if ! echo "$RESP" | jq -e '.usage.aien_token_ids' >/dev/null 2>&1; then echo "gate2 $LABEL port $PORT: no token ids in the reply -> ERROR (not a verdict): $(echo "$RESP" | cut -c1-200)"; exit 2; fi
GOT=$(echo "$RESP" | jq -c '.usage.aien_token_ids')
WANT=$(jq -c '.generated_ids' $G)
PT=$(echo "$RESP" | jq -r '.usage.prompt_tokens // "?"')
TEXT=$(echo "$RESP" | jq -r '.choices[0].message.content // ""')
if [ "$GOT" = "$WANT" ]; then V=PASS; else V=FAIL; fi
jq -n --arg port "$PORT" --arg model "$MODEL" --argjson got "$GOT" --argjson want "$WANT" --arg text "$TEXT" --arg pt "$PT" --arg v "$V" --arg wall "$(echo "$T1 - $T0" | bc)" --arg at "$(date -u +%FT%TZ)" \
  '{at:$at, port:$port, model:$model, prompt_tokens:$pt, golden_prompt_tokens:24, got_ids:$got, golden_ids:$want, text:$text, wall_s:$wall, identical:($got==$want), verdict:$v}' > $E/gate2-$LABEL.json
echo "gate2 $LABEL port $PORT model $MODEL prompt_tokens $PT wall ${T1}: ids $( [ "$GOT" = "$WANT" ] && echo identical || echo DIFFER ) -> $V"
[ "$V" = PASS ]
