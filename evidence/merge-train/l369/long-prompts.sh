#!/bin/bash
# L369 on the merge train (L387): long prompts through the batch door. The chip prefill reads a prompt in chunks of 128;
# the words must be the reference build's (prompts token by token), request by request.
#   long-prompts.sh ref <port> <dir>   reference: each request ALONE, one at a time, against the reference build
#   long-prompts.sh run <port> <out>   the train case: the 3 long prompts (~256/~2k/~8k), one SAMPLED request on the ~2k
#                                      prompt (temperature 0.8, top_p 0.95, seed 7) and the suite's 32 short prompts all
#                                      at once (short rows join while the long ones are read), then turn 2 on the ~2k
#                                      conversation; PASS = every reply's ids == the reference (long, sampled, turn 2)
#                                      and == the store's alone replies (short). Exit 0 PASS, 1 FAIL, 2 no verdict.
# Reference: $REFS_DIR (default the train's CURRENT reference)/l369-long. Adapted from L369's targets/l369/door/door-gate.sh.
set -u
D=$(cd "$(dirname "$0")" && pwd); MT=$(cd "$D/.." && pwd); REPO=$(cd "$MT/../.." && pwd)
MODEL_NAME=$(jq -r .model_dir "$MT/suite.json")
REFS_DIR=${REFS_DIR:-$HOME/.local/state/aien-q36/merge-train/refs/$MODEL_NAME/CURRENT}
mapfile -t S < "$REPO/$(jq -r .prompts "$MT/suite.json")"
body() { # <model> <max_tokens> <messages json> [sampled]
  if [ "${4:-}" = sampled ]; then
    jq -cn --arg m "$1" --argjson n "$2" --argjson msg "$3" '{model:$m, messages:$msg, temperature:0.8, top_p:0.95, seed:7, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}'
  else
    jq -cn --arg m "$1" --argjson n "$2" --argjson msg "$3" '{model:$m, messages:$msg, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}'
  fi; }
ask() { # <port> <model> <max_tokens> <messages json> <out> [sampled]
  local t0 b busy=0; t0=$(date +%s.%N); b=$(body "$2" "$3" "$4" "${6:-}")
  while curl -s -m 1800 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' -d "$b" > "$5.raw"; grep -q '"server_busy"' "$5.raw" && [ $busy -lt 2400 ]; do busy=$((busy+1)); sleep 0.5; done
  jq --arg t0 "$t0" --arg t1 "$(date +%s.%N)" '{ids:.usage.aien_token_ids, text:.choices[0].message.content, finish:.choices[0].finish_reason, prompt_tokens:.usage.prompt_tokens, t0:$t0, t1:$t1}' "$5.raw" > "$5" 2>/dev/null || echo '{"ids":null}' > "$5"; }
ntok() { if [ "$1" = 0 ]; then echo 24; else echo $(( 8 + ($1 * 7) % 41 )); fi; }
user() { jq -cn --rawfile c "$1" '[{role:"user", content:$c}]'; }
userline() { jq -cn --arg c "$1" '[{role:"user", content:$c}]'; }
turn2() { # <port> <model> <dir>
  local r; r=$(jq -r .text "$3/long-2048.json")
  jq -cn --rawfile a "$D/long-2048.txt" --arg r "$r" '[{role:"user",content:$a},{role:"assistant",content:$r},{role:"user",content:"Now say it again in five words."}]' > "$3/turn2.messages"
  ask "$1" "$2" 16 "$(cat "$3/turn2.messages")" "$3/turn2.json"; }
model() { curl -s -m 10 "http://127.0.0.1:$1/v1/models" | jq -r '.data[0].id // empty'; }
case ${1:?ref|run} in
ref)
  P=${2:?port}; O=${3:?dir}; mkdir -p "$O"; M=$(model "$P"); [ -n "$M" ] || { echo "long-prompts: no answer from $P"; exit 2; }
  for n in 256 2048 8192; do ask "$P" "$M" 24 "$(user "$D/long-$n.txt")" "$O/long-$n.json"; done
  ask "$P" "$M" 24 "$(user "$D/long-2048.txt")" "$O/sampled-2048.json" sampled
  turn2 "$P" "$M" "$O"
  for f in "$O"/*.json; do [ "$(jq -c .ids "$f")" != null ] || { echo "long-prompts ref: $f has no ids"; exit 2; }; done
  echo "$(date -u +%FT%TZ) port $P model $M $(curl -s -m 10 "http://127.0.0.1:$P/health" | jq -c '{build:(.build // .version // null)}') ${REF_NOTE:-}" > "$O/done"
  echo "long-prompts ref $O: $(for n in 256 2048 8192; do jq -r .prompt_tokens "$O/long-$n.json"; done | tr '\n' ' ')prompt tokens"
  ;;
run)
  P=${2:?port}; O=${3:?out}/l369-long; R=$REFS_DIR/l369-long; A=$REFS_DIR/alone; mkdir -p "$O"
  [ -s "$R/done" ] || { echo "long-prompts: no reference at $R"; exit 2; }
  M=$(model "$P"); [ -n "$M" ] || { echo "long-prompts: no answer from $P"; exit 2; }
  for n in 8192 2048 256; do ask "$P" "$M" 24 "$(user "$D/long-$n.txt")" "$O/long-$n.json" & sleep 0.3; done
  ask "$P" "$M" 24 "$(user "$D/long-2048.txt")" "$O/sampled-2048.json" sampled &
  for ((i=0; i<${#S[@]}; i++)); do ( sleep "$(echo "scale=3; ($i % 8) * 0.25" | bc)"; ask "$P" "$M" "$(ntok $i)" "$(userline "${S[$i]}")" "$O/short-$i.json" ) & done
  wait
  turn2 "$P" "$M" "$O"
  same=0; bad=""
  for f in "$R"/*.json; do n=$(basename "$f"); a=$(jq -c .ids "$f"); b=$(jq -c .ids "$O/$n" 2>/dev/null)
    if [ "$a" != null ] && [ "$a" = "$b" ]; then same=$((same+1)); else bad="$bad $n"; fi; done
  for ((i=0; i<${#S[@]}; i++)); do a=$(jq -c .ids "$A/$i.json" 2>/dev/null); b=$(jq -c .ids "$O/short-$i.json" 2>/dev/null)
    if [ -n "$a" ] && [ "$a" != null ] && [ "$a" = "$b" ]; then same=$((same+1)); else bad="$bad short-$i"; fi; done
  nulls=$(cat "$O"/*.json | jq -s '[.[] | select(.ids == null)] | length')
  V=FAIL; [ -z "$bad" ] && [ "$same" -gt 0 ] && V=PASS
  echo "long-prompts run: $same identical, differ [$bad ], no-ids $nulls; prompt tokens $(for n in 256 2048 8192; do jq -r .prompt_tokens "$O/long-$n.json"; done | tr '\n' ' ')-> $V"
  [ $V = PASS ] && exit 0; [ "$nulls" -gt 0 ] && exit 2; exit 1
  ;;
esac
