#!/bin/bash
# L376 chip gate: two Hermes-shaped chats on one 35B batch door with the team board on.
# Usage: gate.sh PORT OUTDIR SERVER_LOG BOARD_DIR
# A: posts a fact with the built-in board_post (answered in-server, never sent to the client).
# B: on its next turn, gets the fact in the appended block and answers from it.
# Receipts: join lines (reuse N tok) for B after a delivery and for A after a built-in round;
# C (delivery off) is the control that puts a number on the block's own tokens; D is B's turn
# with the block pasted by hand (delivery off): identical token ids. Eight readers post at once.
set -uo pipefail
P=$1; O=$2; SLOG=$3; BD=$4; mkdir -p "$O"
U="http://127.0.0.1:$P"
MODEL=$(curl -s "$U/v1/models" | jq -r '.data[0].id')
SYS="You are Hermes, an agent on the Spark working with other agents. Use the tools when they help. Be brief."
TOOLS='[{"type":"function","function":{"name":"read_file","description":"Read a file on the Spark","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]'
FACT="the release sha is 706dff5f"
fail=0
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/gate.log"; }
check() { if [ "$1" = 0 ]; then say "PASS $2"; else say "FAIL $2"; fail=1; fi; }
limit() { say "LIMIT $*"; }
# req NAME CONV AGENT MESSAGES_JSON [EXTRA_JSON]
req() {
  local n=$1 c=$2 a=$3 m=$4 x=${5:-"{}"}
  local mark; mark=$(wc -l < "$SLOG")
  jq -n --arg model "$MODEL" --arg sys "$SYS" --arg c "$c" --arg a "$a" --argjson m "$m" --argjson tools "$TOOLS" --argjson x "$x" \
    '{model:$model, temperature:0, max_tokens:300, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true,
      aien_conversation:$c, aien_agent:$a, tools:$tools, messages:([{role:"system",content:$sys}] + $m)} + $x' > "$O/$n.req.json"
  local t0 t1 code; t0=$(date +%s.%N)
  code=$(curl -s -o "$O/$n.json" -w '%{http_code}' --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$O/$n.req.json" "$U/v1/chat/completions")
  t1=$(date +%s.%N)
  sleep 0.3
  tail -n +$((mark + 1)) "$SLOG" | grep -E "join:|built-in tool|board" > "$O/$n.server.log"
  say "$n http $code wall $(echo "$t1 - $t0" | bc) s prompt $(jq -r '.usage.prompt_tokens' "$O/$n.json") completion $(jq -r '.usage.completion_tokens' "$O/$n.json") | $(grep -o 'join: .*' "$O/$n.server.log" | head -1)"
}
content() { jq -r '.choices[0].message.content // ""' "$O/$1.json"; }
ids() { jq -c '[.. | objects | select(has("aien_token_ids")) | .aien_token_ids] | first' "$O/$1.json"; }
ptok() { jq -r '.usage.prompt_tokens' "$O/$1.json"; }
reuse() { grep -o 'reuse [0-9]* tok' "$O/$1.server.log" | head -1 | awk '{print $2}'; }
plen() { grep -o 'prompt [0-9]* tok' "$O/$1.server.log" | head -1 | awk '{print $2}'; }
board_json() { curl -s "$U/v1/aien/board?since=${1:-0}&limit=200"; }

say "=== gate on $P, model $MODEL, board dir $BD"
curl -s "$U/health" > "$O/health.json"
check "$([ "$(jq -r '.board.tools' "$O/health.json")" = true ]; echo $?)" "health: board on with built-in tools ($(jq -c .board "$O/health.json"))"

# 1. A and B at once (two rows). A posts the fact with the built-in tool; B just starts.
M_B1='[{"role":"user","content":"Say hello in five words."}]'
M_A1="[{\"role\":\"user\",\"content\":\"Use the board_post tool to post this exact fact to the team board: $FACT. After the tool answers, reply with the single word done.\"}]"
req B1 L376-B agent-B "$M_B1" &
req A1 L376-A agent-A "$M_A1" &
wait
A1_ROUNDS=$(grep -c "built-in tool round" "$O/A1.server.log")
if board_json | jq -e --arg f "$FACT" '.entries[] | select(.author == "agent-A" and (.text | contains("706dff5f")))' > /dev/null; then
  check 0 "A1: the fact is on the board, posted by agent-A through the built-in tool ($A1_ROUNDS built-in round lines)"
else
  limit "A1: the model did not post with board_post (content: $(content A1 | head -c 120)); posting the fact by HTTP as agent-A"
  curl -s -X POST -H 'Content-Type: application/json' --data "{\"author\":\"agent-A\",\"text\":\"$FACT\"}" "$U/v1/aien/board" > "$O/A1.http-post.json"
fi
check "$(jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("board_post") == null' "$O/A1.json" > /dev/null; echo $?)" "A1: no board_post call reached the client"
check "$(grep -q "board" "$O/A1.server.log" || grep -q "board" "$O/B1.server.log" || true; [ "$(jq -r '.board.entries' <(curl -s "$U/health"))" -ge 1 ]; echo $?)" "board has at least one entry"

# 2. B's next turn: the fact arrives in the block; B answers from it. Join line: reuse >= B1's prompt
#    minus the generation prompt (the block is appended, nothing before it is re-read).
B1_TEXT=$(content B1)
M_B2=$(jq -n --arg b "$B1_TEXT" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":"According to the team board, what is the release sha? Reply with the sha only."}]')
req B2 L376-B agent-B "$M_B2"
check "$(content B2 | grep -q "706dff5f"; echo $?)" "B2 used the fact A posted: '$(content B2 | head -c 80)'"
B1P=$(ptok B1); B2R=$(reuse B2); B2P=$(plen B2)
check "$([ -n "$B2R" ] && [ "$B2R" -ge $((B1P - 16)) ]; echo $?)" "B2 join: reuse $B2R tok of prompt $B2P (B1 prompt $B1P): the row kept B1, only the new text was fed"

# 3. Control C: the same two turns with delivery off; the block's own tokens = B2 prompt - C2 prompt.
req C1 L376-C agent-C "$M_B1" '{"aien_board":false}'
req C2 L376-C agent-C "$M_B2" '{"aien_board":false}'
C2P=$(ptok C2); BLOCK_TOK=$((B2P - C2P)); B2_FED=$((B2P - B2R)); C2_FED=$((C2P - $(reuse C2)))
say "receipt: block tokens = $BLOCK_TOK (B2 prompt $B2P - C2 prompt $C2P); fed at B2 = $B2_FED, fed at C2 = $C2_FED, difference $((B2_FED - C2_FED))"
check "$([ $((B2_FED - C2_FED)) -eq "$BLOCK_TOK" ]; echo $?)" "tokens re-read per board update = the block's own tokens ($BLOCK_TOK)"

# 4. D: B2 again with the block pasted by hand (delivery off): identical token ids and words.
KEY="id-$(printf %s L376-B | sha256sum | cut -c1-16)"
BLOCK=$(jq -r '.patches[-1].text' "$BD/conversations/$KEY.json")
check "$([ -n "$BLOCK" ] && [ "$BLOCK" != null ]; echo $?)" "B's record has the delivered block ($(printf %s "$BLOCK" | wc -c) chars)"
M_D=$(jq -n --arg b "$B1_TEXT" --arg blk "$BLOCK" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":("According to the team board, what is the release sha? Reply with the sha only.\n\n" + $blk)}]')
req D L376-B agent-B "$M_D" '{"aien_board":false}'
check "$([ "$(ids D)" = "$(ids B2)" ] && [ "$(ids D)" != null ]; echo $?)" "D: token ids identical to B2 ($(ids B2 | jq length) ids)"
check "$([ "$(content D)" = "$(content B2)" ]; echo $?)" "D: words identical to B2"
check "$([ "$(ptok D)" = "$B2P" ]; echo $?)" "D: prompt tokens identical to B2 ($B2P)"

# 5. Eight readers post at once; B sees all eight next turn.
for i in 1 2 3 4 5 6 7 8; do
  curl -s -X POST -H 'Content-Type: application/json' --data "{\"author\":\"reader-$i\",\"text\":\"part $i of 8: read ok, 1 question answered\"}" "$U/v1/aien/board" > "$O/reader-$i.json" &
done
wait
NREAD=$(board_json | jq '[.entries[] | select(.author | startswith("reader-"))] | length')
check "$([ "$NREAD" = 8 ]; echo $?)" "eight readers' notes on the board ($NREAD)"
B2_TEXT=$(content B2)
M_B3=$(jq -n --arg b "$B1_TEXT" --arg b2 "$B2_TEXT" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":"According to the team board, what is the release sha? Reply with the sha only."},{"role":"assistant","content":$b2},{"role":"user","content":"How many reader parts reported on the team board since your last turn? Reply with the number only."}]')
req B3 L376-B agent-B "$M_B3"
check "$(content B3 | grep -q "8"; echo $?)" "B3 counted the readers: '$(content B3 | head -c 60)'"
check "$([ "$(jq -r '.patches[-1].text' "$BD/conversations/$KEY.json" | grep -c 'reader-')" = 8 ]; echo $?)" "B3's block carries the eight reader lines"
B3R=$(reuse B3); B3P=$(plen B3)
check "$([ -n "$B3R" ] && [ "$B3R" -ge $((B2P - 16)) ]; echo $?)" "B3 join: reuse $B3R of $B3P (B2 prompt $B2P): append-only across three turns"

# 6. A's next turn after the built-in round: the row kept the snapshot at the first round's header.
A1_TEXT=$(content A1)
M_A2=$(jq -n --arg u "$(jq -r '.[0].content' <<< "$M_A1")" --arg a "$A1_TEXT" '[{"role":"user","content":$u},{"role":"assistant","content":$a},{"role":"user","content":"Now read /etc/hostname with read_file."}]')
req A2 L376-A agent-A "$M_A2"
A1P=$(ptok A1); A2R=$(reuse A2); A2P=$(plen A2)
check "$([ -n "$A2R" ] && [ "$A2R" -ge $((A1P - 16)) ]; echo $?)" "A2 join: reuse $A2R of $A2P (A1 prompt $A1P): the continuation kept the row's snapshot"
check "$(jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("read_file") != null' "$O/A2.json" > /dev/null; echo $?)" "A2: the client's read_file call was forwarded"

# 7. Mixed round: a built-in post and a client call in one reply (model-dependent; recorded either way).
A2_CALLS=$(jq -c '.choices[0].message.tool_calls // []' "$O/A2.json")
M_A3=$(jq -n --argjson m "$M_A2" --arg a "$(content A2)" --argjson calls "$A2_CALLS" '$m + [{"role":"assistant","content":$a,"tool_calls":$calls}] + ($calls | map({"role":"tool","tool_call_id":.id,"content":"spark"})) + [{"role":"user","content":"Post to the team board with board_post that the hostname is spark, and in the same reply also call read_file on /etc/os-release."}]')
req A3 L376-A agent-A "$M_A3"
if board_json | jq -e '.entries[] | select(.author == "agent-A" and (.text | test("spark")))' > /dev/null \
   && jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("read_file") != null' "$O/A3.json" > /dev/null; then
  say "PASS A3 mixed round: board_post answered in-server, read_file forwarded"
else
  limit "A3 mixed round not produced by the model (tool_calls: $(jq -c '(.choices[0].message.tool_calls // []) | map(.function.name)' "$O/A3.json"), board spark entry: $(board_json | jq '[.entries[] | select(.text | test("spark"))] | length'))"
fi
A3R=$(reuse A3); A3P=$(plen A3); A2PT=$(ptok A2)
say "A3 join: reuse $A3R of $A3P (A2 prompt $A2PT)"

board_json > "$O/board-final.json"
cp "$BD/conversations/"*.json "$O/" 2>/dev/null
say "=== gate done, fail=$fail"
exit $fail
