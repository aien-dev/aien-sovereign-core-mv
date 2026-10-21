#!/bin/bash
# L376 chip gate: two Hermes-shaped chats on one 35B batch door with the team board on.
# Usage: gate.sh PORT OUTDIR SERVER_LOG BOARD_DIR
# B1 runs first on an empty board (no block). A1 posts a fact with the built-in board_post
# (answered in-server, never sent to the client; the receipt is the "continuation" join line:
# the hidden round re-read nothing). B2 (the other chat's next turn) gets the fact in the appended
# block and answers from it. C (delivery off) is the control that puts a number on the block's own
# tokens; D is B2 with the block pasted by hand (delivery off): identical token ids and words.
# Eight readers post at once; B3 and A2 run together (two rows) and both keep their rows.
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
  tail -n +$((mark + 1)) "$SLOG" | grep -E "join:|built-in|board" > "$O/$n.server.log"
  say "$n http $code wall $(echo "$t1 - $t0" | bc) s prompt $(jq -r '.usage.prompt_tokens' "$O/$n.json") completion $(jq -r '.usage.completion_tokens' "$O/$n.json") | $(grep -o 'join: .*' "$O/$n.server.log" | head -1)"
}
content() { jq -r '.choices[0].message.content // ""' "$O/$1.json"; }
ids() { jq -c '[.. | objects | select(has("aien_token_ids")) | .aien_token_ids] | first' "$O/$1.json"; }
ptok() { jq -r '.usage.prompt_tokens' "$O/$1.json"; }
# join line N (1-based) of a request's server log: prompt, reuse, kind
jl() { grep -o 'join: .*' "$O/$1.server.log" | sed -n "${2:-1}p"; }
reuse() { jl "$1" "${2:-1}" | grep -o 'reuse [0-9]* tok' | awk '{print $2}'; }
plen() { jl "$1" "${2:-1}" | grep -o 'prompt [0-9]* tok' | awk '{print $2}'; }
kind() { jl "$1" "${2:-1}" | grep -o 'tok ([a-z]*' | sed 's/tok (//'; }
board_json() { curl -s "$U/v1/aien/board?since=${1:-0}&limit=200"; }
KEY_B="id-$(printf %s L376-B | sha256sum | cut -c1-16)"

say "=== gate on $P, model $MODEL, board dir $BD"
curl -s "$U/health" > "$O/health.json"
check "$([ "$(jq -r '.board.tools' "$O/health.json")" = true ]; echo $?)" "health: board on with built-in tools ($(jq -c .board "$O/health.json"))"

# 1. B1 first, on an empty board: no block, no patch (so B2 is the turn the fact arrives at).
M_B1='[{"role":"user","content":"Say hello in five words."}]'
req B1 L376-B agent-B "$M_B1"
check "$([ "$(jq -r '.patches | length' "$BD/conversations/$KEY_B.json" 2>/dev/null || echo 0)" = 0 ]; echo $?)" "B1: no block on an empty board"

# 2. A1 posts the fact with the built-in tool: a hidden round that continues the same row.
M_A1="[{\"role\":\"user\",\"content\":\"Use the board_post tool to post this exact fact to the team board: $FACT. After the tool answers, reply with the single word done.\"}]"
req A1 L376-A agent-A "$M_A1"
A1_OK=0
if board_json | jq -e '.entries[] | select(.author == "agent-A" and (.text | contains("706dff5f")))' > /dev/null; then
  A1_OK=1
  check 0 "A1: the fact is on the board, posted by agent-A through the built-in tool"
  A1P=$(plen A1 1); A1R2=$(reuse A1 2); A1P2=$(plen A1 2); A1K2=$(kind A1 2)
  E1=$(grep -o 'built-in tools: [0-9+]* tok' "$O/A1.server.log" | grep -o '[0-9]*' | head -1)
  check "$(jl A1 2 | grep -q continuation; echo $?)" "A1: the second join line is the continuation ($(jl A1 2))"
  check "$([ -n "$A1R2" ] && [ -n "$E1" ] && [ "$A1R2" = $((A1P + E1)) ]; echo $?)" "A1 continuation re-read nothing: reuse $A1R2 = round-1 prompt $A1P + round-1 reply $E1 tok ($A1K2); fed $((A1P2 - A1R2)) tok = the tool answer and the header"
else
  limit "A1: the model did not post with board_post (content: $(content A1 | head -c 120)); posting the fact by HTTP as agent-A; the continuation receipt and A2's snapshot check are LIMIT"
  curl -s -X POST -H 'Content-Type: application/json' --data "{\"author\":\"agent-A\",\"text\":\"$FACT\"}" "$U/v1/aien/board" > "$O/A1.http-post.json"
fi
check "$(jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("board_post") == null' "$O/A1.json" > /dev/null; echo $?)" "A1: no board_post call reached the client"
check "$([ "$(jq -r '.board.entries' <(curl -s "$U/health"))" -ge 1 ]; echo $?)" "board has at least one entry"

# 3. B's next turn: the fact arrives in the block; B answers from it.
B1_TEXT=$(content B1)
M_B2=$(jq -n --arg b "$B1_TEXT" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":"According to the team board, what is the release sha? Reply with the sha only."}]')
req B2 L376-B agent-B "$M_B2"
check "$(content B2 | grep -q "706dff5f"; echo $?)" "B2 used the fact A posted: '$(content B2 | head -c 80)'"
check "$([ "$(jq -r '.patches[-1].turn' "$BD/conversations/$KEY_B.json")" = 2 ]; echo $?)" "B2: the block landed at turn 2 (B's record)"
B1P=$(ptok B1); B2R=$(reuse B2); B2P=$(plen B2); B2K=$(kind B2)

# 4. Control C: the same two turns with delivery off; the block's own tokens = B2 prompt - C2 prompt.
#    C1 == B1 at temperature 0 (same prompt), so the join relation (reuse - previous prompt) must be
#    identical for B2 and C2: nothing before the block is re-read. No slack.
req C1 L376-C agent-C "$M_B1" '{"aien_board":false}'
req C2 L376-C agent-C "$M_B2" '{"aien_board":false}'
C1P=$(ptok C1); C2P=$(ptok C2); C2R=$(reuse C2); C2K=$(kind C2)
BLOCK_TOK=$((B2P - C2P)); B2_FED=$((B2P - B2R)); C2_FED=$((C2P - C2R))
say "receipt: block tokens = $BLOCK_TOK (B2 prompt $B2P - C2 prompt $C2P); fed at B2 = $B2_FED, fed at C2 = $C2_FED, difference $((B2_FED - C2_FED))"
check "$([ "$(content C1)" = "$B1_TEXT" ]; echo $?)" "C1 words == B1 words (the control is the same conversation without the board)"
check "$([ $((B2_FED - C2_FED)) -eq "$BLOCK_TOK" ]; echo $?)" "tokens re-read per board update = the block's own tokens ($BLOCK_TOK)"
check "$([ $((B2R - B1P)) -eq $((C2R - C1P)) ] && [ "$B2K" = "$C2K" ]; echo $?)" "B2 join: reuse $B2R of $B2P ($B2K) after B1 prompt $B1P, exactly the control's relation (C2 reuse $C2R of $C2P, $C2K, after C1 prompt $C1P)"
# GEN: tokens of the generation header the row's snapshot does not cover (0 when the row stayed live).
GEN=$(( C1P - C2R > 0 ? C1P - C2R : 0 ))

# 5. D: B2 again with the block pasted by hand (delivery off): identical token ids and words.
BLOCK=$(jq -r '.patches[-1].text' "$BD/conversations/$KEY_B.json")
check "$([ -n "$BLOCK" ] && [ "$BLOCK" != null ]; echo $?)" "B's record has the delivered block ($(printf %s "$BLOCK" | wc -c) chars)"
M_D=$(jq -n --arg b "$B1_TEXT" --arg blk "$BLOCK" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":("According to the team board, what is the release sha? Reply with the sha only.\n\n" + $blk)}]')
req D L376-B agent-B "$M_D" '{"aien_board":false}'
check "$([ "$(ids D)" = "$(ids B2)" ] && [ "$(ids D)" != null ]; echo $?)" "D: token ids identical to B2 ($(ids B2 | jq length) ids)"
check "$([ "$(content D)" = "$(content B2)" ]; echo $?)" "D: words identical to B2"
check "$([ "$(ptok D)" = "$B2P" ]; echo $?)" "D: prompt tokens identical to B2 ($B2P)"

# 6. Eight readers post at once (by HTTP: the L371 call site is a ledger item for L371, LIMIT here).
for i in 1 2 3 4 5 6 7 8; do
  curl -s -X POST -H 'Content-Type: application/json' --data "{\"author\":\"reader-$i\",\"text\":\"part $i of 8: read ok, 1 question answered\"}" "$U/v1/aien/board" > "$O/reader-$i.json" &
done
wait
NREAD=$(board_json | jq '[.entries[] | select(.author | startswith("reader-"))] | length')
check "$([ "$NREAD" = 8 ]; echo $?)" "eight readers' notes on the board ($NREAD)"
limit "the eight notes are posted by curl, not by L371's reader code (its call site: POST /v1/aien/board or Board::post)"

# 7. B3 and A2 together (two rows). B3 sees all eight; A2 (after A1's built-in round) keeps its row.
B2_TEXT=$(content B2)
M_B3=$(jq -n --arg b "$B1_TEXT" --arg b2 "$B2_TEXT" '[{"role":"user","content":"Say hello in five words."},{"role":"assistant","content":$b},{"role":"user","content":"According to the team board, what is the release sha? Reply with the sha only."},{"role":"assistant","content":$b2},{"role":"user","content":"How many reader parts reported on the team board since your last turn? Reply with the number only."}]')
A1_TEXT=$(content A1)
M_A2=$(jq -n --arg u "$(jq -r '.[0].content' <<< "$M_A1")" --arg a "$A1_TEXT" '[{"role":"user","content":$u},{"role":"assistant","content":$a},{"role":"user","content":"Now read /etc/hostname with read_file."}]')
req B3 L376-B agent-B "$M_B3" &
req A2 L376-A agent-A "$M_A2" &
wait
B3_NUM=$(content B3 | tr -d ' .\n\r')
check "$([ "$B3_NUM" = 8 ]; echo $?)" "B3 counted the readers exactly: '$(content B3 | head -c 60)'"
check "$([ "$(jq -r '.patches[-1].text' "$BD/conversations/$KEY_B.json" | grep -c 'reader-')" = 8 ]; echo $?)" "B3's block carries the eight reader lines"
B3R=$(reuse B3); B3P=$(plen B3); B3K=$(kind B3)
check "$([ -n "$B3R" ] && [ "$B3R" -ge $((B2P - GEN)) ]; echo $?)" "B3 join: reuse $B3R of $B3P ($B3K) >= B2 prompt $B2P - header $GEN: append-only across three turns"
A1PT=$(ptok A2 > /dev/null; ptok A1); A2R=$(reuse A2); A2P=$(plen A2); A2K=$(kind A2)
if [ "$A1_OK" = 1 ]; then
  check "$([ -n "$A2R" ] && [ "$A2R" -ge $((A1PT - GEN)) ]; echo $?)" "A2 join: reuse $A2R of $A2P ($A2K) >= A1 round-1 prompt $A1PT - header $GEN: the continuation kept the row's snapshot at the first round's header"
else
  limit "A2 join: reuse $A2R of $A2P ($A2K), A1 prompt $A1PT: A1 had no built-in round, so this proves nothing about keep_snapshot"
fi
check "$(jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("read_file") != null' "$O/A2.json" > /dev/null; echo $?)" "A2: the client's read_file call was forwarded"

# 8. Mixed round: a built-in post and a client call in one reply (model-dependent; recorded either way).
A2_CALLS=$(jq -c '.choices[0].message.tool_calls // []' "$O/A2.json")
M_A3=$(jq -n --argjson m "$M_A2" --arg a "$(content A2)" --argjson calls "$A2_CALLS" '$m + [{"role":"assistant","content":$a,"tool_calls":$calls}] + ($calls | map({"role":"tool","tool_call_id":.id,"content":"spark"})) + [{"role":"user","content":"Post to the team board with board_post that the hostname is spark, and in the same reply also call read_file on /etc/os-release."}]')
req A3 L376-A agent-A "$M_A3"
if board_json | jq -e '.entries[] | select(.author == "agent-A" and (.text | test("spark")))' > /dev/null \
   && jq -e '(.choices[0].message.tool_calls // []) | map(.function.name) | index("read_file") != null' "$O/A3.json" > /dev/null; then
  A3R=$(reuse A3); A3P=$(plen A3); A2PT=$(ptok A2)
  check "$([ -n "$A3R" ] && [ "$A3R" -ge $((A2PT - GEN)) ]; echo $?)" "A3 mixed round: board_post answered in-server, read_file forwarded; join reuse $A3R of $A3P >= A2 prompt $A2PT - header $GEN"
else
  limit "A3 mixed round not produced by the model (tool_calls: $(jq -c '(.choices[0].message.tool_calls // []) | map(.function.name)' "$O/A3.json"), board spark entry: $(board_json | jq '[.entries[] | select(.text | test("spark"))] | length')); join: $(jl A3 1)"
fi

board_json > "$O/board-final.json"
cp "$BD/conversations/"*.json "$O/" 2>/dev/null
say "=== gate done, fail=$fail"
exit $fail
