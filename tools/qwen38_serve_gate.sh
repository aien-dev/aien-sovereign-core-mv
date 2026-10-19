#!/bin/bash
# Correctness gate for the live Qwen3.8-27B server (sc #642): greedy ids through the real HTTP
# door against the host reference golden, a negative control that must fail, and next-turn resume
# (snapshot and live) against cold recomputation. Usage: qwen38_serve_gate.sh <base-url> <golden.json> <out-dir>
set -uo pipefail
URL=${1:-http://127.0.0.1:18091}; GOLDEN=${2:-$HOME/handoffs/qwen38-ref/gen-chat-max-24.json}; OUT=${3:?out dir}
LOG=${4:-$HOME/.local/state/aien-q38/serve.log}
mkdir -p "$OUT"
N=$(jq '.generated_ids | length' "$GOLDEN")
WANT=$(jq -c '.generated_ids' "$GOLDEN")
WANT_PROMPT=$(jq '.prompt_ids | length' "$GOLDEN")
Q1="Write a C function that returns the larger of two ints."
Q1X="Write a C function that returns the smaller of two ints."
Q2="Now make it work for three ints."
Q3="Thanks."
# ask <name> <messages-json>: greedy, thinking off, ids returned; saves the reply as <name>.json
ask() {
  jq -n --argjson m "$2" --argjson n "$N" \
    '{messages:$m, temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}' \
    > "$OUT/$1.req.json"
  # Other clients share the server: retry while it answers busy (one generation at a time).
  for try in $(seq 1 240); do
    curl -s -m 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/$1.req.json" > "$OUT/$1.json"
    jq -e '.error' "$OUT/$1.json" >/dev/null 2>&1 || break
    sleep 5
  done
}
# The server's own record of how a reply's prompt was fed (the qwen38 line before its id).
reuse() { grep -B1 "$(jq -r '.id' "$OUT/$1.json"):" "$LOG" | grep -o 'reuse [A-Za-z]*([0-9]*)\|reuse Cold' | head -1; }
ids() { jq -c '.usage.aien_token_ids' "$OUT/$1.json"; }
content() { jq -r '.choices[0].message.content' "$OUT/$1.json"; }
m1=$(jq -n --arg q "$Q1" '[{role:"user",content:$q}]')
ask golden "$m1"
G_IDS=$(ids golden); G_PT=$(jq '.usage.prompt_tokens' "$OUT/golden.json")
C1=$(content golden)
# Turn 2b: the client sends back an edited reply, so the live state diverges and only the
# snapshot (taken before turn 1's assistant header) can be reused.
m2b=$(jq -n --arg q "$Q1" --arg q2 "$Q2" '[{role:"user",content:$q},{role:"assistant",content:"Sure."},{role:"user",content:$q2}]')
ask turn2b_resumed "$m2b"
# Turn 1 again (cold: nothing held matches) so turn 2 below continues from it.
ask golden2 "$m1"
m2=$(jq -n --arg q "$Q1" --arg a "$C1" --arg q2 "$Q2" '[{role:"user",content:$q},{role:"assistant",content:$a},{role:"user",content:$q2}]')
ask turn2_resumed "$m2"
C2=$(content turn2_resumed)
m3=$(jq -n --arg q "$Q1" --arg a "$C1" --arg q2 "$Q2" --arg a2 "$C2" --arg q3 "$Q3" \
  '[{role:"user",content:$q},{role:"assistant",content:$a},{role:"user",content:$q2},{role:"assistant",content:$a2},{role:"user",content:$q3}]')
ask turn3_resumed "$m3"
# Negative control: a different question must not give the golden ids. It also leaves no state
# that the turn 2 / turn 3 prompts can reuse, so the next two requests run cold.
ask control "$(jq -n --arg q "$Q1X" '[{role:"user",content:$q}]')"
ask turn2_cold "$m2"
ask turn2b_cold "$m2b"
ask control2 "$(jq -n --arg q "$Q1X" '[{role:"user",content:$q}]')"
ask turn3_cold "$m3"
pass=1
check() { if [ "$2" = 1 ]; then echo "PASS $1"; else echo "FAIL $1"; pass=0; fi; }
{
  # A busy chip (503) voids the run instead of failing it silently as null ids.
  for f in "$OUT"/*.json; do case $f in *.req.json) continue;; esac
    jq -e '.usage.aien_token_ids' "$f" >/dev/null || echo "VOID $(basename "$f"): $(head -c 200 "$f")"; done
  echo "golden ids want $WANT"
  echo "golden ids got  $G_IDS"
  check "golden: $N greedy ids equal the host reference" "$([ "$G_IDS" = "$WANT" ] && echo 1)"
  check "golden: prompt is the reference's $WANT_PROMPT tokens" "$([ "$G_PT" = "$WANT_PROMPT" ] && echo 1)"
  check "golden repeat: same ids" "$([ "$(ids golden2)" = "$WANT" ] && echo 1)"
  check "control (smaller): ids differ from the golden (must)" "$([ "$(ids control)" != "$WANT" ] && [ "$(ids control)" != null ] && echo 1)"
  echo "turn2 resumed $(ids turn2_resumed)"; echo "turn2 cold    $(ids turn2_cold)"
  for n in golden turn2b_resumed turn2_resumed turn3_resumed control turn2_cold turn2b_cold turn3_cold; do echo "$n: $(reuse $n)"; done
  check "turn 2b (edited reply) reused the snapshot" "$(reuse turn2b_resumed | grep -q Snapshot && echo 1)"
  check "turn 2 resumed request reused a state" "$(reuse turn2_resumed | grep -q 'Live\|Snapshot' && echo 1)"
  echo "turn2b resumed $(ids turn2b_resumed)"; echo "turn2b cold    $(ids turn2b_cold)"
  check "turn 2b resumed (snapshot) == cold" "$([ "$(ids turn2b_resumed)" = "$(ids turn2b_cold)" ] && [ "$(ids turn2b_cold)" != null ] && echo 1)"
  check "turn 3 resumed request reused a state" "$(reuse turn3_resumed | grep -q 'Live\|Snapshot' && echo 1)"
  check "cold requests ran cold" "$([ "$(reuse turn2_cold)" = "reuse Cold" ] && [ "$(reuse turn2b_cold)" = "reuse Cold" ] && [ "$(reuse turn3_cold)" = "reuse Cold" ] && echo 1)"
  check "turn 2 resumed == cold" "$([ "$(ids turn2_resumed)" = "$(ids turn2_cold)" ] && [ "$(ids turn2_cold)" != null ] && echo 1)"
  echo "turn3 resumed $(ids turn3_resumed)"; echo "turn3 cold    $(ids turn3_cold)"
  check "turn 3 resumed == cold" "$([ "$(ids turn3_resumed)" = "$(ids turn3_cold)" ] && [ "$(ids turn3_cold)" != null ] && echo 1)"
  check "turn 2 differs from turn 1 (no stale reply)" "$([ "$(ids turn2_cold)" != "$G_IDS" ] && echo 1)"
  echo "golden text: $C1"
  [ $pass = 1 ] && echo "GATE PASS" || echo "GATE FAIL"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
grep -q '^GATE PASS$' "$OUT/summary.txt"
