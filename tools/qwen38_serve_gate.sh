#!/bin/bash
# Correctness gate for the live Qwen3.8-27B server (sc #642): greedy ids through the real HTTP
# door against the host reference golden, a negative control that must fail, and next-turn resume
# (snapshot and live) against cold recomputation.
# Usage: qwen38_serve_gate.sh [base-url] [golden.json] <out-dir> [server-log]
#   base-url     default http://127.0.0.1:18091
#   golden.json  default: the host reference shipped in this repository
#                (crates/aien-inference-abi/fixtures/qwen38/gen-chat-max-24.json); "" also means default
#   server-log   default $AIEN_SERVE_LOG, else the log tools/serve-27b.sh writes:
#                ${XDG_STATE_HOME:-$HOME/.local/state}/aien/serve-27b.log
# Writes <out-dir>/summary.txt (names the server binary sha256 and model folder it tested) and
# <out-dir>/receipt.json (build, binary sha256, model checksum list, machine state, per-request
# figures, MATCH or MISMATCH against the shipped golden). See tools/qwen38_receipt.sh for how the
# server is found (SERVER_BIN and MODEL_DIR override). Needs curl, jq, sha256sum and awk.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${1:-http://127.0.0.1:18091}
GOLDEN=${2:-$REPO/crates/aien-inference-abi/fixtures/qwen38/gen-chat-max-24.json}
OUT=${3:?usage: qwen38_serve_gate.sh [base-url] [golden.json] <out-dir> [server-log]}
LOG=${4:-${AIEN_SERVE_LOG:-${XDG_STATE_HOME:-$HOME/.local/state}/aien/serve-27b.log}}
for t in curl jq; do command -v "$t" >/dev/null || { echo "qwen38_serve_gate.sh: needs $t" >&2; exit 2; }; done
[ -f "$GOLDEN" ] || { echo "qwen38_serve_gate.sh: golden file not found: $GOLDEN" >&2; exit 2; }
[ -f "$LOG" ] || { echo "qwen38_serve_gate.sh: server log not found: $LOG (pass it as the 4th argument or set AIEN_SERVE_LOG)" >&2; exit 2; }
mkdir -p "$OUT"
# shellcheck source=tools/qwen38_receipt.sh
. "$REPO/tools/qwen38_receipt.sh"
q38_server_facts "$URL"
Q38_MACHINE_START=$(q38_machine_json)
: > "$OUT/runs.jsonl"
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
  local w=""
  for try in $(seq 1 240); do
    w=$(curl -s -m 900 -o "$OUT/$1.json" -w "%{time_total}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @"$OUT/$1.req.json")
    # Retry only "busy" answers (server_busy, chip lock); any other error is the result.
    jq -e '.error | tostring | test("busy"; "i")' "$OUT/$1.json" >/dev/null 2>&1 || break
    sleep 5
  done
  q38_run_json "$1" "$OUT/$1.json" "$w" "$LOG" >> "$OUT/runs.jsonl"
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
  echo "server $URL pid $Q38_PID"
  echo "server binary sha256 $Q38_BIN_SHA256 (${Q38_BIN:-unknown})"
  echo "server build $Q38_BUILD"
  echo "model dir $Q38_MODEL_DIR"
  echo "golden $GOLDEN sha256 $(sha256sum "$GOLDEN" | cut -d" " -f1)"
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
G_MATCH=MISMATCH; [ "$G_IDS" = "$WANT" ] && G_MATCH=MATCH
echo "golden ids against the shipped reference: $G_MATCH" >> "$OUT/summary.txt"
VERDICT=$(grep -q '^GATE PASS$' "$OUT/summary.txt" && echo PASS || echo FAIL)
EXTRA=$(jq -n --arg golden "$GOLDEN" --arg gsha "$(sha256sum "$GOLDEN" | cut -d' ' -f1)" \
  --arg m "$G_MATCH" --arg v "$VERDICT" --rawfile s "$OUT/summary.txt" \
  '{reference:{file:$golden, sha256:$gsha}, golden_ids:$m, verdict:$v, summary:($s | rtrimstr("\n") | split("\n"))}')
q38_write_receipt "$OUT/receipt.json" gate "$URL" "$OUT/runs.jsonl" "$EXTRA"
cat "$OUT/summary.txt"
echo "receipt: $OUT/receipt.json"
[ "$VERDICT" = PASS ]
