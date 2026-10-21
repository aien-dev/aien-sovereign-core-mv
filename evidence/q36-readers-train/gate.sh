#!/bin/bash
# L371 eight readers on the merge train. Usage: gate.sh PORT OUT
# Exit 0 PASS, 1 FAIL (a check failed), 2 ERROR (no answer). Checks:
#  1 /v1/models lists <id>-readers.
#  2 a short input on the readers id is the plain path: token ids == the same request on the plain id.
#  3 a long input (roadmap-excerpt.md, ~6k tokens) on the readers id: aien_readers with >= 2 parts, contiguous lines,
#    (a part may start on the line the one before ended on: a very long line is cut at spaces), every part has notes, the answer names M22 (OMEGA_OPTIMIZER, which only the last part holds).
#  4 red control: the same long input on the plain id carries no aien_readers (the check in 3 can tell).
set -uo pipefail
P=$1; O=$2; H=$(cd "$(dirname "$0")"; pwd); mkdir -p "$O"; U=http://127.0.0.1:$P
fail() { echo "FAIL: $*"; exit 1; }
post() { curl -s --max-time 1800 -H 'Content-Type: application/json' --data-binary @"$1" "$U/v1/chat/completions" > "$2" || { echo "ERROR: no answer for $1"; exit 2; }; }
curl -s --max-time 30 "$U/v1/models" > "$O/models.json" || { echo "ERROR: /v1/models"; exit 2; }
M=$(jq -r '.data[0].id' "$O/models.json"); MR=$M-readers
jq -e --arg m "$MR" '[.data[].id] | index($m)' "$O/models.json" >/dev/null || fail "1: $MR not in /v1/models"
echo "PASS 1: $MR listed"
for m in "$M" "$MR"; do
  jq -n --arg m "$m" '{model:$m, temperature:0, max_tokens:16, aien_return_token_ids:true, chat_template_kwargs:{enable_thinking:false}, messages:[{role:"user",content:"Say hello in five words."}]}' > "$O/short-$m.req"
  post "$O/short-$m.req" "$O/short-$m.json"
done
a=$(jq -c .usage.aien_token_ids "$O/short-$M.json"); b=$(jq -c .usage.aien_token_ids "$O/short-$MR.json")
[ "$a" != null ] && [ "$a" = "$b" ] || fail "2: short input ids differ: plain $a readers $b"
jq -e '.aien_readers == null' "$O/short-$MR.json" >/dev/null || fail "2: short input went to the readers"
echo "PASS 2: short input on the readers id == plain ($a)"
for m in "$MR" "$M"; do
  jq -n --arg m "$m" --rawfile doc "$H/roadmap-excerpt.md" '{model:$m, temperature:0, max_tokens:120, chat_template_kwargs:{enable_thinking:false}, messages:[{role:"user",content:($doc + "\n\nWhat is milestone M22 called? One line.")}]}' > "$O/long-$m.req"
  t0=$(date +%s); post "$O/long-$m.req" "$O/long-$m.json"; echo "long $m: $(( $(date +%s) - t0 )) s"
done
R="$O/long-$MR.json"
jq -e '.aien_readers.readers >= 2' "$R" >/dev/null || fail "3: no aien_readers on the long input: $(head -c 300 "$R")"
jq -e '.aien_readers.parts | (all(.[]; (.notes | length) > 0)) and ([range(1; length) as $i | ((.[$i].lines[0] - .[$i-1].lines[1]) as $d | $d == 0 or $d == 1)] | all)' "$R" >/dev/null \
  || fail "3: parts not contiguous or a part without notes"
jq -r '.choices[0].message | "\(.content) \(.reasoning_content)"' "$R" | grep -q OMEGA_OPTIMIZER || fail "3: answer misses OMEGA_OPTIMIZER: $(jq -r '.choices[0].message.content' "$R" | head -c 200)"
echo "PASS 3: $(jq -r '.aien_readers.readers' "$R") readers, $(jq -r '.aien_readers.input_tokens' "$R") tokens, map $(jq -r '.aien_readers.map_ms' "$R") ms, answer names OMEGA_OPTIMIZER"
jq -e '.aien_readers == null' "$O/long-$M.json" >/dev/null || fail "4: the plain id read with readers"
jq -r '.choices[0].message.content' "$O/long-$M.json" | grep -q OMEGA_OPTIMIZER || echo "note 4: plain answer misses OMEGA_OPTIMIZER"
echo "PASS 4: the plain id has no aien_readers (red control: check 3 tells the two apart)"
