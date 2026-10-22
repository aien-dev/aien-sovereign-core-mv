#!/bin/bash
# L371 eight readers on the merge train. Usage: gate.sh PORT OUT
# Exit 0 PASS, 1 FAIL (a check failed), 2 ERROR (no answer). Checks:
#  1 /v1/models lists <id>-readers.
#  2 a short input on the readers id is the plain path: token ids == the same request on the plain id.
#  3 a long input (roadmap-excerpt.md, ~6k tokens) on the readers id: aien_readers with >= 2 parts, contiguous lines,
#    (a part may start on the line the one before ended on: a very long line is cut at spaces), every part has notes,
#    ${L371_READERS:-8} readers (the case needs that many free rows), and the final answer (content, not reasoning) joins
#    two facts from far-apart parts: M0 is DOCTRINE_V1 (line 39) and the retired 28-milestone roadmap ended at
#    SOVEREIGN_MACHINE_CLOSURE (line 121). Each name appears once in the excerpt.
#  4 red control: the same long input on the plain id carries no aien_readers (the check in 3 can tell).
set -uo pipefail
NR=${L371_READERS:-8}
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
  jq -n --arg m "$m" --rawfile doc "$H/roadmap-excerpt.md" '{model:$m, temperature:0, max_tokens:160, chat_template_kwargs:{enable_thinking:false}, messages:[{role:"user",content:($doc + "\n\nAnswer in two lines. (1) What is milestone M0 called? (2) Which milestone did the retired 28-milestone workstream roadmap end at? Give the names in capitals as written.")}]}' > "$O/long-$m.req"
  t0=$(date +%s); post "$O/long-$m.req" "$O/long-$m.json"; echo "long $m: $(( $(date +%s) - t0 )) s"
done
R="$O/long-$MR.json"
jq -e '.aien_readers != null' "$R" >/dev/null || fail "3: no aien_readers on the long input: $(head -c 300 "$R")"
jq -e --argjson n "$NR" '.aien_readers.readers == $n and (.aien_readers.parts | length) == $n' "$R" >/dev/null \
  || fail "3: $(jq -r .aien_readers.readers "$R") readers, want $NR (were $NR rows free?)"
jq -e '.aien_readers.parts | (all(.[]; (.notes | length) > 0)) and ([range(1; length) as $i | ((.[$i].lines[0] - .[$i-1].lines[1]) as $d | $d == 0 or $d == 1)] | all)' "$R" >/dev/null \
  || fail "3: parts not contiguous or a part without notes"
A=$(jq -r '.choices[0].message.content // ""' "$R")
for f in DOCTRINE_V1 SOVEREIGN_MACHINE_CLOSURE; do grep -q "$f" <<<"$A" || fail "3: final answer misses $f: $(head -c 300 <<<"$A")"; done
echo "PASS 3: $(jq -r '.aien_readers.readers' "$R") readers, $(jq -r '.aien_readers.input_tokens' "$R") tokens, map $(jq -r '.aien_readers.map_ms' "$R") ms, answer joins DOCTRINE_V1 (part with line 39) and SOVEREIGN_MACHINE_CLOSURE (part with line 121)"
jq -e '.aien_readers == null' "$O/long-$M.json" >/dev/null || fail "4: the plain id read with readers"
jq -r '.choices[0].message.content' "$O/long-$M.json" | grep -q SOVEREIGN_MACHINE_CLOSURE || echo "note 4: plain answer misses SOVEREIGN_MACHINE_CLOSURE"
echo "PASS 4: the plain id has no aien_readers (red control: check 3 tells the two apart)"
