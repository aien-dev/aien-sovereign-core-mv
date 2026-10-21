#!/bin/bash
# L313: the bf16-state door's alone replies (alone-bf16/) against the f32 door's (../q36-batch-20261021/alone/), token ids
# per prompt: identical or the first differing position. -> drift.json
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; A=../q36-batch-20261021/alone
for f in alone-bf16/[0-9]*.json alone-bf16/turn2.json; do
  b=$(basename $f .json)
  jq -cn --arg p "$b" --argjson x "$(jq -c .ids $A/$b.json)" --argjson y "$(jq -c .ids $f)" \
    '{prompt:$p, f32_tokens:($x|length), bf16_tokens:($y|length), identical:($x==$y),
      first_diff:([range(0; [($x|length),($y|length)]|min)] | map(select($x[.] != $y[.])) | first // null)}'
done | jq -s '{rows:., identical:(map(select(.identical))|length), prompts:length}' > drift.json
jq -c '{identical, prompts, differing:[.rows[]|select(.identical|not)|{prompt,first_diff,f32_tokens}]}' drift.json
