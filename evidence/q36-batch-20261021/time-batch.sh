#!/bin/bash
# L354 timing: N conversations at once (N = 1 2 4 8 16 32, REPS each) against the batch door on <port>, every reply
# greedy and MAX tokens long (prompts.txt, each tagged with a unique "[run r n i]" so no row reuses another's prefix).
# Total tok/s = every reply's completion tokens / wall from the first request out to the last reply in.
#   time-batch.sh <port> <label> [max_tokens 128]   -> <label>/time.jsonl (one line per N and rep)
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; PORT=${1:?port}; L=${2:?label}; MAX=${3:-128}; REPS=${REPS:-2}
mapfile -t P < prompts.txt
M=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | jq -r '.data[0].id // empty'); [ -n "$M" ] || { echo "no server on $PORT"; exit 2; }
mkdir -p "$L"; : > "$L/time.jsonl"
one() { # <tag> <prompt index> <out>
  jq -cn --arg m "$M" --argjson n $MAX --arg c "[$1] ${P[$2]}" '{model:$m, messages:[{role:"user",content:$c}], temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}}' \
    | curl -s -m 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @- > "$3"; }
one warm 0 "$L/warm.json"
for N in 1 2 4 8 16 32; do for r in $(seq 1 $REPS); do
  t0=$(date +%s.%N)
  for ((i=0; i<N; i++)); do one "run $r $N $i" $((i % ${#P[@]})) "$L/n$N-r$r-$i.json" & done; wait
  t1=$(date +%s.%N)
  toks=$(cat "$L"/n$N-r$r-*.json | jq -s '[.[].usage.completion_tokens // 0] | add')
  ok=$(cat "$L"/n$N-r$r-*.json | jq -s '[.[] | select(.usage.completion_tokens != null)] | length')
  jq -cn --argjson n $N --argjson r $r --argjson toks $toks --argjson ok $ok --arg t0 $t0 --arg t1 $t1 \
    '{n:$n, rep:$r, replies_ok:$ok, completion_tokens:$toks, wall_s:(($t1|tonumber)-($t0|tonumber)), tok_s:($toks/(($t1|tonumber)-($t0|tonumber)))}' | tee -a "$L/time.jsonl"
done; done
