#!/bin/bash
# L387 timing receipts on one port (the live door before and after the swap, same flags): prompt reading tok/s at
# ~2k/~8k/~20k tokens (one request, max_tokens 1: prompt_tokens / wall) and decode tok/s at N=1 and N=8 at once
# (greedy, 128 tokens, the 32 short prompts; completion tokens / wall, L354's time-batch method). REPS each (default 2).
# Every request carries a unique tag at the start, so no row reuses another's prefix. in_flight on /health is logged
# before each run (the live door also serves Renda and the courier: a run with in_flight > 0 is marked shared).
#   time-motor.sh <port> <label>    -> <label>/time.jsonl; run it under spark-window (spark-window run ...)
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; P=${1:?port}; L=${2:?label}; REPS=${REPS:-2}
M=$(curl -s -m 10 "http://127.0.0.1:$P/v1/models" | jq -r '.data[0].id // empty'); [ -n "$M" ] || { echo "no server on $P"; exit 2; }
mkdir -p "$L"; : > "$L/time.jsonl"; mapfile -t S < "$D/short.txt"
busy() { curl -s -m 5 "http://127.0.0.1:$P/health" | jq -r '.in_flight // -1'; }
req() { # <content> <max_tokens> <out>
  jq -cn --arg m "$M" --argjson n "$2" --rawfile c /dev/stdin '{model:$m, messages:[{role:"user",content:$c}], temperature:0, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}}' <<<"$1" \
    | curl -s -m 1800 "http://127.0.0.1:$P/v1/chat/completions" -H 'Content-Type: application/json' -d @- > "$3"; }
emit() { tee -a "$L/time.jsonl"; }
req "[warm $L] ${S[0]}" 8 "$L/warm.json"
for r in $(seq 1 "$REPS"); do
  for f in long-2048.txt long-8192.txt long-20k.txt; do
    b=$(busy); t0=$(date +%s.%N); req "[time $L read $r $f] $(cat "$D/$f")" 1 "$L/read-$f-r$r.json"; t1=$(date +%s.%N)
    pt=$(jq '.usage.prompt_tokens // 0' "$L/read-$f-r$r.json")
    jq -cn --arg k read --arg f "$f" --argjson r "$r" --argjson pt "$pt" --arg t0 "$t0" --arg t1 "$t1" --argjson b "$b" \
      '{kind:$k, prompt:$f, rep:$r, prompt_tokens:$pt, wall_s:(($t1|tonumber)-($t0|tonumber)), tok_s:($pt/(($t1|tonumber)-($t0|tonumber))), in_flight_before:$b}' | emit
  done
  for N in 1 8; do
    b=$(busy); t0=$(date +%s.%N)
    for ((i=0; i<N; i++)); do req "[time $L dec $r $N $i] ${S[$i]}" 128 "$L/dec-n$N-r$r-$i.json" & done; wait
    t1=$(date +%s.%N)
    toks=$(cat "$L"/dec-n$N-r$r-*.json | jq -s '[.[].usage.completion_tokens // 0] | add')
    ok=$(cat "$L"/dec-n$N-r$r-*.json | jq -s '[.[] | select(.usage.completion_tokens != null)] | length')
    jq -cn --arg k decode --argjson n "$N" --argjson r "$r" --argjson toks "$toks" --argjson ok "$ok" --arg t0 "$t0" --arg t1 "$t1" --argjson b "$b" \
      '{kind:$k, n:$n, rep:$r, replies_ok:$ok, completion_tokens:$toks, wall_s:(($t1|tonumber)-($t0|tonumber)), tok_s:($toks/(($t1|tonumber)-($t0|tonumber))), in_flight_before:$b}' | emit
  done
done
echo "time-motor $L done"
