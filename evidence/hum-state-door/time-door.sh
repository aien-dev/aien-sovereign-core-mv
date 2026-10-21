#!/bin/bash
# L313 timing inside ONE spark-window (chip + quiet locks): the same door binary with the bf16 state switch off, then on
# (8 slots, context 2048, 127.0.0.1:18098, loaded under q36-extra by L354's serve-batch.sh), N = 1 and 8 conversations at
# once, REPS 3 each, every reply greedy and 128 tokens (L354's time-batch.sh logic, N list cut to 1 and 8).
#   time-door.sh <sha8>     -> time-off-<sha8>/time.jsonl, time-on-<sha8>/time.jsonl, log time-door.log
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; G=$D/../q36-batch-20261021; S=$HOME/.local/state/aien-q36; SHA=${1:?sha8}; export PORT=18098
log() { echo "$(date -u +%FT%TZ) $*" >> time-door.log; }
if [ -z "${SPARK_WINDOW:-}" ]; then
  log "waiting for a window"; spark-window open l313-door --ledger L313 --lease 60 --wait >> time-door.log 2>&1 || { log "no window"; exit 3; }
  spark-window run "$D/time-door.sh" "$SHA"; rc=$?; spark-window close >> time-door.log 2>&1; log "window closed (rc $rc)"; exit $rc
fi
up() { for i in $(seq 1 1000); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && return 0; kill -0 $(cat $S/batch-$PORT.pid) 2>/dev/null || return 1; sleep 5; done; return 1; }
down() { local p=$(cat $S/batch-$PORT.pid); ps -o args= -p $p | grep -q "port $PORT" && kill $p; for i in $(seq 1 30); do ps -p $p >/dev/null || break; sleep 2; done; log "stopped $p"; }
timeit() { # <label>
  local L=$1 M; mapfile -t P < $G/prompts.txt
  M=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | jq -r '.data[0].id // empty')
  mkdir -p "$L"; : > "$L/time.jsonl"
  one() { jq -cn --arg m "$M" --arg c "[$1] ${P[$2]}" '{model:$m, messages:[{role:"user",content:$c}], temperature:0, max_tokens:128, chat_template_kwargs:{enable_thinking:false}}' \
    | curl -s -m 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @- > "$3"; }
  one warm 0 "$L/warm.json"
  for N in 1 8; do for r in 1 2 3; do
    t0=$(date +%s.%N)
    for ((i=0; i<N; i++)); do one "run $r $N $i" $((i % ${#P[@]})) "$L/n$N-r$r-$i.json" & done; wait
    t1=$(date +%s.%N)
    toks=$(cat "$L"/n$N-r$r-*.json | jq -s '[.[].usage.completion_tokens // 0] | add')
    ok=$(cat "$L"/n$N-r$r-*.json | jq -s '[.[] | select(.usage.completion_tokens != null)] | length')
    jq -cn --argjson n $N --argjson r $r --argjson toks $toks --argjson ok $ok --arg t0 $t0 --arg t1 $t1 \
      '{n:$n, rep:$r, replies_ok:$ok, completion_tokens:$toks, wall_s:(($t1|tonumber)-($t0|tonumber)), tok_s:($toks/(($t1|tonumber)-($t0|tonumber)))}' >> "$L/time.jsonl"
  done; done
}
log "window $SPARK_WINDOW"
for mode in off on; do
  b=0; [ $mode = on ] && b=1
  log "$mode: load ($SHA, 8 slots)"; (cd $G && Q36_EXTRA_WAIT=300 AIEN_OMEGA_GDN_STATE_BF16=$b ./serve-batch.sh $SHA 8 2048 l313-time-$mode) >> time-door.log 2>&1
  up || { log "$mode: server did not come up: $(tail -1 $S/batch-$PORT.log | cut -c1-200)"; down; continue; }
  log "$mode: up, $(grep VmRSS /proc/$(pgrep -f "aien-serve-http.*--port $PORT" | head -1)/status | tr -s ' ')"
  timeit time-$mode-$SHA; log "$mode: timing done"; down
done
log done
