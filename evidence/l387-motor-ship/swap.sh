#!/bin/bash
# L387 swap of 18091 to the ship build, inside ONE spark-window: timing receipts on the live door before (old build),
# stop old (clean SIGTERM), start new (same flags + switches), live gate2 (gate-batch.sh 18091 18091) + tool call,
# timing after. Back-out on any failure: the old bin with its switches. q36-extra admits no load during the restart.
#   swap.sh <new bin-dir name> [inner]
set -u
T=$HOME/workspace/targets/l387; S=$HOME/.local/state/aien-q36; E=$HOME/workspace/sc-motor-ship/evidence/l387-motor-ship
G=$HOME/workspace/sc-motor-ship/evidence/q36-batch-20261021
NEW=${1:?new bin}; OLD=bin-16b0e6f8; OLD_ENV="AIEN_Q36_STROKE=1"; NEW_ENV="AIEN_Q36_STROKE=1"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $T/swap.log; }
board() { echo "$(date -u +%FT%H:%MZ | sed 's/T.*T/T/') L387 to all|coordinator|renda|L040: $*" >> $HOME/handoffs/BOARD.md; }
if [ "${2:-}" != inner ]; then
  log "waiting for a window (l387-ship)"
  spark-window open l387-ship --ledger L387 --lease 100 --wait >> $T/swap.log 2>&1 || { log "no window"; exit 3; }
  spark-window run $T/swap.sh "$NEW" inner; rc=$?
  spark-window close >> $T/swap.log 2>&1; log "window closed (rc $rc)"; exit $rc
fi
up() { for i in $(seq 1 ${2:-120}); do curl -s -m 5 127.0.0.1:18091/v1/models | grep -q qwen3.6 && return 0; sleep 2; done; return 1; }
idle() { for i in $(seq 1 ${1:-150}); do [ "$(curl -s -m 5 127.0.0.1:18091/health | jq -r '.in_flight // 0')" = 0 ] && return 0; sleep 2; done; return 1; }
stop_live() { local p; p=$(cat $S/serve.pid); ps -o args= -p $p | grep -q 'port 18091' || { log "serve.pid $p is not the 18091 server"; return 1; }
  kill -TERM $p; for i in $(seq 1 60); do kill -0 $p 2>/dev/null || { log "stopped $p (SIGTERM, ${i}s)"; return 0; }; sleep 1; done; log "pid $p still alive after 60 s"; return 1; }
start_live() { # <bin> <env...>
  local b=$1; shift; env "$@" Q36_BATCH_BIN=$b $S/start-35b-batch.sh >> $T/swap.log 2>&1; }
backout() { log "BACK OUT: $*"; touch /tmp/aien-q36-live-restart; stop_live; start_live $OLD $OLD_ENV; up 120 && log "old $OLD back on 18091" || log "OLD DID NOT COME UP"; rm -f /tmp/aien-q36-live-restart; board "18091 BACKED OUT to $OLD ($OLD_ENV): $*"; ledger note L387 "BACK OUT $(date -u +%H:%MZ): $*; 18091 = $OLD $OLD_ENV" >/dev/null; exit 1; }
log "window $SPARK_WINDOW; before timing on $OLD"
idle 60 || log "18091 not idle before timing (in_flight $(curl -s -m 5 127.0.0.1:18091/health | jq -r .in_flight))"
READ_REPS=${READ_REPS:-1} DEC_REPS=${DEC_REPS:-3} $E/time-motor.sh 18091 $E/before-${OLD#bin-} >> $T/swap.log 2>&1
board "SWAP NOW: 18091 restarts on the ship build $NEW ($NEW_ENV), back in under 3 min; requests get connection refused meanwhile, retry. Back-out line: $OLD $OLD_ENV."
idle 150 || log "18091 still busy after 300 s: swapping anyway (in_flight $(curl -s -m 5 127.0.0.1:18091/health | jq -r .in_flight))"
touch /tmp/aien-q36-live-restart; t0=$(date +%s)
stop_live || { rm -f /tmp/aien-q36-live-restart; log "could not stop old: no swap"; exit 1; }
start_live $NEW $NEW_ENV
if ! up 120; then t1=$(date +%s); log "new build not serving after $((t1-t0)) s: $(tail -3 $S/serve.log | tr '\n' ' ' | cut -c1-300)"; backout "new build $NEW did not come up"; fi
t1=$(date +%s); rm -f /tmp/aien-q36-live-restart; log "SWAPPED: 18091 = $NEW ($NEW_ENV), downtime $((t1-t0)) s"
ledger note L387 "$(date -u +%H:%MZ) SWAPPED 18091 to $NEW ($NEW_ENV), downtime $((t1-t0)) s; back-out line: Q36_BATCH_BIN=$OLD $OLD_ENV start-35b-batch.sh" >/dev/null
(cd $G && ./gate-batch.sh 18091 18091 live-${NEW#bin-}-$(date -u +%m%dT%H%MZ)) >> $T/swap.log 2>&1; grc=$?
log "live gate2 rc $grc"
[ $grc = 0 ] || backout "live gate2 FAIL rc $grc on $NEW"
M=$(curl -s -m 10 127.0.0.1:18091/v1/models | jq -r '.data[0].id')
jq -cn --arg m "$M" '{model:$m, messages:[{role:"user",content:"What is the weather in Paris right now? Use the tool."}], temperature:0, max_tokens:200, chat_template_kwargs:{enable_thinking:false}, tools:[{type:"function",function:{name:"get_weather",description:"Current weather for a city",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}]}' \
  | curl -s -m 300 127.0.0.1:18091/v1/chat/completions -H 'Content-Type: application/json' -d @- > $E/live-tool-call.json
tn=$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' $E/live-tool-call.json)
log "live tool call: ${tn:-NONE} $(jq -c '.choices[0].message.tool_calls[0].function.arguments // .choices[0].message.content' $E/live-tool-call.json | cut -c1-100)"
[ "$tn" = get_weather ] || backout "live tool call did not parse on $NEW"
board "18091 serves the ship build $NEW ($NEW_ENV): live gate2 PASS + tool call parsed. Back-out line: Q36_BATCH_BIN=$OLD $OLD_ENV start-35b-batch.sh."
log "after timing on $NEW"
idle 60 || true
READ_REPS=${READ_REPS:-1} DEC_REPS=${DEC_REPS:-3} $E/time-motor.sh 18091 $E/after-${NEW#bin-} >> $T/swap.log 2>&1
log "SWAP DONE"
