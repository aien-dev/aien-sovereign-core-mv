#!/bin/bash
# L355 tok/s before/after, same command on both: greedy chat request, 256 tokens, thinking off, 3 rounds alternating
# 18091 (live FP8 server, untouched) and $WP (my build with AIEN_Q36_W4_STORE=default, started and stopped here).
# Run inside: spark-window open l355-w4 --ledger L355 --gb10 none --wait ; spark-window run ./timing.sh ; spark-window close
set -u
WP=${WP:-18093}   # test port (18093 by the brief; another free port when an agent holds it)
E=$HOME/workspace/sc-l355/evidence/l355-w4; cd $E
BIN=${BIN:-$HOME/workspace/targets/l355/sc/release/aien-serve-http}
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
OUT=$E/timing.jsonl; SLOG=$E/serve-w4.log.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/timing.log.txt; }
ss -ltn | grep -q "127.0.0.1:$WP " && { log "$WP busy, stop"; exit 2; }
log "start W4 server on $WP bin $(sha256sum $BIN | cut -c1-16) memavail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
AIEN_Q36_W4_STORE=default setsid nohup $BIN --engine omega --model-dir $M --port $WP --max-context 131072 --max-tokens-cap 16384 --thinking off > $SLOG 2>&1 < /dev/null &
SP=$!; echo $SP > $E/serve-w4.pid
for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:$WP/v1/models >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { log "W4 server died"; tail -5 $SLOG; exit 3; }; sleep 5; done
log "W4 server up after $((i*5)) s, rss_kb $(awk '/VmRSS/{print $2}' /proc/$SP/status)"
log "gate2: $(./gate2-http-w4.sh $WP w4e-$WP)"
req() { # port label round
  local port=$1 label=$2 r=$3 model body resp
  model=$(curl -s -m 10 http://127.0.0.1:$port/v1/models | jq -r '.data[0].id')
  body=$(jq -cn --arg m "$model" '{model:$m, messages:[{role:"user", content:"Explain how a suspension bridge carries its load, step by step, in detail."}], temperature:0, max_tokens:256, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}')
  local t0=$(date +%s.%N); resp=$(curl -s -m 900 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d "$body"); local t1=$(date +%s.%N)
  local id=$(echo "$resp" | jq -r '.id'); local log=$([ $port = 18091 ] && echo $HOME/.local/state/aien-q36/serve.log || echo $SLOG)
  sleep 1; local line=$(grep -F "$id:" $log | tail -1); local tps=$(echo "$line" | sed -n 's/.*decode [0-9]* ms (\([0-9.]*\) tok\/s).*/\1/p')
  local chip=$(grep -B1 -F "$id:" $log | grep -o 'graph chip [0-9.]* ms/token' | tail -1 | grep -o '[0-9.]*')
  jq -cn --arg label $label --arg r $r --arg id "$id" --arg tps "$tps" --arg chip "$chip" --argjson ct "$(echo "$resp" | jq '.usage.completion_tokens')" --arg wall "$(echo "$t1-$t0" | bc)" --argjson ids "$(echo "$resp" | jq -c '.usage.aien_token_ids')" \
    '{label:$label, round:($r|tonumber), id:$id, decode_tok_s:($tps|tonumber? // null), graph_chip_ms_per_token:($chip|tonumber? // null), completion_tokens:$ct, wall_s:($wall|tonumber), ids:$ids}' | tee -a $OUT | cut -c1-200
}
req $WP w4-warm 0 >/dev/null
for r in 1 2 3; do req 18091 fp8 $r; req $WP w4e $r; done
log "stop W4 server $SP"; kill $SP; wait $SP 2>/dev/null
log "timing done"
