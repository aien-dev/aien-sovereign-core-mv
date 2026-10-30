#!/bin/bash
# serve-gate.sh: L314 exact two-pass head, the pinned server on 18093 (never 18091).
#   1. ON  (default): gate2-http.sh (golden ids) + 4 chat prompts x 128 greedy tokens + 1 sampled request
#   2. MUTANT 5 (MASK inverted): gate2-http.sh must FAIL
#   3. OFF (AIEN_Q36_EXACT_HEAD=0, same binary): the same 4 + 1 requests; ON ids must equal OFF ids
# Waits for 18093 to be free and MemAvailable >= 50 GB (the server keeps 8 GB free on top of its 39.9 GB) before each start. Stops only the servers it started.
set -u
S=$HOME/.local/state/aien-q36; BIN=bin-e1f02b29; PORT=18093
E=$HOME/workspace/sc-hum-exact-serve/evidence/hum-exact-head/serve; mkdir -p $E
G2=$HOME/handoffs/motor-scripts/gate2-http.sh
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/serve-gate.txt; }
PROMPTS=("Explain in three paragraphs how a bicycle derailleur shifts gears."
         "Write a Rust function that parses a semantic version string like 1.2.3 and returns a struct."
         "Review this code and list bugs: int avg(int *a, int n) { int s; for (int i = 0; i <= n; i++) s += a[i]; return s / n; }"
         "What is 17 times 23? Show the steps, then give a one-line summary.")

start() { # $1 label, rest: env assignments
  local label=$1; shift
  while ss -ltn | grep -q "127.0.0.1:$PORT "; do sleep 20; done
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 48 ]; do sleep 20; done
  (cd $S/$BIN && sha256sum -c --quiet SHA256SUMS) || { log "$label: checksum FAIL"; exit 3; }
  log "$label: start $BIN on $PORT ($*), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"
  local try up=0
  for try in 1 2 3 4 5 6 7 8 9 10; do
    while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 48 ]; do sleep 15; done
    (cd $S/$BIN && env "$@" setsid nohup ./aien-serve-http --engine omega --model-dir $HOME/models/qwen3.6-35b-a3b-fp8-95a723d0 \
       --port $PORT --max-context 131072 --max-tokens-cap 16384 --thinking off > $E/server-$label.log 2>&1 < /dev/null & echo $! > $E/server.pid)
    local pid; pid=$(cat $E/server.pid)
    for i in $(seq 1 240); do
      curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && { up=1; break; }
      ps -p $pid > /dev/null || break; sleep 5
    done
    [ $up -eq 1 ] && break
    cp $E/server-$label.log $E/server-$label-fail$try.log
    log "$label: start try $try failed: $(grep -h 'failed' $E/server-$label.log | tail -1 | cut -c1-200)"
    ps -p $pid > /dev/null && stop; sleep 30
  done
  [ $up -eq 1 ] || { log "$label: server never came up, giving up"; exit 4; }
  grep -h "exact two-pass\|engine ready\|TEST MUTANT" $E/server-$label.log | tee -a $E/serve-gate.txt
}
stop() { # the process listening on $PORT, only if it is our aien-serve-http (setsid forks: server.pid is not it)
  local pid; pid=$(ss -ltnpH "sport = :$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -z "$pid" ] && { log "stop: nothing listens on $PORT"; return; }
  if ps -o args= -p $pid | grep -q "aien-serve-http.*--port $PORT"; then kill $pid; else log "stop: pid $pid on $PORT is not ours"; return; fi
  for i in $(seq 1 30); do ps -p $pid > /dev/null || break; sleep 2; done
  log "stopped pid $pid"
}
gate2() { # $1 label; retries an ERROR (chip busy) up to 5 times
  local rc
  for t in 1 2 3 4 5; do bash $G2 $PORT $1 2>&1 | tee -a $E/serve-gate.txt; rc=${PIPESTATUS[0]}; [ $rc -ne 2 ] && break; sleep 60; done
  cp $HOME/workspace/sc-motor/evidence/motor-chip-20261021/gate2-$1.json $E/ 2>/dev/null
  return $rc
}
ask() { # $1 label $2 index $3 temperature
  local m body
  m=$(curl -s -m 10 http://127.0.0.1:$PORT/v1/models | jq -r '.data[0].id')
  body=$(jq -cn --arg m "$m" --arg c "${PROMPTS[$2]}" --argjson t "$3" '{model:$m, messages:[{role:"user", content:$c}], temperature:$t, seed:7, max_tokens:128, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}')
  for t in 1 2 3 4 5; do
    curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$body" > $E/ask-$1-$2-t$3.json
    jq -e '.usage.aien_token_ids' $E/ask-$1-$2-t$3.json > /dev/null 2>&1 && break; sleep 60
  done
  log "$1 prompt $2 t=$3: $(jq -c '.usage.aien_token_ids | length' $E/ask-$1-$2-t$3.json 2>/dev/null) ids"
}
battery() { for i in 0 1 2 3; do ask $1 $i 0; done; ask $1 0 1.0; }

# phases: serve-gate.sh [on] [mutant5] [off] (default all); a skipped phase reads its earlier rc from $E/rc-<phase>
PH=${*:-on mutant5 off}
case " $PH " in *" on "*) start on; gate2 l314-exact-on; echo $? > $E/rc-on; battery on; stop;; esac
case " $PH " in *" mutant5 "*) start mutant5 AIEN_Q36_TEST_MUTANT=5; gate2 l314-exact-mutant5; echo $? > $E/rc-mutant5; stop;; esac
case " $PH " in *" off "*) start off AIEN_Q36_EXACT_HEAD=0; gate2 l314-exact-off; echo $? > $E/rc-off; battery off; stop;; esac
ON=$(cat $E/rc-on 2>/dev/null || echo 9); MU=$(cat $E/rc-mutant5 2>/dev/null || echo 9); OFF=$(cat $E/rc-off 2>/dev/null || echo 9)
same=0; total=0
for f in $E/ask-on-*.json; do
  g=${f/ask-on-/ask-off-}; total=$((total+1))
  a=$(jq -c '.usage.aien_token_ids' $f); b=$(jq -c '.usage.aien_token_ids' $g)
  [ -n "$a" ] && [ "$a" != null ] && [ "$a" = "$b" ] && same=$((same+1)) || log "DIFFER: $(basename $f)"
done
V=FAIL; [ $ON -eq 0 ] && [ $MU -eq 1 ] && [ $same -eq $total ] && V=PASS
log "VERDICT $V: gate2 on rc $ON (want 0), mutant5 rc $MU (want 1), off rc $OFF, ON == OFF ids on $same / $total requests"
echo DONE > $E/serve-gate.done
