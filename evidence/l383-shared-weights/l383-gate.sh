#!/bin/bash
# l383-gate.sh: shared weights on the chip (L383). Never touches 18091; stops only the servers it started.
#  0. OWNER: one 35B test server (port 18183, 2 rows x 8192) loads under q36-extra (L383) and publishes its
#     weights (--weights-publish). It stands in for the live server (18091 is stopped for tests).
#  1. IDENTITY: copy A (18184, --weights-from) vs the owner (its own upload, same binary): gate2 (golden ids)
#     on both; 4 chat prompts x 128 greedy tokens + 1 sampled (t=1.0, seed 7) on each, then the same on both
#     AT ONCE (two processes reading the same pages); every id list must match the owner's.
#  2. TIMED START: copy A stopped, then started again inside a spark-window (--ledger L383, --gb10 none: the copy
#     takes the chip lock itself; quiet lock held so no build competes for the CPU): wall time to /v1/models, the
#     server's own "ready in", its own device allocations, rss, MemAvailable drop.
#  3. COPIES THAT FIT: copies B, C (18185, 18186) beside the owner and A: gate2 on each; their memory.
#  4. CRASH: kill -9 copy A in the middle of a 512-token request; the owner and copy B still pass gate2.
#  5. MUTANT: a copy with AIEN_L383_TEST_WRITE_SHARED=1 writes one byte into a shared weight: it must die by
#     SIGSEGV (read-only mapping) without "LANDED"; the owner and copy B still pass gate2 afterwards.
#  6. OWNER EXITS: the owner is stopped; copy B still passes gate2 (the copies hold their own references);
#     a new copy is refused cleanly (no publisher).
set -u
B=${B:-$HOME/workspace/targets/l383/bin-ad9d6370}
E=${E:-$HOME/workspace/sc-l383/evidence/l383-shared-weights}; mkdir -p $E
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
S=$HOME/.local/state/aien-l383; mkdir -p $S; chmod 700 $S; SOCK=$S/weights.sock
OWNER=18183; CA=18184; CB=18185; CC=18186; CM=18187; CN=18188
G2=$HOME/handoffs/motor-scripts/gate2-http.sh; G2E=$HOME/workspace/sc-motor/evidence/motor-chip-20261021
SERVE_ARGS="--engine omega --model-dir $M --max-context 8192 --max-tokens-cap 4096 --thinking off --batch-slots 2"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $E/gate.txt; }
availm() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
pid_on() { ss -ltnpH "sport = :$1" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
up() { curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q '"id"'; }
stop() { # $1 port: only our aien-serve-http on that port
  local pid; pid=$(pid_on $1); [ -z "$pid" ] && return 0
  ps -o args= -p $pid | grep -q "aien-serve-http.*--port $1" || { log "stop: pid $pid on $1 is not ours"; return 0; }
  kill $pid; for i in $(seq 1 30); do ps -p $pid > /dev/null || break; sleep 2; done; log "stopped $1 (pid $pid)"
}
QPID=""
cleanup() { for p in $CA $CB $CC $CM $CN $OWNER; do stop $p; done; [ -n "$QPID" ] && kill $QPID 2>/dev/null; rm -f $S/gate.pid; }
gate2() { # $1 port $2 label -> rc 0 PASS, 1 FAIL; retries an ERROR (rc 2) 5 times
  local rc=2
  for t in 1 2 3 4 5; do bash $G2 $1 $2 2>&1 | tee -a $E/gate.txt; rc=${PIPESTATUS[0]}; [ $rc -ne 2 ] && break; sleep 30; done
  cp $G2E/gate2-$2.json $E/ 2>/dev/null; echo "$2 $rc" >> $E/gate2.tsv; return $rc
}
PROMPTS=("Explain in three paragraphs how a bicycle derailleur shifts gears."
         "Write a Rust function that parses a semantic version string like 1.2.3 and returns a struct."
         "Review this code and list bugs: int avg(int *a, int n) { int s; for (int i = 0; i <= n; i++) s += a[i]; return s / n; }"
         "What is 17 times 23? Show the steps, then give a one-line summary.")
ask() { # $1 port $2 label $3 index $4 temperature $5 max_tokens
  local m body f=$E/ask-$2-$3-t$4.json
  m=$(curl -s -m 10 http://127.0.0.1:$1/v1/models | jq -r '.data[0].id')
  body=$(jq -cn --arg m "$m" --arg c "${PROMPTS[$3]}" --argjson t "$4" --argjson n "${5:-128}" '{model:$m, messages:[{role:"user", content:$c}], temperature:$t, seed:7, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}')
  for t in 1 2 3; do
    curl -s -m 900 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' -d "$body" > $f
    jq -e '.usage.aien_token_ids' $f > /dev/null 2>&1 && break; sleep 20
  done
  log "$2 prompt $3 t=$4: $(jq -c '.usage.aien_token_ids | length' $f 2>/dev/null) ids"
}
battery() { for i in 0 1 2 3; do ask $1 $2 $i 0; done; ask $1 $2 0 1.0; }
same_as() { # $1 label vs $2 label -> prints "same/total", logs differences
  local same=0 total=0 f g a b
  for f in $E/ask-$2-*.json; do
    g=${f/ask-$2-/ask-$1-}; total=$((total+1))
    a=$(jq -c '.usage.aien_token_ids' $f 2>/dev/null); b=$(jq -c '.usage.aien_token_ids' $g 2>/dev/null)
    if [ -n "$a" ] && [ "$a" != null ] && [ "$a" = "$b" ]; then same=$((same+1)); else log "DIFFER $1 vs $2: $(basename $g)" >&2; fi
  done
  echo "$same/$total"
}
start_copy() { # $1 port $2 label, rest env: start a copy on the shared weights, wait until it answers, record its receipt
  local port=$1 label=$2; shift 2
  local a0 t0 t1 pid i
  a0=$(availm); t0=$(date +%s.%N)
  (cd $B && exec env "$@" ./aien-serve-http $SERVE_ARGS --port $port --weights-from $SOCK > $E/copy-$label.log 2>&1 < /dev/null) &
  pid=$!; echo $pid > $E/copy-$label.pid
  for i in $(seq 1 1200); do up $port && break; ps -p $pid > /dev/null || break; sleep 0.25; done
  t1=$(date +%s.%N)
  if ! up $port; then wait $pid 2>/dev/null; local rc=$?; log "copy $label ($port): not up, exit $rc: $(tail -2 $E/copy-$label.log | tr '\n' ' ' | cut -c1-300)"; echo $rc > $E/copy-$label.rc; return 1; fi
  sleep 3
  local a1 rss; a1=$(availm); rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status)
  local ready; ready=$(grep -h 'engine ready' $E/copy-$label.log | sed 's/.*engine ready/engine ready/')
  log "copy $label ($port, pid $pid): answering after $(echo "$t1 - $t0" | bc) s; MemAvailable $a0 -> $a1 MiB (drop $((a0 - a1)) MiB); VmRSS $rss MiB; $ready"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$port" "$(echo "$t1 - $t0" | bc)" "$((a0 - a1))" "$rss" "$ready" >> $E/copies.tsv
}

# ---- sub-command used inside the spark-window (phase 2) ----
if [ "${1:-}" = timed-start ]; then start_copy $CA timed; exit $?; fi

echo $$ > $S/gate.pid
trap cleanup EXIT
(cd $B && sha256sum -c --quiet SHA256SUMS) || { log "checksum FAIL"; exit 3; }
log "=== L383 gate: build $(tr '\n' ' ' < $B/BUILD.txt)"
# ---- 0. owner ----
if up $OWNER; then log "owner port $OWNER already answers: not ours, giving up"; trap - EXIT; exit 4; fi
rm -f $SOCK
log "owner: queued in q36-extra (L383) on $OWNER, MemAvailable $(availm) MiB"
(cd $B && Q36_EXTRA_LEDGER=L383 Q36_EXTRA_WAIT=14400 exec q36-extra ./aien-serve-http $SERVE_ARGS --port $OWNER --weights-publish $SOCK > $E/owner.log 2>&1 < /dev/null) &
QPID=$!
until up $OWNER; do ps -p $QPID > /dev/null || { log "owner: gone before answering: $(tail -3 $E/owner.log | tr '\n' ' ' | cut -c1-300)"; exit 5; }; sleep 10; done
log "owner: $(grep -h 'published\|engine ready\|exact two-pass' $E/owner.log | tr '\n' ' ' | cut -c1-700)"
[ -S $SOCK ] && log "socket $(stat -c '%A %U' $SOCK) $SOCK" || { log "no socket at $SOCK"; exit 6; }
gate2 $OWNER l383-owner; G_OWNER=$?
# ---- 1. identity ----
start_copy $CA a || exit 7
log "copy a: $(grep -h 'exact two-pass\|shared from' $E/copy-a.log | tr '\n' ' ' | cut -c1-300)"
gate2 $CA l383-copy-a; G_A=$?
battery $OWNER owner; battery $CA a
battery $OWNER ownerpar & battery $CA apar & wait
ID_A=$(same_as a owner); ID_OP=$(same_as ownerpar owner); ID_AP=$(same_as apar owner)
log "IDENTITY: copy a == owner $ID_A, owner at once == owner alone $ID_OP, copy a at once == owner alone $ID_AP"
# ---- 2. timed start inside a window ----
stop $CA
WIN=no
if timeout 2400 spark-window open l383-copy-start --ledger L383 --gb10 none --lease 20 --wait > $E/window-open.txt 2>&1; then
  WIN=$(spark-window id 2>/dev/null || echo yes)
  spark-window run bash $0 timed-start; spark-window close > $E/window-close.txt 2>&1
else
  log "no window within 40 min: timed start runs WITHOUT a window (box state recorded)"; uptime >> $E/gate.txt; start_copy $CA timed
fi
log "timed start window: $WIN"
up $CA || start_copy $CA a2
# ---- 3. copies that fit ----
start_copy $CB b; gate2 $CB l383-copy-b; G_B=$?
start_copy $CC c; gate2 $CC l383-copy-c; G_C=$?
N_UP=0; for p in $CA $CB $CC; do up $p && N_UP=$((N_UP+1)); done
log "COPIES: $N_UP test copies answering beside the owner, MemAvailable $(availm) MiB"
for p in $OWNER $CA $CB $CC; do pid=$(pid_on $p); [ -n "$pid" ] && log "  port $p pid $pid VmRSS $(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status) MiB"; done
# ---- 4. crash ----
ask $CA crash 0 0 512 > /dev/null 2>&1 &
sleep 4; pa=$(pid_on $CA); [ -n "$pa" ] && kill -9 $pa && log "CRASH: killed copy a (pid $pa) mid-request"
wait
gate2 $OWNER l383-owner-after-crash; G_OC=$?; gate2 $CB l383-copy-b-after-crash; G_BC=$?
# ---- 5. mutant ----
start_copy $CM mutant AIEN_L383_TEST_WRITE_SHARED=1; MRC=$(cat $E/copy-mutant.rc 2>/dev/null || echo up)
grep -q "L383 TEST MUTANT: writing" $E/copy-mutant.log && MW=1 || MW=0
grep -q "LANDED" $E/copy-mutant.log && ML=1 || ML=0
log "MUTANT: exit $MRC (139 = SIGSEGV), tried to write $MW, write landed $ML"
stop $CM
gate2 $OWNER l383-owner-after-mutant; G_OM=$?; gate2 $CB l383-copy-b-after-mutant; G_BM=$?
# ---- 6. owner exits ----
stop $OWNER; sleep 5
gate2 $CB l383-copy-b-owner-gone; G_BG=$?
start_copy $CN nopub; NRC=$(cat $E/copy-nopub.rc 2>/dev/null || echo up)
log "NO PUBLISHER: new copy exit $NRC: $(tail -1 $E/copy-nopub.log | cut -c1-200)"
# ---- verdict ----
TS=$(awk -F'\t' '$1=="timed"{print $3}' $E/copies.tsv | tail -1)
V=PASS
[ $G_OWNER -eq 0 ] && [ $G_A -eq 0 ] && [ $G_B -eq 0 ] && [ $G_C -eq 0 ] || V=FAIL
for x in "$ID_A" "$ID_OP" "$ID_AP"; do [ "${x%/*}" = "${x#*/}" ] && [ "${x#*/}" != 0 ] || V=FAIL; done
[ $N_UP -ge 3 ] || V=FAIL
[ $G_OC -eq 0 ] && [ $G_BC -eq 0 ] && [ $G_OM -eq 0 ] && [ $G_BM -eq 0 ] && [ $G_BG -eq 0 ] || V=FAIL
[ "$MRC" = 139 ] && [ $MW -eq 1 ] && [ $ML -eq 0 ] || V=FAIL
[ "$NRC" != up ] || V=FAIL
log "VERDICT $V: gate2 owner $G_OWNER a $G_A b $G_B c $G_C (want 0); identity a $ID_A, at once owner $ID_OP a $ID_AP; copies up $N_UP (want >= 3); timed start ${TS:-?} s (window $WIN); after crash owner $G_OC b $G_BC; mutant exit $MRC wrote $MW landed $ML, after it owner $G_OM b $G_BM; owner gone b $G_BG; no publisher exit $NRC"
echo $V > $E/gate.done
