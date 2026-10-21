#!/bin/bash
# exact-batch-gate.sh <sha8> [slots]: L314 exact head inside L354's batch door (strokes, 8 slots), through L354's own gate.
# Run as:  q36-extra ./exact-batch-gate.sh <sha8>   (L366: one extra 35B beside 18091, memory checked inside the lock;
# the test server inherits the lock and oom_score_adj 1000 and lives only inside this script).
#   real   exact head ON (default): every concurrent greedy reply == the same prompt alone on 18091 (plain head,
#          q36-batch-20261021/alone/), turn 2 == alone, prompt 0 == host golden  -> must PASS (rc 0)
#   mut5   AIEN_Q36_TEST_MUTANT=5 (exact head MASK inverted)                    -> must FAIL (rc 1)
# Rows that read logits (last prompt token, sampled requests) run the full head per stroke slot, so strokes mix
# both kinds whenever a row joins mid-spell (gate-batch.sh staggers joins).
set -u
SHA=${1:?sha8}; SLOTS=${2:-8}; export PORT=18094
D=$(cd "$(dirname "$0")" && pwd); G=$D/../../q36-batch-20261021; S=$HOME/.local/state/aien-q36
R=$D/gate-$SHA.result; L=$D/gate-$SHA.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
up() { for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' && return 0; kill -0 $(cat $S/batch-$PORT.pid) 2>/dev/null || return 1; sleep 5; done; return 1; }
down() { local p=$(cat $S/batch-$PORT.pid); ps -o args= -p $p | grep -q "port $PORT" && kill $p; for i in $(seq 1 30); do ps -p $p >/dev/null || break; sleep 2; done; log "stopped $p"; }
run() { # <label> <want rc> [env]
  local label=$1 want=$2 rc=2; shift 2
  for t in 1 2 3; do
    # the live batch door holds ~58 GB with 8 slots: wait (up to 40 min) for the 45.4 GB this load needs before each try
    for w in $(seq 1 240); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 46 ] && break; sleep 10; done
    log "$label: start bin-$SHA $SLOTS slots ($*) MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"
    env "$@" $G/serve-batch.sh $SHA $SLOTS 2048 $label >> $L 2>&1
    if ! up; then log "$label: server did not come up: $(tail -1 $S/batch-$PORT.log | cut -c1-300)"; down; sleep 60; continue; fi
    grep -h "exact two-pass\|TEST MUTANT\|stroke" $S/batch-$PORT.log | tail -3 >> $L
    $G/gate-batch.sh 18091 $PORT $label >> $L 2>&1; rc=$?; down
    [ $rc = 2 ] && { log "$label: no verdict, retry"; sleep 60; continue; }
    break
  done
  log "$label: rc $rc (want $want)"; echo "$label rc=$rc want=$want" >> $R
}
: > $R
run l314-real$SLOTS-$SHA 0
run l314-mut5-$SLOTS-$SHA 1 AIEN_Q36_TEST_MUTANT=5
V=FAIL; grep -q "rc=2" $R && V="NO VERDICT (a run never loaded)"; grep -q "real.*rc=0 want=0" $R && grep -q "mut5.*rc=1 want=1" $R && V=PASS
log "VERDICT $V"; echo "VERDICT $V" >> $R
