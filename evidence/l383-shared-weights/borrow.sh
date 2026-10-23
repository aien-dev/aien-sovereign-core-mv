#!/bin/bash
# borrow.sh PORT OUT SERVER_LOG [SOCK]: L383 merge-train case. Test copies borrow the train server's weights
# (it runs with AIEN_SERVE_WEIGHTS_PUBLISH=SOCK) instead of loading their own 35 GB. No extra 35B load.
#  1. copy A starts with --weights-from SOCK (2 rows x 8192): its start time and own memory are the receipt;
#     golden-chat-max-24 ids on A == the host golden; 4 prompts greedy + 1 sampled on A == the same on the
#     train server, alone and with both servers answering at once.
#  2. copies B, C start beside it (each refuses by itself if memory is short; that is reported, not hidden).
#  3. crash: kill -9 A in the middle of a 512-token request; the train server and B still give the golden.
#  4. mutant: a copy with AIEN_L383_TEST_WRITE_SHARED=1 writes one byte into a shared weight and must die by
#     SIGSEGV (read-only mapping) without "LANDED"; the train server still gives the golden afterwards.
# Exit 0 PASS, 1 FAIL (identity, crash or mutant), 2 ERROR (no publisher, no copy came up).
set -u
PORT=${1:?port}; OUT=${2:?out}; SLOG=${3:-}; SOCK=${4:-$HOME/.local/state/aien-mt/w-$1.sock} # the merge train publishes there
mkdir -p "$OUT"; R=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
GOLD=$R/evidence/q36-host-20261020/gen-chat-max-24.json
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/borrow.txt"; }
availm() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
pid_on() { ss -ltnpH "sport = :$1" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
up() { curl -s -m 5 "http://127.0.0.1:$1/v1/models" | grep -q '"id"'; }
OPID=$(pid_on "$PORT"); [ -n "$OPID" ] || { log "ERROR: nothing listens on $PORT"; exit 2; }
[ -S "$SOCK" ] || { log "ERROR: no weights socket at $SOCK (the train server must run with AIEN_SERVE_WEIGHTS_PUBLISH=$SOCK)"; exit 2; }
EXE=$(readlink -f /proc/$OPID/exe); MD=$(tr '\0' '\n' < /proc/$OPID/cmdline | grep -A1 -x -- --model-dir | tail -1)
log "owner: port $PORT pid $OPID exe $EXE model $MD socket $(stat -c '%A' "$SOCK")"
free_port() { local p; for p in $(seq 18201 18299); do ss -ltnH "sport = :$p" | grep -q . || { echo $p; return; }; done; }
COPIES=""
cleanup() { local p q; for p in $COPIES; do q=$(pid_on $p); [ -n "$q" ] && ps -o args= -p $q | grep -q -- "--weights-from" && kill $q; done; }
trap cleanup EXIT
start_copy() { # $1 label, rest env -> sets CP (port) and CPID; returns 1 if it did not come up (exit code in $OUT/copy-<label>.rc)
  local label=$1; shift; CP=$(free_port); local a0 t0 t1 i
  a0=$(availm); t0=$(date +%s.%N)
  ( exec env "$@" "$EXE" --engine omega --model-dir "$MD" --port $CP --max-context 8192 --max-tokens-cap 4096 --thinking off \
      --batch-slots 2 --weights-from "$SOCK" > "$OUT/copy-$label.log" 2>&1 < /dev/null ) &
  CPID=$!; COPIES="$COPIES $CP"
  for i in $(seq 1 1200); do up $CP && break; ps -p $CPID > /dev/null || break; sleep 0.25; done
  t1=$(date +%s.%N)
  if ! up $CP; then wait $CPID 2>/dev/null; echo $? > "$OUT/copy-$label.rc"; log "copy $label: not up (exit $(cat "$OUT/copy-$label.rc")): $(tail -1 "$OUT/copy-$label.log" | cut -c1-240)"; return 1; fi
  sleep 3
  local a1 rss ready; a1=$(availm); rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$CPID/status)
  ready=$(grep -h 'engine ready' "$OUT/copy-$label.log" | sed 's/.*engine ready/engine ready/')
  log "copy $label: port $CP pid $CPID answering after $(echo "$t1 - $t0" | bc) s; MemAvailable drop $((a0 - a1)) MiB; VmRSS $rss MiB; $ready"
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$(echo "$t1 - $t0" | bc)" "$((a0 - a1))" "$rss" "$ready" >> "$OUT/copies.tsv"
}
chat() { # $1 port $2 out-file $3 content $4 temperature $5 max_tokens
  local m body; m=$(curl -s -m 10 "http://127.0.0.1:$1/v1/models" | jq -r '.data[0].id')
  body=$(jq -cn --arg m "$m" --arg c "$3" --argjson t "$4" --argjson n "$5" '{model:$m, messages:[{role:"user", content:$c}], temperature:$t, seed:7, max_tokens:$n, chat_template_kwargs:{enable_thinking:false}, aien_return_token_ids:true}')
  curl -s -m 900 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" > "$2"
  jq -c '.usage.aien_token_ids' "$2" 2>/dev/null
}
golden() { # $1 port $2 label -> 0 identical
  local got want; got=$(chat $1 "$OUT/golden-$2.json" "Write a C function that returns the larger of two ints." 0 24); want=$(jq -c .generated_ids "$GOLD")
  if [ -n "$got" ] && [ "$got" = "$want" ]; then log "golden $2 (port $1): identical"; return 0; fi
  log "golden $2 (port $1): DIFFER got $got"; return 1
}
PROMPTS=("Explain in three paragraphs how a bicycle derailleur shifts gears."
         "Write a Rust function that parses a semantic version string like 1.2.3 and returns a struct."
         "Review this code and list bugs: int avg(int *a, int n) { int s; for (int i = 0; i <= n; i++) s += a[i]; return s / n; }"
         "What is 17 times 23? Show the steps, then give a one-line summary.")
battery() { local i; for i in 0 1 2 3; do chat $1 "$OUT/ask-$2-$i.json" "${PROMPTS[$i]}" 0 64 > /dev/null; done; chat $1 "$OUT/ask-$2-s.json" "${PROMPTS[0]}" 1.0 64 > /dev/null; }
same() { local n=0 t=0 f a b; for f in "$OUT"/ask-$1-*.json; do t=$((t+1)); a=$(jq -c .usage.aien_token_ids "$f" 2>/dev/null); b=$(jq -c .usage.aien_token_ids "${f/ask-$1-/ask-$2-}" 2>/dev/null); [ -n "$a" ] && [ "$a" != null ] && [ "$a" = "$b" ] && n=$((n+1)); done; echo "$n/$t"; }
V=PASS; fail() { V=FAIL; log "FAIL: $*"; }
# 1. copy A, identity
start_copy a || exit 2; A=$CP
golden $A copy-a || fail "copy a golden"
battery $PORT owner; battery $A a
battery $PORT ownerpar & battery $A apar & wait
for pair in "a owner" "ownerpar owner" "apar owner"; do s=$(same $pair); log "identity ${pair% *} == owner alone: $s"; [ "${s%/*}" = "${s#*/}" ] || fail "identity ${pair% *} $s"; done
# 2. copies B, C
NUP=1; start_copy b && { NUP=$((NUP+1)); BP=$CP; golden $BP copy-b || fail "copy b golden"; }
start_copy c && { NUP=$((NUP+1)); golden $CP copy-c || fail "copy c golden"; }
log "COPIES: $NUP answering beside the train server, MemAvailable $(availm) MiB"
# 3. crash
chat $A "$OUT/crash.json" "${PROMPTS[0]}" 0 512 > /dev/null 2>&1 & sleep 4
pa=$(pid_on $A); [ -n "$pa" ] && kill -9 $pa && log "CRASH: killed copy a (pid $pa) mid-request"; wait
golden $PORT owner-after-crash || fail "train server after crash"
[ -n "${BP:-}" ] && { golden $BP copy-b-after-crash || fail "copy b after crash"; }
# 4. mutant
start_copy mutant AIEN_L383_TEST_WRITE_SHARED=1 && fail "mutant copy came up (the write did not kill it)"
MRC=$(cat "$OUT/copy-mutant.rc" 2>/dev/null || echo up)
grep -q "L383 TEST MUTANT: writing" "$OUT/copy-mutant.log" && MW=1 || MW=0; grep -q LANDED "$OUT/copy-mutant.log" && ML=1 || ML=0
log "MUTANT: exit $MRC (139 = SIGSEGV), tried $MW, landed $ML"
[ "$MRC" = 139 ] && [ $MW = 1 ] && [ $ML = 0 ] || fail "mutant exit $MRC tried $MW landed $ML"
golden $PORT owner-after-mutant || fail "train server after mutant"
log "VERDICT $V (copies answering $NUP; receipts in copies.tsv)"
[ $V = PASS ]
