#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NEXT-PHASE-2 fault harness (ACCEPTANCE-v2.md, amended by ACCEPTANCE-v3.md and ACCEPTANCE-v4.md).
# Shell + jq only, no Python.
#
#   run-faults.sh fixture FIX_ROOT
#       F0: one real S0..S3 on the gpu build (AIEN_BIN), then Shutdown.
#       Needs AIEN_MODEL_PATH, AIEN_TOKENIZER_PATH, AIEN_REQUIRE_BLACKWELL=1,
#       and a GPU window: run it under `quietlock hold` (<= 20 min) with the
#       announce and release whispers, as NEXT-PHASE-1 did.
#   run-faults.sh cases FIX_ROOT RUN_ROOT [CASE ...]
#       Every case of ACCEPTANCE-v2 section 5 on the cpu-fault build
#       (AIEN_BIN built with the compose archive, no GPU archive, feature
#       fault-hold). Never touches the GPU: a daemon whose Backend line is not
#       CPU-reference stops the harness. CASE = control C1a C1b C2a C2b C2c C2d
#       C3a C3b C4 C5a C5b C5c C6a..C6h C6c-ctl C6i C7a C7b C7c C8a C8b (default:
#       all). REPS (default 3); C6i always runs once (ACCEPTANCE-v3 DECIDED 4).
#       C8a needs FIX_OLD = the v2 fixture F0v2 (no record mark), else NOT_RUN.
#
# Each run starts from a byte copy of F0 in RUN_ROOT/<case>-<rep>/ with its own
# provenance and state dirs, and writes result.json there; RUN_ROOT/results.jsonl
# collects them. make-receipt.sh turns RUN_ROOT into the content-addressed receipt.
set -u
BIN=${AIEN_BIN:?AIEN_BIN}
MODE=${1:?fixture|cases}
FIX=${2:?FIX_ROOT}
REPS=${REPS:-3}
GOAL='Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.'
CONSTRAINT='Project constraints: every change stays inside the authorized workspace; one file per change; plain text only; no network.'

now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
tstat() { stat -c '%i %s %.9Y %.9Z' "$1" 2>/dev/null || echo absent; }
tree_list() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum); }
tree_digest() { tree_list "$1" | sha256sum | cut -d' ' -f1; }
# The home's files: compose dir, machine root, record mark (ACCEPTANCE-v3 2.1).
home_digest() { (tree_list "$1/compose"; sha256sum "$1/compose.machine-root"; [ -f "$1/compose.cortex-mark" ] && sha256sum "$1/compose.cortex-mark") | sha256sum | cut -d' ' -f1; }
mark_records() { od -An -tu8 -j56 -N8 "$1" 2>/dev/null | tr -d ' '; }

DPID=; NOTRUN=
# Never leave a daemon of ours behind, whatever stops the harness.
trap '[ -n "$DPID" ] && kill -0 "$DPID" 2>/dev/null && kill -9 "$DPID" 2>/dev/null' EXIT
# start_daemon NAME: wait for the socket and the Reconcile: line (rc 0), or
# for the process to exit (rc 1).
start_daemon() {
  local log=$R/$1.log i=0
  [ -n "${KEEP_SOCK:-}" ] || rm -f "$R/aien.sock"   # C7c keeps a left socket file
  (cd "$R" && exec setsid "$BIN" daemon >"$log" 2>&1 </dev/null) &
  DPID=$!
  while kill -0 "$DPID" 2>/dev/null && [ $i -lt 1200 ]; do
    if [ -S "$R/aien.sock" ] && grep -q '^Reconcile:' "$log"; then break; fi
    sleep 0.1; i=$((i + 1))
  done
  if ! kill -0 "$DPID" 2>/dev/null; then wait "$DPID" 2>/dev/null; DRC=$?; return 1; fi
  if [ "$MODE" = cases ] && ! grep -q 'Backend:.*CPU-reference' "$log"; then
    echo "harness: daemon $1 is not the CPU build; stopping (no GPU use allowed here)" >&2
    kill -9 "$DPID"; exit 9
  fi
  [ -S "$R/aien.sock" ] && grep -q '^Reconcile:' "$log"
}
stop_daemon() {
  "$BIN" compose shutdown >/dev/null 2>&1
  local i=0; while kill -0 "$DPID" 2>/dev/null && [ $i -lt 300 ]; do sleep 0.1; i=$((i + 1)); done
  wait "$DPID" 2>/dev/null; rm -f "$R/aien.sock"
}
kill_daemon() { kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; rm -f "$R/aien.sock"; }
reconcile_line() { grep -m1 '^Reconcile:' "$R/$1.log"; }

# cx ID ARGS...: one `aien compose` call; stdout kept as steps/ID.json.
cx() {
  local id=$1; shift
  "$BIN" compose "$@" >"$R/steps/$id.json" 2>"$R/steps/$id.err"
  local rc=$?; echo "$rc" >"$R/steps/$id.rc"; return $rc
}
j() { jq -r "$2" "$R/steps/$1.json" 2>/dev/null; }
refusal() { j "$1" '.error // ""' | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\):.*/\1/p'; }
authorize() { cx "$1" authorize --report "$R/s3-report.json" --workspace "$WS" --approver drake --desk 1; j "$1" '.authorization.id // empty'; }
execute() { cx "$1" execute --report "$R/s3-report.json" --workspace "$WS" --authorization "$2"; }

# exec_hold POINT ID AUTH: execute in the background, stopped at POINT.
EPID=; ERC=
exec_hold() {
  rm -f "$R/hold"
  AIEN_FAULT_HOLD=$1 AIEN_FAULT_HOLD_FILE=$R/hold "$BIN" compose execute --report "$R/s3-report.json" \
    --workspace "$WS" --authorization "$3" >"$R/steps/$2.json" 2>"$R/steps/$2.err" &
  EPID=$!
  local i=0
  while [ ! -s "$R/hold" ] && kill -0 "$EPID" 2>/dev/null && [ $i -lt 600 ]; do sleep 0.05; i=$((i + 1)); done
  [ -s "$R/hold" ]
}
release() { rm -f "$R/hold"; wait "$EPID"; ERC=$?; echo "$ERC" >"$R/steps/$1.rc"; }
kill_executor() { kill -9 "$EPID" 2>/dev/null; wait "$EPID"; ERC=$?; echo "$ERC" >"$R/steps/$1.rc"; rm -f "$R/hold"; }

CHECKS='[]'
chk() { CHECKS=$(jq -c --arg n "$1" --argjson ok "$2" --argjson v "${3:-null}" '. + [{check:$n, ok:$ok, value:$v}]' <<<"$CHECKS"); }
eq() { [ "$1" = "$2" ] && echo true || echo false; }
q() { jq -Rn --arg s "$1" '$s'; }

ledger() { cx "$1" effects >/dev/null; }
# phases ID PHASE INTENT_OR_AUTH_KEY VALUE: count effect notes of PHASE whose KEY == VALUE.
phases() { j "$1" "[.effects[].text | fromjson? | select(.phase == \"$2\" and .$3 == $4)] | length"; }
intent_state() { j "$1" "[.ledger.intents[] | select(.authorization == $2)] | .[0].state // \"none\""; }

# ---------------------------------------------------------------- fixture
if [ "$MODE" = fixture ]; then
  : "${AIEN_MODEL_PATH:?}" "${AIEN_TOKENIZER_PATH:?}"
  R=$FIX; rm -rf "$R"; mkdir -p "$R"/ws/docs "$R"/outside "$R"/compose "$R"/prov "$R"/state "$R"/steps
  R=$(cd "$R" && pwd); WS=$R/ws
  export AIEN_COMPOSE_DIR=$R/compose AIEN_PROVENANCE_DIR=$R/prov AIEN_RUNTIME_SOCK=$R/aien.sock \
         AIEN_RUNTIME_STATE_DIR=$R/state AIEN_REQUIRE_CHECKPOINT=1
  printf '# Demo project\n\nA small local project used by the NEXT-PHASE-2 campaign.\n' >"$WS/README.md"
  printf 'Plan: keep notes short and local.\n' >"$WS/docs/plan.txt"
  printf 'sentinel outside the authorized workspace\n' >"$R/outside/sentinel.txt"
  # sc#328: the daemon requires the approval desk key (copied into every run with the fixture).
  "$BIN" compose desk-key --create 1 >/dev/null || { echo "desk key create failed"; exit 2; }
  t0=$(now_ms)
  start_daemon daemon-fixture || { echo "fixture daemon failed"; tail -5 "$R/daemon-fixture.log"; exit 2; }
  t1=$(now_ms)
  cx S0 recall
  cx S1 remember --text "$CONSTRAINT"
  cx S2 inspect --workspace "$WS"
  cx S3 propose --goal "$GOAL" --workspace "$WS"
  jq '.report' "$R/steps/S3.json" >"$R/s3-report.json"
  stop_daemon
  jq -n --arg backend "$(grep -m1 'Backend:' "$R/daemon-fixture.log" | sed 's/\x1b\[[0-9;]*m//g; s/^ *Backend: //')" \
    --arg reconcile "$(reconcile_line daemon-fixture)" --argjson start_ms $((t1 - t0)) \
    --slurpfile s3 "$R/steps/S3.json" --arg files "$( (tree_list "$R/compose"; sha256sum "$R/compose.machine-root"; [ -f "$R/compose.cortex-mark" ] && sha256sum "$R/compose.cortex-mark"; tree_list "$WS") | sha256sum | cut -d' ' -f1)" \
    '{backend:$backend, reconcile:$reconcile, daemon_start_ms:$start_ms, s3_ok:($s3[0].ok // false),
      s3_committed:($s3[0].report.committed // false), proposal_path:($s3[0].report.proposal_path // null),
      machine_id:($s3[0].report.machine_id // null), files_sha256:$files}' >"$R/fixture.json"
  cat "$R/fixture.json"
  exit 0
fi

# ---------------------------------------------------------------- cases
ROOT=${3:?RUN_ROOT}; shift 3
mkdir -p "$ROOT"; ROOT=$(cd "$ROOT" && pwd); FIX=$(cd "$FIX" && pwd)
jq -e '.s3_committed == true' "$FIX/fixture.json" >/dev/null || { echo "fixture F0 has no committed proposal: cases NOT_RUN"; exit 3; }
PPATH=$(jq -r .proposal_path "$FIX/fixture.json")
CASES=${*:-control C1a C1b C2a C2b C2c C2d C3a C3b C4 C5a C5b C5c C6a C6b C6c C6c-ctl C6d C6e C6f C6g C6h C6i C7a C7b C7c C8a C8b}

new_run() {   # NAME [FIXTURE]
  local from=${2:-$FIX}
  R=$ROOT/$1; rm -rf "$R"; mkdir -p "$R"/prov "$R"/state "$R"/steps
  cp -a "$from/compose" "$from/compose.machine-root" "$from/ws" "$from/outside" "$from/s3-report.json" "$R/"
  [ -f "$from/compose.cortex-mark" ] && cp -a "$from/compose.cortex-mark" "$R/"
  WS=$R/ws; T=$WS/$PPATH; CHECKS='[]'; INJ=; OUTCOME=; NOTRUN=; NOTAPPL=; NO_R6=
  export AIEN_COMPOSE_DIR=$R/compose AIEN_PROVENANCE_DIR=$R/prov AIEN_RUNTIME_SOCK=$R/aien.sock \
         AIEN_RUNTIME_STATE_DIR=$R/state AIEN_REQUIRE_CHECKPOINT=0
  unset AIEN_MODEL_PATH AIEN_TOKENIZER_PATH AIEN_REQUIRE_BLACKWELL AIEN_GPU_BACKEND AIEN_FAULT_HOLD AIEN_FAULT_HOLD_FILE KEEP_SOCK
  T0=$(tstat "$T"); WS0=$(tree_list "$WS"); OUT0=$(tree_digest "$R/outside")
}

# pre_state ID: machine id, record count and prefix digest before the fault.
pre_state() {
  cx "$1" recall >/dev/null
  N0=$(j "$1" .recall.records_total); MID0=$(j "$1" .machine_id); MIDF0=$(sha "$R/compose/machine.id")
  cx "$1-prefix" recall --prefix "$N0" >/dev/null; PD0=$(j "$1-prefix" .recall.prefix_digest)
}

# rules ID: R1..R4 of ACCEPTANCE-v2 section 3, on the final state (daemon up).
rules() {
  cx "$1-recall" recall --prefix "$N0" >/dev/null
  chk R4_machine_id "$(eq "$(j "$1-recall" .machine_id)" "$MID0")" "$(jq -nc --arg a "$MID0" --arg b "$(j "$1-recall" .machine_id)" '[$a,$b]')"
  chk R4_machine_id_file "$(eq "$(sha "$R/compose/machine.id")" "$MIDF0")"
  chk R4_prefix_digest "$(eq "$(j "$1-recall" .recall.prefix_digest)" "$PD0")" "$(jq -nc --arg a "$PD0" --arg b "$(j "$1-recall" .recall.prefix_digest)" '[$a,$b]')"
  ledger "$1-ledger"
  local auths intents ok_true receipts_true done_n changed
  auths=$(j "$1-recall" '[.authorizations[] | select(.verified) | {id, g: (.text | fromjson? // {})}]')
  intents=$(j "$1-ledger" '.ledger.intents')
  chk R1_intents_cite_earlier_matching_grants "$(jq -n --argjson a "$auths" --argjson i "$intents" \
    '[$i[] | . as $r | ($a | map(select(.id == $r.authorization)) | .[0].g) as $g
      | select($g == null or $r.authorization >= $r.intent or $g.path != $r.path
               or $g.content_sha256 != $r.content_sha256 or $g.target != $r.target)] | length == 0')" "$(jq -nc --argjson i "$intents" '[$i[].intent]')"
  changed=$(diff <(echo "$WS0") <(tree_list "$WS") | sed -n 's/^[<>] [0-9a-f]*  \.\///p' | sort -u | jq -R . | jq -sc .)
  chk R1_workspace_only_proposal_path "$(jq -n --argjson c "$changed" --arg p "$PPATH" '$c - [$p] | length == 0')" "$changed"
  chk R1_outside_unchanged "$(eq "$(tree_digest "$R/outside")" "$OUT0")"
  chk R2_one_intent_per_authorization "$(jq -n --argjson i "$intents" '[$i[].authorization] | length == (unique | length)')"
  done_n=$(jq -n --argjson i "$intents" '[$i[] | select(.state == "DONE")] | length')
  receipts_true=0
  for p in "$R"/prov/*.json; do
    [ -f "$p" ] || continue
    [ "$(jq -r '.tool + ":" + (.success|tostring)' "$p")" = "write_file:true" ] && receipts_true=$((receipts_true + 1))
  done
  chk R2_success_receipts_le_done_intents "$( [ "$receipts_true" -le "$done_n" ] && echo true || echo false)" "[$receipts_true,$done_n]"
  ok_true=$(for f in "$R"/steps/*.json; do jq -c 'select(.step == "S5" and .ok == true) | .state' "$f" 2>/dev/null; done | sort -u | paste -sd, -)
  chk R3_execute_ok_only_when_done "$( [ -z "$ok_true" ] || [ "$ok_true" = '"DONE"' ] && echo true || echo false)" "$(q "$ok_true")"
  chk R3_done_means_content_on_disk "$(jq -n --argjson i "$intents" --arg t "$(sha "$T")" \
    '[$i[] | select(.state == "DONE") | .content_sha256] | unique | (length == 0) or (. == [$t])')"
}

finish() {   # NAME REP
  local verdict
  # R6 (ACCEPTANCE-v3 section 3): no E_MARK in any start line, recall or execute
  # answer of a control, C1..C5, C6c-ctl, C7a, C7c (ACCEPTANCE-v4) or C8 run.
  case $1 in control-*|C1*|C2*|C3*|C4*|C5*|C6c-ctl|C7a|C7c|C8*)
    local em; em=$(cd "$R" && grep -l 'E_MARK' daemon-*.log steps/*.json 2>/dev/null | paste -sd, -)
    chk R6_no_E_MARK "$( [ -z "$em" ] && echo true || echo false)" "$(q "$em")";;
  esac
  verdict=$(jq -r 'if length > 0 and all(.ok) then "PASS" else "FAIL" end' <<<"$CHECKS")
  # C6d guard (DECIDED 3): an uninjected run whose checks all hold is
  # NOT_APPLICABLE, never PASS; a failed check stays FAIL.
  [ -n "$NOTAPPL" ] && [ "$verdict" = PASS ] && verdict=NOT_APPLICABLE
  [ -n "$NOTRUN" ] && verdict=NOT_RUN
  local ev
  ev=$(cd "$R" && find steps -maxdepth 1 -type f -name '*.json' -o -maxdepth 1 -name 'daemon-*.log' | sort | while read -r f; do
    jq -nc --arg f "$f" --arg s "$(sha "$R/$f")" '{file:$f, sha256:$s}'; done | jq -sc .)
  ev=$(jq -c --argjson more "$(cd "$R" && for f in daemon-*.log; do [ -f "$f" ] && jq -nc --arg f "$f" --arg s "$(sha "$R/$f")" '{file:$f, sha256:$s}'; done | jq -sc .)" '. + $more' <<<"$ev")
  jq -n --arg row "$1" --argjson rep "$2" --arg inj "$INJ" --arg out "$OUTCOME" --argjson checks "$CHECKS" \
    --arg verdict "$verdict" --argjson ev "$ev" --arg root "$R" \
    '{row:$row, rep:$rep, injected_at:$inj, outcome:$out, verdict:$verdict, checks:$checks, evidence_digests:$ev, run_root:$root}' \
    >"$R/result.json"
  jq -c . "$R/result.json" >>"$ROOT/results.jsonl"
  echo "$1 rep $2: $verdict  $OUTCOME"
  jq -r '.checks[] | select(.ok | not) | "   FAILED \(.check) \(.value)"' "$R/result.json"
}

up() { start_daemon "$1" || { chk "daemon_$1_up" false "$(q "$(tail -3 "$R/$1.log" | tr '\n' ' ')")"; return 1; }; chk "daemon_$1_up" true; }

run_control() {
  INJ="none (uninjected control)"
  up daemon-1 || return
  chk reconcile_at_start_nothing_open "$(reconcile_line daemon-1 | grep -q 'checked 0' && echo true || echo false)" "$(q "$(reconcile_line daemon-1)")"
  pre_state pre
  A1=$(authorize A1)
  execute X1 "$A1"; chk execute_done "$(eq "$(j X1 .state)" DONE)" "$(q "$(j X1 .state)")"
  T1=$(tstat "$T")
  chk written "$( [ "$T1" != "$T0" ] && echo true || echo false)"
  execute X2 "$A1"; chk retry_refused_AlreadySpent "$(eq "$(refusal X2)" AlreadySpent)" "$(q "$(refusal X2)")"
  chk retry_no_write "$(eq "$(tstat "$T")" "$T1")"
  stop_daemon; up daemon-2 || return
  chk restart_reconcile_nothing_open "$(reconcile_line daemon-2 | grep -q 'checked 0' && echo true || echo false)" "$(q "$(reconcile_line daemon-2)")"
  rules final; stop_daemon
  OUTCOME="DONE once; retry AlreadySpent; restart clean"
}

run_C1() {   # $1 = a|b
  INJ="hold after_write; SIGKILL daemon; release (ack lost)$([ "$1" = b ] && echo '; harness overwrites target before restart')"
  up daemon-1 || return; pre_state pre
  A1=$(authorize A1)
  exec_hold after_write X1 "$A1" || { chk hold_reached false; return; }
  T1=$(tstat "$T"); chk written_before_ack "$( [ "$T1" != "$T0" ] && echo true || echo false)"
  kill_daemon; release X1
  chk executor_exit_3 "$(eq "$ERC" 3)" "$ERC"
  chk executor_state_UNRESOLVED "$(eq "$(j X1 .state)" UNRESOLVED)" "$(q "$(j X1 .state)")"
  chk executor_ok_false "$(eq "$(j X1 .ok)" false)"
  chk receipt_success_false "$(eq "$(j X1 .receipt.success)" false)"
  chk written_once_so_far "$(eq "$(tstat "$T")" "$T1")"
  if [ "$1" = b ]; then
    printf 'harness bytes, not the proposal\n' >"$T"; T1=$(tstat "$T"); TH=$(sha "$T")
  fi
  up daemon-2 || return
  local rl; rl=$(reconcile_line daemon-2)
  ledger L1; local st; st=$(intent_state L1 "$A1")
  cx RC1 recall >/dev/null
  local I; I=$(j L1 "[.ledger.intents[] | select(.authorization == $A1)] | .[0].intent")
  chk intent_has_no_ack "$(eq "$(phases RC1 ack intent "$I")" 0)"
  if [ "$1" = a ]; then
    chk restart_reconcile_DONE "$(echo "$rl" | grep -q 'DONE 1,' && echo true || echo false)" "$(q "$rl")"
    chk ledger_DONE "$(eq "$st" DONE)" "$(q "$st")"
    execute X2 "$A1"; chk retry_refused_AlreadySpent "$(eq "$(refusal X2)" AlreadySpent)" "$(q "$(refusal X2)")"
    OUTCOME="ack lost -> UNRESOLVED (exit $ERC) -> restart reconcile $st; retry $(refusal X2)"
  else
    chk restart_reconcile_UNRESOLVED "$(echo "$rl" | grep -q 'UNRESOLVED 1 recorded' && echo true || echo false)" "$(q "$rl")"
    chk ledger_UNRESOLVED "$(eq "$st" UNRESOLVED)" "$(q "$st")"
    execute X2 "$A1"; chk retry_refused_ReconciliationRequired "$(eq "$(refusal X2)" ReconciliationRequired)" "$(q "$(refusal X2)")"
    ledger L2; chk R5_effects_lists_UNRESOLVED "$(eq "$(intent_state L2 "$A1")" UNRESOLVED)"
    cx D1 reconcile --intent "$I" --declare not_done --approver drake
    chk declare_NOT_DONE "$(eq "$(j D1 '.reconcile.outcomes[0].state')" NOT_DONE)" "$(q "$(j D1 '.reconcile.outcomes[0].state')")"
    execute X3 "$A1"; chk retry_after_declare_AlreadySpent "$(eq "$(refusal X3)" AlreadySpent)" "$(q "$(refusal X3)")"
    chk harness_bytes_kept "$(eq "$(sha "$T")" "$TH")"
    OUTCOME="ack lost + world changed -> restart UNRESOLVED; retry $(refusal X2); operator NOT_DONE; retry $(refusal X3)"
  fi
  chk no_second_write "$(eq "$(tstat "$T")" "$T1")"
  rules final; stop_daemon
}

run_C2a() {
  INJ="hold before_intent; SIGKILL daemon; release"
  up daemon-1 || return; pre_state pre
  A1=$(authorize A1)
  cx count1 recall >/dev/null; local n1; n1=$(j count1 .recall.records_total)
  exec_hold before_intent X1 "$A1" || { chk hold_reached false; return; }
  kill_daemon; release X1
  chk executor_refused "$(eq "$(j X1 .ok)" false)" "$(q "$(j X1 .error)")"
  chk no_write "$(eq "$(tstat "$T")" "$T0")"
  up daemon-2 || return
  # ACCEPTANCE-v3 C2a (rule frozen, G1): a daemon open appends exactly the 5
  # start-up records (class 1, kind 65538, subjects 1..5), nothing else.
  local n2 h1 h2 ids; n1=${n1:-0}; ids=$(seq -s, $((n1 + 1)) $((n1 + 5)))
  cx count2 recall --ids "$ids" >/dev/null
  n2=$(j count2 .recall.records_total); h1=$(j count1 '.recall.host | length'); h2=$(j count2 '.recall.host | length')
  chk host_record_count_unchanged "$(eq "$h2" "$h1")" "[$h1,$h2]"
  chk records_total_is_before_plus_5 "$(eq "$n2" $((n1 + 5)))" "{\"before_kill\":$n1,\"after_restart\":$n2}"
  local seen; seen=$(j count2 '[.recall.cited[] | {id, cls, kind, subject}]')
  chk startup_records_are_5_entity_creates "$(jq -n --argjson s "$seen" --argjson lo $((n1 + 1)) \
    '($s | length == 5) and ([$s[] | select(.cls == 1 and .kind == 65538)] | length == 5)
     and ([$s[].subject] | sort == [1,2,3,4,5]) and ([$s[].id] | sort == [range($lo; $lo + 5)])')" "$(jq -nc --argjson s "$seen" '$s')"
  ledger L1; chk no_intent "$(eq "$(intent_state L1 "$A1")" none)"
  execute X2 "$A1"; chk same_grant_executes_after_restart "$(eq "$(j X2 .state)" DONE)" "$(q "$(j X2 .state)")"
  T1=$(tstat "$T"); execute X3 "$A1"
  chk no_second_write "$(eq "$(tstat "$T")" "$T1")"
  OUTCOME="refused while daemon dead (no intent, no write); after restart DONE once"
  rules final; stop_daemon
}

run_C2bcd() {   # b|c|d
  local point=after_intent; [ "$1" = d ] && point=after_write
  case $1 in
    b) INJ="hold after_intent; SIGKILL executor (daemon up)";;
    c) INJ="hold after_intent; SIGKILL daemon then executor; restart";;
    d) INJ="hold after_write; SIGKILL executor (daemon up)";;
  esac
  up daemon-1 || return; pre_state pre
  A1=$(authorize A1)
  exec_hold "$point" X1 "$A1" || { chk hold_reached false; return; }
  [ "$1" = c ] && kill_daemon
  kill_executor X1
  T1=$(tstat "$T")
  if [ "$1" = d ]; then chk written "$( [ "$T1" != "$T0" ] && echo true || echo false)"; else chk no_write "$(eq "$T1" "$T0")"; fi
  local want=NOT_DONE; [ "$1" = d ] && want=DONE
  if [ "$1" = c ]; then
    up daemon-2 || return
    chk restart_reconcile_NOT_DONE "$(reconcile_line daemon-2 | grep -q 'NOT_DONE 1,' && echo true || echo false)" "$(q "$(reconcile_line daemon-2)")"
    cx RC2 recall >/dev/null
    chk recall_shows_intent "$(eq "$(phases RC2 intent authorization "$A1")" 1)"
    chk recall_shows_reconcile_record "$(eq "$(phases RC2 reconcile authorization "$A1")" 1)"
  else
    execute X2 "$A1"; chk retry_refused_ReconciliationRequired "$(eq "$(refusal X2)" ReconciliationRequired)" "$(q "$(refusal X2)")"
    cx RC reconcile
    chk reconcile_records_$want "$(eq "$(j RC '.reconcile.outcomes[0].state')" $want)" "$(q "$(j RC '.reconcile.outcomes[0].state')")"
  fi
  ledger L1; chk ledger_$want "$(eq "$(intent_state L1 "$A1")" $want)"
  execute X3 "$A1"; chk retry_refused_AlreadySpent "$(eq "$(refusal X3)" AlreadySpent)" "$(q "$(refusal X3)")"
  chk no_write_by_retry "$(eq "$(tstat "$T")" "$T1")"
  OUTCOME="executor killed at $point -> $(intent_state L1 "$A1"); retry $(refusal X3)"
  if [ "$1" = b ]; then
    A2=$(authorize A2); execute X4 "$A2"
    chk new_grant_DONE "$(eq "$(j X4 .state)" DONE)" "$(q "$(j X4 .state)")"
    cx RC3 recall >/dev/null
    local na; na=$(( $(j RC3 '.authorizations | length') - $(j pre '.authorizations | length') ))
    chk approvals_2_both_counted "$(eq "$na" 2)" "$na"
    OUTCOME="$OUTCOME; new grant A2 DONE (approvals $na)"
  fi
  rules final; stop_daemon
}

run_C3a() {
  INJ="cpu-fault daemon started with AIEN_REQUIRE_BLACKWELL=1"
  local before; before=$(home_digest "$R")
  export AIEN_REQUIRE_BLACKWELL=1
  if start_daemon daemon-1; then chk daemon_refused false; kill_daemon; else chk daemon_refused true "$DRC"; fi
  unset AIEN_REQUIRE_BLACKWELL
  chk exit_nonzero "$( [ "${DRC:-0}" != 0 ] && echo true || echo false)" "${DRC:-0}"
  chk names_gb10_requirement "$(grep -q 'GB10 GPU is required' "$R/daemon-1.log" && echo true || echo false)" "$(q "$(grep -m1 Fatal "$R/daemon-1.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  chk no_socket "$( [ -S "$R/aien.sock" ] && echo false || echo true)"
  chk home_unchanged "$(eq "$(home_digest "$R")" "$before")"
  chk workspace_unchanged "$(eq "$(tree_list "$WS")" "$WS0")"
  OUTCOME="refused: $(grep -m1 Fatal "$R/daemon-1.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-120)"
}

run_C3b() {
  INJ="cpu-fault daemon, no GPU requirement; S3 propose"
  up daemon-1 || return; pre_state pre
  chk backend_cpu_reference "$(grep -q 'Backend:.*CPU-reference' "$R/daemon-1.log" && echo true || echo false)" "$(q "$(grep -m1 'Backend:' "$R/daemon-1.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  cx P1 propose --goal "$GOAL" --workspace "$WS"
  local committed; committed=$(j P1 'if .report.committed == null then "none" else (.report.committed | tostring) end')
  chk propose_outcome_explicit "$( { [ "$committed" = true ] || { [ "$committed" = false ] && [ -n "$(j P1 '.report.proposer_error // .report.uncommitted_proposal // empty')" ]; } || [ "$(j P1 .ok)" = false ]; } && echo true || echo false)" "$(q "committed=$committed $(j P1 '.report.proposer_error // .error // ""' | cut -c1-160)")"
  chk no_gpu_claim "$(grep -qi 'OmegaGb10Backend (native' "$R/daemon-1.log" "$R/steps/P1.json" && echo false || echo true)"
  ledger L1; chk no_intents "$(eq "$(j L1 '.ledger.intents | length')" 0)"
  chk workspace_unchanged "$(eq "$(tree_list "$WS")" "$WS0")"
  OUTCOME="CPU-reference daemon; propose committed=$committed"
  rules final; stop_daemon
}

run_C4() {
  INJ="authorize A1; stop; execute; Shutdown+restart; execute; resume; execute; authorize A2; execute"
  up daemon-1 || return; pre_state pre
  A1=$(authorize A1)
  cx STOP stop --approver drake
  execute X1 "$A1"; chk stopped_refused "$(eq "$(refusal X1)" Stopped)" "$(q "$(refusal X1)")"
  stop_daemon; up daemon-2 || return
  execute X2 "$A1"; chk stop_survives_restart "$(eq "$(refusal X2)" Stopped)" "$(q "$(refusal X2)")"
  cx RESUME resume --approver drake
  execute X3 "$A1"; chk old_grant_stale "$(eq "$(refusal X3)" Stale)" "$(q "$(refusal X3)")"
  chk zero_writes_until_A2 "$(eq "$(tstat "$T")" "$T0")"
  A2=$(authorize A2); execute X4 "$A2"
  chk new_grant_DONE "$(eq "$(j X4 .state)" DONE)" "$(q "$(j X4 .state)")"
  T1=$(tstat "$T"); execute X5 "$A2"
  chk written_once "$(eq "$(tstat "$T")" "$T1")"
  OUTCOME="$(refusal X1), $(refusal X2) after restart, $(refusal X3) after resume; A2 $(j X4 .state)"
  rules final; stop_daemon
}

run_C5() {   # a|b|c
  up daemon-1 || return; pre_state pre
  A1=$(authorize A1)
  case $1 in
  a) INJ="revoke A1 before execute"
     cx RV revoke --authorization "$A1" --approver drake
     chk revoke_true "$(eq "$(j RV .control.revoked)" true)"
     execute X1 "$A1"; chk refused_Revoked "$(eq "$(refusal X1)" Revoked)" "$(q "$(refusal X1)")"
     chk no_write "$(eq "$(tstat "$T")" "$T0")"
     OUTCOME="revoked:true; execute $(refusal X1)";;
  b) INJ="execute A1 to DONE, then revoke A1"
     execute X1 "$A1"; chk execute_done "$(eq "$(j X1 .state)" DONE)"
     cx n1 recall >/dev/null
     cx RV revoke --authorization "$A1" --approver drake
     chk revoke_false "$(eq "$(j RV .control.revoked)" false)"
     chk nothing_recorded "$(eq "$(j RV .control.recorded)" null)"
     cx n2 recall >/dev/null
     chk record_count_unchanged "$(eq "$(j n2 .recall.records_total)" "$(j n1 .recall.records_total)")"
     OUTCOME="revoke after spend: revoked:$(j RV .control.revoked), nothing recorded";;
  c) INJ="authorize A1; harness writes other bytes to the target; execute A1"
     mkdir -p "$(dirname "$T")"; printf 'changed by someone else after the grant\n' >"$T"; local th; th=$(tstat "$T")
     execute X1 "$A1"; chk refused_Stale "$(eq "$(refusal X1)" Stale)" "$(q "$(refusal X1)")"
     chk target_untouched "$(eq "$(tstat "$T")" "$th")"
     OUTCOME="world changed after grant: $(refusal X1)";;
  esac
  rules final; stop_daemon
}

c6_base() {   # control run plus one unspent grant A2, daemon stopped
  up daemon-1 || return 1; pre_state pre
  A1=$(authorize A1); execute X1 "$A1"
  chk base_execute_done "$(eq "$(j X1 .state)" DONE)"
  S1=$(stat -c %s "$R/compose/cortex.cx")
  [ "${1:-}" = save_mark ] && cp "$R/compose.cortex-mark" "$R/mark-before-A2" 2>/dev/null
  A2=$(authorize A2)
  stop_daemon
  T1=$(tstat "$T")
  mkdir -p "$R/pre-damage"
  (cd "$R" && cp -a --parents compose compose.machine-root state pre-damage/ 2>/dev/null
   [ -f compose.cortex-mark ] && cp -a compose.cortex-mark pre-damage/)
  return 0
}

# c6_score NAME GRANT FILE: the v2 C6 rule on the damaged home (ACCEPTANCE-v2
# section 5), plus the v3 mark rule (a refused start leaves the mark as it was).
c6_score() {
  local d=$1 g=$2 f=$3
  local dmg ds; dmg=$(sha "$R/damaged.bin"); ds=$(stat -c %s "$R/damaged.bin" 2>/dev/null || echo 0)
  local refused=false
  if start_daemon "$d"; then
    local rl; rl=$(reconcile_line "$d")
    cx RCL-$d recall
    local rok; rok=$(j RCL-$d .ok)
    execute X-$d "$g"
    local named=false
    if [ "$rok" = false ] && j RCL-$d .error | grep -Eq 'E_TORN|E_DIGEST|E_REPLAY|E_IDENTITY|E_MARK|expected 32|refused|CorruptLedger|digest'; then named=true; fi
    chk named_refusal_on_recall "$named" "$(q "$(j RCL-$d '.error // "recall succeeded"' | cut -c1-240)")"
    chk start_reconcile_reports "$( { echo "$rl" | grep -q 'refused' || [ "$named" = false ]; } && echo true || echo false)" "$(q "$(echo "$rl" | cut -c1-240)")"
    chk execute_grant_refused "$(eq "$(j X-$d .ok)" false)" "$(q "$(j X-$d '.error // .state' | cut -c1-200)")"
    echo "$rl" | grep -q 'refused' && refused=true
    stop_daemon
    OUTCOME="opened; recall ok=$rok; execute ok=$(j X-$d .ok); $(echo "$rl" | cut -c1-160)"
  else
    refused=true
    chk daemon_refused_named "$(grep -Eq 'is damaged|Fatal' "$R/$d.log" && echo true || echo false)" "$(q "$(grep -m1 Fatal "$R/$d.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200)")"
    OUTCOME="daemon refused to start: $(grep -m1 Fatal "$R/$d.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-140)"
  fi
  chk no_write_on_damaged_state "$(eq "$(tstat "$T")" "$T1")"
  if [ "$f" != compose/jspace/jspace.meta ] || [ -f "$R/damaged.bin" ]; then
    # No silent repair or reset: every damaged byte is still there (records the
    # daemon appends after a refusal-free open are allowed and shown).
    local cs; cs=$(stat -c %s "$R/$f" 2>/dev/null || echo 0)
    chk damaged_bytes_kept "$( [ "$cs" -ge "$ds" ] && cmp -s -n "$ds" "$R/damaged.bin" "$R/$f" && echo true || echo false)" "{\"damaged_sha256\":\"$dmg\",\"damaged_bytes\":$ds,\"now_bytes\":$cs}"
  fi
  # ACCEPTANCE-v3 section 5, C6a..h: compose.cortex-mark unchanged by a refused start.
  if [ "$refused" = true ] && [ -f "$R/damaged-mark.bin" ]; then
    chk mark_unchanged_by_refused_start "$(cmp -s "$R/damaged-mark.bin" "$R/compose.cortex-mark" && echo true || echo false)" "$(q "$(sha "$R/compose.cortex-mark")")"
  fi
}

run_C6() {   # a..i
  c6_base || return
  local f
  case $1 in
  a) f=compose/cortex.cx; INJ="truncate cortex.cx by 1 byte (inside the last record)"
     truncate -s -1 "$R/$f";;
  b) f=compose/cortex.cx; INJ="flip one byte inside the last record of cortex.cx"
     local sz; sz=$(stat -c %s "$R/$f"); local off=$(( S1 + (sz - S1) / 2 ))
     printf "$(printf '\\x%02x' $(( $(od -An -tu1 -j $off -N1 "$R/$f") ^ 0xff )))" | dd of="$R/$f" bs=1 seek=$off conv=notrunc 2>/dev/null;;
  c) f=compose/cortex.cx; INJ="cut cortex.cx at the record boundary before the last record (A2 dropped)"
     truncate -s "$S1" "$R/$f";;
  d) run_C6d; return;;
  e) f=compose/jspace/jspace.meta; INJ="flip one byte in the middle of jspace.meta"
     local sz; sz=$(stat -c %s "$R/$f"); local off=$(( sz / 2 ))
     printf "$(printf '\\x%02x' $(( $(od -An -tu1 -j $off -N1 "$R/$f") ^ 0xff )))" | dd of="$R/$f" bs=1 seek=$off conv=notrunc 2>/dev/null;;
  f) f=compose.machine-root; INJ="truncate the machine root to 16 bytes"
     truncate -s 16 "$R/$f";;
  g) f=compose/machine.id; INJ="flip the first byte of machine.id"
     printf "$(printf '\\x%02x' $(( $(od -An -tu1 -j 0 -N1 "$R/$f") ^ 0xff )))" | dd of="$R/$f" bs=1 seek=0 conv=notrunc 2>/dev/null;;
  h) f=state/processed_operations.json; INJ="write '{' into processed_operations.json"
     printf '{' >"$R/$f";;
  i) f=compose/jspace/jspace.meta; INJ="delete jspace/jspace.meta (daemon stopped)"
     rm -f "$R/$f";;
  esac
  if [ -f "$R/$f" ]; then cp "$R/$f" "$R/damaged.bin"; else rm -f "$R/damaged.bin"; fi
  [ -f "$R/compose.cortex-mark" ] && cp "$R/compose.cortex-mark" "$R/damaged-mark.bin"
  if [ -f "$R/$f" ] && cmp -s "$R/pre-damage/$f" "$R/$f"; then
    NOTRUN="injection changed nothing: $f is $(stat -c %s "$R/$f") bytes before and after"; OUTCOME=$NOTRUN; return
  fi
  if [ ! -f "$R/$f" ] && [ ! -f "$R/pre-damage/$f" ]; then
    NOTRUN="injection changed nothing: $f was already absent"; OUTCOME=$NOTRUN; return
  fi
  c6_score daemon-2 "$A2" "$f"
  case $1 in
  c) # ACCEPTANCE-v3 C6c: the cut is named E_MARK_TRUNCATED, nothing is rewritten.
     chk start_line_refused_E_MARK_TRUNCATED "$(reconcile_line daemon-2 | grep -q 'Reconcile: refused.*E_MARK_TRUNCATED' && echo true || echo false)" "$(q "$(reconcile_line daemon-2 | cut -c1-240)")"
     chk recall_refused_E_MARK_TRUNCATED "$( [ "$(j RCL-daemon-2 .ok)" = false ] && j RCL-daemon-2 .error | grep -q E_MARK_TRUNCATED && echo true || echo false)" "$(q "$(j RCL-daemon-2 '.error // "recall succeeded"' | cut -c1-240)")"
     chk cortex_cx_identical_to_damaged "$(cmp -s "$R/damaged.bin" "$R/compose/cortex.cx" && echo true || echo false)"
     chk mark_identical_to_damaged_state "$(cmp -s "$R/damaged-mark.bin" "$R/compose.cortex-mark" && echo true || echo false)";;
  i) # ACCEPTANCE-v3 C6i: no silent re-creation; cortex.cx only grows by appends.
     chk jspace_meta_still_absent "$( [ -e "$R/$f" ] && echo false || echo true)" "$(q "$(tstat "$R/$f")")"
     local p0; p0=$(stat -c %s "$R/pre-damage/compose/cortex.cx")
     chk cortex_cx_prefix_identical "$( [ "$(stat -c %s "$R/compose/cortex.cx")" -ge "$p0" ] && cmp -s -n "$p0" "$R/pre-damage/compose/cortex.cx" "$R/compose/cortex.cx" && echo true || echo false)" "{\"pre_bytes\":$p0,\"now_bytes\":$(stat -c %s "$R/compose/cortex.cx")}";;
  esac
}

# C6d guard (ACCEPTANCE-v3 DECIDED 3). jspace.data is never written by compose
# at omega 62b6a28 (G6); spill_end is the u64 at offset 48 of jspace.meta.
c6d_trim() {
  local f=$R/compose/jspace/jspace.data se; se=$(od -An -tu8 -j48 -N8 "$R/compose/jspace/jspace.meta" | tr -d ' ')
  if [ "${se:-0}" -gt 0 ]; then truncate -s $((se - 1)) "$f"; INJ="truncate jspace.data to spill_end-1 = $((se - 1)) bytes"
  else truncate -s -1 "$f"; INJ="truncate jspace.data by 1 byte (spill_end 0)"; fi
}
run_C6d() {
  local f=compose/jspace/jspace.data ds se
  ds=$(stat -c %s "$R/$f"); se=$(od -An -tu8 -j48 -N8 "$R/compose/jspace/jspace.meta" | tr -d ' ')
  chk guard_read "$( [ -n "$ds" ] && [ -n "$se" ] && echo true || echo false)" "{\"data_bytes\":${ds:-null},\"spill_end\":${se:-null}}"
  if [ "${ds:-0}" -gt 0 ] || [ "${se:-0}" -gt 0 ]; then
    c6d_trim; cp "$R/$f" "$R/damaged.bin"; cp "$R/compose.cortex-mark" "$R/damaged-mark.bin" 2>/dev/null
    c6_score daemon-2 "$A2" "$f"; return
  fi
  INJ="none: jspace.data 0 bytes and spill_end 0 (guard); case run uninjected"
  up daemon-2 || return
  cx RCL recall; chk recall_ok "$(eq "$(j RCL .ok)" true)"
  execute X2 "$A2"; chk uninjected_A2_DONE "$(eq "$(j X2 .state)" DONE)" "$(q "$(j X2 '.state // .error')")"
  A3=$(authorize A3)
  stop_daemon
  local de; de=$(stat -c %s "$R/$f")
  if [ "$de" -eq 0 ]; then
    chk end_of_case_jspace_data_still_0_bytes true 0
    NOTAPPL=1; OUTCOME="NOT_APPLICABLE: jspace.data 0 bytes, spill_end 0 before and after the case (compose never spills at omega 62b6a28)"
    return
  fi
  # The guard fired late: inject now and score on the fresh grant A3.
  chk end_of_case_jspace_data_still_0_bytes false "$de"
  T1=$(tstat "$T")
  rm -rf "$R/pre-damage"; mkdir -p "$R/pre-damage"
  (cd "$R" && cp -a --parents compose compose.machine-root state pre-damage/ 2>/dev/null; cp -a compose.cortex-mark pre-damage/ 2>/dev/null)
  c6d_trim; cp "$R/$f" "$R/damaged.bin"; cp "$R/compose.cortex-mark" "$R/damaged-mark.bin" 2>/dev/null
  chk injected_late_after_guard true "$de"
  c6_score daemon-3 "$A3" "$f"
}

# C6c-ctl: the on-disk state of a process that died after the A2 append and
# before its mark update (simulated: the older mark is put back).
run_C6c_ctl() {
  c6_base save_mark || return
  [ -f "$R/mark-before-A2" ] || { NOTRUN="no record mark before A2"; OUTCOME=$NOTRUN; return; }
  INJ="put back the record mark copied just before authorize A2 (daemon stopped)"
  cp "$R/mark-before-A2" "$R/compose.cortex-mark"
  local m; m=$(mark_records "$R/compose.cortex-mark")
  up daemon-2 || return
  chk start_log_mark_advanced_M_to_M_plus_1 "$(grep -q "record mark advanced $m->$((m + 1)) " "$R/daemon-2.log" && echo true || echo false)" "$(q "$(grep -m1 'Cortex mark:' "$R/daemon-2.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  cx RCL recall; chk recall_ok "$(eq "$(j RCL .ok)" true)" "$(q "$(j RCL '.error // "ok"' | cut -c1-200)")"
  execute X2 "$A2"; chk A2_DONE "$(eq "$(j X2 .state)" DONE)" "$(q "$(j X2 '.state // .error')")"
  local t2; t2=$(tstat "$T"); execute X3 "$A2"
  chk A2_retry_AlreadySpent "$(eq "$(refusal X3)" AlreadySpent)" "$(q "$(refusal X3)")"
  chk written_once "$(eq "$(tstat "$T")" "$t2")"
  OUTCOME="mark M=$m put back; $(grep -m1 'Cortex mark:' "$R/daemon-2.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-80); A2 $(j X2 .state)"
  rules final; stop_daemon
}

run_C7a() {
  # ACCEPTANCE-v4 C7a: the start-up reconcile returns an error (release builds
  # abort on panic, so the error arm is the path they take).
  INJ="daemon started with AIEN_FAULT_HOLD=reconcile_error (start-up reconcile returns an error)"
  export AIEN_FAULT_HOLD=reconcile_error
  start_daemon daemon-1; local up1=$?
  unset AIEN_FAULT_HOLD
  chk daemon_serves_socket_up "$( [ $up1 -eq 0 ] && [ -S "$R/aien.sock" ] && echo true || echo false)" "$(q "$(tail -2 "$R/daemon-1.log" | tr '\n' ' ' | cut -c1-200)")"
  [ $up1 -eq 0 ] || return
  local rl; rl=$(reconcile_line daemon-1)
  chk start_line_refused_names_refusal "$(echo "$rl" | grep -q '^Reconcile: refused: fault hold reconcile_error.*effect commands refuse until a successful reconcile (aien compose reconcile)$' && echo true || echo false)" "$(q "$(echo "$rl" | cut -c1-240)")"
  pre_state pre
  A1=$(authorize A1); chk authorize_answers "$( [ -n "$A1" ] && echo true || echo false)" "$(q "$A1")"
  execute X1 "$A1"; chk first_execute_refused_ReconcileFailed "$(eq "$(refusal X1)" ReconcileFailed)" "$(q "$(j X1 '.error // .state' | cut -c1-200)")"
  ledger L1; chk no_intent "$(eq "$(intent_state L1 "$A1")" none)"
  chk no_write "$(eq "$(tstat "$T")" "$T0")"
  cx RC reconcile; chk reconcile_ok "$(eq "$(j RC .ok)" true)" "$(q "$(j RC '.error // "ok"' | cut -c1-200)")"
  execute X2 "$A1"; chk second_execute_DONE "$(eq "$(j X2 .state)" DONE)" "$(q "$(j X2 '.state // .error')")"
  local t1; t1=$(tstat "$T"); chk written "$( [ "$t1" != "$T0" ] && echo true || echo false)"
  execute X3 "$A1"
  chk written_once "$(eq "$(tstat "$T")" "$t1")"
  stop_daemon; up daemon-2 || return
  chk restart_reconcile_normal "$(reconcile_line daemon-2 | grep -q 'checked 0' && echo true || echo false)" "$(q "$(reconcile_line daemon-2)")"
  OUTCOME="start reconcile refused (error hook); execute $(refusal X1); reconcile ok=$(j RC .ok); execute $(j X2 .state); restart clean"
  rules final; stop_daemon
}

run_C7c() {
  # ACCEPTANCE-v4 C7c: a real panic at start. Expected: daemon exits, no
  # socket served, no write; the next start clears the left socket file.
  INJ="daemon started with AIEN_FAULT_HOLD=reconcile_panic (real panic); restart without the hook, left socket file kept"
  local h0; h0=$(home_digest "$R")
  export AIEN_FAULT_HOLD=reconcile_panic
  DRC=; start_daemon daemon-1
  unset AIEN_FAULT_HOLD
  if kill -0 "$DPID" 2>/dev/null; then
    chk daemon_exits_by_itself false "$(q "still running: $(tail -2 "$R/daemon-1.log" | tr '\n' ' ' | cut -c1-160)")"
    kill_daemon; return
  fi
  chk daemon_exits_by_itself true
  chk exit_status_nonzero "$( [ -n "$DRC" ] && [ "$DRC" -ne 0 ] && echo true || echo false)" "$(q "$DRC")"
  chk log_names_fault_hold "$(grep -q 'fault hold reconcile_panic' "$R/daemon-1.log" && echo true || echo false)" "$(q "$(grep -m1 'panicked at' "$R/daemon-1.log" | cut -c1-160)")"
  cx NS recall
  chk no_process_serves_socket "$(j NS '(.ok == false) and ((.error // "") | test("Failed to connect"))')" "$(q "$(j NS '.error // "connected"' | cut -c1-200)")"
  chk socket_file_left_recorded_not_scored true "$(q "$( [ -S "$R/aien.sock" ] && echo present || echo absent)")"
  local sockleft; sockleft=$( [ -S "$R/aien.sock" ] && echo present || echo absent)
  chk home_identical "$(eq "$(home_digest "$R")" "$h0")"
  chk target_unchanged "$(eq "$(tstat "$T")" "$T0")"
  chk provenance_empty "$( [ -z "$(ls -A "$R/prov")" ] && echo true || echo false)"
  KEEP_SOCK=1; up daemon-2; local up2=$?; unset KEEP_SOCK
  [ $up2 -eq 0 ] || return
  chk restart_reconcile_normal "$(reconcile_line daemon-2 | grep -q 'checked 0' && echo true || echo false)" "$(q "$(reconcile_line daemon-2)")"
  pre_state pre; chk recall_ok "$(eq "$(j pre .ok)" true)"
  ledger L1; chk ledger_no_intent "$(eq "$(j L1 '.ledger.intents | length')" 0)" "$(q "$(j L1 '.ledger.intents | length')")"
  OUTCOME="panic at start: exit $DRC, no server, home identical, socket file $sockleft; restart clean"
  rules final; stop_daemon
}

run_C7b() {
  c6_base || return
  INJ="cut cortex.cx at the record boundary before the last record (C6c), then recover"
  cp "$R/compose.cortex-mark" "$R/damaged-mark.bin"
  truncate -s "$S1" "$R/compose/cortex.cx"
  up daemon-2 || return
  local rl; rl=$(reconcile_line daemon-2)
  chk start_refused_E_MARK_TRUNCATED "$(echo "$rl" | grep -q 'Reconcile: refused.*E_MARK_TRUNCATED' && echo true || echo false)" "$(q "$(echo "$rl" | cut -c1-240)")"
  cx REC recover
  chk recover_mark_lost_1 "$(eq "$(j REC .repair.mark_lost)" 1)" "$(q "$(j REC '.repair.mark_lost // .error')")"
  local kept; kept=$(j REC '.repair.mark_kept_as // ""')
  chk mark_kept_as_lost_seq_file "$(echo "$kept" | grep -Eq '\.cortex-mark\.lost-[0-9]+$' && [ -f "$kept" ] && echo true || echo false)" "$(q "$kept")"
  chk kept_mark_identical_to_damaged_state "$(cmp -s "$R/damaged-mark.bin" "$kept" && echo true || echo false)"
  local rr; rr=$(j REC '.repair.mark_repair_record // 0')
  cx RCR recall --ids "$rr"
  chk repair_constraint_record "$(j RCR "[.recall.cited[] | select(.note == \"constraint\") | .text | fromjson? | select(.repair == \"cortex-mark\" and .lost == 1)] | length == 1")" "$(q "$(j RCR '.recall.cited[0].text // .error' | cut -c1-240)")"
  A3=$(authorize A3); chk authorize_A3_answers "$( [ -n "$A3" ] && echo true || echo false)" "$(q "$A3")"
  execute X3 "$A3"; chk first_execute_refused_ReconcileFailed "$(eq "$(refusal X3)" ReconcileFailed)" "$(q "$(j X3 '.error // .state' | cut -c1-200)")"
  chk no_write_before_reconcile "$(eq "$(tstat "$T")" "$T1")"
  cx RC reconcile; chk reconcile_ok "$(eq "$(j RC .ok)" true)" "$(q "$(j RC '.error // "ok"' | cut -c1-200)")"
  execute X4 "$A3"; chk second_execute_A3_DONE "$(eq "$(j X4 .state)" DONE)" "$(q "$(j X4 '.state // .error')")"
  local t4; t4=$(tstat "$T"); execute X5 "$A3"
  chk written_once "$(eq "$(tstat "$T")" "$t4")"
  OUTCOME="start refused (mark); recover mark_lost=$(j REC .repair.mark_lost); execute $(refusal X3); reconcile ok=$(j RC .ok); execute A3 $(j X4 .state)"
  rules final; stop_daemon
}

run_C8a() {
  INJ="v2 home without a record mark (copy of F0v2)"
  [ -f "$R/compose.cortex-mark" ] && { NOTRUN="FIX_OLD has a record mark"; OUTCOME=$NOTRUN; return; }
  up daemon-1 || return
  local na; na=$(grep -c 'Cortex mark: mark adopted (seq 1, records [0-9]*)' "$R/daemon-1.log")
  chk first_start_one_adopted_line "$(eq "$na" 1)" "$(q "$(grep -m1 'Cortex mark:' "$R/daemon-1.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  chk mark_128_bytes "$(eq "$(stat -c %s "$R/compose.cortex-mark" 2>/dev/null)" 128)"
  pre_state pre; chk recall_ok "$(eq "$(j pre .ok)" true)"
  stop_daemon; up daemon-2 || return
  chk second_start_no_adopt_or_advance "$(grep -Eq 'mark adopted|record mark advanced' "$R/daemon-2.log" && echo false || echo true)" "$(q "$(grep -m1 'Cortex mark:' "$R/daemon-2.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  cx R2 recall; chk second_recall_ok "$(eq "$(j R2 .ok)" true)"
  OUTCOME="$(grep -m1 'Cortex mark:' "$R/daemon-1.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-80); second start quiet"
  rules final; stop_daemon
}

run_C8b() {
  run_control
  INJ="after the control steps: delete compose.cortex-mark (daemon stopped)"
  rm -f "$R/compose.cortex-mark"
  up daemon-3 || return
  chk start_one_adopted_line "$(eq "$(grep -c 'Cortex mark: mark adopted' "$R/daemon-3.log")" 1)" "$(q "$(grep -m1 'Cortex mark:' "$R/daemon-3.log" | sed 's/\x1b\[[0-9;]*m//g')")"
  cx R3 recall; chk recall_ok "$(eq "$(j R3 .ok)" true)"
  local t1; t1=$(tstat "$T"); execute X9 "$A1"
  chk reexecute_AlreadySpent "$(eq "$(refusal X9)" AlreadySpent)" "$(q "$(refusal X9)")"
  chk no_second_write "$(eq "$(tstat "$T")" "$t1")"
  OUTCOME="control done; mark deleted; restart adopted; re-execute $(refusal X9)"
  rules final2; stop_daemon
}

old_fixture_ok() {   # F0v2 files sha re-check (ACCEPTANCE-v3 section 4)
  local want got; want=$(jq -r .files_sha256 "$1/fixture.json" 2>/dev/null)
  got=$( (tree_list "$1/compose"; sha256sum "$1/compose.machine-root"; [ -f "$1/compose.cortex-mark" ] && sha256sum "$1/compose.cortex-mark"; tree_list "$1/ws") | sha256sum | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ]
}

for c in $CASES; do
  case $c in
    control) for k in C1 C2 C4 C5 C6 C7 C8; do new_run "control-$k"; run_control; finish "control-$k" 1; done;;
    C1a|C1b) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C1 "${c#C1}"; finish "$c" "$r"; done;;
    C2a) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C2a; finish "$c" "$r"; done;;
    C2b|C2c|C2d) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C2bcd "${c#C2}"; finish "$c" "$r"; done;;
    C3a) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C3a; finish "$c" "$r"; done;;
    C3b) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C3b; finish "$c" "$r"; done;;
    C4) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C4; finish "$c" "$r"; done;;
    C5a|C5b|C5c) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C5 "${c#C5}"; finish "$c" "$r"; done;;
    C6i) new_run "$c-1"; run_C6 i; finish "$c" 1;;   # exactly once (DECIDED 4)
    C6c-ctl) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C6c_ctl; finish "$c" "$r"; done;;
    C6?) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C6 "${c#C6}"; finish "$c" "$r"; done;;
    C7a) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C7a; finish "$c" "$r"; done;;
    C7b) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C7b; finish "$c" "$r"; done;;
    C7c) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C7c; finish "$c" "$r"; done;;
    C8a) for r in $(seq 1 "$REPS"); do
           if [ -z "${FIX_OLD:-}" ] || ! old_fixture_ok "$FIX_OLD"; then
             new_run "$c-$r"; NOTRUN="FIX_OLD (F0v2) missing or its files sha256 does not match its fixture.json"; OUTCOME=$NOTRUN
           else new_run "$c-$r" "$FIX_OLD"; run_C8a; fi
           finish "$c" "$r"; done;;
    C8b) for r in $(seq 1 "$REPS"); do new_run "$c-$r"; run_C8b; finish "$c" "$r"; done;;
    *) echo "unknown case $c"; exit 2;;
  esac
  # never leave a daemon behind
  [ -n "$DPID" ] && kill -0 "$DPID" 2>/dev/null && kill_daemon
done
exit 0
