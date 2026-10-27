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

# CAND-4 qualification: harness-side tests for coverage cases 2 and 5 (ACCEPTANCE-CAND4.md
# section 6.1). Shell + jq only. No candidate code, no hook: the frozen cpu_fault binary is
# started, killed with SIGKILL at a measured moment and restarted. Never touches the GPU.
#   run-windows.sh FIX_ROOT RUN_ROOT
#     FIX_ROOT  the Q2 fixture F0 (fixture.json with s3_committed true)
#     RUN_ROOT  output; results.jsonl gets one line per run:
#               {row, rep, verdict, outcome, env:"CPU fault injection", checks, run_root}
# AIEN_BIN = the cpu_fault binary. Rows (windows.decl.json):
#   W-ctl-nokill        control: propose to the end, no kill; measures the propose time TP
#   W-ctl-wrong-expect  control: the restart evaluator given a wrong prefix digest must say FAIL
#   W-ctl-damaged-mark  control: a damaged record mark after a clean authorize must be seen
#   W2 x3               case 2: SIGKILL the daemon at 0.3 / 0.5 / 0.7 x TP into `compose propose`
#   W5 x1               case 5: 40 trials, authorize + SIGKILL at a random 0-50 ms offset
# Verdicts: PASS, FAIL, NOT_RUN (the injection did not happen: nothing is claimed).
set -u
BIN=${AIEN_BIN:?AIEN_BIN}
FIX=${1:?FIX_ROOT} ROOT=${2:?RUN_ROOT}
TRIALS=${W5_TRIALS:-40}
mkdir -p "$ROOT"; ROOT=$(cd "$ROOT" && pwd); FIX=$(cd "$FIX" && pwd)
jq -e '.s3_committed == true' "$FIX/fixture.json" >/dev/null || { echo "fixture F0 has no committed proposal: NOT_RUN"; exit 3; }
PPATH=$(jq -r .proposal_path "$FIX/fixture.json")
GOAL='Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.'

now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
tree_list() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum); }
# the socket path must be shorter than SUN_LEN (108): sockets live in a short temp dir
SOCKD=$(mktemp -d /tmp/c4w.XXXXXX); SOCK=$SOCKD/s
DPID=
trap 'rm -rf "$SOCKD"; [ -n "$DPID" ] && kill -0 "$DPID" 2>/dev/null && kill -9 "$DPID" 2>/dev/null' EXIT
start_daemon() {
  local log=$R/$1.log i=0
  rm -f "$SOCK"
  (cd "$R" && exec setsid "$BIN" daemon >"$log" 2>&1 </dev/null) &
  DPID=$!
  while kill -0 "$DPID" 2>/dev/null && [ $i -lt 1200 ]; do
    if [ -S "$SOCK" ] && grep -q '^Reconcile:' "$log"; then break; fi
    sleep 0.1; i=$((i + 1))
  done
  kill -0 "$DPID" 2>/dev/null || { wait "$DPID" 2>/dev/null; return 1; }
  if ! grep -q 'Backend:.*CPU-reference' "$log"; then
    echo "harness: daemon $1 is not the CPU build; stopping (no GPU use allowed here)" >&2
    kill -9 "$DPID"; exit 9
  fi
  [ -S "$SOCK" ] && grep -q '^Reconcile:' "$log"
}
stop_daemon() {
  "$BIN" compose shutdown >/dev/null 2>&1
  local i=0; while kill -0 "$DPID" 2>/dev/null && [ $i -lt 300 ]; do sleep 0.1; i=$((i + 1)); done
  wait "$DPID" 2>/dev/null; rm -f "$SOCK"
}
kill_daemon() { kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; rm -f "$SOCK"; }
cx() { local id=$1; shift; "$BIN" compose "$@" >"$R/steps/$id.json" 2>"$R/steps/$id.err"; local rc=$?; echo "$rc" >"$R/steps/$id.rc"; return $rc; }
j() { jq -r "$2" "$R/steps/$1.json" 2>/dev/null; }
eq() { [ "$1" = "$2" ] && echo true || echo false; }
q() { jq -Rn --arg s "$1" '$s'; }
CHECKS='[]'; NOTRUN=
chk() { CHECKS=$(jq -c --arg n "$1" --argjson ok "$2" --argjson v "${3:-null}" '. + [{check:$n, ok:$ok, value:$v}]' <<<"$CHECKS"); }

new_run() {   # NAME
  R=$ROOT/$1; rm -rf "$R"; mkdir -p "$R"/prov "$R"/state "$R"/steps
  cp -a "$FIX/compose" "$FIX/compose.machine-root" "$FIX/ws" "$FIX/outside" "$FIX/s3-report.json" "$R/"
  [ -f "$FIX/compose.cortex-mark" ] && cp -a "$FIX/compose.cortex-mark" "$R/"
  WS=$R/ws; CHECKS='[]'; NOTRUN=
  export AIEN_COMPOSE_DIR=$R/compose AIEN_PROVENANCE_DIR=$R/prov AIEN_RUNTIME_SOCK=$SOCK \
         AIEN_RUNTIME_STATE_DIR=$R/state AIEN_REQUIRE_CHECKPOINT=0
  unset AIEN_MODEL_PATH AIEN_TOKENIZER_PATH AIEN_REQUIRE_BLACKWELL AIEN_GPU_BACKEND AIEN_FAULT_HOLD AIEN_FAULT_HOLD_FILE
  # sc#328: the daemon requires the approval desk key (the fixture may carry one).
  [ -f "$R/compose/approval-desk.key" ] || "$BIN" compose desk-key --create 1 >/dev/null
  WS0=$(tree_list "$WS")
}
pre_state() {   # sets N0, MID0, PD0 (daemon up)
  cx pre recall >/dev/null; N0=$(j pre .recall.records_total); MID0=$(j pre .machine_id)
  cx pre-prefix recall --prefix "$N0" >/dev/null; PD0=$(j pre-prefix .recall.prefix_digest)
}
# eval_restart TAG WANT_PD: the restart evaluator. Daemon TAG is up (or not). Returns 0 iff
# every consistency check holds: up, recall ok, no E_MARK, machine id and the prefix digest
# over the first N0 records equal the pre-fault values, workspace unchanged.
EVAL_NOTES=
eval_restart() {
  local tag=$1 wantpd=$2 ok=0 em
  cx "$tag-recall" recall --prefix "$N0" >/dev/null
  em=$(cd "$R" && grep -l 'E_MARK' daemon-*.log steps/*.json 2>/dev/null | paste -sd, -)
  chk "${tag}_recall_ok" "$(eq "$(j "$tag-recall" .ok)" true)" "$(q "$(j "$tag-recall" '.error // "ok"' | cut -c1-160)")"
  chk "${tag}_no_E_MARK" "$( [ -z "$em" ] && echo true || echo false)" "$(q "$em")"
  chk "${tag}_machine_id" "$(eq "$(j "$tag-recall" .machine_id)" "$MID0")"
  chk "${tag}_prefix_digest" "$(eq "$(j "$tag-recall" .recall.prefix_digest)" "$wantpd")"
  chk "${tag}_workspace_unchanged" "$(eq "$(tree_list "$WS")" "$WS0")"
  jq -e 'all(.ok)' <<<"$CHECKS" >/dev/null
}
emit() {   # ROW REP VERDICT OUTCOME
  jq -nc --arg row "$1" --argjson rep "$2" --arg v "$3" --arg o "$4" --argjson c "$CHECKS" --arg root "$R" \
    '{row:$row, rep:$rep, verdict:$v, outcome:$o, env:"CPU fault injection", checks:$c, run_root:$root}' >>"$ROOT/results.jsonl"
  echo "$1 rep $2: $3  $4"
}
verdict_of() { if [ -n "$NOTRUN" ]; then echo NOT_RUN; else jq -r 'if length > 0 and all(.ok) then "PASS" else "FAIL" end' <<<"$CHECKS"; fi; }
fsleep() { sleep "$(awk "BEGIN{printf \"%.3f\", $1/1000}")"; }

TP_MS=; SETTLE=
# ---- W-ctl-nokill: baseline, also measures the propose time and whether propose settles
new_run W-ctl-nokill
start_daemon d1 || { chk daemon_up false; emit W-ctl-nokill 1 FAIL "daemon did not start"; exit 0; }
pre_state
t0=$(now_ms); cx P1 propose --goal "$GOAL" --workspace "$WS"; TP_MS=$(( $(now_ms) - t0 ))
cx post recall >/dev/null; N1=$(j post .recall.records_total)
SETTLE=$(( N1 > N0 ? 1 : 0 ))
chk propose_answered "$(eq "$(j P1 'has("report") or has("error")')" true)" "$(q "$(j P1 '.report.committed // .error // "none"' | cut -c1-120)")"
chk propose_ms_measured "$( [ "$TP_MS" -gt 0 ] && echo true || echo false)" "$TP_MS"
chk propose_wrote_records "$(eq "$SETTLE" 1)" "[$N0,$N1]"
N0=$N1; cx pre-prefix recall --prefix "$N0" >/dev/null; PD0=$(j pre-prefix .recall.prefix_digest)
stop_daemon; start_daemon d2 || chk daemon_2_up false
eval_restart restart "$PD0"
emit W-ctl-nokill 1 "$(verdict_of)" "propose took ${TP_MS} ms, records $N0 after, records written by propose: $SETTLE"
stop_daemon
echo "$TP_MS $SETTLE" >"$ROOT/tp"

# ---- W-ctl-wrong-expect: the evaluator must reject a wrong expected prefix digest
new_run W-ctl-wrong-expect
start_daemon d1; pre_state; stop_daemon; start_daemon d2
if eval_restart restart "$(printf '0%.0s' $(seq 64))"; then v=FAIL o="the evaluator accepted a wrong prefix digest"; else v=PASS o="the evaluator rejected a wrong prefix digest (a control: PASS means it can see the difference)"; fi
emit W-ctl-wrong-expect 1 $v "$o"; stop_daemon

# ---- W-ctl-damaged-mark: a damaged record mark after a clean authorize must be seen
new_run W-ctl-damaged-mark
start_daemon d1; pre_state
cx A1 authorize --report "$R/s3-report.json" --workspace "$WS" --approver drake --desk 1
stop_daemon
if [ -f "$R/compose.cortex-mark" ]; then
  printf 'X' | dd of="$R/compose.cortex-mark" bs=1 seek=70 conv=notrunc 2>/dev/null
  start_daemon d2; cx pre2 recall >/dev/null
  em=$(cd "$R" && grep -l 'E_MARK' daemon-*.log steps/*.json 2>/dev/null | paste -sd, -)
  if [ -n "$em" ]; then v=PASS o="a damaged mark shows E_MARK in: $em"; else v=FAIL o="a damaged mark was not reported"; fi
  stop_daemon
else v=FAIL o="no record mark to damage"; fi
emit W-ctl-damaged-mark 1 $v "$o"

# ---- W2: SIGKILL the daemon inside `compose propose`, then restart
if [ "$SETTLE" != 1 ] || [ "$TP_MS" -lt 20 ]; then why="control propose took ${TP_MS} ms and wrote records=$SETTLE: no window to hit on this build"; else why=; fi
rep=0
for f in 300 500 700; do
  rep=$((rep + 1)); new_run "W2-$rep"
  if [ -n "$why" ]; then NOTRUN=$why; emit W2 $rep NOT_RUN "$why"; continue; fi
  start_daemon d1 || { chk daemon_1_up false; emit W2 $rep FAIL "daemon did not start"; continue; }
  pre_state
  "$BIN" compose propose --goal "$GOAL" --workspace "$WS" >"$R/steps/P1.json" 2>"$R/steps/P1.err" & CPID=$!
  fsleep $(( TP_MS * f / 1000 ))
  alive=no; kill -0 "$CPID" 2>/dev/null && alive=yes
  kill_daemon; wait "$CPID" 2>/dev/null; crc=$?
  chk kill_in_flight "$(eq "$alive" yes)" "$(jq -nc --argjson tp "$TP_MS" --argjson f "$f" --argjson crc "$crc" '{tp_ms:$tp, at_permille:$f, client_rc:$crc}')"
  [ "$alive" = yes ] || NOTRUN="the propose client had already finished when the kill was sent"
  if start_daemon d2; then chk daemon_2_up true; else
    chk daemon_2_up false "$(q "$(grep -m1 -E 'Fatal|E_[A-Z_]+' "$R/d2.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160)")"
    cx REC recover >/dev/null 2>&1; chk recover_ran "$(eq "$(cat "$R/steps/REC.rc")" 0)"
    start_daemon d3 && chk daemon_3_up true || chk daemon_3_up false
  fi
  eval_restart restart "$PD0"
  NA=$(j restart-recall .recall.records_total)
  emit W2 $rep "$(verdict_of)" "kill at $((f / 10))% of ${TP_MS} ms; client alive $alive; records before $N0, after restart $NA; $(grep -m1 'Reconcile:' "$R/d2.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-80)"
  stop_daemon
done

# ---- W5: authorize + SIGKILL at a random 0-50 ms, restart, TRIALS times
new_run W5; W5R=$R; ALLCHECKS='[]'; adv=0; landed=0; bad=0; offs=
for t in $(seq 1 "$TRIALS"); do
  new_run "W5/t$t"
  off=$(( RANDOM % 51 )); offs="$offs $off"
  if ! start_daemon d1; then bad=$((bad + 1)); ALLCHECKS=$(jq -c --argjson t "$t" '. + [{check:"trial_\($t)_daemon_1_up", ok:false}]' <<<"$ALLCHECKS"); continue; fi
  pre_state
  "$BIN" compose authorize --report "$R/s3-report.json" --workspace "$WS" --approver drake --desk 1 >"$R/steps/A1.json" 2>"$R/steps/A1.err" & CPID=$!
  fsleep "$off"
  kill_daemon; wait "$CPID" 2>/dev/null
  start_daemon d2 || { chk daemon_2_up false; cx REC recover >/dev/null 2>&1; start_daemon d3; }
  eval_restart restart "$PD0" || bad=$((bad + 1))
  grep -q 'record mark advanced' "$R/d2.log" 2>/dev/null && adv=$((adv + 1))
  ALLCHECKS=$(jq -c --argjson t "$t" --argjson off "$off" --argjson ad "$(grep -q 'record mark advanced' "$R/d2.log" 2>/dev/null && echo true || echo false)" \
    --argjson ok "$(jq 'all(.ok)' <<<"$CHECKS")" '. + [{check:"trial_\($t)_restart_consistent", ok:$ok, value:{offset_ms:$off, record_mark_advanced:$ad}}]' <<<"$ALLCHECKS")
  stop_daemon
done
R=$W5R; CHECKS=$ALLCHECKS
chk all_trials_consistent "$(eq "$bad" 0)" "$bad"
chk window_hit_at_least_once "$( [ "$adv" -ge 1 ] && echo true || echo false)" "$(q "$adv of $TRIALS restarts logged record mark advanced")"
[ "$adv" -ge 1 ] || NOTRUN="no trial hit the window (no restart logged: record mark advanced)"
emit W5 1 "$(verdict_of)" "$TRIALS trials, offsets (ms)${offs}; restarts that advanced the mark: $adv; inconsistent: $bad"
