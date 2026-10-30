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

# WHOLE-SYSTEM-E2E held-fault rows (aien-sovereign-core#379, lane L2 of aien-architecture#190).
# Rows and predictions: docs/campaigns/whole-system-e2e/FAULTS-v1.md (committed before this script; never edited by it).
#
#   scripts/whole_system_e2e_faults.sh run        build a fault-hold CLI, run the rows, write one run folder
#
# Shell plus jq. No Python, no systemd, no accelerator: every daemon must print the CPU-reference backend line.
# Needs CLAUDE_JOB_DIR (build target and state under $CLAUDE_JOB_DIR/tmp/e2e-faults), the prebuilt compose archive
# at $CLAUDE_JOB_DIR/tmp/librx_compose.a, jq, sha256sum, hostname. Model: WSE2E_MODEL_DIR (default the CAND-4 snapshot).
# Idempotent: it rebuilds its own job dir from scratch each run (rows/, fixture/ are removed first), and it only
# ever signals the daemon and executor processes it started itself (pids kept in DPID and EPID).
set -u
[ "${1:-}" = run ] || { echo "usage: $0 run" >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
JOB=${CLAUDE_JOB_DIR:?CLAUDE_JOB_DIR must be set}
LIB=$JOB/tmp/librx_compose.a
D=$JOB/tmp/e2e-faults
BIN=$D/target/release/aien-cli
MDIR=${WSE2E_MODEL_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/ 2>/dev/null | head -1)}
UTC0=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${WSE2E_OUT:-$REPO/docs/campaigns/whole-system-e2e/runs/FAULTS-$UTC0}
for t in jq sha256sum hostname; do command -v $t >/dev/null || { echo "need $t" >&2; exit 2; }; done
[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 2; }
[ -n "$MDIR" ] && [ -f "$MDIR/model.safetensors" ] && [ -f "$MDIR/tokenizer.json" ] || { echo "model dir '$MDIR' lacks model.safetensors or tokenizer.json" >&2; exit 2; }
QUIET=$HOME/workspace/.spark-quiet
while [ -s "$QUIET" ]; do echo "quiet flag set; waiting"; sleep 30; done
DIRTY=$(git -C "$REPO" status --porcelain -- . ":!docs/campaigns/whole-system-e2e/runs" | wc -l)
[ "$DIRTY" -eq 0 ] || [ -n "${WSE2E_ALLOW_DIRTY:-}" ] || { echo "working tree has $DIRTY uncommitted changes; commit them first" >&2; exit 2; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
mkdir -p "$D"
( cd "$REPO" && env -u AIEN_CUDAPP_DIR -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_FORCE_CPU_STUB \
    AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" cargo build --release -p aien-cli --features aien-cli/fault-hold ) >"$D/build.log" 2>&1 \
  || { echo "build failed, see $D/build.log" >&2; exit 2; }
grep -q "building the stub" "$D/build.log" && { echo "stub compose build; refusing" >&2; exit 2; }
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
for p in AIEN_FAULT_HOLD reconcile_panic reconcile_error; do grep -aqF "$p" "$BIN" || { echo "binary lacks the $p hold; refusing" >&2; exit 2; }; done
W_SHA=$(sha256sum "$(readlink -f "$MDIR/model.safetensors")" | cut -d' ' -f1)
T_SHA=$(sha256sum "$(readlink -f "$MDIR/tokenizer.json")" | cut -d' ' -f1)
C_W=$(sed -n 's/^model-safetensors-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_T=$(sed -n 's/^tokenizer-json-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_ID=$(sed -n 's/^model-id = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
HOST=$(hostname)

rm -rf "$D/rows" "$D/fixture"; mkdir -p "$D/rows" "$D/fixture" "$OUT/chain" "$OUT/artifacts"
cp "$REPO/docs/campaigns/whole-system-e2e/FAULTS-v1.md" "$OUT/FAULTS-v1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/FAULTS-v1.1.md" "$OUT/FAULTS-v1.1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/FAULTS-v1.2.md" "$OUT/FAULTS-v1.2.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/FAULTS-v1.2.1.md" "$OUT/FAULTS-v1.2.1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/FAULTS-v1.3.md" "$OUT/FAULTS-v1.3.as-run.md"

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
tstat() { stat -c '%i %s %.9Y' "$1" 2>/dev/null || echo absent; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

OBJ6='Write a file named done.md in the outbox folder containing one line: finished.'
OBJ_ID=$(printf 'AIEN_E2E_FAULTS_OBJECTIVE_V1\n%s' "$OBJ6" | sha256sum | cut -d' ' -f1)
RUN_ID=$(printf '%s\n%s\n%s\n%s' "$OBJ_ID" "$BIN_SHA" "$W_SHA" "$HOST" | sha256sum | cut -d' ' -f1)
# v1.3 (FAULTS-v1.3.md, sc#408): the id `aien-cli objective record` gives OBJ6 (sc#388 formula); not the chain id above.
REC_ID=$(printf 'AIEN_E2E_OBJECTIVE_V1\n%s' "$OBJ6" | sha256sum | cut -d' ' -f1)
OBJ6_SHA=$(printf '%s' "$OBJ6" | sha256sum | cut -d' ' -f1)

# ---- the receipt chain (same format as scripts/whole_system_e2e.sh) -----------------------------------------
N=0; PREV=$(printf '0%.0s' $(seq 1 64)); CHAIN=$OUT/chain; ROWS=""; CUR_BACKEND=""
rcpt() { # step status evidence [fields json]
  N=$((N + 1)); local f; f=$(printf '%s/%03d-%s.json' "$CHAIN" "$N" "$1")
  jq -n --arg oid "$OBJ_ID" --arg rid "$RUN_ID" --arg step "$1" --arg status "$2" --arg ev "$3" --argjson fields "${4:-{\}}" \
    --arg utc "$(utc)" --arg prev "$PREV" --arg commit "$COMMIT" --arg bin "$BIN_SHA" --arg w "$W_SHA" --arg t "$T_SHA" \
    --arg backend "${CUR_BACKEND:-no daemon started}" --arg host "$HOST" \
    '{objective_id:$oid, run_id:$rid, step:$step, status:$status, utc:$utc, prev_receipt_sha256:$prev, evidence:$ev,
      repo_commit:$commit, daemon_sha256:$bin, cli_sha256:$bin, model_weights_digest:$w, tokenizer_sha256:$t,
      backend_line:$backend, host:$host} + $fields' >"$f"
  PREV=$(sha_of "$f")
  case $1 in CTRL-*|E[0-9]*|FAULT-*) ROWS="$ROWS$1 $2"$'\n' ;; esac
  echo "[$(date -u +%H:%M:%S)] $1 $2: $3" | cut -c1-400 | tee -a "$OUT/console.log"
}

# ---- per-row environment (same variables as the E2E harness) -------------------------------------------------
R=""; WS=""; SOCK=""; ART=""; E=(); REPORT=""
mkenv() { # row dir R already exists
  SOCK=$R/s; WS=$D/work/ws   # the proposal binds the workspace path, so every row uses the same path (re-seeded per row)
  E=(env -u AIEN_CUDAPP_DIR -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_REQUIRE_BLACKWELL
     -u AIEN_GPU_BACKEND -u AIEN_MODEL_DIR -u AIEN_MODEL_PATH -u AIEN_TOKENIZER_PATH -u AIEN_FORCE_CPU_STUB
     -u AIEN_FAULT_HOLD -u AIEN_FAULT_HOLD_FILE -u AIEN_ALLEN_SUBJECT -u AIEN_ALLEN_ADOPT
     -u AIEN_COMPOSE_MAX_TOKENS -u AIEN_COMPOSE_DOC_MAX_TOKENS
     HOME="$R/home" NO_COLOR=1 AIEN_RUNTIME_SOCK="$SOCK" AIEN_RUNTIME_STATE_DIR="$R/state"
     AIEN_COMPOSE_DIR="$R/compose" AIEN_PROVENANCE_DIR="$R/prov" AIEN_REQUIRE_CHECKPOINT=1
     AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=1 AIEN_COMPOSE_EDIT_BUDGET_MS=599000 AIEN_COMPOSE_DOC_BUDGET_MS=599000)
}
cli() { "${E[@]}" "$BIN" "$@"; }

# ---- daemon and executor control (only our own pids) ---------------------------------------------------------
DPID=""; EPID=""; DRC=""; DLOG=/dev/null; DN=0; KILL_RC=""; EKILL_RC=""; ERC=""
start_daemon() { # [VAR=val ...]; sets DPID, CUR_BACKEND; on early exit sets DRC
  DN=$((DN + 1)); DLOG=$R/logs/daemon-$DN.log; DRC=""; rm -f "$SOCK"
  "${E[@]}" AIEN_MODEL_PATH="$MDIR/model.safetensors" AIEN_TOKENIZER_PATH="$MDIR/tokenizer.json" "$@" "$BIN" daemon >"$DLOG" 2>&1 &
  DPID=$!
  local i
  for i in $(seq 1 600); do
    sleep 2
    if [ ! -d "/proc/$DPID" ]; then wait "$DPID" 2>/dev/null; DRC=$?; DPID=""; echo "daemon $DN exited early (rc $DRC): $(tail -c 300 "$DLOG" | tr '\n' ' ')"; return 1; fi
    if [ -S "$SOCK" ] && grep -q -E "Replay reconcile:|^Reconcile:" "$DLOG"; then break; fi
  done
  [ -S "$SOCK" ] || { echo "daemon $DN not serving after 20 min"; return 1; }
  CUR_BACKEND=$(grep -m1 '^  Backend:' "$DLOG" | sed 's/^ *//')
  case $CUR_BACKEND in *CPU-reference*) ;; *) echo "not the CPU reference backend: '$CUR_BACKEND'"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1 ;; esac
  return 0
}
reconcile_line() { grep -m1 '^Reconcile:' "$DLOG" | cut -c1-300; }
stop_graceful() { [ -n "$DPID" ] || return 0; cli compose shutdown >"$R/logs/shutdown-$DN.json" 2>&1; local i=0; while [ -d "/proc/$DPID" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i + 1)); done; wait "$DPID" 2>/dev/null; DPID=""; rm -f "$SOCK"; }
kill_daemon() { [ -n "$DPID" ] || return 1; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; KILL_RC=$?; DPID=""; rm -f "$SOCK"; [ "$KILL_RC" -eq 137 ]; }
kill_executor() { [ -n "$EPID" ] || return 1; kill -9 "$EPID" 2>/dev/null; wait "$EPID" 2>/dev/null; EKILL_RC=$?; EPID=""; rm -f "$R/hold"; [ "$EKILL_RC" -eq 137 ]; }
trap '[ -n "$DPID" ] && kill -9 "$DPID" 2>/dev/null; [ -n "$EPID" ] && kill -9 "$EPID" 2>/dev/null' EXIT

# ---- steps ---------------------------------------------------------------------------------------------------
authorize() { # tag -> grant id on stdout (empty if refused)
  cli compose authorize --report "$REPORT" --workspace "$WS" --approver drake --desk 1 >"$ART/$1.json" 2>"$ART/$1.err"
  [ "$(jq -r '.ok' "$ART/$1.json" 2>/dev/null)" = true ] && jq -r '.authorization.id' "$ART/$1.json"
}
T_STATE=none; T_REF=""; T_TEXT=""
execute() { # tag grant: sets T_STATE, T_REF (refusal name), T_TEXT
  cli compose execute --report "$REPORT" --workspace "$WS" --authorization "$2" >"$ART/$1.json" 2>"$ART/$1.err"
  T_STATE=$(jq -r '.state // "none"' "$ART/$1.json" 2>/dev/null); [ -n "$T_STATE" ] || T_STATE=none
  T_TEXT=$(jq -r '.error // ""' "$ART/$1.json" 2>/dev/null | head -c 220 | tr '\n' ' ')
  [ -n "$T_TEXT" ] || T_TEXT=$(head -c 220 "$ART/$1.err" | tr '\n' ' ')
  T_REF=$(printf '%s' "$T_TEXT" | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\):.*/\1/p')
  [ -n "$T_REF" ] || T_REF=$(printf '%s' "$T_TEXT" | grep -o 'E_[A-Z_]*' | head -1)
  [ "$T_STATE" = DONE ]
}
hold_execute() { # point tag grant: start the executor held at point; true when the hold was reached
  rm -f "$R/hold"
  "${E[@]}" AIEN_FAULT_HOLD="$1" AIEN_FAULT_HOLD_FILE="$R/hold" "$BIN" compose execute --report "$REPORT" --workspace "$WS" --authorization "$3" \
    >"$ART/$2.json" 2>"$ART/$2.err" &
  EPID=$!
  local i=0
  while [ ! -s "$R/hold" ] && kill -0 "$EPID" 2>/dev/null && [ $i -lt 1200 ]; do sleep 0.05; i=$((i + 1)); done
  [ -s "$R/hold" ]
}
release_executor() { rm -f "$R/hold"; wait "$EPID" 2>/dev/null; ERC=$?; EPID=""; }
count_effects() { find "$WS" -name done.md -type f 2>/dev/null | wc -l; }
effect_file() { find "$WS" -name done.md -type f 2>/dev/null | head -1; }
ledger() { cli compose effects >"$ART/$1.json" 2>/dev/null; }
intent_state() { jq -r "[.ledger.intents[] | select(.authorization == $2)] | .[0].state // \"none\"" "$ART/$1.json" 2>/dev/null; }
intent_count() { jq -r "[.ledger.intents[] | select(.authorization == $2)] | length" "$ART/$1.json" 2>/dev/null; }

row_open() { # name [obj]: fresh row dir from the fixture, daemon 1 up, grant G minted
  RN=$1; R=$D/rows/$RN; ART=$OUT/artifacts/$RN; DN=0
  rm -rf "$R"; mkdir -p "$R"/{state,prov,logs} "$ART"
  cp -a "$D/fixture/compose" "$D/fixture/home" "$R/"; rm -rf "$D/work/ws"; cp -a "$D/fixture/ws" "$D/work/ws"
  cp -a "$D/fixture/compose.machine-root" "$R/"; [ -e "$D/fixture/compose.cortex-mark" ] && cp -a "$D/fixture/compose.cortex-mark" "$R/"
  cp "$D/fixture/propose.json" "$ART/propose.json"; REPORT=$ART/propose.json
  if [ "${2:-}" = obj ]; then   # v1.3: the report naming REC_ID, and the record in this row's provenance dir
    cp "$D/fixture/propose-obj.json" "$ART/propose.json"; mkdir -p "$R/prov/objectives"
    cp -a "$D/fixture/prov/objectives/$REC_ID.json" "$R/prov/objectives/" || return 2
  fi
  mkenv
  start_daemon || return 1
  G=$(authorize authorize-1); [ -n "$G" ] || { echo "authorize refused: $(head -c 200 "$ART/authorize-1.json")"; return 2; }
  return 0
}
row_close() { stop_graceful; cp "$R"/logs/daemon-*.log "$ART/" 2>/dev/null; }
fields() { # kill_point restarts eb ea dup grant extra-json
  jq -nc --arg kp "$1" --argjson rs "$2" --argjson eb "$3" --argjson ea "$4" --argjson dup "$5" --argjson g "${6:-null}" --argjson x "${7:-{\}}" \
    '{kill_point:$kp, restart_count:$rs, effects_before_kill:$eb, effects_after_restart:$ea, duplicates:$dup, grant_id:$g} + $x'
}
dupflag() { # stat_first stat_last count -> 0/1
  if [ "$3" -le 1 ] && [ "$1" = "$2" ]; then echo 0; else echo 1; fi
}

# ---- fixture: one real propose --------------------------------------------------------------------------------
echo "run folder: $OUT" | tee "$OUT/console.log"
R=$D/fixture; ART=$OUT/artifacts/fixture; rm -rf "$D/work"; mkdir -p "$R"/{state,prov,logs,home,compose} "$D/work/ws/inbox" "$D/work/ws/outbox" "$ART"; mkenv
printf 'Harbour notes. Three boats came in before noon.\n' >"$WS/inbox/harbour-notes.txt"
cli compose desk-key --create 1 >"$ART/deskkey.json" 2>&1   # prints id and path, never the key
ABORT=""
# v1.3 (FAULTS-v1.3.md): the objective is recorded before the propose, and the propose names it (sc#388 check).
cli objective record --text "$OBJ6" >"$ART/objective-record.json" 2>"$ART/objective-record.err"
if start_daemon; then
  PROP_OK=0
  for a in 1 2; do
    cli compose propose --goal "$OBJ6" --workspace "$WS" --context work --objective-id "$REC_ID" >"$ART/propose.json" 2>"$ART/propose.err"
    [ "$(jq -r '.report.committed // false' "$ART/propose.json" 2>/dev/null)" = true ] && { PROP_OK=1; break; }
  done
  stop_graceful; cp "$R"/logs/daemon-*.log "$ART/" 2>/dev/null
  if [ $PROP_OK = 1 ]; then
    # v1 to v1.2 rows: the same report without the top-level objective_id (the legacy path, as before).
    cp "$ART/propose.json" "$D/fixture/propose-obj.json"; jq -c 'del(.objective_id)' "$ART/propose.json" >"$D/fixture/propose.json"; cp -a "$WS" "$D/fixture/ws"
    PRE_ST=PASS; [ "$W_SHA" = "$C_W" ] && [ "$T_SHA" = "$C_T" ] || PRE_ST=FAIL
    rcpt PRE $PRE_ST "fault-hold build $BIN_SHA; backend '$CUR_BACKEND'; fixture proposal committed (path $(jq -r '.report.proposal_path // "?"' "$ART/propose.json")); model $C_ID" \
      "$(jq -nc --arg cw "$C_W" --arg ct "$C_T" --arg cid "$C_ID" --arg mdir "$MDIR" --arg q "$([ -s "$QUIET" ] && echo held || echo not-held)" --arg f "cargo feature aien-cli/fault-hold" --arg ok "$PRE_ST" \
        '{candidate_model_id:$cid, candidate_weights_sha256:$cw, candidate_tokenizer_sha256:$ct, model_dir:$mdir, quiet_flag:$q, build_feature:$f, weights_match_candidate:($ok=="PASS")}')"
  else
    rcpt PRE FAIL "fixture proposal did not commit: $(jq -r '.report.proposer_error // .error // "no proposal"' "$ART/propose.json" 2>/dev/null | cut -c1-200)" '{}'; ABORT=1
  fi
else
  rcpt PRE FAIL "fixture daemon did not start: $(tail -c 300 "$DLOG" 2>/dev/null | tr '\n' ' ')" '{}'; ABORT=1
fi

# ---- E6-before_intent ----------------------------------------------------------------------------------------
if [ -z "$ABORT" ]; then
  KP=before_intent
  if row_open E6-before_intent && hold_execute $KP held "$G"; then
    kill_daemon; DK=$KILL_RC; EB=$(count_effects); release_executor; XRC=$ERC
    XOK=$(jq -r 'if has("ok") then (.ok|tostring) else "none" end' "$ART/held.json" 2>/dev/null); AFTER_REL=$(count_effects)
    XERR=$(head -c 160 "$ART/held.err" 2>/dev/null | tr "\n" " "); XREF=no; { [ "$XOK" = false ] || { [ "${XRC:-0}" -ne 0 ] && [ -z "$XOK" -o "$XOK" = none ]; }; } && XREF=yes
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; IS1=$(intent_state L1 "$G")
      execute again "$G"; ST1=$T_STATE; F=$(effect_file); S1=$(tstat "${F:-/nonexistent}")
      execute third "$G"; THIRD=$T_REF; S2=$(tstat "${F:-/nonexistent}"); EA=$(count_effects); ledger L2
      DUP=$(dupflag "$S1" "$S2" "$EA")
      if [ "$DK" -eq 137 ] && [ "$XREF" = yes ] && [ "$EB" -eq 0 ] && [ "$AFTER_REL" -eq 0 ] && [ "$IS1" = none ] && [ "$ST1" = DONE ] && [ "$EA" -eq 1 ] && [ "$THIRD" = AlreadySpent ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-before_intent $ST "daemon SIGKILL rc $DK while the executor held at $KP; executor released: refused=$XREF (ok=$XOK, rc=$XRC, stderr: $XERR), files $AFTER_REL; restart ($RL); intent before retry: $IS1; same grant: $ST1; third execute: ${THIRD:-none}; files $EA; duplicates $DUP" \
        "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg ok "$XOK" --arg xerr "$XERR" --arg th "${THIRD:-}" --arg is "$IS1" --arg rl "$RL" '{executor_ok_after_release:$ok, executor_stderr:$xerr, intent_after_restart_before_retry:$is, third_execute_refusal:$th, restart_reconcile_line:$rl}')")"
    else rcpt E6-before_intent FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-before_intent FAIL "setup failed (daemon, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close
fi

# ---- E6-after_intent -----------------------------------------------------------------------------------------
if [ -z "$ABORT" ]; then
  KP=after_intent
  if row_open E6-after_intent && hold_execute $KP held "$G"; then
    kill_daemon; DK=$KILL_RC; kill_executor; EK=$EKILL_RC; EB=$(count_effects)
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; IS1=$(intent_state L1 "$G")
      execute r1 "$G"; R1_ST=$T_STATE; R1=$T_REF; E1=$(count_effects)
      OUTCOME=unknown; RECON=""; NEWGRANT=no
      if [ "$R1_ST" = DONE ]; then OUTCOME=DONE_without_refusal
      elif [ "$R1" = ReconciliationRequired ]; then
        cli compose reconcile >"$ART/reconcile.json" 2>&1; RECON=$(jq -r '.reconcile.outcomes[0].state // .error // "none"' "$ART/reconcile.json" 2>/dev/null | cut -c1-120)
        execute r2 "$G"; R2=$T_REF; E2=$(count_effects)
        if [ "$T_STATE" = DONE ] && [ "$E2" -eq 1 ]; then OUTCOME="a_ReconciliationRequired_reconcile_${RECON}_then_DONE"
        elif [ "$R2" = AlreadySpent ] && [ "$E2" -eq 1 ]; then OUTCOME="b_ReconciliationRequired_reconcile_${RECON}_then_AlreadySpent_file_once"
        elif [ "$R2" = AlreadySpent ] && [ "$E2" -eq 0 ]; then OUTCOME="c_ReconciliationRequired_reconcile_${RECON}_then_AlreadySpent_file_absent"; NEWGRANT=need
        fi
      elif [ "$R1" = AlreadySpent ]; then
        if [ "$E1" -eq 1 ]; then OUTCOME=b_AlreadySpent_file_once
        elif [ "$E1" -eq 0 ]; then OUTCOME=c_AlreadySpent_file_absent_reconcile_at_start; NEWGRANT=need; fi
      else OUTCOME="unexpected_refusal_${R1:-none}"; fi
      F=$(effect_file); S1=$(tstat "${F:-/nonexistent}")
      if [ $NEWGRANT = need ]; then
        G2=$(authorize authorize-2)
        if [ -n "$G2" ] && execute new "$G2"; then NEWGRANT=DONE; else NEWGRANT=failed; fi
        F=$(effect_file); S1=$(tstat "${F:-/nonexistent}"); execute newagain "${G2:-0}"
      fi
      EA=$(count_effects); S2=$(tstat "${F:-/nonexistent}"); ledger L2
      DUP=$(dupflag "$S1" "$S2" "$EA")
      case $OUTCOME in a_*|b_*|c_*) NAMED=1 ;; *) NAMED=0 ;; esac
      OKC=1; [ "$NEWGRANT" = failed ] && OKC=0; [ "$NEWGRANT" = need ] && OKC=0
      if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 0 ] && [ $NAMED = 1 ] && [ $OKC = 1 ] && [ "$EA" -eq 1 ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-after_intent $ST "daemon and executor SIGKILL (rc $DK, $EK) at $KP, files $EB; restart ($RL); intent state $IS1; same grant: state $R1_ST refusal '${R1:-none}'; outcome $OUTCOME; new grant: $NEWGRANT; files at end $EA; duplicates $DUP" \
        "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg o "$OUTCOME" --arg r1 "$R1" --arg is "$IS1" --arg rc "$RECON" --arg ng "$NEWGRANT" --arg rl "$RL" '{outcome:$o, first_retry_refusal:$r1, intent_after_restart:$is, reconcile_state:$rc, new_grant_result:$ng, restart_reconcile_line:$rl}')")"
    else rcpt E6-after_intent FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-after_intent FAIL "setup failed (daemon, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close
fi

# ---- E6-after_write and CTRL-E6 / CTRL-E6b (the controls reuse the after_write crash) ------------------------
crash_after_write() { # name [obj]: row_open, hold after_write, kill both; sets G, EB, F, SW, DK, EK
  row_open "$1" "${2:-}" || return 1
  hold_execute after_write held "$G" || return 1
  kill_daemon; DK=$KILL_RC; kill_executor; EK=$EKILL_RC
  EB=$(count_effects); F=$(effect_file); SW=$(tstat "${F:-/nonexistent}")
  return 0
}
if [ -z "$ABORT" ]; then
  KP=after_write
  if crash_after_write E6-after_write; then
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; IS1=$(intent_state L1 "$G")
      execute again "$G"; AG=$T_REF; ST1=$T_STATE; EA=$(count_effects); S2=$(tstat "$F"); ledger L2; IS2=$(intent_state L2 "$G"); NI=$(intent_count L2 "$G")
      DUP=$(dupflag "$SW" "$S2" "$EA")
      if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 1 ] && [ "$IS2" = DONE ] && [ "$NI" -eq 1 ] && [ "$ST1" != DONE ] && [ "$AG" = AlreadySpent ] && [ "$EA" -eq 1 ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-after_write $ST "daemon and executor SIGKILL (rc $DK, $EK) at $KP, files $EB; restart ($RL); intent after restart $IS1; same grant: ${AG:-none} (state $ST1); intent final $IS2 (count $NI); files $EA; duplicates $DUP" \
        "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg r "$AG" --arg is "$IS2" --argjson ni "$NI" --arg rl "$RL" '{same_grant_refusal:$r, intent_state_final:$is, intents_for_grant:$ni, restart_reconcile_line:$rl}')")"
    else rcpt E6-after_write FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-after_write FAIL "setup failed (daemon, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close
fi

ctrl_row() { # step rowname remove_mark(0/1)
  local step=$1 rn=$2 rm_mark=$3 KP=after_write
  if crash_after_write "$rn"; then
    mv "$R/compose" "$R/compose.removed"; mkdir -p "$R/compose"; cp -a "$R/compose.removed/approval-desk.key" "$R/compose/"   # v1.1: the desk key lives in the home and stays
    if [ "$rm_mark" = 1 ]; then mv "$R/compose.cortex-mark" "$R/compose.cortex-mark.removed" 2>/dev/null; fi
    START="served"; REFTEXT=""
    if start_daemon; then
      RL=$(reconcile_line)
      execute ctl "$G"; REFTEXT=$T_TEXT; REFN=$T_REF; CST=$T_STATE
      ledger L1
    else START="refused to start (rc ${DRC:-?})"; REFTEXT=$(tail -c 300 "$DLOG" | tr '\n' ' '); REFN=$(printf '%s' "$REFTEXT" | grep -o 'E_[A-Z_]*\|ReconciliationRequired\|ReconcileFailed' | head -1); CST=none; RL=""; fi
    EA=$(count_effects); S2=$(tstat "${F:-/nonexistent}"); DUP=$(dupflag "$SW" "$S2" "$EA")
    if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 1 ] && [ "$CST" != DONE ] && [ -n "$REFN" ] && [ "$EA" -eq 1 ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
    rcpt "$step" $ST "after $KP kill, durable state removed (journal, jspace and machine.id gone, desk key kept; record mark removed=$rm_mark); restart: $START ($RL); execute same grant: state $CST, refusal name '${REFN:-none}', text: $REFTEXT; files $EA; duplicates $DUP" \
      "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg s "$START" --arg n "${REFN:-}" --arg t "$REFTEXT" --argjson rm "$rm_mark" --arg cs "$CST" '{daemon_start:$s, refusal_name:$n, refusal_text:$t, record_mark_removed:($rm==1), execute_state:$cs}')")"
  else
    kill_executor; rcpt "$step" FAIL "setup failed (daemon, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close
}
if [ -z "$ABORT" ]; then
  ctrl_row CTRL-E6 CTRL-E6 0
  ctrl_row CTRL-E6b CTRL-E6b 1
fi

# ---- start-up holds: reconcile_error, reconcile_panic --------------------------------------------------------
if [ -z "$ABORT" ]; then
  KP=reconcile_error
  if row_open FAULT-reconcile_error; then
    stop_graceful   # the grant is in the durable home; restart with the hold
    if start_daemon AIEN_FAULT_HOLD=reconcile_error; then
      RL=$(reconcile_line); ledger L1; IS0=$(intent_state L1 "$G")
      execute e1 "$G"; R1=$T_REF; E1=$(count_effects); ledger L2; IS1=$(intent_state L2 "$G")
      cli compose reconcile >"$ART/reconcile.json" 2>&1; RC_OK=$(jq -r '.ok // false' "$ART/reconcile.json" 2>/dev/null)
      execute e2 "$G"; ST2=$T_STATE; EA=$(count_effects); F=$(effect_file); S1=$(tstat "${F:-/nonexistent}")
      execute e3 "$G"; R3=$T_REF; S2=$(tstat "${F:-/nonexistent}"); DUP=$(dupflag "$S1" "$S2" "$EA")
      case $RL in "Reconcile: refused: fault hold reconcile_error"*) LINE_OK=1 ;; *) LINE_OK=0 ;; esac
      if [ $LINE_OK = 1 ] && [ "$R1" = ReconcileFailed ] && [ "$E1" -eq 0 ] && [ "$IS1" = none ] && [ "$RC_OK" = true ] && [ "$ST2" = DONE ] && [ "$EA" -eq 1 ] && [ "$R3" = AlreadySpent ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt FAULT-reconcile_error $ST "daemon restarted with the $KP hold: '$RL'; execute refused '${R1:-none}', files $E1, intent $IS1; reconcile ok=$RC_OK; same grant then $ST2; files $EA; next execute '${R3:-none}'; duplicates $DUP" \
        "$(fields $KP 1 0 "$EA" "$DUP" "$G" "$(jq -nc --arg r1 "$R1" --arg rl "$RL" --arg r3 "$R3" '{first_execute_refusal:$r1, restart_reconcile_line:$rl, third_execute_refusal:$r3}')")"
    else rcpt FAULT-reconcile_error FAIL "daemon with the hold did not serve: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 0 0 0 "$G")"; fi
  else rcpt FAULT-reconcile_error FAIL "setup failed (daemon or authorize)" "$(fields $KP 0 0 0 0 null)"; fi
  row_close

  KP=reconcile_panic
  if row_open FAULT-reconcile_panic; then
    stop_graceful
    PANIC_UP=no
    if start_daemon AIEN_FAULT_HOLD=reconcile_panic; then PANIC_UP=yes; fi
    PRC=${DRC:-none}; LOG_NAMES=$(grep -c 'fault hold reconcile_panic' "$DLOG")
    if [ $PANIC_UP = yes ]; then kill_daemon; fi
    execute p1 "$G"; P1=$T_TEXT; E1=$(count_effects)
    rm -f "$SOCK"
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; IS1=$(intent_state L1 "$G")
      execute p2 "$G"; ST2=$T_STATE; EA=$(count_effects); F=$(effect_file); S1=$(tstat "${F:-/nonexistent}")
      execute p3 "$G"; R3=$T_REF; S2=$(tstat "${F:-/nonexistent}"); DUP=$(dupflag "$S1" "$S2" "$EA")
      if [ $PANIC_UP = no ] && [ "$PRC" != none ] && [ "$PRC" != 0 ] && [ "$LOG_NAMES" -ge 1 ] && [ -n "$P1" ] && [ "$E1" -eq 0 ] && [ "$IS1" = none ] && [ "$ST2" = DONE ] && [ "$EA" -eq 1 ] && [ "$R3" = AlreadySpent ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt FAULT-reconcile_panic $ST "daemon with the $KP hold: served=$PANIC_UP, exit rc $PRC, log names the hold ($LOG_NAMES lines); execute while nothing serves: '$P1', files $E1; clean restart ($RL); intent $IS1; same grant then $ST2; files $EA; next execute '${R3:-none}'; duplicates $DUP" \
        "$(fields $KP 1 0 "$EA" "$DUP" "$G" "$(jq -nc --arg up "$PANIC_UP" --arg rc "$PRC" --arg p1 "$P1" --arg r3 "$R3" '{daemon_served_with_hold:$up, daemon_exit_rc:$rc, execute_while_down:$p1, third_execute_refusal:$r3}')")"
    else rcpt FAULT-reconcile_panic FAIL "clean restart failed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 0 0 0 "$G")"; fi
  else rcpt FAULT-reconcile_panic FAIL "setup failed (daemon or authorize)" "$(fields $KP 0 0 0 0 null)"; fi
  row_close
fi

# ---- v1.2 (FAULTS-v1.2.md): human controls across a SIGKILL restart: revoke, stop, desk ---------------------
if [ -z "$ABORT" ]; then
  KP=revoke_then_kill
  if row_open E6-revoke_durable; then
    cli compose revoke --authorization "$G" --approver drake >"$ART/revoke.json" 2>&1; REV_OK=$(jq -r '.control.ok // .ok // "none"' "$ART/revoke.json" 2>/dev/null)
    kill_daemon; DK=$KILL_RC
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; IS1=$(intent_state L1 "$G")
      REVLISTED=$(jq -r --argjson g "$G" '((.ledger.revoked // []) | tostring) | test("(^|[^0-9])" + ($g|tostring) + "([^0-9]|$)")' "$ART/L1.json" 2>/dev/null)
      execute r1 "$G"; R1=$T_REF; EA=$(count_effects)
      if [ "$DK" -eq 137 ] && [ "$R1" = Revoked ] && [ "$EA" -eq 0 ] && [ "$IS1" = none ] && [ "$REVLISTED" = true ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-revoke_durable $ST "revoke ok=$REV_OK, daemon SIGKILL (rc $DK), restart ($RL); ledger lists the revocation=$REVLISTED, intent $IS1; execute refused '${R1:-none}', files $EA" \
        "$(fields $KP 1 0 "$EA" 0 "$G" "$(jq -nc --arg r "$R1" --arg l "$REVLISTED" --arg ok "$REV_OK" '{refusal:$r, ledger_lists_revocation:$l, revoke_ok:$ok}')")"
    else rcpt E6-revoke_durable FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 0 0 0 "$G")"; fi
  else rcpt E6-revoke_durable FAIL "setup failed (daemon or authorize)" "$(fields $KP 0 0 0 0 null)"; fi
  row_close

  KP=stop_then_kill
  if row_open E6-stop_durable; then
    cli compose stop --approver drake >"$ART/stop.json" 2>&1; STOP_OK=$(jq -r '.control.ok // .ok // "none"' "$ART/stop.json" 2>/dev/null)
    kill_daemon; DK=$KILL_RC
    if start_daemon; then
      RL=$(reconcile_line); ledger L1; STOPPED=$(jq -c '.ledger.stopped' "$ART/L1.json" 2>/dev/null)
      execute s1 "$G"; R1=$T_REF; E1=$(count_effects)
      cli compose resume --approver drake >"$ART/resume.json" 2>&1; RES_OK=$(jq -r '.control.ok // .ok // "none"' "$ART/resume.json" 2>/dev/null)
      execute s2 "$G"; R2=$T_REF; EA=$(count_effects)
      # v1.2.1: a pre-stop grant never revives (Stale); no claim about a fresh grant after resume
      if [ "$DK" -eq 137 ] && [ "$STOPPED" != null ] && [ -n "$STOPPED" ] && [ "$R1" = Stopped ] && [ "$E1" -eq 0 ] && [ "$R2" = Stale ] && [ "$EA" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-stop_durable $ST "stop ok=$STOP_OK, daemon SIGKILL (rc $DK), restart ($RL); ledger stopped=$STOPPED; execute refused '${R1:-none}', files $E1; resume ok=$RES_OK; same grant after resume: '${R2:-none}', files $EA" \
        "$(fields $KP 1 "$E1" "$EA" 0 "$G" "$(jq -nc --arg r "$R1" --arg r2 "$R2" --arg s "$STOPPED" '{stopped_refusal:$r, after_resume_refusal:$r2, ledger_stopped:$s}')")"
    else rcpt E6-stop_durable FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 0 0 0 "$G")"; fi
  else rcpt E6-stop_durable FAIL "setup failed (daemon or authorize)" "$(fields $KP 0 0 0 0 null)"; fi
  row_close

  KP=desk_after_kill
  if row_open E6-desk_after_restart; then
    kill_daemon; DK=$KILL_RC
    if start_daemon; then
      RL=$(reconcile_line)
      cli compose authorize --report "$REPORT" --workspace "$WS" --approver drake >"$ART/authorize-nodesk.json" 2>"$ART/authorize-nodesk.err"
      ND_OK=$(jq -r '.ok // false' "$ART/authorize-nodesk.json" 2>/dev/null); ND_TXT=$( { jq -r '.error // ""' "$ART/authorize-nodesk.json" 2>/dev/null; head -c 200 "$ART/authorize-nodesk.err"; } | tr '\n' ' ' | cut -c1-200)
      ND_REF=$(printf '%s' "$ND_TXT" | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\).*/\1/p')
      G2=$(authorize authorize-desk); D_TXT=$(jq -r '.error // ""' "$ART/authorize-desk.json" 2>/dev/null | cut -c1-200)
      D_REF=$(printf '%s' "$D_TXT" | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\).*/\1/p')
      execute d1 "$G"; ST1=$T_STATE; EA=$(count_effects); F=$(effect_file); S1=$(tstat "${F:-/nonexistent}")
      execute d2 "$G"; R3=$T_REF; S2=$(tstat "${F:-/nonexistent}"); DUP=$(dupflag "$S1" "$S2" "$EA")
      # v1.2.1: without desk proof -> DeskMacRequired; with it -> AlreadyAuthorized (the live grant survived the kill); the grant finishes once
      if [ "$DK" -eq 137 ] && [ "$ND_REF" = DeskMacRequired ] && [ "$D_REF" = AlreadyAuthorized ] && [ "$ST1" = DONE ] && [ "$EA" -eq 1 ] && [ "$R3" = AlreadySpent ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-desk_after_restart $ST "daemon SIGKILL (rc $DK), restart ($RL); authorize without desk proof: '${ND_REF:-none}'; with the desk proof: '${D_REF:-none}' (${D_TXT}); first grant then $ST1; files $EA; next '${R3:-none}'; duplicates $DUP" \
        "$(fields $KP 1 0 "$EA" "$DUP" "$G" "$(jq -nc --arg a "$ND_REF" --arg b "$D_REF" --arg t "$D_TXT" '{authorize_without_desk_refusal:$a, authorize_with_desk_refusal:$b, authorize_with_desk_text:$t}')")"
    else rcpt E6-desk_after_restart FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 0 0 0 "$G")"; fi
  else rcpt E6-desk_after_restart FAIL "setup failed (daemon or authorize)" "$(fields $KP 0 0 0 0 null)"; fi
  row_close
fi

# ---- v1.3 (FAULTS-v1.3.md, sc#408): objective identity across a SIGKILL restart ---------------------------------
recon_row() { # grant -> the restart reconcile outcome (JSON) for that grant's intent, or {}
  local o; o=$(grep -m1 '^Reconcile:' "$DLOG" | sed 's/^[^[]*//' | jq -c --argjson g "$1" '[.[] | select(.authorization == $g)][0] // {}' 2>/dev/null)
  if [ -n "$o" ]; then echo "$o"; else echo '{}'; fi
}
intent_obj() { jq -r "[.ledger.intents[] | select(.authorization == $2)] | .[0].objective_id // \"none\"" "$ART/$1.json" 2>/dev/null; }
grant_obj() { # tag grant: the objective_id in the daemon's own grant text (compose recall), or none
  cli compose recall --ids "$2" >"$ART/$1.json" 2>"$ART/$1.err"
  local o; o=$(jq -r --argjson g "$2" '[(.recall.cited // [])[], (.authorizations // [])[]] | map(select(.id == $g and .text != null))[0].text | fromjson | .objective_id // "none"' "$ART/$1.json" 2>/dev/null)
  if [ -n "$o" ]; then echo "$o"; else echo none; fi
}
refusal_of() { jq -r '.error // ""' "$1" 2>/dev/null | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\):.*/\1/p'; }
if [ -z "$ABORT" ]; then
  KP=after_intent
  if row_open E6-objective_bound_after_intent obj && hold_execute $KP held "$G"; then
    RECF=$R/prov/objectives/$REC_ID.json; RS0=$(sha_of "$RECF")
    kill_daemon; DK=$KILL_RC; kill_executor; EK=$EKILL_RC; EB=$(count_effects)
    if start_daemon; then
      RO=$(recon_row "$G"); RO_ID=$(jq -r '.objective.objective_id // "none"' <<<"$RO"); RO_REC=$(jq -r '.objective.record // "none"' <<<"$RO"); RO_ST=$(jq -r '.state // "none"' <<<"$RO")
      ledger L1; LOID=$(intent_obj L1 "$G"); GOID=$(grant_obj recall-G "$G")
      RS1=$(sha_of "$RECF"); cli objective show "$REC_ID" >"$ART/objective-show.json" 2>&1; SHOW_SHA=$(jq -r '.objective_text_sha256 // "none"' "$ART/objective-show.json" 2>/dev/null)
      execute r1 "$G"; R1=$T_REF
      G2=$(authorize authorize-2); G2OID=none; NST=none; XOID=none; RCPT_OK=no
      if [ -n "$G2" ]; then
        G2OID=$(grant_obj recall-G2 "$G2")
        execute new "$G2"; NST=$T_STATE; XOID=$(jq -r '.objective_id // "none"' "$ART/new.json" 2>/dev/null)
        RP=$(jq -r '.receipt.path // ""' "$ART/new.json" 2>/dev/null); [ -f "$RP" ] && cp "$RP" "$ART/new-write_file-receipt.json"
        # the receipt holds only digests: recompute the arguments digest with the objective id and compare
        WANT=$(jq -cS --argjson a "$G2" '{path, content_sha256, authorization: $a, intent: .intent.id, objective_id}' "$ART/new.json" 2>/dev/null)
        [ -f "$ART/new-write_file-receipt.json" ] && [ "sha256:$(printf '%s' "$WANT" | sha256sum | cut -d' ' -f1)" = "$(jq -r '.arguments_digest' "$ART/new-write_file-receipt.json")" ] && RCPT_OK=yes
      fi
      F=$(effect_file); S1=$(tstat "${F:-/nonexistent}"); execute newagain "${G2:-0}"; EA=$(count_effects); S2=$(tstat "${F:-/nonexistent}"); ledger L2
      L2OID=$(intent_obj L2 "${G2:-0}"); DUP=$(dupflag "$S1" "$S2" "$EA")
      if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 0 ] && [ "$RO_ID" = "$REC_ID" ] && [ "$RO_REC" = bound ] && [ "$LOID" = "$REC_ID" ] && [ "$GOID" = "$REC_ID" ] \
         && [ "$RS0" = "$RS1" ] && [ "$SHOW_SHA" = "$OBJ6_SHA" ] && [ "$R1" = AlreadySpent ] && [ -n "$G2" ] && [ "$G2OID" = "$REC_ID" ] && [ "$NST" = DONE ] \
         && [ "$XOID" = "$REC_ID" ] && [ "$RCPT_OK" = yes ] && [ "$L2OID" = "$REC_ID" ] && [ "$EA" -eq 1 ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-objective_bound_after_intent $ST "daemon and executor SIGKILL (rc $DK, $EK) at $KP, files $EB; restart reconcile: intent state $RO_ST, objective $RO_ID record '$RO_REC'; ledger intent objective $LOID; grant G objective $GOID; record sha256 before/after equal=$([ "$RS0" = "$RS1" ] && echo yes || echo no), show text digest matches=$([ "$SHOW_SHA" = "$OBJ6_SHA" ] && echo yes || echo no); same grant '${R1:-none}'; new grant ${G2:-none} objective $G2OID: $NST, answer objective $XOID, receipt digest binds it=$RCPT_OK; files $EA; duplicates $DUP" \
        "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg rec "$REC_ID" --argjson ro "$RO" --arg l "$LOID" --arg go "$GOID" --arg rs0 "$RS0" --arg rs1 "$RS1" --arg r1 "$R1" --arg g2 "${G2:-}" --arg g2o "$G2OID" --arg n "$NST" --arg x "$XOID" --arg rc "$RCPT_OK" \
          '{objective_record_id:$rec, restart_reconcile_outcome:$ro, ledger_intent_objective:$l, grant_objective:$go, record_sha256_before_kill:$rs0, record_sha256_after_restart:$rs1, same_grant_refusal:$r1, new_grant:$g2, new_grant_objective:$g2o, new_grant_state:$n, execute_answer_objective:$x, receipt_arguments_digest_binds_objective:($rc=="yes")}')")"
    else rcpt E6-objective_bound_after_intent FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-objective_bound_after_intent FAIL "setup failed (daemon, objective report, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close

  KP=after_intent
  if row_open E6-objective_missing_after_intent obj && hold_execute $KP held "$G"; then
    RECF=$R/prov/objectives/$REC_ID.json
    kill_daemon; DK=$KILL_RC; kill_executor; EK=$EKILL_RC; EB=$(count_effects)
    mv "$RECF" "$R/objective-record.removed.json"
    if start_daemon; then
      RO=$(recon_row "$G"); RO_ID=$(jq -r '.objective.objective_id // "none"' <<<"$RO"); RO_REC=$(jq -r '.objective.record // "none"' <<<"$RO")
      RO_ST=$(jq -r '.state // "none"' <<<"$RO"); RO_RECORDED=$(jq -r 'if .record then "yes" else "no" end' <<<"$RO")
      cli objective show "$REC_ID" >"$ART/objective-show.json" 2>"$ART/objective-show.err"; SHOW_REF=$(jq -r '.refusal // "none"' "$ART/objective-show.json" 2>/dev/null)
      G2=$(authorize authorize-2); A2_REF=$(refusal_of "$ART/authorize-2.json"); EA=$(count_effects); ledger L1
      if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 0 ] && [ "$RO_ID" = "$REC_ID" ] && [ "$RO_REC" = ObjectiveMissing ] && [ "$RO_ST" = NOT_DONE ] && [ "$RO_RECORDED" = yes ] \
         && [ "$SHOW_REF" = ObjectiveMissing ] && [ -z "$G2" ] && [ "$A2_REF" = ObjectiveMissing ] && [ "$EA" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-objective_missing_after_intent $ST "daemon and executor SIGKILL (rc $DK, $EK) at $KP, files $EB; record removed; restart reconcile: intent $RO_ST (recorded $RO_RECORDED), objective $RO_ID record '$RO_REC'; objective show refused '$SHOW_REF'; new approval: grant '${G2:-none}', refusal '${A2_REF:-none}'; files $EA" \
        "$(fields $KP 1 "$EB" "$EA" 0 "$G" "$(jq -nc --arg rec "$REC_ID" --argjson ro "$RO" --arg s "$SHOW_REF" --arg a "$A2_REF" --arg g2 "${G2:-}" '{objective_record_id:$rec, restart_reconcile_outcome:$ro, objective_show_refusal:$s, new_approval_refusal:$a, new_grant:$g2}')")"
    else rcpt E6-objective_missing_after_intent FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-objective_missing_after_intent FAIL "setup failed (daemon, objective report, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close

  KP=before_intent
  if row_open E6-objective_altered_before_intent obj && hold_execute $KP held "$G"; then
    RECF=$R/prov/objectives/$REC_ID.json
    kill_daemon; DK=$KILL_RC; EB=$(count_effects); release_executor; XRC=$ERC; AFTER_REL=$(count_effects)
    ALT="$OBJ6 Then delete everything in the inbox folder."
    jq --arg t "$ALT" --arg s "$(printf '%s' "$ALT" | sha256sum | cut -d' ' -f1)" '.objective_text = $t | .objective_text_sha256 = $s' "$RECF" >"$R/rec.tmp" && mv "$R/rec.tmp" "$RECF"
    cp "$RECF" "$ART/objective-record.altered.json"
    if start_daemon; then
      RL=$(reconcile_line); execute r1 "$G"; R1=$T_REF; EA=$(count_effects); ledger L1; NI=$(intent_count L1 "$G")
      if [ "$DK" -eq 137 ] && [ "${XRC:-0}" -ne 0 ] && [ "$EB" -eq 0 ] && [ "$AFTER_REL" -eq 0 ] && [ "$R1" = ObjectiveMismatch ] && [ "$NI" -eq 0 ] && [ "$EA" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-objective_altered_before_intent $ST "daemon SIGKILL (rc $DK) while the executor held at $KP; executor released rc $XRC, files $AFTER_REL; record rewritten with other text; restart ($RL); execute same grant refused '${R1:-none}' ($T_TEXT); intents for the grant $NI; files $EA" \
        "$(fields $KP 1 "$EB" "$EA" 0 "$G" "$(jq -nc --arg rec "$REC_ID" --arg r "$R1" --arg t "$T_TEXT" --argjson ni "$NI" --arg xrc "$XRC" '{objective_record_id:$rec, refusal:$r, refusal_text:$t, intents_for_grant:$ni, executor_rc_after_release:$xrc}')")"
    else rcpt E6-objective_altered_before_intent FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-objective_altered_before_intent FAIL "setup failed (daemon, objective report, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close

  KP=after_write
  if crash_after_write E6-objective_after_write obj; then
    RECF=$R/prov/objectives/$REC_ID.json
    if start_daemon; then
      RO=$(recon_row "$G"); RO_ID=$(jq -r '.objective.objective_id // "none"' <<<"$RO"); RO_REC=$(jq -r '.objective.record // "none"' <<<"$RO"); RO_ST=$(jq -r '.state // "none"' <<<"$RO")
      ledger L1; LOID=$(intent_obj L1 "$G"); IS1=$(intent_state L1 "$G")
      RTXT=$(jq -r '.objective_text' "$RECF"); RT_SHA=$(printf '%s' "$RTXT" | sha256sum | cut -d' ' -f1); RID2=$(printf 'AIEN_E2E_OBJECTIVE_V1\n%s' "$RTXT" | sha256sum | cut -d' ' -f1)
      execute again "$G"; AG=$T_REF; EA=$(count_effects); S2=$(tstat "$F"); DUP=$(dupflag "$SW" "$S2" "$EA")
      if [ "$DK" -eq 137 ] && [ "$EK" -eq 137 ] && [ "$EB" -eq 1 ] && [ "$RO_ST" = DONE ] && [ "$RO_ID" = "$REC_ID" ] && [ "$RO_REC" = bound ] && [ "$IS1" = DONE ] && [ "$LOID" = "$REC_ID" ] \
         && [ "$RT_SHA" = "$OBJ6_SHA" ] && [ "$RID2" = "$REC_ID" ] && [ "$AG" = AlreadySpent ] && [ "$EA" -eq 1 ] && [ "$DUP" -eq 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6-objective_after_write $ST "daemon and executor SIGKILL (rc $DK, $EK) at $KP, files $EB; restart reconcile: intent $RO_ST, objective $RO_ID record '$RO_REC'; ledger intent $IS1 objective $LOID; record text digest matches the objective=$([ "$RT_SHA" = "$OBJ6_SHA" ] && echo yes || echo no), id recomputes=$([ "$RID2" = "$REC_ID" ] && echo yes || echo no); same grant '${AG:-none}'; files $EA; duplicates $DUP" \
        "$(fields $KP 1 "$EB" "$EA" "$DUP" "$G" "$(jq -nc --arg rec "$REC_ID" --argjson ro "$RO" --arg l "$LOID" --arg is "$IS1" --arg a "$AG" '{objective_record_id:$rec, restart_reconcile_outcome:$ro, ledger_intent_objective:$l, ledger_intent_state:$is, same_grant_refusal:$a}')")"
    else rcpt E6-objective_after_write FAIL "daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" "$(fields $KP 1 "$EB" 0 0 "$G")"; fi
  else
    kill_executor; rcpt E6-objective_after_write FAIL "setup failed (daemon, objective report, authorize or hold not reached)" "$(fields $KP 0 0 0 0 null)"
  fi
  row_close
fi

# ---- manifest ------------------------------------------------------------------------------------------------
printf 'objective_id %s\n%s' "$OBJ_ID" "$ROWS" >"$OUT/VERDICT_ROWS.txt"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >chain/SHA256SUMS )
echo "run folder written: $OUT ($(ls "$CHAIN" | wc -l) chain files)"; cat "$OUT/VERDICT_ROWS.txt"
