#!/usr/bin/env bash
# WHOLE-SYSTEM-E2E harness v1 (aien-architecture#190, lane L1).
# Rows: docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md (committed before this script; never edited by it).
# Contract: aien-architecture docs/plans/release-readiness/ACCEPTANCE_E2E.md (PROPOSED, commit 0631cf39).
#
#   scripts/whole_system_e2e.sh run        build the CLI, run the chain E2..E6 + controls, write one run folder
#   scripts/whole_system_e2e.sh check F    the document checks of rows E3/E3m on file F (prints failed ids)
#   scripts/whole_system_e2e.sh selftest   checks with positive and negative controls (no daemon)
#
# Shell plus jq. No Python, no systemd, no accelerator: the daemon must print the CPU-reference backend
# line or the run stops. Needs CLAUDE_JOB_DIR (build target and run state under $CLAUDE_JOB_DIR/tmp/e2e),
# the prebuilt compose archive at $CLAUDE_JOB_DIR/tmp/librx_compose.a, jq, sha256sum, ss, hostname, iconv.
# Model: WSE2E_MODEL_DIR (default: the CAND-4 Llama 3.2 1B snapshot in the local Hugging Face cache);
# its weights and tokenizer sha256 are compared with release/candidate.toml and written into every receipt.
set -u

# ---- document checks (rows E3, E3m) -----------------------------------------------------------------------
# Prints failed ids, space separated: D1 missing/unreadable, D2 first line not "# ", D3 over 200 words,
# D4 a required name missing. $1 file, $2.. required names.
check_doc() {
  local f=$1; shift; local failed=""
  [ -s "$f" ] || { echo "D1"; return; }
  iconv -f UTF-8 -t UTF-8 "$f" >/dev/null 2>&1 || failed="$failed D1"
  [ "$(head -c 2 "$f")" = "# " ] || failed="$failed D2"
  [ "$(wc -w <"$f")" -le 200 ] || failed="$failed D3"
  local n; for n in "$@"; do grep -qF -- "$n" "$f" || { failed="$failed D4"; break; }; done
  echo "${failed# }"
}
selftest() {
  local t; t=$(mktemp -d); local rc=0
  printf '# Report\n\nabout alpha.txt, beta.txt and gamma.txt\n' >"$t/good.md"
  printf 'Report\n\nabout alpha.txt only\n' >"$t/bad.md"
  [ -z "$(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)" ] || { echo "selftest: good document failed: $(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)"; rc=1; }
  [ "$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)" = "D2 D4" ] || { echo "selftest: bad document gave '$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)', want 'D2 D4'"; rc=1; }
  [ "$(check_doc "$t/absent.md")" = "D1" ] || { echo "selftest: absent file not D1"; rc=1; }
  rm -rf "$t"; [ $rc -eq 0 ] && echo "selftest PASS"; return $rc
}
case ${1:-} in
  check) shift; check_doc "$@"; exit 0 ;;
  selftest) selftest; exit $? ;;
  run) ;;
  *) echo "usage: $0 run | check FILE [NAME..] | selftest" >&2; exit 2 ;;
esac

# ---- setup -----------------------------------------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOB=${CLAUDE_JOB_DIR:?CLAUDE_JOB_DIR must be set}
LIB=$JOB/tmp/librx_compose.a
D=$JOB/tmp/e2e
BIN=$D/target/release/aien-cli
MDIR=${WSE2E_MODEL_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/ 2>/dev/null | head -1)}
UTC0=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${WSE2E_OUT:-$REPO/docs/campaigns/whole-system-e2e/runs/RUN-1-dry-$UTC0}
for t in jq sha256sum ss hostname iconv; do command -v $t >/dev/null || { echo "need $t" >&2; exit 2; }; done
[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 2; }
[ -n "$MDIR" ] && [ -f "$MDIR/model.safetensors" ] && [ -f "$MDIR/tokenizer.json" ] || { echo "model dir '$MDIR' lacks model.safetensors or tokenizer.json" >&2; exit 2; }
QUIET=$HOME/workspace/.spark-quiet
while [ -s "$QUIET" ]; do echo "quiet flag set; waiting"; sleep 30; done
DIRTY=$(git -C "$REPO" status --porcelain -- . ":!docs/campaigns/whole-system-e2e/runs" | wc -l)
[ "$DIRTY" -eq 0 ] || [ -n "${WSE2E_ALLOW_DIRTY:-}" ] || { echo "working tree has $DIRTY uncommitted changes; commit them first" >&2; exit 2; }
COMMIT=$(git -C "$REPO" rev-parse HEAD)
mkdir -p "$D"
( cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_FORCE_CPU_STUB \
    AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" cargo build --release -p aien-cli ) >"$D/build.log" 2>&1 \
  || { echo "build failed, see $D/build.log" >&2; exit 2; }
grep -q "building the stub" "$D/build.log" && { echo "stub compose build; refusing" >&2; exit 2; }
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
W_SHA=$(sha256sum "$(readlink -f "$MDIR/model.safetensors")" | cut -d' ' -f1)
T_SHA=$(sha256sum "$(readlink -f "$MDIR/tokenizer.json")" | cut -d' ' -f1)
C_W=$(sed -n 's/^model-safetensors-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_T=$(sed -n 's/^tokenizer-json-sha256 = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
C_ID=$(sed -n 's/^model-id = "\(.*\)"/\1/p' "$REPO/release/candidate.toml")
HOST=$(hostname)

RUN=$D/run; [ -e "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
mkdir -p "$RUN"/{state,compose,prov,home,logs,ws-contract/inbox,ws-contract/outbox,ws/inbox,ws/outbox} "$OUT/chain" "$OUT/artifacts"
SOCK=$RUN/s; LOGS=$RUN/logs; ART=$OUT/artifacts; WS=$RUN/ws; WSC=$RUN/ws-contract
E=(env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_REQUIRE_BLACKWELL
   -u AIEN_GPU_BACKEND -u AIEN_MODEL_DIR -u AIEN_MODEL_PATH -u AIEN_TOKENIZER_PATH -u AIEN_FORCE_CPU_STUB
   -u AIEN_FAULT_HOLD -u AIEN_FAULT_HOLD_FILE -u AIEN_ALLEN_SUBJECT -u AIEN_ALLEN_ADOPT
   -u AIEN_COMPOSE_MAX_TOKENS -u AIEN_COMPOSE_DOC_MAX_TOKENS
   HOME="$RUN/home" NO_COLOR=1 AIEN_RUNTIME_SOCK="$SOCK" AIEN_RUNTIME_STATE_DIR="$RUN/state"
   AIEN_COMPOSE_DIR="$RUN/compose" AIEN_PROVENANCE_DIR="$RUN/prov" AIEN_REQUIRE_CHECKPOINT=1
   AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=1 AIEN_COMPOSE_EDIT_BUDGET_MS=599000 AIEN_COMPOSE_DOC_BUDGET_MS=599000)
cli() { "${E[@]}" "$BIN" "$@"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
tstat() { stat -c '%i %s %Y' "$1" 2>/dev/null || echo absent; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---- the fixture and the identities ------------------------------------------------------------------------
F1=harbour-notes.txt; F2=orchard-ledger.txt; F3=windmill-log.txt
mkfix() { # dir
  printf 'Harbour notes. Three boats came in before noon; the tide table was wrong by twenty minutes.\n' >"$1/inbox/$F1"
  printf 'Orchard ledger. Forty apple trees, twelve pear trees, two rows of quince along the south wall.\n' >"$1/inbox/$F2"
  printf 'Windmill log. The brake band was replaced; the sails turned eleven hours on Tuesday.\n' >"$1/inbox/$F3"
}
mkfix "$WSC"; mkfix "$WS"
# v1.1 (ACCEPTANCE-v1.1.md): the contract objective in the wording of aien-architecture#196 (the runtime refused
# "under 200 words" in dry run 1); the mechanics objective names topics, not files (the destination classifier
# took every file name as a write target in dry run 1); E4 recalls the first topic word.
OBJ='Read the three text files in the inbox folder and write a Markdown report named report.md in the outbox folder. Start with a Markdown heading line that begins with "# ". Use at most 200 words. Mention each file by name.'
T1=harbour; T2=orchard; T3=windmill
OBJ_M="Write a Markdown report named report.md about the $T1, the $T2 and the $T3. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention the $T1, the $T2 and the $T3."
OBJ2='Append one line to report.md naming the topic you reported on first.'
OBJ6='Write a file named done.md in the outbox folder containing one line: finished.'
OBJC='Write a file named ctrl.md in the outbox folder containing one line: control.'
OBJ_ID=$(printf 'AIEN_E2E_OBJECTIVE_V1\n%s' "$OBJ" | sha256sum | cut -d' ' -f1)
RUN_ID=$(printf '%s\n%s\n%s\n%s' "$OBJ_ID" "$BIN_SHA" "$W_SHA" "$HOST" | sha256sum | cut -d' ' -f1)

# ---- the receipt chain -------------------------------------------------------------------------------------
N=0; PREV=$(printf '0%.0s' $(seq 1 64)); CHAIN=$OUT/chain; ROWS=""
rcpt() { # step status evidence [fields json]
  N=$((N + 1)); local f; f=$(printf '%s/%03d-%s.json' "$CHAIN" "$N" "$1")
  jq -n --arg oid "$OBJ_ID" --arg rid "$RUN_ID" --arg step "$1" --arg status "$2" --arg ev "$3" --argjson fields "${4:-{\}}" \
    --arg utc "$(utc)" --arg prev "$PREV" --arg commit "$COMMIT" --arg bin "$BIN_SHA" --arg w "$W_SHA" --arg t "$T_SHA" \
    --arg backend "${CUR_BACKEND:-no daemon started}" --arg host "$HOST" \
    '{objective_id:$oid, run_id:$rid, step:$step, status:$status, utc:$utc, prev_receipt_sha256:$prev, evidence:$ev,
      repo_commit:$commit, daemon_sha256:$bin, cli_sha256:$bin, model_weights_digest:$w, tokenizer_sha256:$t,
      backend_line:$backend, host:$host} + $fields' >"$f"
  PREV=$(sha_of "$f")
  case $1 in CTRL-*|E[0-9]*) ROWS="$ROWS$1 $2"$'\n' ;; esac
  echo "[$(date -u +%H:%M:%S)] $1 $2: $3" | cut -c1-300 | tee -a "$OUT/console.log"
}

# ---- daemon control ----------------------------------------------------------------------------------------
DN=0; DPID=""; CUR_BACKEND=""; RESTARTS=0; LAST_START_UTC=""; DLOG=/dev/null
start_daemon() { # [VAR=val ...]
  DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN.log; rm -f "$SOCK"
  "${E[@]}" AIEN_MODEL_PATH="$MDIR/model.safetensors" AIEN_TOKENIZER_PATH="$MDIR/tokenizer.json" "$@" "$BIN" daemon >"$DLOG" 2>&1 &
  DPID=$!
  local i
  for i in $(seq 1 600); do
    sleep 2
    [ -d "/proc/$DPID" ] || { DPID=""; echo "daemon $DN exited early: $(tail -c 300 "$DLOG" | tr '\n' ' ')"; return 1; }
    if [ -S "$SOCK" ] && grep -q -E "Replay reconcile:|^Reconcile:" "$DLOG"; then break; fi
  done
  [ -S "$SOCK" ] || { echo "daemon $DN not serving after 20 min"; return 1; }
  LAST_START_UTC=$(utc)
  CUR_BACKEND=$(grep -m1 '^  Backend:' "$DLOG" | sed 's/^ *//')
  case $CUR_BACKEND in *CPU-reference*) ;; *) echo "not the CPU reference backend: '$CUR_BACKEND'"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1 ;; esac
  return 0
}
stop_graceful() { [ -n "$DPID" ] || return 0; cli compose shutdown >"$LOGS/shutdown-$DN.json" 2>&1; local i=0; while [ -d "/proc/$DPID" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i + 1)); done; wait "$DPID" 2>/dev/null; DPID=""; rm -f "$SOCK"; }
KILL_RC=""
sigkill() { [ -n "$DPID" ] || return 1; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; KILL_RC=$?; DPID=""; rm -f "$SOCK"; [ "$KILL_RC" -eq 137 ]; }
trap '[ -n "$DPID" ] && kill -9 "$DPID" 2>/dev/null' EXIT

# One task: propose -> authorize (desk) -> execute. $1 tag, $2 workspace, $3 goal, [$4 context] [$5 noexec]
# Sets T_STATE, T_GRANT, T_PATH, T_CSHA, T_MEMCTX, T_MEMN, T_EV, T_RECEIPT (the compose effect receipt json or null)
T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""
task() {
  local tag=$1 ws=$2 goal=$3 ctx=${4:-work}
  T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""
  cli compose propose --goal "$goal" --workspace "$ws" --context "$ctx" >"$ART/$tag-propose.json" 2>"$ART/$tag-propose.err"
  if [ "$(jq -r '.report.committed // false' "$ART/$tag-propose.json" 2>/dev/null)" != true ]; then
    T_EV="propose did not commit: $(jq -r '.report.proposer_error // .error // "no proposal"' "$ART/$tag-propose.json" 2>/dev/null | cut -c1-200)"; return 1
  fi
  T_MEMCTX=$(jq -r '.report.memory.context' "$ART/$tag-propose.json"); T_MEMN=$(jq -r '.report.memory.items_included // 0' "$ART/$tag-propose.json")
  cli compose authorize --report "$ART/$tag-propose.json" --workspace "$ws" --approver drake --desk 1 >"$ART/$tag-authorize.json" 2>"$ART/$tag-authorize.err"
  [ "$(jq -r '.ok' "$ART/$tag-authorize.json" 2>/dev/null)" = true ] || { T_EV="authorize refused: $(jq -c . "$ART/$tag-authorize.json" 2>/dev/null | cut -c1-250)"; return 1; }
  T_GRANT=$(jq -r '.authorization.id' "$ART/$tag-authorize.json")
  [ "${5:-}" = noexec ] && return 0
  execute "$tag" "$ws"
}
execute() { # tag ws [suffix]: execute with T_GRANT; sets T_STATE, T_PATH, T_CSHA, T_RECEIPT, T_REFUSAL
  local tag=$1 ws=$2 s=${3:-}
  cli compose execute --report "$ART/$tag-propose.json" --workspace "$ws" --authorization "$T_GRANT" >"$ART/$tag-execute$s.json" 2>"$ART/$tag-execute$s.err"
  T_STATE=$(jq -r '.state // "none"' "$ART/$tag-execute$s.json" 2>/dev/null); [ -n "$T_STATE" ] || T_STATE=none
  T_REFUSAL=$(jq -r '.error // ""' "$ART/$tag-execute$s.json" 2>/dev/null | sed -n 's/^EFFECT_REFUSED \([A-Za-z]*\):.*/\1/p')
  [ -n "$T_REFUSAL" ] || T_REFUSAL=$(head -c 200 "$ART/$tag-execute$s.err" | tr '\n' ' ')
  T_PATH=$(jq -r '.path // ""' "$ART/$tag-execute$s.json" 2>/dev/null); T_CSHA=$(jq -r '.content_sha256 // ""' "$ART/$tag-execute$s.json" 2>/dev/null)
  local rp; rp=$(jq -r '.receipt // ""' "$ART/$tag-execute$s.json" 2>/dev/null)
  if [ -n "$rp" ] && [ -f "$rp" ]; then T_RECEIPT=$(jq -c . "$rp"); else T_RECEIPT=null; fi
  [ "$T_STATE" = DONE ]
}
find_written() { # ws name: path of the written file (the engine chooses the folder); empty if none
  find "$1" -name "$2" -type f 2>/dev/null | head -1
}
doc_fields() { # file state csha grant failed_ids -> json
  local f=$1 disk=none; [ -f "$f" ] && disk=$(sha_of "$f")
  jq -nc --arg p "$f" --arg st "$2" --arg c "$3" --arg d "$disk" --argjson g "${4:-null}" --arg failed "$5" --argjson rc "${T_RECEIPT:-null}" \
    --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" \
    '{written_path:$p, effect_state:$st, content_sha256:$c, disk_sha256:$d, grant_id:$g, desk:"required", checks_failed:$failed,
      compose_receipt:$rc, memory_context:$mc, memory_items_included:$mn}'
}

# ---- PRE ---------------------------------------------------------------------------------------------------
echo "run folder: $OUT" | tee "$OUT/console.log"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md" "$OUT/ACCEPTANCE-v1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.1.md" "$OUT/ACCEPTANCE-v1.1.as-run.md" 2>/dev/null
FIX=$(cd "$WS/inbox" && sha256sum -- * | LC_ALL=C sort -k2 | jq -R -s 'split("\n") | map(select(length>0) | split("  ") | {name:.[1], sha256:.[0]})')
PRE_OK=1; PRE_EV="host $HOST; model $C_ID"
[ "$W_SHA" = "$C_W" ] && [ "$T_SHA" = "$C_T" ] || { PRE_OK=0; PRE_EV="$PRE_EV; model digests differ from release/candidate.toml"; }

# ---- CTRL-E3a: desk required, no key -> the daemon refuses to start ----------------------------------------
DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN-nokey.log; rm -f "$SOCK"
"${E[@]}" AIEN_MODEL_PATH="$MDIR/model.safetensors" AIEN_TOKENIZER_PATH="$MDIR/tokenizer.json" "$BIN" daemon >"$DLOG" 2>&1 &
NK=$!; for i in $(seq 1 300); do [ -d "/proc/$NK" ] || break; [ -S "$SOCK" ] && break; sleep 1; done
if [ -d "/proc/$NK" ]; then kill -9 "$NK" 2>/dev/null; wait "$NK" 2>/dev/null; NK_RC=killed; else wait "$NK" 2>/dev/null; NK_RC=$?; fi
NK_LINE=$(grep -i -m1 -E 'desk' "$DLOG" | cut -c1-200)
rm -f "$SOCK"

# ---- the real daemon ---------------------------------------------------------------------------------------
cli compose desk-key --create 1 >"$ART/deskkey.json" 2>&1   # prints id and path, never the key
if ! start_daemon; then
  rcpt PRE FAIL "daemon did not start: $(tail -c 300 "$DLOG" 2>/dev/null | tr '\n' ' ')" "$(jq -nc --argjson fix "$FIX" '{inbox_files:$fix}')"
  ABORT=1
else
  ABORT=""
  ROUTE=$(ip route 2>/dev/null | head -5 | tr '\n' ';')
  rcpt PRE "$([ $PRE_OK = 1 ] && echo PASS || echo FAIL)" "$PRE_EV; backend '$CUR_BACKEND'" \
    "$(jq -nc --argjson fix "$FIX" --arg route "$ROUTE" --arg q "$([ -s "$QUIET" ] && echo held || echo not-held)" --arg cw "$C_W" --arg ct "$C_T" --arg cid "$C_ID" --arg mdir "$MDIR" \
      '{inbox_files:$fix, ip_route:$route, quiet_flag:$q, candidate_model_id:$cid, candidate_weights_sha256:$cw, candidate_tokenizer_sha256:$ct, model_dir:$mdir, network_check:"recorded in the NET receipt"}')"
fi
if [ "$NK_RC" != killed ] && [ "$NK_RC" != 0 ] && [ -n "$NK_LINE" ]; then
  rcpt CTRL-E3a PASS "daemon without a desk key exited $NK_RC before serving; log: $NK_LINE" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
else
  rcpt CTRL-E3a FAIL "daemon without a desk key: exit '$NK_RC', desk line '$NK_LINE'" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
fi
rcpt E1 NOT_RUN "RUN-1 runs from a release build of this tree; nothing installed; RUN-3 measures E1" '{"signature_check":"none","artifact_sha256":null}'

# ---- E2 ----------------------------------------------------------------------------------------------------
rcpt E2 NOT_RUN "harness-side objective receipt written before any work (this receipt); installation-side record before work does not exist yet (lane L6)" \
  "$(jq -nc --arg t "$OBJ" --arg s "$(printf '%s' "$OBJ" | sha256sum | cut -d' ' -f1)" '{objective_text:$t, objective_text_sha256:$s, recorded_before_work:true, operator_surface:"harness + aien-cli compose propose", installation_record:"NOT_RUN (lane L6)"}')"

# ---- E3: the contract objective ----------------------------------------------------------------------------
if [ -z "$ABORT" ]; then
  if task E3 "$WSC" "$OBJ"; then
    P=$(find_written "$WSC" report.md); [ -n "$P" ] || P=$WSC/${T_PATH:-outbox/report.md}
    FAILED=$(check_doc "$P" "$F1" "$F2" "$F3"); DSHA=none; [ -f "$P" ] && DSHA=$(sha_of "$P")
    G=$T_GRANT; CS=$T_CSHA; execute E3 "$WSC" -again >/dev/null 2>&1; AGAIN=$T_REFUSAL
    [ -f "$P" ] && cp "$P" "$ART/E3-report.md"
    if [ -z "$FAILED" ] && [ "$DSHA" = "$CS" ] && [ "$AGAIN" = AlreadySpent ]; then ST=PASS; else ST=FAIL; fi
    rcpt E3 $ST "state DONE; written ${P#$WSC/}; checks failed [${FAILED:-none}]; second execute: ${AGAIN:-none}" \
      "$(doc_fields "$P" DONE "$CS" "$G" "$FAILED" | jq -c --arg a "$AGAIN" '. + {second_execute_refusal:$a}')"
  else
    rcpt E3 FAIL "$T_EV" "$(doc_fields none "$T_STATE" "" "$T_GRANT" "")"
  fi
else rcpt E3 NOT_RUN "no daemon" '{}'; fi

# ---- E3m: the mechanics objective --------------------------------------------------------------------------
E3M_OK=0; REPORT=""
if [ -z "$ABORT" ]; then
  if task E3m "$WS" "$OBJ_M"; then
    REPORT=$(find_written "$WS" report.md); [ -n "$REPORT" ] || REPORT=$WS/${T_PATH:-report.md}
    FAILED=$(check_doc "$REPORT" "$T1" "$T2" "$T3"); DSHA=none; [ -f "$REPORT" ] && DSHA=$(sha_of "$REPORT")
    G=$T_GRANT; CS=$T_CSHA; execute E3m "$WS" -again >/dev/null 2>&1; AGAIN=$T_REFUSAL
    [ -f "$REPORT" ] && cp "$REPORT" "$ART/E3m-report.md"
    if [ -z "$FAILED" ] && [ "$DSHA" = "$CS" ] && [ "$AGAIN" = AlreadySpent ]; then ST=PASS; E3M_OK=1; else ST=FAIL; fi
    rcpt E3m $ST "state DONE; written ${REPORT#$WS/}; checks failed [${FAILED:-none}]; second execute: ${AGAIN:-none}" \
      "$(doc_fields "$REPORT" DONE "$CS" "$G" "$FAILED" | jq -c --arg a "$AGAIN" --arg o "$OBJ_M" '. + {second_execute_refusal:$a, objective_text:$o}')"
  else
    rcpt E3m FAIL "$T_EV" "$(doc_fields none "$T_STATE" "" "$T_GRANT" "")"
  fi
else rcpt E3m NOT_RUN "no daemon" '{}'; fi

# ---- E4: remember, SIGKILL, restart, second objective ------------------------------------------------------
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ]; then
  cli compose remember --text "E4 memory item: the topic reported on first is the $T1" >"$ART/E4-remember.json" 2>&1
  REM_OK=$(jq -r '.ok // false' "$ART/E4-remember.json" 2>/dev/null)
  LINES_BEFORE=$(wc -l <"$REPORT"); SHA_BEFORE=$(sha_of "$REPORT")
  if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
  if start_daemon; then
    RESTARTS=$((RESTARTS + 1)); RESTART_UTC=$LAST_START_UTC
    REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
    if task E4 "$WS" "$OBJ2" work; then
      cp "$REPORT" "$ART/E4-report.md" 2>/dev/null
      LINES_AFTER=$(wc -l <"$REPORT"); NEWLINE=$(tail -n 1 "$REPORT")
      RECALL_UTC=$(utc)
      OK=1; WHY=""
      [ "$REM_OK" = true ] || { OK=0; WHY="$WHY remember-refused"; }
      [ "$T_MEMCTX" = work ] && [ "${T_MEMN:-0}" -ge 1 ] || { OK=0; WHY="$WHY memory-report($T_MEMCTX/$T_MEMN)"; }
      [ "$LINES_AFTER" -eq $((LINES_BEFORE + 1)) ] || { OK=0; WHY="$WHY lines($LINES_BEFORE->$LINES_AFTER)"; }
      printf '%s' "$NEWLINE" | grep -qiF -- "$T1" || { OK=0; WHY="$WHY last-line-lacks-$T1"; }
      [ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] || { OK=0; WHY="$WHY report-unchanged"; }
      rcpt E4 "$([ $OK = 1 ] && echo PASS || echo FAIL)" "$KILLED then restart ($REPL); second objective state $T_STATE; memory $T_MEMCTX/$T_MEMN; lines $LINES_BEFORE->$LINES_AFTER; problems [${WHY# }]" \
        "$(jq -nc --arg store "compose home (AIEN_COMPOSE_DIR records; canonical store decided by lane L3)" --arg item "$(sha_of "$ART/E4-remember.json")" --arg rk sigkill --arg ru "$RESTART_UTC" --arg cu "$RECALL_UTC" --arg nl "$NEWLINE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg st "$T_STATE" --argjson g "$T_GRANT" --argjson rc "${T_RECEIPT:-null}" --arg p "${T_PATH:-}" \
          '{memory_store:$store, item_sha256:$item, recall_after_restart:($mn>=1), restart_kind:$rk, restart_utc:$ru, recall_utc:$cu, appended_line:$nl, memory_context:$mc, memory_items_included:$mn, effect_state:$st, grant_id:$g, written_path:$p, compose_receipt:$rc}')"
    else
      rcpt E4 FAIL "$KILLED then restart ($REPL); second objective: $T_EV" "$(jq -nc --arg rk sigkill --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" '{restart_kind:$rk, effect_state:$st, memory_context:$mc, memory_items_included:$mn}')"
    fi
  else
    rcpt E4 FAIL "$KILLED; daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"restart_kind":"sigkill"}'; ABORT=1
  fi
else rcpt E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- E6: authorize, SIGKILL before execute, restart, finish once -------------------------------------------
if [ -z "$ABORT" ]; then
  if task E6 "$WS" "$OBJ6" work noexec; then
    G=$T_GRANT
    if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
    execute E6 "$WS" -dead >/dev/null 2>&1; DEAD_ST=$T_STATE; DEAD_REF=$T_REFUSAL
    DONE1=$(find_written "$WS" done.md); DEAD_WRITE=$([ -n "$DONE1" ] && echo yes || echo no)
    if start_daemon; then
      RESTARTS=$((RESTARTS + 1)); REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
      T_GRANT=$G; execute E6 "$WS"; ST1=$T_STATE; RC1=$T_RECEIPT
      DONE1=$(find_written "$WS" done.md); [ -n "$DONE1" ] || DONE1=$WS/${T_PATH:-outbox/done.md}
      S1=$(tstat "$DONE1"); CONTENT_OK=$([ -f "$DONE1" ] && grep -qi finished "$DONE1" && echo yes || echo no)
      T_GRANT=$G; execute E6 "$WS" -third >/dev/null 2>&1; THIRD=$T_REFUSAL; S2=$(tstat "$DONE1")
      DUP=$([ "$S1" = "$S2" ] && echo 0 || echo 1)
      if [ "$DEAD_WRITE" = no ] && [ "$ST1" = DONE ] && [ "$CONTENT_OK" = yes ] && [ "$THIRD" = AlreadySpent ] && [ "$DUP" = 0 ]; then ST=PASS; else ST=FAIL; fi
      rcpt E6 $ST "$KILLED after authorize; execute while dead: state $DEAD_ST (${DEAD_REF:-no refusal text}), wrote=$DEAD_WRITE; restart ($REPL); same grant: $ST1; third execute: ${THIRD:-none}; duplicates $DUP" \
        "$(jq -nc --arg kp after_authorize_before_execute --argjson ea "$([ "$ST1" = DONE ] && echo 1 || echo 0)" --argjson dup "$DUP" --argjson g "$G" --arg d "$DEAD_REF" --arg t "$THIRD" --argjson r "${RC1:-null}" --arg p "${DONE1#$WS/}" \
          '{kill_point:$kp, restart_count:1, effects_before_kill:0, effects_after_restart:$ea, duplicates:$dup, grant_id:$g, refusal_while_dead:$d, third_execute_refusal:$t, written_path:$p, compose_receipt:$r}')"
    else
      rcpt E6 FAIL "$KILLED; daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"kill_point":"after_authorize_before_execute"}'; ABORT=1
    fi
  else
    rcpt E6 FAIL "$T_EV" '{"kill_point":"after_authorize_before_execute"}'
  fi
else rcpt E6 NOT_RUN "no daemon" '{}'; fi

# ---- CTRL-E3b: revoke before spend -> Revoked --------------------------------------------------------------
if [ -z "$ABORT" ]; then
  if task CTRL-E3b "$WS" "$OBJC" work noexec; then
    G=$T_GRANT
    cli compose revoke --authorization "$G" --approver drake >"$ART/CTRL-E3b-revoke.json" 2>&1
    REV_OK=$(jq -r '.ok // false' "$ART/CTRL-E3b-revoke.json" 2>/dev/null)
    T_GRANT=$G; execute CTRL-E3b "$WS" >/dev/null 2>&1; REF=$T_REFUSAL
    CT=$(find_written "$WS" ctrl.md); ABSENT=$([ -z "$CT" ] && echo yes || echo no)
    if [ "$REV_OK" = true ] && [ "$REF" = Revoked ] && [ "$ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
    rcpt CTRL-E3b $ST "revoke ok=$REV_OK; execute refused '${REF:-none}'; ctrl.md absent=$ABSENT" "$(jq -nc --argjson g "$G" --arg r "$REF" --arg a "$ABSENT" '{grant_id:$g, refusal:$r, file_absent:$a}')"
  else rcpt CTRL-E3b FAIL "$T_EV" '{}'; fi
else rcpt CTRL-E3b NOT_RUN "no daemon" '{}'; fi

# ---- network check (rule 1), before the destructive control -----------------------------------------------
NET=$(ss -tnp state established 2>/dev/null | grep -c "pid=${DPID:-0}," || true)
rcpt NET "$([ "${NET:-0}" = 0 ] && echo PASS || echo FAIL)" "established TCP sockets of the daemon pid: ${NET:-0}" "$(jq -nc --argjson n "${NET:-0}" '{daemon_established_tcp:$n}')"

# ---- CTRL-E4: store removed -> named refusal, never a silent write -----------------------------------------
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ]; then
  stop_graceful
  SHA_BEFORE=$(sha_of "$REPORT"); mv "$RUN/compose" "$RUN/compose.removed-ctrl-e4"; mkdir -p "$RUN/compose"
  cli compose desk-key --create 1 >"$ART/CTRL-E4-deskkey.json" 2>&1
  if start_daemon; then
    if task CTRL-E4 "$WS" "$OBJ2" work; then
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=FAIL
      EV="store removed, yet the second objective ran to $T_STATE (memory $T_MEMCTX/$T_MEMN), report changed=$WROTE: silent, not a named refusal"
    else
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=$([ "$WROTE" = no ] && echo PASS || echo FAIL)
      EV="store removed: refused ($T_EV); report changed=$WROTE"
    fi
    rcpt CTRL-E4 $ST "$EV" "$(jq -nc --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg w "$WROTE" '{effect_state:$st, memory_context:$mc, memory_items_included:$mn, report_changed:$w}')"
  else rcpt CTRL-E4 FAIL "daemon did not start after the store was removed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{}'; fi
else rcpt CTRL-E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- the rest: NOT_RUN by lane -----------------------------------------------------------------------------
rcpt E5 NOT_RUN "no independent verifier yet (lane L4); VERDICT_ROWS.txt and SHA256SUMS written for it" '{"verifier_sha256":null}'
rcpt CTRL-E1 NOT_RUN "no release artifact in RUN-1 (lane L5)" '{}'
rcpt CTRL-E2 NOT_RUN "no installation-side objective record (lane L6)" '{}'
rcpt CTRL-E5 NOT_RUN "no verifier (lane L4)" '{}'
rcpt CTRL-E6 NOT_RUN "no durable objective state to remove (lane L2)" '{}'
stop_graceful
cp "$LOGS"/daemon-*.log "$OUT/artifacts/" 2>/dev/null
printf 'objective_id %s\ncontract_sha256 8ba653848287bc77db253466f6d3ff37c70a42be55601ad47eb1f838a41d20f0\n%s' "$OBJ_ID" "$ROWS" >"$OUT/VERDICT_ROWS.txt"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >chain/SHA256SUMS )
echo "run folder written: $OUT ($(ls "$CHAIN" | wc -l) chain files)"; cat "$OUT/VERDICT_ROWS.txt"
