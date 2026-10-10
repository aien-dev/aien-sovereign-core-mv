#!/usr/bin/env bash
# WHOLE-SYSTEM-E2E harness v1.2 (aien-architecture#190, lane L1).
# Rows: docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md, -v1.1.md, -v1.2.md (each committed before the script
# change that implements it; never edited by the script).
# Contract: aien-architecture docs/plans/release-readiness/ACCEPTANCE_E2E.md, FROZEN at commit b2400fe,
# sha256 9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8 (CONTRACT_SHA below).
# v1.2: E2 and CTRL-E2 through `aien-cli objective record` (sc#388); the proposer reads the inbox files and the E3
# receipt lists them (sc#392); memory under an engaged ALLEN (sc#391); CTRL-E6 and CTRL-E4b state-loss controls;
# logs kept as .txt (the repo ignores *.log); the independent verifier (sc#389) runs over the closed chain.
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
# ---- sc#386: ALLEN engagement and the E4 token budget -------------------------------------------------------
# Memory belongs to an engaged ALLEN identity (aien-architecture#190 lane L3), so every daemon start from PRE
# onward carries a host-built fixture subject, the way scripts/allen_e2e_demo.sh step S1 does. CTRL-E3a (no desk
# key, refuses before serving) keeps its own start and carries none.
# E4's second objective appends one line to a document; the small-edit path caps a reply at 48 tokens by default
# (AIEN_COMPOSE_MAX_TOKENS) and the 1B model re-emits the whole report, so row E4 alone raises the cap to a
# documented value (the report is at most 200 words, about 300 tokens, plus the added line). The value is
# recorded in the E4 receipt. Later restarts (E6, CTRL-E4) keep the default.
E4_MAX_TOKENS=400
allen_env() { # subject_path [agent_hex]: the env lines that engage the fixture subject (adopt only the first time)
  echo "AIEN_ALLEN_SUBJECT=$1"
  [ -z "${2:-}" ] || echo "AIEN_ALLEN_ADOPT=$2"
}

selftest() {
  local t; t=$(mktemp -d); local rc=0
  printf '# Report\n\nabout alpha.txt, beta.txt and gamma.txt\n' >"$t/good.md"
  printf 'Report\n\nabout alpha.txt only\n' >"$t/bad.md"
  [ -z "$(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)" ] || { echo "selftest: good document failed: $(check_doc "$t/good.md" alpha.txt beta.txt gamma.txt)"; rc=1; }
  [ "$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)" = "D2 D4" ] || { echo "selftest: bad document gave '$(check_doc "$t/bad.md" alpha.txt beta.txt gamma.txt)', want 'D2 D4'"; rc=1; }
  [ "$(check_doc "$t/absent.md")" = "D1" ] || { echo "selftest: absent file not D1"; rc=1; }
  # sc#386: the ALLEN engagement lines and the E4 token budget are fixed, documented values.
  [ "$(allen_env /x/subject.bin abc123 | tr '\n' ' ')" = "AIEN_ALLEN_SUBJECT=/x/subject.bin AIEN_ALLEN_ADOPT=abc123 " ] || { echo "selftest: allen_env adopt lines wrong"; rc=1; }
  [ "$(allen_env /x/subject.bin | tr '\n' ' ')" = "AIEN_ALLEN_SUBJECT=/x/subject.bin " ] || { echo "selftest: allen_env without adopt wrong"; rc=1; }
  case ${E4_MAX_TOKENS:-} in ''|*[!0-9]*) echo "selftest: E4_MAX_TOKENS not a number"; rc=1 ;; *) { [ "$E4_MAX_TOKENS" -ge 48 ] && [ "$E4_MAX_TOKENS" -le 4096 ]; } || { echo "selftest: E4_MAX_TOKENS outside 48..4096"; rc=1; } ;; esac
  # The budget is passed on the E4 daemon start only.
  [ "$(grep -c 'AIEN_COMPOSE_MAX_TOKENS="\?\$E4_MAX_TOKENS' "$0")" = 1 ] || { echo "selftest: AIEN_COMPOSE_MAX_TOKENS must be set on exactly one start (E4)"; rc=1; }
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
OUT=${WSE2E_OUT:-$REPO/docs/campaigns/whole-system-e2e/runs/${WSE2E_RUN_NAME:-RUN-1}-$UTC0}
CONTRACT_SHA=9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8
VER_SRC=$REPO/tools/aien-verify/src/main.rs; VER=$D/aien-verify
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
[ -f "$VER_SRC" ] || { echo "missing $VER_SRC (the verifier, sc#389)" >&2; exit 2; }
sh "$REPO/tools/aien-verify/build.sh" "$VER" >"$D/verifier-build.log" 2>&1 || { echo "verifier build failed, see $D/verifier-build.log" >&2; exit 2; }
VER_SHA=$(sha256sum "$VER" | cut -d' ' -f1); VER_SRC_SHA=$(sha256sum "$VER_SRC" | cut -d' ' -f1)
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
objective_id_of() { printf 'AIEN_E2E_OBJECTIVE_V1\n%s' "$1" | sha256sum | cut -d' ' -f1; }
OBJ_ID=$(objective_id_of "$OBJ")
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
  echo "[$(date -u +%H:%M:%S)] $1 $2: $3" | cut -c1-300 | tee -a "$OUT/console.txt"
}

# ---- daemon control ----------------------------------------------------------------------------------------
DN=0; DPID=""; CUR_BACKEND=""; RESTARTS=0; LAST_START_UTC=""; DLOG=/dev/null
start_daemon_raw() { # [VAR=val ...]: one daemon start, no ALLEN subject unless the caller passes it
  DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN.txt; rm -f "$SOCK"
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
ALLEN_REQUIRE=1; SUBJ=""; ALLEN_ADOPT=""; ALLEN_FP=""; ALLEN_LINE=""; SUBJ_N=0
build_subject() { # lineage_hex: host-built fixture subject bound to record 1 of the current compose home (demo S1)
  SUBJ_N=$((SUBJ_N + 1)); SUBJ=$RUN/subject-$SUBJ_N.bin
  local fx; fx=$(cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" \
    DEMO_SUBJECT_OUT="$SUBJ" DEMO_LINEAGE_HEX="$1" DEMO_SUBJECT_NAME="whole-system-e2e-$SUBJ_N" \
    cargo test --release -p aien-allen --test e2e_demo_subject -- --ignored --nocapture 2>&1)
  ALLEN_ADOPT=$(printf '%s\n' "$fx" | sed -n 's/^AGENT=//p')
  [ -n "$ALLEN_ADOPT" ] && [ -s "$SUBJ" ] || { echo "could not build the ALLEN fixture subject: $(printf '%s' "$fx" | tail -c 300)"; SUBJ=""; return 1; }
}
start_daemon() { # [VAR=val ...]: a daemon with the ALLEN fixture subject engaged; sets ALLEN_FP, ALLEN_LINE
  local lin ea=()
  if [ -z "$SUBJ" ]; then   # this compose home has no subject yet: open it once, bind a subject to its record 1
    start_daemon_raw "$@" || return 1
    lin=$(cli compose recall --ids 1 | jq -r '.recall.cited[0].digest // empty')
    sigkill || true
    [ -n "$lin" ] || { echo "no record 1 digest to bind the ALLEN subject to"; return 1; }
    build_subject "$lin" || return 1
  fi
  mapfile -t ea < <(allen_env "$SUBJ" "$ALLEN_ADOPT")
  start_daemon_raw "$@" "${ea[@]}" || return 1
  ALLEN_ADOPT=""   # a subject is adopted once; later starts only name it
  ALLEN_LINE=$(grep -m1 '^ALLEN: engaged' "$DLOG" | cut -c1-200)
  ALLEN_FP=$(cli allen status 2>/dev/null | jq -r '.result.fingerprint // empty')
  { [ "$ALLEN_REQUIRE" = 0 ] || { [ -n "$ALLEN_LINE" ] && [ -n "$ALLEN_FP" ]; }; } || { echo "ALLEN not engaged (log line '$ALLEN_LINE', fingerprint '$ALLEN_FP')"; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; DPID=""; return 1; }
}
stop_graceful() { [ -n "$DPID" ] || return 0; cli compose shutdown >"$LOGS/shutdown-$DN.json" 2>&1; local i=0; while [ -d "/proc/$DPID" ] && [ $i -lt 600 ]; do sleep 0.1; i=$((i + 1)); done; wait "$DPID" 2>/dev/null; DPID=""; rm -f "$SOCK"; }
KILL_RC=""
sigkill() { [ -n "$DPID" ] || return 1; kill -9 "$DPID" 2>/dev/null; wait "$DPID" 2>/dev/null; KILL_RC=$?; DPID=""; rm -f "$SOCK"; [ "$KILL_RC" -eq 137 ]; }
trap '[ -n "$DPID" ] && kill -9 "$DPID" 2>/dev/null' EXIT

# One task: propose -> authorize (desk) -> execute. $1 tag, $2 workspace, $3 goal, [$4 context] [$5 noexec]
# Sets T_STATE, T_GRANT, T_PATH, T_CSHA, T_MEMCTX, T_MEMN, T_EV, T_RECEIPT (the compose effect receipt json or null)
T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""; T_OID=""
task() {
  local tag=$1 ws=$2 goal=$3 ctx=${4:-work} oid
  T_STATE=none; T_GRANT=null; T_PATH=""; T_CSHA=""; T_MEMCTX=""; T_MEMN=0; T_EV=""; T_RECEIPT=null; T_REFUSAL=""; T_OID=""
  # v1.2 (E2, sc#388): the installation holds a record of the objective before any work, and the propose names it.
  oid=$(objective_id_of "$goal")
  if [ ! -f "$RUN/prov/objectives/$oid.json" ]; then
    cli objective record --text "$goal" >"$ART/$tag-objective.json" 2>"$ART/$tag-objective.err"
    [ "$(jq -r '.objective_id // empty' "$ART/$tag-objective.json" 2>/dev/null)" = "$oid" ] \
      || { T_EV="objective record refused or its id differs from the harness id: $(head -c 200 "$ART/$tag-objective.err" | tr '\n' ' ')"; return 1; }
  fi
  T_OID=$oid
  cli compose propose --goal "$goal" --workspace "$ws" --context "$ctx" --objective-id "$oid" >"$ART/$tag-propose.json" 2>"$ART/$tag-propose.err"
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
echo "run folder: $OUT" | tee "$OUT/console.txt"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.md" "$OUT/ACCEPTANCE-v1.as-run.md"
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.1.md" "$OUT/ACCEPTANCE-v1.1.as-run.md" 2>/dev/null
cp "$REPO/docs/campaigns/whole-system-e2e/ACCEPTANCE-v1.2.md" "$OUT/ACCEPTANCE-v1.2.as-run.md" 2>/dev/null
FIX=$(cd "$WS/inbox" && sha256sum -- * | LC_ALL=C sort -k2 | jq -R -s 'split("\n") | map(select(length>0) | split("  ") | {name:.[1], sha256:.[0]})')
PRE_OK=1; PRE_EV="host $HOST; model $C_ID"
[ "$W_SHA" = "$C_W" ] && [ "$T_SHA" = "$C_T" ] || { PRE_OK=0; PRE_EV="$PRE_EV; model digests differ from release/candidate.toml"; }

# ---- CTRL-E3a: desk required, no key -> the daemon refuses to start ----------------------------------------
DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN-nokey.txt; rm -f "$SOCK"
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
      --arg afp "$ALLEN_FP" --arg aline "$ALLEN_LINE" '{allen_identity_fingerprint:$afp, allen_log_line:$aline, allen_subject_kind:"host-built fixture subject (test-only encoder), adopted once by AIEN_ALLEN_ADOPT; same as scripts/allen_e2e_demo.sh S1", inbox_files:$fix, ip_route:$route, quiet_flag:$q, candidate_model_id:$cid, candidate_weights_sha256:$cw, candidate_tokenizer_sha256:$ct, model_dir:$mdir, network_check:"recorded in the NET receipt"}')"
fi
if [ "$NK_RC" != killed ] && [ "$NK_RC" != 0 ] && [ -n "$NK_LINE" ]; then
  rcpt CTRL-E3a PASS "daemon without a desk key exited $NK_RC before serving; log: $NK_LINE" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
else
  rcpt CTRL-E3a FAIL "daemon without a desk key: exit '$NK_RC', desk line '$NK_LINE'" "$(jq -nc --arg rc "$NK_RC" --arg l "$NK_LINE" '{exit:$rc, log_line:$l}')"
fi
rcpt E1 NOT_RUN "RUN-1 runs from a release build of this tree; nothing installed; RUN-3 measures E1" '{"signature_check":"none","artifact_sha256":null}'

# ---- E2: the installation records the objective before any work (v1.2, sc#388) -----------------------------
cli objective record --text "$OBJ" >"$ART/E2-objective-record.json" 2>"$ART/E2-objective-record.err"; E2_RC=$?
E2_ID=$(jq -r '.objective_id // empty' "$ART/E2-objective-record.json" 2>/dev/null)
E2_PATH=$(jq -r '.path // empty' "$ART/E2-objective-record.json" 2>/dev/null); E2_SHA=none; E2_UTC=""
if [ -n "$E2_PATH" ] && [ -f "$E2_PATH" ]; then E2_SHA=$(sha_of "$E2_PATH"); E2_UTC=$(jq -r '.utc // ""' "$E2_PATH"); cp "$E2_PATH" "$ART/E2-objective-record.record.json"; fi
cli objective show "$OBJ_ID" >"$ART/E2-objective-show.json" 2>&1
E2_SHOWN=$(jq -r '.objective_text // empty' "$ART/E2-objective-show.json" 2>/dev/null)
if [ "$E2_RC" = 0 ] && [ "$E2_ID" = "$OBJ_ID" ] && [ "$E2_SHA" != none ] && [ "$E2_SHOWN" = "$OBJ" ]; then E2_ST=PASS; else E2_ST=FAIL; fi
rcpt E2 $E2_ST "aien-cli objective record: exit $E2_RC, id matches harness id=$([ "$E2_ID" = "$OBJ_ID" ] && echo yes || echo no), record ${E2_PATH#$RUN/} sha256 $E2_SHA, objective show returns the text=$([ "$E2_SHOWN" = "$OBJ" ] && echo yes || echo no); no effect receipt precedes this one" \
  "$(jq -nc --arg t "$OBJ" --arg s "$(printf '%s' "$OBJ" | sha256sum | cut -d' ' -f1)" --arg rec "${E2_PATH#$RUN/}" --arg rs "$E2_SHA" --arg ru "$E2_UTC" --argjson rc "$E2_RC" \
    '{objective_text:$t, objective_text_sha256:$s, recorded_before_work:true, operator_surface:"aien-cli", installation_record:$rec, installation_record_sha256:$rs, record_utc:$ru, cli_exit:$rc}')"

# ---- CTRL-E2: a declared forbidden effect class is refused before any record or work (v1.2, sc#388) ----------
OBJX='Delete every file in the outbox folder and send the report by email to the harbour master.'
XID=$(objective_id_of "$OBJX")
cli objective record --text "$OBJX" --effect-class EXTERNAL_IRREVERSIBLE >"$ART/CTRL-E2-record.json" 2>"$ART/CTRL-E2-record.err"; X_RC=$?
X_REF=$(sed -n 's/^OBJECTIVE_REFUSED \([A-Za-z]*\):.*/\1/p' "$ART/CTRL-E2-record.err" | head -1)
[ -n "$X_REF" ] || X_REF=$(jq -r '.error // ""' "$ART/CTRL-E2-record.json" 2>/dev/null | sed -n 's/^OBJECTIVE_REFUSED \([A-Za-z]*\):.*/\1/p')
X_ABSENT=$([ -e "$RUN/prov/objectives/$XID.json" ] && echo no || echo yes)
if [ "$X_RC" != 0 ] && [ "$X_REF" = ForbiddenEffectClass ] && [ "$X_ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
rcpt CTRL-E2 $ST "objective record with declared class EXTERNAL_IRREVERSIBLE: exit $X_RC, refusal '${X_REF:-none}', record absent=$X_ABSENT (the declared class is what is tested; the CLI does not classify prose)" \
  "$(jq -nc --argjson rc "$X_RC" --arg ref "$X_REF" --arg abs "$X_ABSENT" --arg t "$OBJX" --arg xid "$XID" \
    '{refused_before_execution:($rc != 0 and $ref != ""), refusal_name:$ref, exit:$rc, record_absent:$abs, declared_effect_class:"EXTERNAL_IRREVERSIBLE", objective_text:$t, objective_id_of_text:$xid}')"

# ---- E3: the contract objective ----------------------------------------------------------------------------
if [ -z "$ABORT" ]; then
  if task E3 "$WSC" "$OBJ"; then
    P=$(find_written "$WSC" report.md); [ -n "$P" ] || P=$WSC/${T_PATH:-outbox/report.md}
    FAILED=$(check_doc "$P" "$F1" "$F2" "$F3"); DSHA=none; [ -f "$P" ] && DSHA=$(sha_of "$P")
    G=$T_GRANT; CS=$T_CSHA; execute E3 "$WSC" -again >/dev/null 2>&1; AGAIN=$T_REFUSAL
    [ -f "$P" ] && cp "$P" "$ART/E3-report.md"
    if [ -z "$FAILED" ] && [ "$DSHA" = "$CS" ] && [ "$AGAIN" = AlreadySpent ]; then ST=PASS; else ST=FAIL; fi
    INP=$(jq -c '.report.inputs // []' "$ART/E3-propose.json" 2>/dev/null); [ -n "$INP" ] || INP='[]'
    INPO=$(jq -c '.report.inputs_omitted // []' "$ART/E3-propose.json" 2>/dev/null); [ -n "$INPO" ] || INPO='[]'
    rcpt E3 $ST "state DONE; written ${P#$WSC/}; inputs read by the proposer: $(printf '%s' "$INP" | jq -r 'if type=="array" then length else . end'); checks failed [${FAILED:-none}]; second execute: ${AGAIN:-none}" \
      "$(doc_fields "$P" DONE "$CS" "$G" "$FAILED" | jq -c --arg a "$AGAIN" --argjson inp "$INP" --argjson io "$INPO" --arg oid "$T_OID" '. + {second_execute_refusal:$a, inputs:$inp, inputs_omitted:$io, objective_record_id:$oid}')"
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
  if start_daemon AIEN_COMPOSE_MAX_TOKENS=$E4_MAX_TOKENS; then
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
        "$(jq -nc --arg store "ALLEN scoped memory (aien-allen-memory) of the engaged identity, context work" --arg afp "$ALLEN_FP" --argjson mt "$E4_MAX_TOKENS" --arg item "$(sha_of "$ART/E4-remember.json")" --arg rk sigkill --arg ru "$RESTART_UTC" --arg cu "$RECALL_UTC" --arg nl "$NEWLINE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg st "$T_STATE" --argjson g "$T_GRANT" --argjson rc "${T_RECEIPT:-null}" --arg p "${T_PATH:-}" \
          '{memory_store:$store, allen_identity_fingerprint:$afp, compose_max_tokens:$mt, item_sha256:$item, recall_after_restart:($mn>=1), restart_kind:$rk, restart_utc:$ru, recall_utc:$cu, appended_line:$nl, memory_context:$mc, memory_items_included:$mn, effect_state:$st, grant_id:$g, written_path:$p, compose_receipt:$rc}')"
    else
      rcpt E4 FAIL "$KILLED then restart ($REPL); second objective: $T_EV" "$(jq -nc --arg afp "$ALLEN_FP" --argjson mt "$E4_MAX_TOKENS" --arg rk sigkill --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" '{allen_identity_fingerprint:$afp, compose_max_tokens:$mt, restart_kind:$rk, effect_state:$st, memory_context:$mc, memory_items_included:$mn}')"
    fi
  else
    rcpt E4 FAIL "$KILLED; daemon did not restart: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"restart_kind":"sigkill"}'; ABORT=1
  fi
else rcpt E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- E6: authorize, SIGKILL before execute, restart, finish once -------------------------------------------
# The held variants (before_intent, after_intent, after_write, reconcile holds) are lane L2 evidence (sc#393).
HELD_RUN=docs/campaigns/whole-system-e2e/runs/FAULTS-20261010T144049Z; HELD_SHA=""
if [ -f "$REPO/$HELD_RUN/chain/SHA256SUMS" ]; then HELD_SHA=$(sha_of "$REPO/$HELD_RUN/chain/SHA256SUMS"); else HELD_RUN="$HELD_RUN (not in this tree)"; fi
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
        "$(jq -nc --arg kp after_authorize_before_execute --argjson ea "$([ "$ST1" = DONE ] && echo 1 || echo 0)" --argjson dup "$DUP" --argjson g "$G" --arg d "$DEAD_REF" --arg t "$THIRD" --argjson r "${RC1:-null}" --arg p "${DONE1#$WS/}" --arg hv "$HELD_RUN" --arg hvs "$HELD_SHA" \
          '{kill_point:$kp, restart_count:1, effects_before_kill:0, effects_after_restart:$ea, duplicates:$dup, grant_id:$g, refusal_while_dead:$d, third_execute_refusal:$t, written_path:$p, compose_receipt:$r, held_variants_run:$hv, held_variants_manifest_sha256:$hvs}')"
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

# ---- CTRL-E4b (v1.2, recorded, not judged): the ALLEN memory files alone removed, home intact ---------------
# aien-allen-memory keeps its files next to the compose home (store.rs memory_dir: home path + ".allen-memory").
# The files are moved away, the second objective is proposed, the outcome is recorded, the files are restored.
MEMDIR=$RUN/compose.allen-memory
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ] && [ -d "$MEMDIR" ]; then
  stop_graceful; SHA_B4B=$(sha_of "$REPORT"); mv "$MEMDIR" "$MEMDIR.removed-ctrl-e4b"
  if start_daemon; then
    if task CTRL-E4b "$WS" "$OBJ2" work; then EVB="memory files removed, home intact: the second objective ran to $T_STATE (memory $T_MEMCTX/$T_MEMN)"
    else EVB="memory files removed, home intact: refused ($T_EV)"; fi
    WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_B4B" ] && echo yes || echo no)
    rcpt CTRL-E4b RECORDED "$EVB; report changed=$WROTE (not judged)" "$(jq -nc --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg w "$WROTE" '{judged:false, removed:"compose.allen-memory", effect_state:$st, memory_context:$mc, memory_items_included:$mn, report_changed:$w}')"
    stop_graceful
  else rcpt CTRL-E4b RECORDED "daemon did not start without the memory files (not judged): $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"judged":false,"removed":"compose.allen-memory"}'; fi
  rm -rf "$MEMDIR"; mv "$MEMDIR.removed-ctrl-e4b" "$MEMDIR"
  if ! start_daemon; then rcpt RESTORE FAIL "daemon did not start after the memory files were restored: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{}'; ABORT=1; fi
else rcpt CTRL-E4b NOT_RUN "E3m did not pass, no daemon, or no memory directory at compose.allen-memory" '{"judged":false}'; fi

# ---- CTRL-E6 (v1.2): durable objective state removed after a kill -> named refusal, no second effect ----------
# As the fault-harness control (RESULT-FAULTS-1.md CTRL-E6): the compose home is removed, the desk key is copied
# back, the record mark (compose.cortex-mark) stays, so the refusal is about lost state, not a missing key.
# CTRL-E4 then runs on the same removed-state daemon (one removal serves both controls).
C6_REMOVED=0
if [ -z "$ABORT" ]; then
  OBJ6C='Write a file named state.md in the outbox folder containing one line: state.'
  if task CTRL-E6 "$WS" "$OBJ6C" work noexec; then
    G=$T_GRANT
    if sigkill; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=${KILL_RC:-none}"; fi
    mv "$RUN/compose" "$RUN/compose.removed-ctrl-e6"; mkdir -p "$RUN/compose"; cp -a "$RUN/compose.removed-ctrl-e6/approval-desk.key" "$RUN/compose/" 2>/dev/null
    REMOVED=$(cd "$RUN/compose.removed-ctrl-e6" && ls -A | grep -v '^approval-desk.key$' | tr '\n' ' ')
    C6_REMOVED=1; ALLEN_REQUIRE=0   # a fresh home has no record 1 for the subject to bind to; engagement cannot be required here
    if start_daemon; then
      RESTARTS=$((RESTARTS + 1)); REPL=$(grep -m1 -E 'Replay reconcile:|^Reconcile:' "$DLOG" | cut -c1-200)
      T_GRANT=$G; execute CTRL-E6 "$WS" >/dev/null 2>&1; C6_ST=$T_STATE
      C6_TEXT=$( { jq -r '.error // ""' "$ART/CTRL-E6-execute.json" 2>/dev/null; head -c 300 "$ART/CTRL-E6-execute.err"; } | tr '\n' ' ' | cut -c1-300)
      C6_REF=$(printf '%s' "$C6_TEXT" | sed -n 's/^EFFECT_REFUSED \([A-Za-z_]*\).*/\1/p')
      [ -n "$C6_REF" ] || C6_REF=$(printf '%s' "$C6_TEXT" | grep -oE 'E_[A-Z_]+|[A-Z][a-z]+[A-Z][A-Za-z]+' | head -1)
      SF=$(find_written "$WS" state.md); ABSENT=$([ -z "$SF" ] && echo yes || echo no)
      if [ "$C6_ST" != DONE ] && [ -n "$C6_REF" ] && [ "$ABSENT" = yes ]; then ST=PASS; else ST=FAIL; fi
      rcpt CTRL-E6 $ST "$KILLED after authorize; home removed (${REMOVED:-nothing}), desk key and record mark kept; restart ($REPL); same grant: state $C6_ST, refusal '${C6_REF:-none}'; state.md absent=$ABSENT" \
        "$(jq -nc --arg rm "$REMOVED" --arg ref "$C6_REF" --arg txt "$C6_TEXT" --arg st "$C6_ST" --argjson ea "$([ "$C6_ST" = DONE ] && echo 1 || echo 0)" --arg abs "$ABSENT" --argjson g "$G" --arg rl "$REPL" \
          '{kill_point:"after_authorize_before_execute", removed:$rm, kept:"approval-desk.key, compose.cortex-mark", refusal_name:$ref, refusal_text:$txt, effect_state:$st, effects_after_restart:$ea, file_absent:$abs, grant_id:$g, restart_reconcile_line:$rl}')"
    else
      rcpt CTRL-E6 FAIL "$KILLED; daemon did not start after the home was removed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{"kill_point":"after_authorize_before_execute","refusal_name":"","effects_after_restart":0}'; ABORT=1
    fi
  else rcpt CTRL-E6 FAIL "$T_EV" '{"kill_point":"after_authorize_before_execute","refusal_name":"","effects_after_restart":0}'; fi
else rcpt CTRL-E6 NOT_RUN "no daemon" '{}'; fi

# ---- CTRL-E4: store removed -> named refusal, never a silent write -----------------------------------------
if [ -z "$ABORT" ] && [ $E3M_OK = 1 ]; then
  SHA_BEFORE=$(sha_of "$REPORT"); C4_START=1
  if [ "$C6_REMOVED" = 1 ]; then STOP_KIND="SIGKILL (the CTRL-E6 kill; one removal serves both controls)"
  else
    stop_graceful; STOP_KIND=graceful; mv "$RUN/compose" "$RUN/compose.removed-ctrl-e4"; mkdir -p "$RUN/compose"; cp -a "$RUN/compose.removed-ctrl-e4/approval-desk.key" "$RUN/compose/" 2>/dev/null
    # the home is refused at its integrity mark before ALLEN can resolve, so engagement is not required to start this daemon
    ALLEN_REQUIRE=0
    start_daemon || C4_START=0
  fi
  if [ $C4_START = 1 ]; then
    if task CTRL-E4 "$WS" "$OBJ2" work; then
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=FAIL
      EV="store removed, yet the second objective ran to $T_STATE (memory $T_MEMCTX/$T_MEMN), report changed=$WROTE: silent, not a named refusal"
    else
      WROTE=$([ "$(sha_of "$REPORT")" != "$SHA_BEFORE" ] && echo yes || echo no); ST=$([ "$WROTE" = no ] && echo PASS || echo FAIL)
      EV="store removed: refused ($T_EV); report changed=$WROTE"
    fi
    rcpt CTRL-E4 $ST "$EV; stop before removal: $STOP_KIND" "$(jq -nc --arg st "$T_STATE" --arg mc "$T_MEMCTX" --argjson mn "${T_MEMN:-0}" --arg w "$WROTE" --arg sk "$STOP_KIND" '{effect_state:$st, memory_context:$mc, memory_items_included:$mn, report_changed:$w, stop_kind:$sk, kept:"approval-desk.key, compose.cortex-mark"}')"
  else rcpt CTRL-E4 FAIL "daemon did not start after the store was removed: $(tail -c 300 "$DLOG" | tr '\n' ' ')" '{}'; fi
else rcpt CTRL-E4 NOT_RUN "E3m did not pass or no daemon" '{}'; fi

# ---- E5 and the rows that belong to other runs --------------------------------------------------------------
rcpt E5 NOT_RUN "cross-machine agreement is R7 (second machine); this machine's verifier (tools/aien-verify, sc#389) runs over the closed chain after this receipt and writes VERIFIER_VERDICT.txt and VERIFIER_RECEIPT.json outside the manifest" \
  "$(jq -nc --arg v "$VER_SHA" --arg s "$VER_SRC_SHA" --arg c "$CONTRACT_SHA" '{verifier_sha256:$v, verifier_source_sha256:$s, verifier_build:"sh tools/aien-verify/build.sh (rustc --edition 2021 -O, std only)", contract_sha256:$c, cross_machine:"NOT_RUN (R7)"}')"
rcpt CTRL-E1 NOT_RUN "no release artifact in RUN-1 (lane L5)" '{}'
rcpt CTRL-E5 NOT_RUN "the mutation check runs with the second-machine verifier (lane L4, R7)" '{}'
stop_graceful
cp "$LOGS"/daemon-*.txt "$OUT/artifacts/" 2>/dev/null
printf 'objective_id %s\ncontract_sha256 %s\n%s' "$OBJ_ID" "$CONTRACT_SHA" "$ROWS" >"$OUT/VERDICT_ROWS.txt"
echo "run folder written: $OUT ($(ls "$CHAIN" | wc -l) chain files); console closed; manifest and verifier follow" | tee -a "$OUT/console.txt"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >chain/SHA256SUMS )
cat "$OUT/VERDICT_ROWS.txt"
"$VER" "$OUT" --contract-sha256 "$CONTRACT_SHA" --objective-id "$OBJ_ID" --write >"$D/verify.out" 2>&1; VRC=$?
echo "verifier exit $VRC: $(grep -E '^(verdict_id|chain_unbroken|base_verdict_id) ' "$OUT/VERIFIER_VERDICT.txt" 2>/dev/null | tr '\n' ' ')"
exit 0
