#!/usr/bin/env bash
# Run one OWN-30B HTTP or benchmark session inside a bounded Spark quiet window
# (#625), and record whether the GPU was really uncontended.
#
# What it does, in order:
#   1. Preflight: GPU compute processes, GPU utilization, AIEN listeners and the
#      identity of what is being measured (repo HEAD, server binary SHA-256,
#      model path and size, Omega pin, OWN30B_DECODE_GRAPH).
#   2. Refuses (exit 3, command not run) when a GPU compute process other than
#      the declared server (--server-pid) is present, or when nvidia-smi cannot
#      say, unless --allow-contended REASON is given. An override runs the
#      command, records the reason and marks the run not comparable.
#   3. Runs CMD under `quietlock hold` (bounded, owner and reason recorded;
#      quietlock releases the flag when CMD and its process group exit).
#   4. Postflight: the same GPU query again. A foreign process seen then also
#      marks the run not comparable.
#   5. Writes OUT_DIR/manifest.json (schema own30b-quiet-run/v1).
#
# It never takes /tmp/aien-gb10.lock: aien-serve-http takes that lock itself
# for each generation, and holding it from outside would deadlock the server.
# The quiet flag and the GPU lock serve different purposes.
#
# File locks only coordinate cooperating workloads. A GPU process that ignores
# the quiet flag is detected (preflight and postflight), never excluded, and
# this script never kills, pauses or renices anything. A process that starts
# and stops between the two samples is not seen.
#
# Usage:
#   scripts/bench/own30b-quiet-run.sh --owner ID --out DIR [options] -- CMD...
# Options:
#   --minutes N            quiet hold bound, 1 to 20 [10] (quietlock needs an
#                          approval token above 20; this script does not go there)
#   --reason TEXT          recorded in the quiet flag and the manifest
#   --server-pid PID       the server under test; allowed on the GPU (repeatable)
#   --server-bin PATH      server binary, hashed into the manifest
#   --model PATH           model file or directory, path and size recorded
#   --omega-pin TEXT       Omega pin of the native backend, recorded as given
#   --allow-contended TEXT run even when the GPU is not idle; TEXT is recorded
# Environment: OWN30B_QR_QUIETLOCK [quietlock], OWN30B_QR_NVIDIA_SMI
# [nvidia-smi], OWN30B_QR_SS [ss] (the self-test points these at fakes).
#
# Exit: CMD's exit status; 2 usage; 3 refused as contended; 75 quiet flag
# already held (command not run); 128+N when interrupted by signal N (the held
# command is stopped with SIGTERM through quietlock and the flag released).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "own30b-quiet-run: $*" >&2; exit 2; }
QL=${OWN30B_QR_QUIETLOCK:-quietlock}
SMI=${OWN30B_QR_NVIDIA_SMI:-nvidia-smi}
SS=${OWN30B_QR_SS:-ss}

OWNER="" OUT="" MINUTES=10 REASON="OWN-30B uncontended measurement (#625)"
SERVER_BIN="" MODEL="" OMEGA_PIN="" OVERRIDE="" SERVER_PIDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --owner) OWNER=${2:-}; shift 2 ;;
    --out) OUT=${2:-}; shift 2 ;;
    --minutes) MINUTES=${2:-}; shift 2 ;;
    --reason) REASON=${2:-}; shift 2 ;;
    --server-pid) SERVER_PIDS+=("${2:-}"); shift 2 ;;
    --server-bin) SERVER_BIN=${2:-}; shift 2 ;;
    --model) MODEL=${2:-}; shift 2 ;;
    --omega-pin) OMEGA_PIN=${2:-}; shift 2 ;;
    --allow-contended) OVERRIDE=${2:-}; [ -n "$OVERRIDE" ] || die "--allow-contended needs a reason"; shift 2 ;;
    --) shift; break ;;
    *) die "unknown argument $1 (usage: --owner ID --out DIR [options] -- CMD...)" ;;
  esac
done
[ $# -gt 0 ] || die "no command after --"
[ -n "$OWNER" ] || die "--owner is required"
[ -n "$OUT" ] || die "--out is required"
case "$MINUTES" in '' | *[!0-9]*) die "--minutes must be 1 to 20" ;; esac
[ "$MINUTES" -ge 1 ] && [ "$MINUTES" -le 20 ] || die "--minutes must be 1 to 20"
for p in ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"}; do
  case "$p" in '' | *[!0-9]*) die "--server-pid must be a number" ;; esac
done
command -v jq >/dev/null 2>&1 || die "jq not found"
command -v "$QL" >/dev/null 2>&1 || die "quietlock not found ($QL)"
mkdir -p "$OUT" || die "cannot create $OUT"
[ ! -e "$OUT/manifest.json" ] || die "$OUT/manifest.json exists: use a new --out"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
allowed_json=$(printf '%s\n' ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"} | jq -R -s -c 'split("\n") | map(select(length > 0) | tonumber)')
CMD_JSON=$(jq -n -c '$ARGS.positional' --args -- "$@")

# gpu_sample: one JSON object with the raw GPU line, the compute processes and
# the foreign ones (not a declared server). available=false when nvidia-smi
# is missing or fails; that never counts as idle.
gpu_sample() {
  local gpu apps
  if ! gpu=$("$SMI" --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null) ||
    ! apps=$("$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null); then
    jq -n -c --arg t "$(now)" '{at: $t, available: false, gpu: null, compute_apps: [], foreign: []}'
    return
  fi
  jq -n -c --arg t "$(now)" --arg gpu "$gpu" --arg apps "$apps" --argjson allowed "$allowed_json" '
    ($apps | split("\n") | map(select(test("\\S")) | split(", ")
      | {pid: (.[0] | tonumber? // .[0]), name: .[1], used_memory: .[2]})) as $a
    | {at: $t, available: true, gpu: $gpu, compute_apps: $a,
       foreign: [$a[] | select(.pid as $p | $allowed | index($p) | not)]}'
}

listeners() {
  local out
  if out=$("$SS" -ltnpH 2>/dev/null); then
    printf '%s\n' "$out" | grep -i aien | jq -R -s -c 'split("\n") | map(select(length > 0) | gsub("\\s+"; " "))'
  else
    echo null
  fi
}

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

identity() {
  local head dirty bin_sha="" model_kib=""
  head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
  if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then dirty=true; else dirty=false; fi
  if [ -n "$SERVER_BIN" ]; then [ -f "$SERVER_BIN" ] && bin_sha=$(sha256 "$SERVER_BIN"); fi
  if [ -n "$MODEL" ]; then [ -e "$MODEL" ] && model_kib=$(du -skL "$MODEL" 2>/dev/null | cut -f1); fi
  jq -n -c --arg head "$head" --argjson dirty "$dirty" --arg bin "$SERVER_BIN" --arg sha "$bin_sha" \
    --arg model "$MODEL" --arg kib "$model_kib" --arg pin "$OMEGA_PIN" --arg graph "${OWN30B_DECODE_GRAPH-}" '
    {repo_head: $head, repo_dirty: $dirty,
     server_bin: (if $bin == "" then null else {path: $bin, sha256: (if $sha == "" then null else $sha end)} end),
     model: (if $model == "" then null else {path: $model, kib: ($kib | tonumber? // null)} end),
     omega_pin: (if $pin == "" then null else $pin end),
     own30b_decode_graph: (if $graph == "" then null else $graph end)}'
}

STARTED=$(now)
PRE=$(gpu_sample)
LISTEN=$(listeners)
IDENT=$(identity)
REASONS='[]'
add_reason() { REASONS=$(jq -c --arg r "$1" '. + [$r]' <<<"$REASONS"); }
if [ "$(jq -r .available <<<"$PRE")" != true ]; then
  add_reason "preflight: nvidia-smi unavailable, GPU idleness not established"
fi
if [ "$(jq '.foreign | length' <<<"$PRE")" -gt 0 ]; then
  add_reason "preflight: foreign GPU compute process present: $(jq -r '[.foreign[] | "\(.pid) \(.name)"] | join(", ")' <<<"$PRE")"
fi

POST=null RC=0 OUTCOME=""
write_manifest() {
  jq -n --arg owner "$OWNER" --arg reason "$REASON" --argjson minutes "$MINUTES" \
    --arg started "$STARTED" --arg ended "$(now)" --argjson rc "$RC" --arg outcome "$OUTCOME" \
    --argjson reasons "$REASONS" --arg override "$OVERRIDE" --argjson pre "$PRE" --argjson post "$POST" \
    --argjson listeners "$LISTEN" --argjson identity "$IDENT" --argjson allowed "$allowed_json" --argjson cmd "$CMD_JSON" '
    {schema: "own30b-quiet-run/v1", owner: $owner, reason: $reason, hold_minutes: $minutes,
     command: $cmd, started: $started, ended: $ended, exit_status: $rc, outcome: $outcome,
     comparable: ($reasons | length == 0), not_comparable_reasons: $reasons,
     override: (if $override == "" then null else {reason: $override} end),
     server_pids: $allowed, identity: $identity, preflight: ($pre + {aien_listeners: $listeners}),
     postflight: $post,
     note: "File locks coordinate cooperating workloads only; a GPU process that ignores the quiet flag is detected by the preflight and postflight samples, never excluded."}' \
    >"$OUT/manifest.json"
}

if [ "$(jq length <<<"$REASONS")" -gt 0 ] && [ -z "$OVERRIDE" ]; then
  RC=3 OUTCOME="refused: GPU not idle"
  write_manifest
  echo "own30b-quiet-run: refused, GPU not idle (see $OUT/manifest.json); --allow-contended REASON runs it as not comparable" >&2
  exit 3
fi
[ -z "$OVERRIDE" ] || add_reason "override: $OVERRIDE"

if ! "$QL" check >/dev/null 2>&1; then
  RC=75 OUTCOME="refused: quiet flag held by someone else"
  write_manifest
  echo "own30b-quiet-run: quiet flag already held; nothing run" >&2
  exit 75
fi

# quietlock runs in the background so a signal to this script reaches it at
# once. A non-interactive shell starts background jobs with SIGINT ignored, so
# every stop request is passed on as SIGTERM, which quietlock forwards to the
# held command's process group before it releases the flag.
SIG=""
"$QL" hold --owner "$OWNER" --minutes "$MINUTES" --reason "$REASON" -- "$@" &
QPID=$!
on_signal() { SIG=$1; kill -TERM "$QPID" 2>/dev/null; }
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP
while :; do
  wait "$QPID"; RC=$?
  kill -0 "$QPID" 2>/dev/null || break
done
trap - INT TERM HUP

POST=$(gpu_sample)
if [ "$(jq '.foreign | length' <<<"$POST")" -gt 0 ]; then
  add_reason "postflight: foreign GPU compute process present: $(jq -r '[.foreign[] | "\(.pid) \(.name)"] | join(", ")' <<<"$POST")"
fi
if [ -n "$SIG" ]; then
  OUTCOME="interrupted: SIG$SIG"
  add_reason "interrupted by SIG$SIG"
  case "$SIG" in INT) RC=130 ;; TERM) RC=143 ;; HUP) RC=129 ;; esac
elif [ "$RC" -eq 0 ]; then
  OUTCOME="completed"
else
  OUTCOME="exit $RC"
fi
write_manifest
echo "own30b-quiet-run: $OUTCOME; comparable=$(jq -r .comparable "$OUT/manifest.json"); manifest $OUT/manifest.json" >&2
exit "$RC"
