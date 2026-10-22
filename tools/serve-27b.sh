#!/usr/bin/env bash
# Start the Qwen3.8-27B model server (aien-serve-http on AIEN's own Omega GPU engine) in the
# background, with the flags the project's own live server uses. Text chat only, one request at a
# time, OpenAI-compatible API on http://127.0.0.1:$PORT/v1.
#
# Usage: serve-27b.sh [--foreground]
#   --foreground  run in this terminal (for systemd or a supervisor) instead of detaching
# Settings (environment variables, all optional):
#   BIN        server binary            default $HOME/.local/bin/aien-serve-http
#   MODEL_DIR  Qwen3.8-27B FP8 folder   default $HOME/models/qwen3.8-27b-fp8-017b9c7a
#   PORT       TCP port on 127.0.0.1    default 18091
#   STATE_DIR  log and pid file folder  default ${XDG_STATE_HOME:-$HOME/.local/state}/aien
#   LOG        server log               default $STATE_DIR/serve-27b.log
#   READY_TIMEOUT_SECONDS  API readiness limit, default 180 seconds
#   MAX_CONTEXT, MAX_TOKENS_CAP         default 131072 and 16384
# Exit: 0 started, 1 a check failed (nothing started).
set -euo pipefail

BIN=${BIN:-$HOME/.local/bin/aien-serve-http}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-fp8-017b9c7a}
PORT=${PORT:-18091}
STATE_DIR=${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/aien}
LOG=${LOG:-$STATE_DIR/serve-27b.log}
PIDFILE=$STATE_DIR/serve-27b.pid
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-180}
MAX_CONTEXT=${MAX_CONTEXT:-131072}
MAX_TOKENS_CAP=${MAX_TOKENS_CAP:-16384}
FOREGROUND=0
for a in "$@"; do
    case "$a" in
        --foreground) FOREGROUND=1 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "serve-27b: unknown argument $a (try --help)" >&2; exit 1 ;;
    esac
done

fail() { echo "serve-27b: $*" >&2; exit 1; }

[ -x "$BIN" ] || fail "server binary not found or not executable: $BIN (install it with install.sh --with-server, or set BIN)"
[[ "$READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "READY_TIMEOUT_SECONDS must be a positive integer"
# The binary must have the real Omega engine linked. A CPU stub build would start and then fail
# every request, so refuse it here.
info=$("$BIN" --version 2>&1) || fail "$BIN --version failed; is it an aien-serve-http build from this branch or newer?
$info"
grep -q '^omega gpu engine: native' <<<"$info" && grep -q '^qwen38 engine: native' <<<"$info" \
    || fail "$BIN was built without the Omega GPU engine (CPU stub). Rebuild with install.sh --with-server.
$info"
[ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ] \
    || fail "model folder $MODEL_DIR has no config.json or tokenizer.json. Download Qwen/Qwen3.8-27B-FP8 (revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) there, or set MODEL_DIR."
compgen -G '/dev/nvidia*' >/dev/null || fail "no /dev/nvidia* device files: the NVIDIA driver is not loaded"
mkdir -p "$STATE_DIR"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    fail "a server started by this script is already running (pid $(cat "$PIDFILE")). Stop it first: kill -TERM $(cat "$PIDFILE")"
fi
if command -v ss >/dev/null && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    fail "port $PORT is already in use. Pick another one with PORT=..."
fi
# The server peaked at about 36.7 GiB of resident memory on the current build (VmHWM 38499160 kB,
# evidence/qwen38-serve-memory-20261019/proc-status.txt); warn below 40 GiB free.
avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${avail_kib:-0}" -lt $((40 * 1024 * 1024)) ]; then
    echo "serve-27b: warning: only $((avail_kib / 1024 / 1024)) GiB of memory available; the 27B server peaked at about 36.7 GiB and may be killed by the kernel" >&2
fi

ARGS=(--engine omega --model-dir "$MODEL_DIR" --port "$PORT"
      --max-context "$MAX_CONTEXT" --max-tokens-cap "$MAX_TOKENS_CAP" --thinking off)
echo "$(date -u +%FT%TZ) serve-27b: starting $BIN ${ARGS[*]}" >> "$LOG"
if [ "$FOREGROUND" -eq 1 ]; then
    exec "$BIN" "${ARGS[@]}"
fi
STATUSFILE=$STATE_DIR/serve-27b.$$.exit
rm -f "$STATUSFILE"
setsid nohup bash -c '
    log=$1
    status=$2
    shift 2
    "$@" >> "$log" 2>&1 &
    child=$!
    forward_term() { kill -TERM "$child" 2>/dev/null || true; }
    trap forward_term TERM INT
    while :; do
        wait "$child"
        rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    printf "%s\n" "$rc" > "$status"
    exit "$rc"
' _ "$LOG" "$STATUSFILE" "$BIN" "${ARGS[@]}" < /dev/null &
pid=$!
echo "$pid" > "$PIDFILE"

# Record process identity and verify it before signaling during cleanup.
proc_start_time() {
    local stat tail
    [ -r "/proc/$1/stat" ] || return 1
    stat=$(cat "/proc/$1/stat") || return 1
    tail=${stat##*) }
    set -- $tail
    printf '%s\n' "${20}"
}
start_time=$(proc_start_time "$pid" 2>/dev/null || true)
owns_process() {
    [ -n "$start_time" ] \
        && [ "$(proc_start_time "$pid" 2>/dev/null || true)" = "$start_time" ] \
        && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -Fq -- "$BIN" \
        && [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ]
}
cleanup_owned_process() {
    if owns_process; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 -- "-$pid" 2>/dev/null || break
            sleep 0.1
        done
        # The session ID remains reserved while any member of this owned group exists.
        if kill -0 -- "-$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
    fi
    if [ "$(cat "$PIDFILE" 2>/dev/null || true)" = "$pid" ]; then rm -f "$PIDFILE"; fi
}

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
    if response=$(curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null) \
        && grep -Eq '"id"[[:space:]]*:[[:space:]]*"qwen3\.8-27b-fp8"' <<<"$response"; then
        rm -f "$STATUSFILE"
        cat <<EOF
serve-27b: ready, pid $pid
  log:     $LOG
  model:   qwen3.8-27b-fp8 at http://127.0.0.1:$PORT/v1/models
  stop:    kill -TERM $(cat "$PIDFILE")
EOF
        exit 0
    fi
    if [ -f "$STATUSFILE" ]; then
        child_status=$(cat "$STATUSFILE" 2>/dev/null || echo unknown)
        tail_log=$(tail -n 20 "$LOG" 2>/dev/null || true)
        cleanup_owned_process
        fail "server process exited before API readiness (status $child_status); expected model qwen3.8-27b-fp8 at /v1/models. Recent log:\n$tail_log"
    fi
    sleep 1
done

tail_log=$(tail -n 20 "$LOG" 2>/dev/null || true)
cleanup_owned_process
fail "API readiness timed out after ${READY_TIMEOUT_SECONDS}s; expected model qwen3.8-27b-fp8 at /v1/models. Recent log:\n$tail_log"
