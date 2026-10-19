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
#   MAX_CONTEXT, MAX_TOKENS_CAP         default 131072 and 16384
# Exit: 0 started, 1 a check failed (nothing started).
set -euo pipefail

BIN=${BIN:-$HOME/.local/bin/aien-serve-http}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-fp8-017b9c7a}
PORT=${PORT:-18091}
STATE_DIR=${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/aien}
LOG=${LOG:-$STATE_DIR/serve-27b.log}
PIDFILE=$STATE_DIR/serve-27b.pid
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
setsid nohup "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
cat <<EOF
serve-27b: started, pid $(cat "$PIDFILE")
  log:     $LOG
  loading takes about one to two minutes. It is ready when this lists qwen3.8-27b-fp8:
           curl -s http://127.0.0.1:$PORT/v1/models
  chat:    curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
             -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
  stop:    kill -TERM \$(cat $PIDFILE)
           (it finishes the reply in progress, then exits; that can take a few minutes)
EOF
