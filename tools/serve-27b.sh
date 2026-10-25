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

# Start the Qwen3.8-27B model server (aien-serve-http on AIEN's own Omega GPU engine) in the
# background, with the flags the project's own live server uses. Text chat only, one request at a
# time, OpenAI-compatible API on http://127.0.0.1:$PORT/v1.
#
# Usage: serve-27b.sh [--foreground] [--no-wait]
#   --foreground  run in this terminal (for systemd or a supervisor) instead of detaching
#   --no-wait     detach and return at once, without waiting for the model to load (the exit
#                 code then says nothing about the load; check the log)
# Settings (environment variables, all optional):
#   BIN        server binary            default $HOME/.local/bin/aien-serve-http
#   MODEL_DIR  Qwen3.8-27B FP8 folder   default $HOME/models/qwen3.8-27b-fp8-017b9c7a
#   PORT       TCP port on 127.0.0.1    default 18091
#   STATE_DIR  log and pid file folder  default ${XDG_STATE_HOME:-$HOME/.local/state}/aien
#   LOG        server log               default $STATE_DIR/serve-27b.log
#   MAX_CONTEXT, MAX_TOKENS_CAP         default 131072 and 16384
#   READY_TIMEOUT  seconds to wait for "ready"   default 2400 (lock queue 1800 + load)
# Memory: the loader needs 49,497 MiB (48.3 GiB) of MemAvailable, checked after it holds the GPU lock.
# Ready: http://127.0.0.1:$PORT/v1/models answers and lists qwen3.8-27b-fp8 (needs curl).
# Exit: 0 ready (or started, with --no-wait), 1 a check failed or the server exited without
#       serving, whatever its exit status (for example the loader refused for lack of memory;
#       the status and the reason are printed), 2 not ready when READY_TIMEOUT ran out (the
#       server keeps running).
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
WAIT=1
READY_TIMEOUT=${READY_TIMEOUT:-2400}
for a in "$@"; do
    case "$a" in
        --foreground) FOREGROUND=1 ;;
        --no-wait) WAIT=0 ;;
        -h|--help) sed -n "17,38p" "$0"; exit 0 ;;
        *) echo "serve-27b: unknown argument $a (try --help)" >&2; exit 1 ;;
    esac
done

fail() { echo "serve-27b: $*" >&2; exit 1; }
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "READY_TIMEOUT must be a positive number of seconds"
if [ "$FOREGROUND" -eq 0 ] && [ "$WAIT" -eq 1 ]; then
    command -v curl >/dev/null || fail "curl is needed to see when the server is ready (apt install curl), or pass --no-wait"
fi

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
# The loader needs 41,305 MiB for the model and its working memory (at the default MAX_CONTEXT)
# plus 8,192 MiB kept free for other jobs: 49,497 MiB (48.3 GiB) of MemAvailable. A server that
# starts has peaked at 38,499,160 kB (36.7 GiB) resident on the current build (VmHWM,
# evidence/qwen38-serve-memory-20261019/proc-status.txt). The loader makes the binding check itself
# once it holds the GPU lock; this early look only warns, because the lock wait can be many minutes
# and free memory moves while you wait.
NEED_MIB=49497
avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${avail_kib:-0}" -lt $((NEED_MIB * 1024)) ]; then
    echo "serve-27b: warning: only $((avail_kib / 1024)) MiB of memory available; the loader needs $NEED_MIB MiB (48.3 GiB) free. It checks again when it gets the GPU lock and refuses if still short." >&2
fi

ARGS=(--engine omega --model-dir "$MODEL_DIR" --port "$PORT"
      --max-context "$MAX_CONTEXT" --max-tokens-cap "$MAX_TOKENS_CAP" --thinking off)
echo "$(date -u +%FT%TZ) serve-27b: starting $BIN ${ARGS[*]}" >> "$LOG"
if [ "$FOREGROUND" -eq 1 ]; then
    exec "$BIN" "${ARGS[@]}"
fi
log_start=$(stat -c %s "$LOG")
setsid nohup "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
pid=$!
echo $pid > "$PIDFILE"

# A dead process (a zombie counts: this script is its parent) means the load was refused or failed.
gone() { local s; s=$(ps -o stat= -p "$pid" 2>/dev/null || true); [ -z "$s" ] || [ "${s#Z}" != "$s" ]; }
new_log() { tail -c +$((log_start + 1)) "$LOG"; }
# Ready means the API answers and advertises the 27B model, not merely that the process is alive.
api_ready() {
    local models
    models=$(curl --fail --silent --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null) \
        && grep -Eq '"id"[[:space:]]*:[[:space:]]*"qwen3\.8-27b-fp8"' <<<"$models"
}

if [ "$WAIT" -eq 1 ]; then
    echo "serve-27b: waiting for pid $pid to load the model (it may queue for the GPU lock first; log $LOG)" >&2
    t0=$SECONDS
    until api_ready; do
        if gone; then
            rc=0; wait "$pid" 2>/dev/null || rc=$?
            rm -f "$PIDFILE"
            echo "serve-27b: the server exited (status $rc) before http://127.0.0.1:$PORT/v1/models listed qwen3.8-27b-fp8; it did not start. Last log lines ($LOG):" >&2
            new_log | tail -n 5 | sed 's/^/  /' >&2
            exit 1
        fi
        if [ $((SECONDS - t0)) -ge "$READY_TIMEOUT" ]; then
            echo "serve-27b: not ready after $READY_TIMEOUT s: http://127.0.0.1:$PORT/v1/models does not list qwen3.8-27b-fp8 yet; pid $pid is still running (likely queued for the GPU lock). Watch: tail -f $LOG" >&2
            exit 2
        fi
        [ $(((SECONDS - t0) % 60)) -eq 0 ] && [ $((SECONDS - t0)) -gt 0 ] && echo "serve-27b: still loading after $((SECONDS - t0)) s" >&2
        sleep 1
    done
fi
cat <<EOT
serve-27b: $([ "$WAIT" -eq 1 ] && echo "ready" || echo "started, not waited for"), pid $pid
  log:     $LOG
  loading takes about one minute once the GPU is free; before that it queues for the GPU lock.
  check:   curl -s http://127.0.0.1:$PORT/v1/models      (lists qwen3.8-27b-fp8 when ready)
  chat:    curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
             -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
  stop:    kill -TERM \$(cat $PIDFILE)
           (it finishes the reply in progress, then exits; that can take a few minutes)
EOT
