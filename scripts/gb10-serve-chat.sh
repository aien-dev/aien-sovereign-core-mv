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

# SC-GB10-SERVE-CHAT (tests/gates/sc-gb10-serve-chat.gate): the GPU receipt an Alpha needs.
# Builds aien-serve-http against the real Omega GPU engine (the build install.sh --with-server
# does), starts it on the GB10 with the Qwen3.8-27B checkpoint through tools/serve-27b.sh, waits
# until the API advertises qwen3.8-27b-fp8, sends one short greedy chat request, checks the reply
# is a well-formed, non-empty chat.completion, then stops the server with SIGTERM and waits for
# its "shutdown complete" line.
#
# Run by tests/gates/run.sh gb10-serve-chat, which first materialises the pinned omega, physics
# and aienos sources and writes $PINS/env.sh (scripts/pins.sh --gpu). Arguments: <env.sh>.
#
# Chip lock: under aien-test (a gb10 gate) the runner already holds /tmp/aien-gb10.lock for
# this whole script and says so with AIEN_GB10_LOCK_HELD=1. The server would queue behind that
# lock until its load gave up, so with the marker it takes no lock, but only after verifying in
# /proc/locks that the holder is one of its ancestors; a marker nobody backs is refused (exit 1,
# so this gate FAILS rather than passes). Without the marker the server takes
# /tmp/aien-gb10.lock itself, as the live server does.
#
# NOT_RUN (exit 77 with a not_run_reason observation, aien-test verdict.rs SELF_NOT_RUN_EXIT),
# never PASS: model_missing (no checkpoint at $MODEL_DIR), memory_short (MemAvailable below the
# 49,497 MiB the loader needs, checked before the server is started; the server is not started).
# Every other problem is a FAIL. Observations: model_dir, model_dir_source (both emitted before any
# check, so a NOT_RUN names them too), build_rc, engine_native, mem_available_mib, chip_lock,
# serve_rc, reply_ok, reply_chars, completion_tokens, shutdown_clean.
#
# Settings: MODEL_DIR (an override, receipted as model_dir_source "override"; default
# $AIEN_MODELS_DIR/qwen3.8-27b-fp8-017b9c7a, "models-mount", else $HOME/models/..., "home";
# tests/gates/run.sh sets AIEN_MODELS_DIR to the sandbox home's read-only models/ mount because it
# gives each step a fresh writable HOME; tests/gates/step-home.sh gb10_model_dir, host-tested by
# scripts/test-gb10-model-dir.sh), READY_TIMEOUT (default 2400 s, as tools/serve-27b.sh),
# STOP_TIMEOUT (default 900 s).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 3
ENV_SH=${1:?usage: scripts/gb10-serve-chat.sh <pins env.sh>}
TARGET=${CARGO_TARGET_DIR:-$ROOT/target}
# shellcheck source=tests/gates/step-home.sh
. "$ROOT/tests/gates/step-home.sh" || exit 3
gb10_model_dir
MODEL_ID=qwen3.8-27b-fp8
READY_TIMEOUT=${READY_TIMEOUT:-2400}
STOP_TIMEOUT=${STOP_TIMEOUT:-900}
# The loader's own figure (tools/serve-27b.sh NEED_MIB, qwen38_engine.rs admit_load): 41,305 MiB
# for the model at the default context plus 8,192 MiB kept free.
NEED_MIB=49497

# The leading newline keeps the line whole when a tool left a partial line.
obs() { printf '\nAIEN-OBS %s %s\n' "$1" "$2"; }
not_run() { echo "gb10-serve-chat: NOT_RUN $1: $2" >&2; obs not_run_reason "\"$1\""; exit 77; }
fail() { echo "gb10-serve-chat: FAIL: $*" >&2; exit 1; }
json_str() { local s=${1//\\/\\\\}; printf '"%s"' "${s//\"/\\\"}"; }
# Model provenance goes into the receipt before any check: an override is never mistaken for the default.
obs model_dir "$(json_str "$MODEL_DIR")"
obs model_dir_source "\"$MODEL_DIR_SOURCE\""
for t in curl jq ss od tr; do command -v "$t" >/dev/null || fail "needs $t"; done
# This gate starts its own ephemeral server. Keep its credential in memory and
# inherit it only into that child; do not require operator secrets in gate env.
if [ -z "${AIEN_SERVE_HTTP_BEARER_TOKEN:-}" ]; then
    AIEN_SERVE_HTTP_BEARER_TOKEN=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n') \
        || fail "cannot generate test-server credential"
    [ -n "$AIEN_SERVE_HTTP_BEARER_TOKEN" ] || fail "empty test-server credential"
fi
export AIEN_SERVE_HTTP_BEARER_TOKEN

[ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ] \
    || not_run model_missing "no config.json or tokenizer.json under $MODEL_DIR"

# 1. Memory pre-check, before the release build: a chip that cannot hold the
# model is a NOT_RUN, and it should not cost a 5-minute GPU build to learn.
# MemAvailable counts mapped model pages as reclaimable, so beside a resident model server it
# overstates what a second load can get (2026-10-25T02:47Z: a load admitted at MemAvailable
# 53.5 GiB drove a global OOM that killed the live 35B). Count MemAvailable - Mapped instead.
avail_kib=$(awk '/^MemAvailable:/ {a=$2} /^Mapped:/ {m=$2} END {print a-m}' /proc/meminfo 2>/dev/null)
avail_mib=$(( ${avail_kib:-0} / 1024 ))
obs mem_available_mib "$avail_mib"
[ "$avail_mib" -ge "$NEED_MIB" ] \
    || not_run memory_short "MemAvailable-Mapped $avail_mib MiB, the 27B loader needs $NEED_MIB MiB (another model is resident: the GB10 needs an exclusive window); the server was not started"

# 2. Build, as install.sh --with-server does (pinned sources, release, real GPU engine).
[ -f "$ENV_SH" ] || fail "pins env file $ENV_SH is missing"
# shellcheck disable=SC1090
. "$ENV_SH"
[ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = 1 ] && [ -n "${AIEN_OMEGA_DIR:-}" ] \
    || fail "$ENV_SH does not select the GPU engine (AIEN_OMEGA_DIR unset)"
cargo build --locked --release -p aien-serve-http
rc=$?
obs build_rc "$rc"
[ "$rc" -eq 0 ] || fail "cargo build of aien-serve-http failed (rc $rc)"
BIN=$TARGET/release/aien-serve-http
info=$("$BIN" --version 2>&1) || fail "$BIN --version failed: $info"
echo "$info"
if grep -q '^omega gpu engine: native' <<<"$info" && grep -q '^qwen38 engine: native' <<<"$info"; then
    obs engine_native true
else
    obs engine_native false
    fail "the built server carries the CPU stub, not the Omega GPU engine"
fi

STATE=$(mktemp -d) || fail "mktemp failed"
pid=""
stop_server() {
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    rm -rf "$STATE"
}
trap stop_server EXIT

# 3. Chip lock (see the header). The marker is passed through untouched; the server itself
#    verifies in /proc/locks that an ancestor holds /tmp/aien-gb10.lock and refuses to start
#    otherwise (aien-serve-http chip_reservation.rs). There is no private lock and no override.
if [ "${AIEN_GB10_LOCK_HELD:-}" = 1 ]; then
    obs chip_lock '"held by the caller (AIEN_GB10_LOCK_HELD=1); the server verifies the holder is an ancestor"'
else
    obs chip_lock '"server takes /tmp/aien-gb10.lock"'
fi

# 4. Start through the project's own start script on a free loopback port (never the live 18091).
port=""
for p in $(seq 18190 18199); do
    ss -ltn "sport = :$p" 2>/dev/null | grep -q LISTEN || { port=$p; break; }
done
[ -n "$port" ] || fail "no free port in 18190-18199"
LOG=$STATE/serve-27b.log
BIN=$BIN MODEL_DIR=$MODEL_DIR PORT=$port STATE_DIR=$STATE LOG=$LOG READY_TIMEOUT=$READY_TIMEOUT \
    bash tools/serve-27b.sh
rc=$?
obs serve_rc "$rc"
[ -f "$STATE/serve-27b.pid" ] && pid=$(cat "$STATE/serve-27b.pid")
if [ "$rc" -ne 0 ]; then
    sed 's/^/  log: /' "$LOG" >&2
    if grep -q 'after taking the GPU lock (another model may be loaded); refusing' "$LOG"; then
        not_run memory_short "the loader refused after taking the GPU lock (memory fell below its need meanwhile)"
    fi
    fail "tools/serve-27b.sh exited $rc: the server did not come up serving $MODEL_ID"
fi
[ -n "$pid" ] || fail "tools/serve-27b.sh reported ready but wrote no pid file"

# 5. One short, deterministic chat request: greedy, thinking off, a few tokens.
req=$(jq -nc --arg m "$MODEL_ID" '{model:$m, messages:[{role:"user",content:"Reply with the single word: ready"}],
    temperature:0, max_tokens:16, stream:false, chat_template_kwargs:{enable_thinking:false}}')
code=$(curl -s -m 900 -o "$STATE/reply.json" -w '%{http_code}' "http://127.0.0.1:$port/v1/chat/completions" \
    -H "Authorization: Bearer $AIEN_SERVE_HTTP_BEARER_TOKEN" -H 'Content-Type: application/json' -d "$req") || code=000
echo "reply (HTTP $code):"; cat "$STATE/reply.json"; echo
if [ "$code" = 200 ] && jq -e --arg m "$MODEL_ID" '
        .object == "chat.completion" and .model == $m
        and (.choices | length) == 1
        and .choices[0].message.role == "assistant"
        and (.choices[0].message.content | type == "string" and (gsub("\\s"; "") | length) > 0)
        and (.choices[0].finish_reason | IN("stop", "length"))
        and (.usage.completion_tokens | type == "number" and . > 0)' "$STATE/reply.json" >/dev/null; then
    obs reply_ok true; reply_ok=1
else
    obs reply_ok false; reply_ok=0
fi
obs reply_chars "$(jq '.choices[0].message.content // "" | length' "$STATE/reply.json" 2>/dev/null || echo 0)"
obs completion_tokens "$(jq '.usage.completion_tokens // 0' "$STATE/reply.json" 2>/dev/null || echo 0)"

# 6. Clean shutdown: SIGTERM, the server finishes, drains and logs "shutdown complete".
kill -TERM "$pid" 2>/dev/null
t0=$SECONDS
while kill -0 "$pid" 2>/dev/null && [ $((SECONDS - t0)) -lt "$STOP_TIMEOUT" ]; do sleep 1; done
if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null
    obs shutdown_clean false
    fail "the server was still running $STOP_TIMEOUT s after SIGTERM (killed)"
fi
pid=""
if grep -q 'aien-serve-http: shutdown complete' "$LOG"; then
    obs shutdown_clean true
else
    obs shutdown_clean false
    sed 's/^/  log: /' "$LOG" >&2
    fail "the server exited after SIGTERM without logging 'shutdown complete'"
fi
[ "$reply_ok" = 1 ] || fail "the chat reply was not a well-formed, non-empty chat.completion from $MODEL_ID (HTTP $code)"
sed 's/^/  log: /' "$LOG"
echo "gb10-serve-chat: PASS ($MODEL_ID on port $port answered and shut down cleanly)"
