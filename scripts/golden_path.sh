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

if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$BIN" ]; then
    echo "== Building release binary =="
    # Step 1 of the gate is "clean checkout, one command": scripts/pins.sh --gpu writes
    # target/pins/env.sh so the real omega libraries link (sovereign-core #92). Without it
    # the build carries the CPU stubs and step 2 (GB10) cannot pass.
    if [ -f "$REPO_ROOT/target/pins/env.sh" ]; then
        # shellcheck disable=SC1091
        . "$REPO_ROOT/target/pins/env.sh"
    else
        echo "WARNING: no target/pins/env.sh (run scripts/pins.sh --gpu first); building with the CPU stubs"
    fi#!/usr/bin/env bash
# AIEN 0.1 Native Golden Path acceptance harness.
# Executes the 10-step release gate from docs/NATIVE_GOLDEN_PATH.md against a
# freshly booted daemon and prints one verdict line per step.
#
# Usage:
#   bash scripts/golden_path.sh [--skip-build]
#
# Environment:
#   AIEN_GP_STRICT=1        any PARTIAL step fails the gate (default: PARTIAL warns)
#   AIEN_RUNTIME_SOCK       daemon socket (default /tmp/aien-runtime.sock)
#   AIEN_CORTEX_URL         Cortex base URL (default http://127.0.0.1:18080)
#
# Exit codes: 0 all steps PASS, 1 at least one FAIL, 2 only PARTIAL (non strict).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/target/release/aien-cli"
SOCK="${AIEN_RUNTIME_SOCK:-/tmp/aien-runtime.sock}"
CORTEX_URL="${AIEN_CORTEX_URL:-http://127.0.0.1:18080}"
STRICT="${AIEN_GP_STRICT:-0}"
SKIP_BUILD=0
[ "${1:-}" = "--skip-build" ] && SKIP_BUILD=1

STATE_DIR="$(mktemp -d /tmp/aien-golden-ops.XXXXXX)"
DAEMON_LOG="$(mktemp /tmp/aien-golden-daemon.XXXXXX.log)"
export AIEN_RUNTIME_STATE_DIR="$STATE_DIR"

PASS=0
PARTIAL=0
FAIL=0
declare -a VERDICTS

record() { # record PASS|PARTIAL|FAIL "step" "detail"
    local status="$1" step="$2" detail="$3"
    printf '%-8s %-46s %s\n' "$status" "$step" "$detail"
    VERDICTS+=("$status $step $detail")
    case "$status" in
        PASS) PASS=$((PASS + 1)) ;;
        PARTIAL) PARTIAL=$((PARTIAL + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
    esac
}

# ---------------------------------------------------------------------------
# IPC helper: one JSON-line ControlEnvelope over the runtime socket.
# ---------------------------------------------------------------------------
for tool in socat jq; do command -v "$tool" >/dev/null 2>&1 || { echo "golden_path.sh needs $tool (no Python at run time)" >&2; exit 1; }; done
ipc() { # ipc '<envelope-json>' ; prints response line (socat, no Python)
    local resp
    resp="$(printf '%s\n' "$1" | timeout 125 socat -t 120 - UNIX-CONNECT:"$SOCK" 2>/dev/null | head -n1)"
    if [[ -z "$resp" ]]; then
        printf '{"Error": "no response from %s"}\n' "$SOCK"
    else
        printf '%s\n' "$resp"
    fi
}

status_json() {
    ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$(date +%s%N),\"operator_session\":1,\"command\":\"GetRuntimeStatus\"}"
}

field() { # field '<json>' '<key>' ; unwraps the Status envelope when present
    jq -r --arg k "$2" '(.Status // .) | (.[$k] // "")' <<<"$1" 2>/dev/null
}

status_field() { # status_field '<key>'
    field "$(status_json)" "$1"
}

wait_socket() { # wait for the daemon to answer status queries
    for _ in $(seq 1 100); do
        resp="$(status_json 2>/dev/null)"
        if [[ "$resp" == *'"Status('* ]] || [[ "$resp" == *'active_sequences'* ]]; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

stop_daemon() {
    ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$(date +%s%N),\"operator_session\":1,\"command\":\"Shutdown\"}" >/dev/null 2>&1
    for _ in $(seq 1 50); do
        [ -S "$SOCK" ] || return 0
        sleep 0.2
    done
    pkill -f 'aien-cli.*(daemon|--daemon)' 2>/dev/null
    sleep 0.5
    rm -f "$SOCK"
}

boot_daemon() {
    rm -f "$SOCK"
    "$BIN" daemon >"$DAEMON_LOG" 2>&1 &
    DAEMON_PID=$!
    wait_socket
}

trap 'stop_daemon' EXIT

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$BIN" ]; then
    echo "== Building release binary =="
    (cargo build --release -p aien-cli) || {
        echo "FATAL: release build failed"
        exit 1
    }
fi
[ -x "$BIN" ] || { echo "FATAL: $BIN not found"; exit 1; }

echo "== AIEN 0.1 Native Golden Path =="
echo "socket:     $SOCK"
echo "state dir:  $STATE_DIR"
echo "daemon log: $DAEMON_LOG"
echo

# sc#328: the daemon requires the approval desk key by default; create it in
# this run's compose home before the first boot.
"$BIN" compose desk-key --create 1 >/dev/null || { echo "FATAL: cannot create the approval desk key" >&2; exit 1; }

# Step 1: boot from clean state with one command, never the mock backend.
stop_daemon
if boot_daemon; then
    backend_line="$(grep -m1 'Backend:' "$DAEMON_LOG" || true)"
    if [[ -n "$backend_line" ]] && ! grep -q 'Mock' "$DAEMON_LOG"; then
        record PASS "1 boot clean checkout, native backend" "$backend_line"
    else
        record FAIL "1 boot clean checkout, native backend" "backend line missing or mock present: $(grep -m2 -E 'Backend|Model|error' "$DAEMON_LOG" | tr '\n' ' ')"
    fi
else
    record FAIL "1 boot clean checkout, native backend" "daemon did not answer on $SOCK; log: $DAEMON_LOG"
    print_summary() { :; }
    printf '\nFATAL: cannot continue without a daemon.\n'
    exit 1
fi

# Step 2: real model loaded, active path printed.
model_line="$(grep -m1 'Model:' "$DAEMON_LOG" || true)"
if [[ "$model_line" == *"checkpoint loaded"* ]]; then
    record PASS "2 real model on GB10" "$model_line"
elif [[ "$model_line" == *"reference fallback"* ]]; then
    if [ "$STRICT" = "1" ]; then
        record FAIL "2 real model on GB10" "$model_line"
    else
        record PARTIAL "2 real model on GB10" "$model_line"
    fi
else
    record FAIL "2 real model on GB10" "no Model line in daemon log"
fi

# Step 3: operator prompt retrieves Cortex memory over an authenticated call.
CORTEX_TOKEN="$(atlas-vault get CORTEX_TOKEN 2>/dev/null | tr -d '[:space:]')"
    if [ -n "$CORTEX_TOKEN" ]; then
    cortex_http="$(curl -s -o /dev/null -w '%{http_code}' -G "$CORTEX_URL/api/cortex/search" \
        -H "Authorization: Bearer $CORTEX_TOKEN" --data-urlencode 'q=sovereign runtime golden path' \
        --data-urlencode 'space=atlas-memory' --data-urlencode 'limit=1')"
    if [ "$cortex_http" = "200" ]; then
        record PASS "3 Cortex retrieval for operator prompt" "GET $CORTEX_URL/api/cortex/search -> 200 with vault bearer token"
    else
        record PARTIAL "3 Cortex retrieval for operator prompt" "GET $CORTEX_URL/api/cortex/search -> HTTP $cortex_http"
    fi
else
    record PARTIAL "3 Cortex retrieval for operator prompt" "CORTEX_TOKEN unavailable from atlas-vault; Cortex call skipped"
fi

# Step 4: branch agents with zero-copy KV sharing.
prompt_tokens="$(jq -cn '"golden path swarm probe" | explode' 2>/dev/null || echo '[103,108,9]')"
launch_resp="$(ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$(date +%s%N),\"operator_session\":1,\"command\":{\"LaunchSwarm\":{\"model_handle\":1,\"branch_count\":4,\"max_active_sequences\":8,\"max_tokens_per_branch\":8,\"root_world_id\":0,\"priority\":1,\"prompt_tokens\":$prompt_tokens}}}")"
swarm_id="$(jq -r '.SwarmAccepted.swarm_id // ""' <<<"$launch_resp" 2>/dev/null)"
if [ -n "$swarm_id" ]; then
    # Branches may complete between launch and the first status query, so the
    # fork proof is the copy-on-write fault count, not a transient snapshot.
    swarms="$(status_field 'active_swarms')"
    worlds="$(status_field 'active_worlds')"
    cow="$(status_field 'cow_faults')"
    if [ "${cow:-0}" -gt 0 ] 2>/dev/null; then
        record PASS "4 branch swarm, zero-copy KV fork" "swarm $swarm_id accepted, cow_faults=$cow (post-flight snapshot: swarms=$swarms worlds=$worlds)"
    else
        record FAIL "4 branch swarm, zero-copy KV fork" "swarm $swarm_id accepted but cow_faults=$cow (expected >0; swarms=$swarms worlds=$worlds)"
    fi
else
    record FAIL "4 branch swarm, zero-copy KV fork" "launch response: $launch_resp"
    swarm_id=""
fi

# Step 5: tokens stream end to end until the arena drains.
drained=""
for _ in $(seq 1 120); do
    seqs="$(status_field 'active_sequences')"
    if [ "${seqs:-1}" = "0" ]; then
        drained=1
        break
    fi
    sleep 0.5
done
if [ -n "$drained" ]; then
    cow="$(status_field 'cow_faults')"
    detail="active_sequences drained to 0, cow_faults=$cow (backend: $(grep -m1 'Backend: Native' "$DAEMON_LOG" | sed 's/.*Backend: //'))"
    if [ "${cow:-0}" -gt 0 ] 2>/dev/null; then
        record PASS "5 tokens streamed through scheduler+KV+backend" "$detail"
    else
        record PARTIAL "5 tokens streamed through scheduler+KV+backend" "$detail (drained but no copy-on-write faults recorded)"
    fi
else
    record FAIL "5 tokens streamed through scheduler+KV+backend" "active_sequences still $(status_field 'active_sequences') after 60s"
fi

# Step 6: the actual CLI policy membrane allows a safe effect to execute.
if cargo test -q -p aien-cli --bin aien-cli tools::tests::test_policy_approved_write_reaches_effect_handler --offline >/dev/null 2>&1; then
    record PASS "6 policy approved effect reaches boundary" "policy-approved write reached the file effect handler"
else
    record FAIL "6 policy approved effect reaches boundary" "approved effect-boundary test failed"
fi

# Step 7: the same CLI membrane denies an effect before execution.
if cargo test -q -p aien-cli --bin aien-cli tools::tests::test_path_confinement_in_tool_dispatch --offline >/dev/null 2>&1; then
    record PASS "7 denied effect blocked before execution" "policy denial test passed before the handler ran"
else
    record FAIL "7 denied effect blocked before execution" "policy denial test failed"
fi

# Step 8: world effect commits carry a SHA-256 over canonical bytes; the
# runtime suite exercises commit_draft and the canonical hashing path.
if (cargo test -q -p aien-runtime world --offline >/dev/null 2>&1 \
    && cargo test -q -p aien-cli --bin aien-cli tools::tests::effect_receipt_is_hashed_and_secret_free --offline >/dev/null 2>&1); then
    record PASS "8 world commit SHA-256 provenance" "World and secret-free effect receipt tests passed"
else
    record FAIL "8 world commit SHA-256 provenance" "cargo test -p aien-runtime failed"
fi

# Step 9: idempotency ledger survives restart; replayed operation is rejected.
cancel_op="$(date +%s%N)"
if [ -n "$swarm_id" ]; then
    cancel_resp="$(ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$cancel_op,\"operator_session\":1,\"command\":{\"CancelSwarm\":$swarm_id}}")"
else
    cancel_op="$(date +%s%N)"
    cancel_resp="$(ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$cancel_op,\"operator_session\":1,\"command\":{\"CancelSwarm\":999999}}")"
fi
stop_daemon
if boot_daemon; then
    replay_resp="$(ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$cancel_op,\"operator_session\":1,\"command\":{\"CancelSwarm\":${swarm_id:-999999}}}")"
    if [[ "$replay_resp" == *"already processed"* ]]; then
        record PASS "9 idempotency survives restart" "replayed op $cancel_op rejected: $(jq -r '(.Error // "")[:80]' <<<"$replay_resp" 2>/dev/null)"
    else
        record FAIL "9 idempotency survives restart" "replay response: $replay_resp"
    fi
else
    record FAIL "9 idempotency survives restart" "daemon did not come back after restart"
fi

# Step 10: shut down with zero leaked KV blocks and zero retained branch worlds.
pre_free="$(status_field 'free_kv_blocks')"
pre_total="$(status_field 'total_kv_blocks')"
pre_worlds="$(status_field 'active_worlds')"
if [ "${pre_free:-0}" = "${pre_total:-1}" ] && [ "${pre_worlds:-9}" -le 1 ] 2>/dev/null; then
    record PASS "10 zero leaks at shutdown" "free=$pre_free/$pre_total kv blocks, active_worlds=$pre_worlds"
else
    record FAIL "10 zero leaks at shutdown" "free=$pre_free/$pre_total kv blocks, active_worlds=$pre_worlds"
fi

if ipc "{\"protocol_version\":1,\"request_id\":$RANDOM,\"operation_id\":$(date +%s%N),\"operator_session\":1,\"command\":\"Shutdown\"}" | grep -q 'ShutdownAck'; then
    :
else
    record PARTIAL "10 zero leaks at shutdown" "daemon did not acknowledge Shutdown"
fi

echo
echo "== Summary: $PASS pass, $PARTIAL partial, $FAIL fail =="
if [ "$FAIL" -gt 0 ]; then
    exit 1
elif [ "$PARTIAL" -gt 0 ] && [ "$STRICT" = "1" ]; then
    exit 1
elif [ "$PARTIAL" -gt 0 ]; then
    exit 2
fi
exit 0
