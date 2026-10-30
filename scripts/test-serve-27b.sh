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

# Tests tools/serve-27b.sh without loading a model: a stand-in server binary that refuses to load,
# exits 0 without serving, comes up, advertises the wrong model, or hangs in the GPU-lock queue,
# and a stand-in curl on PATH that answers /v1/models for it. Needs /dev/nvidia* (the script
# checks it) and touches no port, no model and no chip. Checks: an exited server exits 1 with its
# status and reason, ready (the API lists qwen3.8-27b-fp8) exits 0, a server that never lists the
# model exits 2 at READY_TIMEOUT and keeps running, --no-wait returns at once, an unrelated pid in
# the pid file is never signalled, --foreground stays attached, and the memory figure is 49,497 MiB.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/tools/serve-27b.sh
T=$(mktemp -d)
trap 'for p in "$T"/*.pid; do [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null; done; rm -rf "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
compgen -G '/dev/nvidia*' >/dev/null || { echo "SKIP: no /dev/nvidia* device files"; exit 0; }
pass=0
ok() { echo "ok: $*"; pass=$((pass + 1)); }

mkdir "$T/model" "$T/path"; echo '{}' > "$T/model/config.json"; echo '{}' > "$T/model/tokenizer.json"
cat > "$T/fake" <<'EOS'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then echo "omega gpu engine: native"; echo "qwen38 engine: native"; exit 0; fi
case "$FAKE" in
  refuse)     echo "aien-serve-http: qwen38 load failed: needs 41305 MiB plus 8192 MiB kept free (49497 MiB in all), MemAvailable 6144 MiB after taking the GPU lock; refusing" >&2; exit 1 ;;
  early-zero) echo "aien-serve-http: fixture exits 0 before serving" >&2; exit 0 ;;
  ready)      echo "aien-serve-http: serving qwen3.8-27b-fp8 at http://127.0.0.1:1/v1 (engine omega)" >&2; touch "$FAKE_UP"; exec sleep 30 ;;
  wrong)      echo "aien-serve-http: serving fixture-wrong-model" >&2; touch "$FAKE_UP"; exec sleep 30 ;;
  hang)       echo "aien-serve-http: loading Qwen3.8 (qwen3_5) onto the Omega engine" >&2; exec sleep 30 ;;
esac
EOS
# The stand-in API: answers /v1/models only once the stand-in server says it is up.
cat > "$T/path/curl" <<'EOS'
#!/usr/bin/env bash
[ -e "$FAKE_UP" ] || exit 7
case "$FAKE" in
  ready) echo '{"object":"list","data":[{"id":"qwen3.8-27b-fp8","object":"model"}]}' ;;
  *)     echo '{"object":"list","data":[{"id":"fixture-wrong-model","object":"model"}]}' ;;
esac
EOS
chmod +x "$T/fake" "$T/path/curl"
run() { # name, extra args...; sets rc and out
    local name=$1; shift
    out=$(PATH=$T/path:$PATH FAKE=$name FAKE_UP=$T/$name.up BIN=$T/fake MODEL_DIR=$T/model PORT=1 \
          STATE_DIR=$T/$name LOG=$T/$name.log "$S" "$@" 2>&1); rc=$?
    [ -f "$T/$name/serve-27b.pid" ] && cp "$T/$name/serve-27b.pid" "$T/$name.pid"
}

run refuse
[ "$rc" -eq 1 ] || fail "refused load must exit 1, got $rc: $out"
grep -q 'MemAvailable 6144 MiB after taking the GPU lock' <<<"$out" || fail "refusal reason not printed: $out"
grep -q 'exited (status 1)' <<<"$out" || fail "server exit status not printed: $out"
grep -q 'did not start' <<<"$out" || fail "no plain 'did not start' line: $out"
grep -q 'started' <<<"$out" && fail "refused load still printed 'started': $out"
[ ! -e "$T/refuse/serve-27b.pid" ] || fail "pid file left behind after a refusal"
ok "refusal exits 1 with the server's status and reason"

run early-zero
[ "$rc" -eq 1 ] || fail "a server that exits 0 without serving must exit 1, got $rc: $out"
grep -q 'exited (status 0)' <<<"$out" || fail "status 0 not printed: $out"
[ ! -e "$T/early-zero/serve-27b.pid" ] || fail "pid file left behind after an early exit 0"
ok "exit 0 without serving is a failure"

run ready
[ "$rc" -eq 0 ] || fail "ready must exit 0, got $rc: $out"
grep -q 'serve-27b: ready, pid' <<<"$out" || fail "no ready line: $out"
pid=$(cat "$T/ready.pid")
kill -0 "$pid" 2>/dev/null || fail "ready server pid $pid is not alive"
[ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ] || fail "ready server is not in its own session"
kill -TERM "$pid"; for _ in {1..50}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
kill -0 "$pid" 2>/dev/null && fail "TERM on the recorded pid did not stop the server"
ok "ready (API lists qwen3.8-27b-fp8) exits 0; TERM on the recorded pid stops it"

READY_TIMEOUT=3 run hang
[ "$rc" -eq 2 ] || fail "a hang past READY_TIMEOUT must exit 2, got $rc: $out"
grep -q 'not ready after 3 s' <<<"$out" || fail "timeout text missing: $out"
kill -0 "$(cat "$T/hang.pid")" 2>/dev/null || fail "timeout must leave the queued server running"
ok "hang exits 2 at the timeout and leaves the server running"

READY_TIMEOUT=2 run wrong
[ "$rc" -eq 2 ] || fail "an API without qwen3.8-27b-fp8 must not count as ready, got $rc: $out"
grep -q 'does not list qwen3.8-27b-fp8' <<<"$out" || fail "timeout does not name the expected model: $out"
ok "an API advertising another model is not ready"

kill "$(cat "$T/hang.pid")"; sleep 0.3
run hang --no-wait
[ "$rc" -eq 0 ] || fail "--no-wait must exit 0, got $rc: $out"
grep -q 'started, not waited for' <<<"$out" || fail "--no-wait text missing: $out"
ok "--no-wait returns at once"

mkdir -p "$T/owned"
sleep 20 & unrelated=$!
echo "$unrelated" > "$T/owned/serve-27b.pid"
out=$(PATH=$T/path:$PATH FAKE=refuse FAKE_UP=$T/owned.up BIN=$T/fake MODEL_DIR=$T/model PORT=1 \
      STATE_DIR=$T/owned LOG=$T/owned.log "$S" 2>&1); rc=$?
[ "$rc" -eq 1 ] || fail "a live pid in the pid file must refuse the start, got $rc: $out"
kill -0 "$unrelated" 2>/dev/null || fail "the script signalled the unrelated pid from the pid file"
kill "$unrelated"; wait "$unrelated" 2>/dev/null
ok "an unrelated live pid in the pid file is refused, not signalled"

PATH=$T/path:$PATH FAKE=ready FAKE_UP=$T/fg.up BIN=$T/fake MODEL_DIR=$T/model PORT=1 \
    STATE_DIR=$T/fg LOG=$T/fg.log "$S" --foreground >/dev/null 2>&1 &
fg=$!
for _ in {1..50}; do [ -e "$T/fg.up" ] && break; sleep 0.1; done
kill -0 "$fg" 2>/dev/null || fail "--foreground did not stay attached"
[ ! -e "$T/fg/serve-27b.pid" ] || fail "--foreground wrote a pid file"
kill "$fg"; wait "$fg" 2>/dev/null
ok "--foreground stays attached and writes no pid file"

READY_TIMEOUT=soon run refuse
[ "$rc" -eq 1 ] && grep -q 'READY_TIMEOUT must be a positive number' <<<"$out" || fail "a bad READY_TIMEOUT was accepted: $out"
ok "READY_TIMEOUT is checked"

help=$("$S" --help)
head -n 1 <<<"$help" | grep -q '^# Start the Qwen3.8-27B model server' || fail "--help does not start at the usage text: $help"
tail -n 1 <<<"$help" | grep -q 'server keeps running' || fail "--help does not end at the exit codes: $help"
ok "--help prints the usage text"

grep -q '^NEED_MIB=49497$' "$S" || fail "script memory figure is not 49497 MiB (41305 + 8192)"
grep -q 'KEEP_FREE: u64 = 8 << 30' "$ROOT/crates/aien-serve-http/src/qwen38_engine.rs" || fail "loader keep-free changed; update the docs and this test"
ok "memory figure"
echo "PASS ($pass checks)"
