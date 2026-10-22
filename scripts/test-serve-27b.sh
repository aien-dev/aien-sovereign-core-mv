#!/usr/bin/env bash
# Fixture tests for tools/serve-27b.sh. These use a tiny HTTP server and never load a model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aien-test-serve-27b.XXXXXX")"
PORT=18164
STATE="$WORK/state"
MODEL="$WORK/model"
FIXTURE="$WORK/fixture-server"
SCRIPT=${SERVE_SCRIPT:-$ROOT/tools/serve-27b.sh}
mkdir -p "$STATE" "$MODEL"
printf '{}\n' > "$MODEL/config.json"
printf '{}\n' > "$MODEL/tokenizer.json"
cat > "$FIXTURE" <<'PY'
#!/usr/bin/env python3
import http.server
import json
import os
import sys

if sys.argv[1:] == ["--version"]:
    print("aien-serve-http fixture")
    print("omega gpu engine: native")
    print("qwen38 engine: native")
    raise SystemExit(0)
mode = os.environ["FIXTURE_MODE"]
if mode in ("early-nonzero", "early-zero"):
    print("fixture refusal before API readiness", file=sys.stderr, flush=True)
    raise SystemExit(23 if mode == "early-nonzero" else 0)
port = int(sys.argv[sys.argv.index("--port") + 1])
model_id = "fixture-wrong-model" if mode in ("wrong-model", "never") else "qwen3.8-27b-fp8"

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps({"object": "list", "data": [{"id": model_id}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *_):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
chmod +x "$FIXTURE"
cleanup() {
    if [[ -f "$STATE/serve-27b.pid" ]]; then
        pid=$(cat "$STATE/serve-27b.pid")
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        [[ "$pgid" == "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
base_env=("BIN=$FIXTURE" "MODEL_DIR=$MODEL" "PORT=$PORT" "STATE_DIR=$STATE" "LOG=$WORK/server.log")
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then fail "fixture test port $PORT is already in use"; fi

run_mode() {
    local mode=$1 timeout_s=$2
    shift 2
    env "${base_env[@]}" "FIXTURE_MODE=$mode" "READY_TIMEOUT_SECONDS=$timeout_s" "$SCRIPT" "$@"
}

# RED reproduction on the assigned base revision: the old launcher returns success after
# detaching even though the fixture immediately refuses to start.
git -C "$ROOT" show 56d42869b90a1b297a7faafa27b4502c1a104425:tools/serve-27b.sh > "$WORK/baseline-serve-27b.sh"
chmod +x "$WORK/baseline-serve-27b.sh"
status=0
env "${base_env[@]}" FIXTURE_MODE=early-nonzero "$WORK/baseline-serve-27b.sh" >"$WORK/baseline-red.out" 2>&1 || status=$?
[[ $status -eq 0 ]] || fail "base revision did not reproduce false-success launch"
grep -q 'serve-27b: started, pid' "$WORK/baseline-red.out" || fail "base revision omitted false started message"
for _ in {1..20}; do grep -q 'fixture refusal before API readiness' "$WORK/server.log" && break; sleep 0.1; done
grep -q 'fixture refusal before API readiness' "$WORK/server.log" || fail "base fixture did not record its refusal"
echo "RED reproduction fixture: base helper returned 0 and printed started after child refusal"
rm -f "$STATE/serve-27b.pid" "$WORK/server.log"

status=0
run_mode early-nonzero 4 >"$WORK/early-nonzero.out" 2>&1 || status=$?
[[ $status -ne 0 ]] || fail "early nonzero refusal returned success"
grep -q 'status 23' "$WORK/early-nonzero.out" || fail "early refusal did not report child status 23"
grep -q 'fixture refusal before API readiness' "$WORK/early-nonzero.out" || fail "early refusal omitted child log"
[[ ! -e "$STATE/serve-27b.pid" ]] || fail "early refusal left a PID file"
echo "PASS fixture: early child refusal returns nonzero, status 23, and log"

status=0
run_mode early-zero 4 >"$WORK/early-zero.out" 2>&1 || status=$?
[[ $status -ne 0 ]] || fail "exit-zero without readiness returned success"
grep -q 'status 0' "$WORK/early-zero.out" || fail "exit-zero failure did not report status 0"
echo "PASS fixture: child exit 0 without API readiness returns nonzero"

status=0
run_mode never 2 >"$WORK/timeout.out" 2>&1 || status=$?
[[ $status -ne 0 ]] || fail "never-ready fixture returned success"
grep -q 'API readiness timed out after 2s' "$WORK/timeout.out" || fail "never-ready fixture did not report bounded timeout"
grep -q 'expected model qwen3.8-27b-fp8' "$WORK/timeout.out" || fail "timeout omitted expected API identity"
[[ ! -e "$STATE/serve-27b.pid" ]] || fail "timeout left a PID file"
! ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN || fail "timeout did not clean up its fixture server"
echo "PASS fixture: bounded API timeout reports identity and cleans up owned process"

status=0
run_mode ready 5 >"$WORK/ready.out" 2>&1 || status=$?
[[ $status -eq 0 ]] || fail "ready API fixture failed: $(cat "$WORK/ready.out")"
grep -q 'serve-27b: ready' "$WORK/ready.out" || fail "ready API fixture did not report ready"
pid=$(cat "$STATE/serve-27b.pid")
kill -0 "$pid" 2>/dev/null || fail "ready fixture PID is not alive"
curl --fail --silent "http://127.0.0.1:$PORT/v1/models" | grep -q 'qwen3.8-27b-fp8' || fail "fixture API identity is not reachable"
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
[[ "$pgid" == "$pid" ]] || fail "ready fixture is not in its owned session"
kill -TERM "$pid"
for _ in {1..50}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
! ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN || fail "TERM on the recorded PID did not stop the server child"
rm -f "$STATE/serve-27b.pid"
echo "PASS fixture: advertised API succeeds and TERM on the recorded PID stops its owned server"

sleep 20 &
unrelated=$!
printf '%s\n' "$unrelated" > "$STATE/serve-27b.pid"
status=0
run_mode early-nonzero 2 >"$WORK/ownership.out" 2>&1 || status=$?
[[ $status -ne 0 ]] || fail "existing PID owner was accepted"
kill -0 "$unrelated" 2>/dev/null || fail "script signaled unrelated PID from PID file"
kill "$unrelated"; wait "$unrelated" 2>/dev/null || true
rm -f "$STATE/serve-27b.pid"
echo "PASS fixture: existing unrelated PID is not signaled"

run_mode ready 2 >"$WORK/foreground.out" 2>&1 --foreground &
foreground_pid=$!
sleep 0.3
kill -0 "$foreground_pid" 2>/dev/null || fail "foreground mode did not retain the caller process"
[[ ! -e "$STATE/serve-27b.pid" ]] || fail "foreground mode wrote a detached PID file"
kill "$foreground_pid" 2>/dev/null || true
wait "$foreground_pid" 2>/dev/null || true
echo "PASS fixture: foreground mode remains attached and writes no detached PID file"
echo "Fixture cases passed: 6. No real model start or GPU operation was performed."
