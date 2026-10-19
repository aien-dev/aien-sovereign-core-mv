#!/usr/bin/env bash
# Tests tools/serve-27b.sh without loading a model: a stand-in server binary that either refuses to
# load, comes up, or hangs in the GPU-lock queue. Needs /dev/nvidia* (the script checks it) and
# touches no port, no model and no chip. Checks: refusal exits 1 with the reason, ready exits 0,
# a hang exits 2 at READY_TIMEOUT, --no-wait returns at once, and the memory figure is 49,497 MiB.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/tools/serve-27b.sh
T=$(mktemp -d)
trap 'for p in "$T"/*.pid; do [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null; done; rm -rf "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }
compgen -G '/dev/nvidia*' >/dev/null || { echo "SKIP: no /dev/nvidia* device files"; exit 0; }

mkdir "$T/model"; echo '{}' > "$T/model/config.json"; echo '{}' > "$T/model/tokenizer.json"
cat > "$T/fake" <<'EOS'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then echo "omega gpu engine: native"; echo "qwen38 engine: native"; exit 0; fi
case "$FAKE" in
  refuse) echo "aien-serve-http: qwen38 load failed: needs 41305 MiB plus 8192 MiB kept free (49497 MiB in all), MemAvailable 6144 MiB after taking the GPU lock; refusing" >&2; exit 1 ;;
  ready)  echo "aien-serve-http: serving qwen3.8-27b-fp8 at http://127.0.0.1:1/v1 (engine omega)" >&2; exec sleep 30 ;;
  hang)   echo "aien-serve-http: loading Qwen3.8 (qwen3_5) onto the Omega engine" >&2; exec sleep 30 ;;
esac
EOS
chmod +x "$T/fake"
run() { # name, extra args...; sets rc and out
    local name=$1; shift
    out=$(FAKE=$name BIN=$T/fake MODEL_DIR=$T/model PORT=1 STATE_DIR=$T/$name LOG=$T/$name.log "$S" "$@" 2>&1); rc=$?
    [ -f "$T/$name/serve-27b.pid" ] && cp "$T/$name/serve-27b.pid" "$T/$name.pid"
}

run refuse
[ "$rc" -eq 1 ] || fail "refused load must exit 1, got $rc: $out"
grep -q 'MemAvailable 6144 MiB after taking the GPU lock' <<<"$out" || fail "refusal reason not printed: $out"
grep -q 'did not start' <<<"$out" || fail "no plain 'did not start' line: $out"
grep -q 'started' <<<"$out" && fail "refused load still printed 'started': $out"
[ ! -e "$T/refuse/serve-27b.pid" ] || fail "pid file left behind after a refusal"
echo "ok: refusal exits 1 and says why"

run ready
[ "$rc" -eq 0 ] || fail "ready must exit 0, got $rc: $out"
grep -q 'serve-27b: ready, pid' <<<"$out" || fail "no ready line: $out"
echo "ok: ready exits 0"

READY_TIMEOUT=3 run hang
[ "$rc" -eq 2 ] || fail "a hang past READY_TIMEOUT must exit 2, got $rc: $out"
grep -q 'not ready after 3 s' <<<"$out" || fail "timeout text missing: $out"
echo "ok: hang exits 2 at the timeout"

kill "$(cat "$T/hang.pid")"; sleep 0.3
run hang --no-wait
[ "$rc" -eq 0 ] || fail "--no-wait must exit 0, got $rc: $out"
grep -q 'started, not waited for' <<<"$out" || fail "--no-wait text missing: $out"
echo "ok: --no-wait returns at once"

grep -q '^NEED_MIB=49497$' "$S" || fail "script memory figure is not 49497 MiB (41305 + 8192)"
grep -q 'KEEP_FREE: u64 = 8 << 30' "$ROOT/crates/aien-serve-http/src/qwen38_engine.rs" || fail "loader keep-free changed; update the docs and this test"
echo "ok: memory figure"
echo "PASS"
