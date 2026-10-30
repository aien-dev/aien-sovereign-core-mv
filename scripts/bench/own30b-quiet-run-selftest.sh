#!/usr/bin/env bash
# Self-test for scripts/bench/own30b-quiet-run.sh (#625). Bash + jq only.
# Fakes quietlock, nvidia-smi, ss and flock on PATH; the "server" is a
# command that sleeps or exits. Covers: normal completion, child failure,
# SIGINT, SIGTERM, repeat invocations, contended refusal, recorded override,
# a declared server PID, a foreign process seen only after the run, a quiet
# flag already held, and that /tmp/aien-gb10.lock is never touched.
# With QUIETLOCK_REAL=/path/to/quietlock the hold and interrupt cases run
# again against the real binary in a private QUIETLOCK_DIR.
#
#   bash scripts/bench/own30b-quiet-run-selftest.sh   (exit 0 = all checks passed)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN="$HERE/own30b-quiet-run.sh"
T=$(mktemp -d "${TMPDIR:-/tmp}/own30b-quiet-run-selftest.XXXXXX")
if [ "${QR_SELFTEST_KEEP:-0}" = 1 ]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
PASS=0; FAIL=0
check() { # check <name> <command...>
  local name=$1; shift
  if "$@" >/dev/null 2>&1; then PASS=$((PASS + 1)); echo "ok   $name"
  else FAIL=$((FAIL + 1)); echo "FAIL $name"; fi
}
jqt() { jq -e "$@"; }

BIN="$T/bin"; QD="$T/ql"; mkdir -p "$BIN" "$QD"
cat >"$BIN/quietlock" <<'FAKE'
#!/usr/bin/env bash
# Fake quietlock: a flag file, acquire/release lines in a log, and TERM
# forwarded to the held command (the real one forwards to its process group).
D=${FAKE_QL_DIR:?}
case "$1" in
  check) [ ! -e "$D/held" ] || { echo "quietlock: QUIETLOCK_REFUSED fake" >&2; exit 75; }; exit 0 ;;
  hold)
    shift; owner="" minutes=""
    while [ "$1" != -- ]; do case "$1" in --owner) owner=$2 ;; --minutes) minutes=$2 ;; esac; shift 2; done
    shift
    [ ! -e "$D/held" ] || { echo "quietlock: QUIETLOCK_REFUSED fake" >&2; exit 75; }
    : >"$D/held"; echo "acquire $owner $minutes" >>"$D/log"
    "$@" & c=$!
    trap 'echo "forward TERM" >>"$D/log"; kill -TERM $c 2>/dev/null' TERM
    while :; do wait $c; rc=$?; kill -0 $c 2>/dev/null || break; done
    rm -f "$D/held"; echo "release $rc" >>"$D/log"; exit $rc ;;
  *) exit 2 ;;
esac
FAKE
cat >"$BIN/nvidia-smi" <<'FAKE'
#!/usr/bin/env bash
[ ! -e "${FAKE_SMI_DIR:?}/broken" ] || exit 9
case "$1" in
  --query-gpu=*) echo "NVIDIA GB10, 0 %, [N/A], [N/A]" ;;
  --query-compute-apps=*) cat "$FAKE_SMI_DIR/apps" 2>/dev/null; true ;;
esac
FAKE
cat >"$BIN/ss" <<'FAKE'
#!/usr/bin/env bash
echo 'LISTEN 0      128      127.0.0.1:18091 0.0.0.0:* users:(("aien-serve-http",pid=4242,fd=13))'
echo 'LISTEN 0      128      127.0.0.1:18080 0.0.0.0:* users:(("cortex-rs",pid=5060,fd=12))'
FAKE
cat >"$BIN/flock" <<'FAKE'
#!/usr/bin/env bash
echo "flock $*" >>"${FAKE_FLOCK_LOG:?}"; exit 1
FAKE
chmod +x "$BIN"/*
export PATH="$BIN:$PATH" FAKE_QL_DIR="$QD" FAKE_SMI_DIR="$T/smi" FAKE_FLOCK_LOG="$T/flock.log"
mkdir -p "$FAKE_SMI_DIR"
reset() { rm -f "$QD/held" "$QD/log" "$FAKE_SMI_DIR/apps" "$FAKE_SMI_DIR/broken"; }
qr() { # qr <out> [runner args...] -- cmd...  (exit status in $RC)
  local out=$1; shift
  "$RUN" --owner selftest --minutes 2 --out "$T/$out" "$@" 2>"$T/$out.err"; RC=$?
}
M() { echo "$T/$1/manifest.json"; }
not_held() { [ ! -e "$QD/held" ]; }
log_has() { grep -qx "$1" "$QD/log"; }

# 1. idle GPU, normal completion
reset; qr ok1 --server-bin "$BIN/ss" --model "$BIN" --omega-pin abc123 -- true
check "normal: exit 0" test "$RC" -eq 0
check "normal: hold acquired with owner and bound" log_has "acquire selftest 2"
check "normal: hold released" log_has "release 0"
check "normal: no flag left" not_held
check "normal: manifest comparable" jqt '.schema == "own30b-quiet-run/v1" and .comparable and .outcome == "completed" and .exit_status == 0' "$(M ok1)"
check "normal: identity recorded" jqt '(.identity.repo_head | length) >= 7 and (.identity.server_bin.sha256 | length) == 64 and .identity.model.kib >= 0 and .identity.omega_pin == "abc123"' "$(M ok1)"
check "normal: GPU line kept raw (N/A memory)" jqt '.preflight.gpu == "NVIDIA GB10, 0 %, [N/A], [N/A]" and .postflight.available' "$(M ok1)"
check "normal: only AIEN listeners recorded" jqt '.preflight.aien_listeners | length == 1 and (.[0] | test("aien-serve-http"))' "$(M ok1)"
check "normal: command recorded" jqt '.command == ["true"]' "$(M ok1)"

# 2. child failure
reset; qr fail1 -- sh -c 'exit 7'
check "child failure: exit status passed through" test "$RC" -eq 7
check "child failure: hold released" log_has "release 7"
check "child failure: no flag left" not_held
check "child failure: outcome" jqt '.outcome == "exit 7" and .exit_status == 7' "$(M fail1)"

# 3. repeat invocations, and a reused output directory
reset; qr rep1 -- true; r1=$RC; qr rep2 -- true; r2=$RC
check "repeat: both runs succeed" test "$r1$r2" = 00
check "repeat: two acquire/release pairs" test "$(grep -c '^acquire ' "$QD/log")$(grep -c '^release 0' "$QD/log")" = 22
qr rep1 -- true
check "NEGATIVE: reused --out refused" test "$RC" -eq 2

# 4/5. interrupts: the held command is stopped and the flag released
interrupt() { # interrupt <out> <signal> <expected exit>
  reset; rm -f "$T/$1.child"
  # Job control on: a background job of a non-interactive shell starts with
  # SIGINT ignored, which a terminal-run (foreground) runner never has.
  set -m
  "$RUN" --owner selftest --minutes 2 --out "$T/$1" -- sh -c 'echo $$ >"$0"; exec sleep 30' "$T/$1.child" 2>"$T/$1.err" &
  local rp=$! i
  set +m
  for i in $(seq 1 100); do [ -s "$T/$1.child" ] && break; sleep 0.1; done
  kill "-$2" "$rp"; wait "$rp"; RC=$?
  check "$2: runner exits $3" test "$RC" -eq "$3"
  check "$2: no flag left" not_held
  check "$2: held command stopped" bash -c "! kill -0 \$(cat '$T/$1.child')"
  check "$2: manifest says interrupted, not comparable" jqt --arg s "SIG$2" '.outcome == "interrupted: \($s)" and (.comparable | not)' "$(M "$1")"
}
interrupt int1 INT 130
check "INT: passed on as TERM" log_has "forward TERM"
interrupt term1 TERM 143

# 6. contended: refused, nothing run, no hold taken
reset; printf '999, ./max-model, 70000 MiB\n' >"$FAKE_SMI_DIR/apps"
qr busy1 -- touch "$T/busy1.ran"
check "contended: exit 3" test "$RC" -eq 3
check "contended: command not run" test ! -e "$T/busy1.ran"
check "contended: no hold taken" test ! -e "$QD/log"
check "contended: reason names the process" jqt '(.comparable | not) and (.not_comparable_reasons[0] | test("999 ./max-model")) and .override == null' "$(M busy1)"

# 7. contended with override: runs, recorded, not comparable
qr busy2 --allow-contended "MAX model shares the GPU on purpose" -- touch "$T/busy2.ran"
check "override: runs" test "$RC" -eq 0 -a -e "$T/busy2.ran"
check "override: reason recorded, not comparable" jqt '(.comparable | not) and .override.reason == "MAX model shares the GPU on purpose" and (.not_comparable_reasons | any(startswith("override:")))' "$(M busy2)"

# 8. the declared server is allowed on the GPU
reset; printf '4242, ./aien-serve-http, 81717 MiB\n' >"$FAKE_SMI_DIR/apps"
qr srv1 --server-pid 4242 -- true
check "server pid: comparable" jqt '.comparable and .server_pids == [4242] and (.preflight.compute_apps | length) == 1 and (.preflight.foreign | length) == 0' "$(M srv1)"

# 9. a foreign process that appears during the run
reset; qr post1 -- sh -c "printf '777, ./repro2, 9 MiB\n' >'$FAKE_SMI_DIR/apps'"
check "postflight: run completes" test "$RC" -eq 0
check "postflight: foreign process marks not comparable" jqt '(.comparable | not) and (.not_comparable_reasons[0] | test("^postflight: .*777"))' "$(M post1)"

# 10. nvidia-smi unavailable never counts as idle
reset; : >"$FAKE_SMI_DIR/broken"
qr nosmi -- touch "$T/nosmi.ran"
check "no nvidia-smi: refused" test "$RC" -eq 3 -a ! -e "$T/nosmi.ran"
check "no nvidia-smi: recorded" jqt '(.preflight.available | not) and (.not_comparable_reasons[0] | test("unavailable"))' "$(M nosmi)"

# 11. quiet flag already held by someone else
reset; : >"$QD/held"
qr held1 -- touch "$T/held1.ran"
check "quiet held: exit 75, command not run" test "$RC" -eq 75 -a ! -e "$T/held1.ran"
check "quiet held: other holder's flag untouched" test -e "$QD/held"

# 12. usage errors
reset
check "NEGATIVE: --minutes 21 refused" bash -c "! '$RUN' --owner x --minutes 21 --out '$T/u1' -- true"
check "NEGATIVE: missing --owner refused" bash -c "! '$RUN' --out '$T/u2' -- true"
check "NEGATIVE: no command refused" bash -c "! '$RUN' --owner x --out '$T/u3' --"

# 13. the GPU lock is never taken from outside the server
check "never runs flock" test ! -e "$FAKE_FLOCK_LOG"
check "script does not open the GPU lock" bash -c "! grep -v '^#' '$RUN' | grep -q 'aien-gb10.lock'"

# 14. optional: the real quietlock in a private state dir
if [ -n "${QUIETLOCK_REAL:-}" ]; then
  export OWN30B_QR_QUIETLOCK="$QUIETLOCK_REAL" QUIETLOCK_DIR="$T/realq"; mkdir -p "$QUIETLOCK_DIR"
  reset; qr real1 -- true
  check "real quietlock: normal run releases" test "$RC" -eq 0 -a ! -e "$QUIETLOCK_DIR/.spark-quiet"
  qr real2 -- sh -c 'exit 5'
  check "real quietlock: child failure releases" test "$RC" -eq 5 -a ! -e "$QUIETLOCK_DIR/.spark-quiet"
  interrupt realint INT 130
  check "real quietlock: INT releases" test ! -e "$QUIETLOCK_DIR/.spark-quiet"
  interrupt realterm TERM 143
  check "real quietlock: TERM releases" test ! -e "$QUIETLOCK_DIR/.spark-quiet"
fi

echo "own30b-quiet-run-selftest: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
