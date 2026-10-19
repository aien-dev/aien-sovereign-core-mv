#!/bin/bash
# qwen38_gate_run.sh: the Qwen3.8-27B serve and speed gates (QWEN38-SERVE, QWEN38-SPEED; sc
# ledger L006). Starts the model server built from this tree on a test port, waits for it,
# runs the shipped check (tools/qwen38_serve_gate.sh or tools/qwen38_serve_speed.sh), stops the
# server, and prints the result as AIEN-OBS lines for the verification bundle. Called by
# tests/gates/run.sh (steps qwen38-serve and qwen38-speed); works by hand too.
#
# Usage: qwen38_gate_run.sh serve|speed <out-dir>
#   SERVER_BIN   the aien-serve-http binary (required; run.sh builds it)
#   MODEL_DIR    default $HOME/models/qwen3.8-27b-fp8-017b9c7a (never 18091's server, always
#                a fresh process of our own)
#   AIEN_Q38_PORT_RANGE   test ports tried in order, default 18391-18399 (18091 is the live
#                server's port and is never used)
#   AIEN_Q38_MEM_WAIT_S   how long to wait for enough free memory, default 300
#   AIEN_Q38_READY_WAIT_S how long to wait for the server to load, default 900
#   AIEN_Q38_LOCK_WAIT_S  how long to wait for /tmp/aien-gb10.lock when run by hand, default 1800
#
# Memory: the loader asks for 41,305 MiB plus 8,192 MiB kept free (49,497 MiB, about 48.3 GiB of
# MemAvailable) and refuses below that. This script checks /proc/meminfo before it queues for the
# chip and waits up to AIEN_Q38_MEM_WAIT_S for it (and re-checks once the lock is held); still
# short, it FAILS with refusal "memory_short" (no server is started, nothing hangs). A missing
# checkpoint is NOT_RUN model_missing (exit 77), as for PREFILL-E2E-2.
#
# GPU lock: under the judge the runner holds /tmp/aien-gb10.lock for the whole gate and sets
# AIEN_GB10_LOCK_HELD=1, so nothing is locked here (a second flock would wait on its own parent
# forever, sc#580). By hand, the script re-runs itself under the lock: through
# ~/.local/bin/quiet-measure when present (quietlock check first), else plain flock -w. Held
# only for the load, the measurement and the stop.
#
# Exit 0 PASS, 1 FAIL, 77 NOT_RUN (model missing). Observations (AIEN-OBS, bundle fields):
#   both gates:  refusal "memory_short"|"load_failed"|"stub_build"|"not_ready" (only on FAIL),
#                mem_available_kib_at_load, mem_needed_kib, loadavg_at_load "1 5 15",
#                loadavg_at_end, port, server_binary_sha256, server_build (the --version
#                commit line), omega_commit (omega.lock), model_dir, model_sha256_list_sha256
#                (sha256 of docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256, the checkpoint's
#                per-file digests), server_ready_s, server_exit, receipt (path)
#   serve:       golden_ids "MATCH"|"MISMATCH" (24 greedy ids against the shipped host
#                reference), serve_gate_verdict "PASS"|"FAIL" (the full #642 gate: golden,
#                control, snapshot/live/cold resume)
#   speed:       decode_ids "MATCH"|"MISMATCH" (every 256-token decode equals the shipped
#                reference ids), decode_runs (integer), decode_median_chip_ms_per_token (string,
#                e.g. "131.05"), decode_tokens_per_s (string, e.g. "7.63"),
#                ceiling_tokens_per_s "10.15" (docs/qwen38/CEILING.md, memory ceiling),
#                share_of_ceiling (string, e.g. "0.752"), measurement_completed true|false,
#                fraction_computed true|false
# Figures are strings because the bundle's observation grammar takes integers and strings only
# (no floats). PASS for speed is not a speed bar: measurement completed, ids match, fraction
# computed. The number stands beside the ceiling.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KIND=${1:?usage: qwen38_gate_run.sh serve|speed <out-dir>}
OUT=${2:?usage: qwen38_gate_run.sh serve|speed <out-dir>}
case "$KIND" in serve|speed) ;; *) echo "qwen38_gate_run.sh: kind must be serve or speed, got $KIND" >&2; exit 2 ;; esac
GATE="QWEN38-$(tr a-z A-Z <<<"$KIND")"
BIN=${SERVER_BIN:?set SERVER_BIN to the aien-serve-http binary}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-fp8-017b9c7a}
PORT_RANGE=${AIEN_Q38_PORT_RANGE:-18391-18399}
MEM_WAIT_S=${AIEN_Q38_MEM_WAIT_S:-300}
READY_WAIT_S=${AIEN_Q38_READY_WAIT_S:-900}
LOCK_WAIT_S=${AIEN_Q38_LOCK_WAIT_S:-1800}
GPU_LOCK=/tmp/aien-gb10.lock
# The loader's demand: 41,305 MiB for the model plus 8,192 MiB it insists stay free.
NEED_KIB=$(( (41305 + 8192) * 1024 ))
SUMS="$REPO/docs/qwen38/qwen3.8-27b-fp8-017b9c7a.sha256"
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
str() { jq -n --arg s "$1" '$s'; }
mem_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0; }
loadavg() { cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown; }
for t in curl jq sha256sum awk ss; do command -v "$t" >/dev/null || { echo "$GATE: needs $t" >&2; exit 2; }; done

# Memory: wait up to $1 seconds for the loader's demand, then refuse (FAIL, nothing started).
mem_wait() {
  local wait_s=$1 t0 have
  t0=$(date +%s)
  while :; do
    have=$(mem_kib)
    [ "$have" -ge "$NEED_KIB" ] && { MEM_HAVE=$have; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$wait_s" ]; then
      mkdir -p "$OUT"
      obs mem_available_kib_at_load "$have"; obs mem_needed_kib "$NEED_KIB"; obs loadavg_at_load "$(str "$(loadavg)")"
      obs refusal '"memory_short"'
      jq -n --arg gate "$GATE" --argjson have "$have" --argjson need "$NEED_KIB" --arg load "$(loadavg)" \
        --arg at "$(date -u +%FT%TZ)" --arg waited "$wait_s" \
        '{gate:$gate, verdict:"FAIL", refusal:"memory_short", mem_available_kib:$have, mem_needed_kib:$need,
          waited_s:($waited|tonumber), loadavg_1_5_15:$load, at:$at}' > "$OUT/receipt.json"
      obs receipt "$(str "$OUT/receipt.json")"
      echo "$GATE FAIL: memory short: MemAvailable $((have / 1024)) MiB, the loader needs $((NEED_KIB / 1024)) MiB (41,305 MiB model + 8,192 MiB kept free); waited $wait_s s. No server was started."
      exit 1
    fi
    echo "$GATE: MemAvailable $((have / 1024)) MiB < $((NEED_KIB / 1024)) MiB needed; waiting ($(( $(date +%s) - t0 ))/$wait_s s)"
    sleep 15
  done
}

# Memory first, before queuing for the chip: short of it there is nothing to lock for.
mem_wait "$MEM_WAIT_S"

# By hand (no judge): take the chip lock around everything below, then run again inside it.
if [ "${AIEN_GB10_LOCK_HELD:-0}" != 1 ]; then
  if command -v quietlock >/dev/null; then quietlock check || true; fi
  export AIEN_GB10_LOCK_HELD=1
  if command -v quiet-measure >/dev/null; then
    echo "$GATE: taking $GPU_LOCK through quiet-measure (wait up to $LOCK_WAIT_S s)"
    exec quiet-measure flock -w "$LOCK_WAIT_S" "$GPU_LOCK" bash "$0" "$@"
  fi
  echo "$GATE: taking $GPU_LOCK with flock (wait up to $LOCK_WAIT_S s)"
  exec flock -w "$LOCK_WAIT_S" "$GPU_LOCK" bash "$0" "$@"
fi

mkdir -p "$OUT"
rm -f "$OUT/server.log" "$OUT/serve-27b.pid"
obs model_dir "$(str "$MODEL_DIR")"
obs omega_commit "$(str "$(tr -d '[:space:]' < "$REPO/omega.lock")")"
if [ -f "$SUMS" ]; then obs model_sha256_list_sha256 "$(str "$(sha256sum "$SUMS" | cut -d' ' -f1)")"; fi
if [ ! -f "$MODEL_DIR/config.json" ] || [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
  obs not_run_reason '"model_missing"'
  echo "$GATE NOT_RUN: no checkpoint at $MODEL_DIR (config.json, tokenizer.json)"
  exit 77
fi
[ -x "$BIN" ] || { echo "$GATE FAIL: server binary not executable: $BIN"; obs refusal '"stub_build"'; exit 1; }
obs server_binary_sha256 "$(str "$(sha256sum "$BIN" | cut -d' ' -f1)")"
info=$("$BIN" --version 2>&1) || true
obs server_build "$(str "$(grep '^sovereign-core commit:' <<<"$info" | head -1)")"
if ! grep -q '^omega gpu engine: native' <<<"$info" || ! grep -q '^qwen38 engine: native' <<<"$info"; then
  obs refusal '"stub_build"'
  echo "$GATE FAIL: $BIN carries the CPU stub, not the Omega engine:"; echo "$info"
  exit 1
fi

# A free test port from the range (never 18091).
PORT=""
lo=${PORT_RANGE%-*}; hi=${PORT_RANGE#*-}
for p in $(seq "$lo" "$hi"); do
  [ "$p" = 18091 ] && continue
  ss -ltnH "sport = :$p" 2>/dev/null | grep -q LISTEN || { PORT=$p; break; }
done
[ -n "$PORT" ] || { obs refusal '"no_free_port"'; echo "$GATE FAIL: no free port in $PORT_RANGE"; exit 1; }
obs port "$PORT"

# Memory again now that the lock is held (it can move while waiting for the chip): one quick re-check.
mem_wait 30
obs mem_available_kib_at_load "$MEM_HAVE"; obs mem_needed_kib "$NEED_KIB"; obs loadavg_at_load "$(str "$(loadavg)")"

# Start our own server with the project's start script (same flags as the live server).
LOG="$OUT/server.log"
PIDFILE="$OUT/serve-27b.pid"
STOPPED=0
stop_server() {
  [ "$STOPPED" = 1 ] && return 0; STOPPED=1
  local pid rc=""
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; rc=137; fi
  fi
  if grep -q 'shutdown complete' "$LOG" 2>/dev/null; then rc=${rc:-0}; else rc=${rc:-unknown}; fi
  obs server_exit "$(str "$rc")"
  obs loadavg_at_end "$(str "$(loadavg)")"
}
trap stop_server EXIT
t0=$(date +%s)
if ! BIN="$BIN" MODEL_DIR="$MODEL_DIR" PORT="$PORT" STATE_DIR="$OUT" LOG="$LOG" bash "$REPO/tools/serve-27b.sh" > "$OUT/start.out" 2>&1; then
  obs refusal '"load_failed"'; echo "$GATE FAIL: tools/serve-27b.sh refused to start the server:"; cat "$OUT/start.out"; exit 1
fi
pid=$(cat "$PIDFILE")
echo "$GATE: server pid $pid on port $PORT, log $LOG"
ready=0
while [ $(( $(date +%s) - t0 )) -lt "$READY_WAIT_S" ]; do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q 'qwen3.8-27b'; then ready=1; break; fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  if kill -0 "$pid" 2>/dev/null; then
    obs refusal '"not_ready"'; echo "$GATE FAIL: server not ready after $READY_WAIT_S s; last log lines:"
  else
    obs refusal '"load_failed"'; echo "$GATE FAIL: server exited during load; last log lines:"
  fi
  tail -n 20 "$LOG"; exit 1
fi
obs server_ready_s "$(( $(date +%s) - t0 ))"
echo "$GATE: server ready in $(( $(date +%s) - t0 )) s"

# The shipped check through the real HTTP door.
rc=0
case "$KIND" in
  serve)
    SERVER_BIN="$BIN" MODEL_DIR="$MODEL_DIR" bash "$REPO/tools/qwen38_serve_gate.sh" "http://127.0.0.1:$PORT" "" "$OUT/gate" "$LOG" || rc=$?
    R="$OUT/gate/receipt.json"
    ;;
  speed)
    SERVER_BIN="$BIN" MODEL_DIR="$MODEL_DIR" bash "$REPO/tools/qwen38_serve_speed.sh" "http://127.0.0.1:$PORT" "" "$OUT/speed" "$LOG" || rc=$?
    R="$OUT/speed/receipt.json"
    ;;
esac
stop_server
if [ ! -f "$R" ]; then obs refusal '"no_receipt"'; echo "$GATE FAIL: the check wrote no receipt ($R), rc $rc"; exit 1; fi
cp "$R" "$OUT/receipt.json"
obs receipt "$(str "$OUT/receipt.json")"
pass=0
case "$KIND" in
  serve)
    g=$(jq -r '.golden_ids // "unknown"' "$R"); v=$(jq -r '.verdict // "unknown"' "$R")
    obs golden_ids "$(str "$g")"; obs serve_gate_verdict "$(str "$v")"
    [ "$g" = MATCH ] && [ "$v" = PASS ] && [ "$rc" = 0 ] && pass=1
    ;;
  speed)
    ids=$(jq -r '.decode_ids // "unknown"' "$R")
    n=$(jq '[.runs[] | select(.name | startswith("decode"))] | length' "$R")
    med=$(jq -r '.decode_median_chip_ms_per_token // empty' "$R")
    tps=$(jq -r '.decode_tokens_per_s // empty' "$R")
    ceil=$(jq -r '.ceiling_tokens_per_s // empty' "$R")
    share=$(jq -r '.share_of_ceiling // empty' "$R")
    obs decode_ids "$(str "$ids")"; obs decode_runs "$n"
    completed=false; [ -n "$med" ] && [ "$n" -ge 1 ] && completed=true
    fraction=false; [ -n "$share" ] && fraction=true
    obs decode_median_chip_ms_per_token "$(str "$(printf '%.2f' "${med:-0}")")"
    obs decode_tokens_per_s "$(str "$(printf '%.2f' "${tps:-0}")")"
    obs ceiling_tokens_per_s "$(str "${ceil:-10.15}")"
    obs share_of_ceiling "$(str "$(printf '%.3f' "${share:-0}")")"
    obs measurement_completed "$completed"; obs fraction_computed "$fraction"
    [ "$completed" = true ] && [ "$fraction" = true ] && [ "$ids" = MATCH ] && [ "$rc" = 0 ] && pass=1
    echo "$GATE: $(printf '%.2f' "${tps:-0}") tokens/s ($(printf '%.2f' "${med:-0}") ms/token chip, median of $n decodes of 256 greedy tokens) beside the ceiling ${ceil:-10.15} tokens/s: share $(printf '%.3f' "${share:-0}"); ids $ids"
    ;;
esac
echo "--- receipt $OUT/receipt.json ---"; cat "$OUT/receipt.json"; echo
if [ "$pass" = 1 ]; then echo "$GATE PASS"; exit 0; fi
echo "$GATE FAIL (check rc $rc; see the receipt above)"; exit 1
