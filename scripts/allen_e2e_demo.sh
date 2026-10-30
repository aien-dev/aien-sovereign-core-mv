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

# ALLEN end-to-end demo driver (docs/campaigns/allen-e2e/DEMO-v1.md and DEMO-v2.md, arch#162).
# DEMO_VERSION=v1 (default), v2, v3 or v4-gb10 selects the predeclared spec; v2 differs only by its three named changes,
# v3 is v2 unchanged; v4-gb10 is v2 with M-A = Qwen3-4B on the GB10 (sealed sc#277 candidate binary) and M-B = SmolLM2 on CPU
# (DEMO-v4-gb10.md). v4-gb10 sends no signal to a live GB10 daemon: see "stopping a GB10 daemon" below.
#
#   scripts/allen_e2e_demo.sh run      build, run S0-S8 + S3-red, write the receipts
#   scripts/allen_e2e_demo.sh check F  the C1-C5 document checker on file F (prints failed ids)
#   scripts/allen_e2e_demo.sh selftest checker positive and negative controls only (no daemon)
#   scripts/allen_e2e_demo.sh selftest-gb  GB10 graceful-stop logic against fake daemons, kill shadowed (no chip)
#
# The spec is followed as written. Nothing here edits it, retries a model step
# with a changed prompt, or weakens a criterion. Real CPU only: the daemon must
# print the CPU-reference backend line or the run stops (v4-gb10: M-A must print the GB10 line and the
# reservation line). No Python. The GPU is used only by v4-gb10, only by the sealed daemon.
#
# Needs: CLAUDE_JOB_DIR (state under $CLAUDE_JOB_DIR/tmp/demo), the prebuilt omega
# compose archive at $CLAUDE_JOB_DIR/tmp/librx_compose.a, jq, iconv, sha256sum.
set -u

# ---- the document checker (C1-C5), used for S3, S8 and S3-red ----------------
# Prints the failed constraint ids, space separated (empty = all hold).
CANARY='PERSONAL-CANARY-7741'
check_doc() {
  local f=$1 failed="" n
  if [ -f "$f" ] && iconv -f UTF-8 -t UTF-8 "$f" >/dev/null 2>&1; then
    head -n1 "$f" | grep -q '^# ' || failed="$failed C2"
    grep -qi 'tomato' "$f" || failed="$failed C3"
    n=$(wc -w <"$f")
    [ "$n" -le 200 ] || failed="$failed C4"
    ! grep -q "$CANARY" "$f" || failed="$failed C5"
  else
    # No file, or not UTF-8: C1 fails and C2-C4 cannot hold. C5 is not evaluable.
    failed=" C1 C2 C3 C4"
  fi
  echo "${failed# }"
}

selftest() {
  local t rc=0 out
  t=$(mktemp -d)
  printf '# Garden plan\n\nPlant tomatoes in May. Water them weekly.\n' >"$t/good.md"
  printf '# Garden plan\n\nPlant carrots and beans.\n%s\n' "$CANARY" >"$t/red.md"
  out=$(check_doc "$t/good.md"); [ -z "$out" ] || { echo "good control failed: $out"; rc=1; }
  out=$(check_doc "$t/red.md"); [ "$out" = "C3 C5" ] || { echo "red control: '$out'"; rc=1; }
  out=$(check_doc "$t/missing.md"); [ "$out" = "C1 C2 C3 C4" ] || { echo "missing control: '$out'"; rc=1; }
  printf 'no heading tomato\n' >"$t/nohead.md"
  out=$(check_doc "$t/nohead.md"); [ "$out" = "C2" ] || { echo "nohead control: '$out'"; rc=1; }
  # shellcheck disable=SC2046
  seq 1 201 | tr '\n' ' ' | sed 's/^/# tomato /' >"$t/long.md"
  out=$(check_doc "$t/long.md"); [ "$out" = "C4" ] || { echo "long control: '$out'"; rc=1; }
  printf '# t\n\377\376 tomato\n' >"$t/bin.md"
  out=$(check_doc "$t/bin.md"); [ "$out" = "C1 C2 C3 C4" ] || { echo "binary control: '$out'"; rc=1; }
  rm -rf "$t"
  [ $rc -eq 0 ] && echo "selftest ok"
  return $rc
}

# ---- v4-gb10: stopping a GB10 daemon (chip safety) ---------------------------
# HARD RULE: no signal of any kind is ever sent to a process that holds a GB10 channel. A GB10 daemon
# is stopped only through its own shutdown path (the control Shutdown op, sent by `aien-cli compose
# shutdown`, crates/aien-cli/src/compose.rs "shutdown"), and the script then waits with NO timeout for
# the daemon to exit. `aien-cli stop` is not used: it also runs `pkill -f spark-cockpit`. The only
# signal-sending commands left in this file act on CPU daemons (v1-v3, and M-B in v4-gb10); each is
# unreachable while a GB10 daemon is live (GB_LIVE_PID set; see safe_kill9, kill9, start_daemon and the exit trap).
alive() { [ -d "/proc/$1" ]; }   # liveness by /proc, not by a signal
GB_POLL=${GB_POLL:-5}
nvrm_count() { # number of NVRM lines in the kernel log, or "unavailable"
  local t; t=$({ sudo -n dmesg 2>/dev/null || dmesg 2>/dev/null; }) || true
  if [ -z "$t" ]; then echo unavailable; else printf '%s\n' "$t" | grep -aic nvrm || true; fi
}
gb_send_shutdown() { cli compose shutdown >"$LOGS/shutdown-$DN.json" 2>&1 && grep -q '"shutdown": *true' "$LOGS/shutdown-$DN.json"; }
gb_life_record() { # appends this daemon life's facts (from its log) to GB_LIVES
  local log=$DLOG nb na delta bk rv
  bk=$(grep -a -m1 '^  Backend:' "$log" | sed 's/^ *//')
  rv=$(grep -a -m1 'GB10_SERVING_RESERVATION reserved bytes=' "$log" | sed 's/^ *//' | cut -c1-300)
  nb=${NVRM_BEFORE:-unavailable}; na=$(nvrm_count)
  case "$nb$na" in *unavailable*) delta=unavailable ;; *) delta=$((na - nb)) ;; esac
  GB_LAST_DELTA=$delta
  GB_LIVES=$(printf '%s' "$GB_LIVES" | jq -c --argjson n "$DN" --arg log "$(basename "$log")" --arg bk "$bk" --arg rv "$rv" \
    --argjson fb "$(grep -aic fallback "$log")" \
    --argjson nvl "$(grep -a -i nvrm "$log" | grep -avc WARNING)" \
    --argjson af "$(grep -a 'GB10_SERVING_ALLOC' "$log" | grep -acE 'failures=[1-9]|failures_since_warmup=[1-9]|refused')" \
    --argjson rm "$(grep -aiE 'NV_ERR_NO_MEMORY|status 0x51|RM_ALLOC.*fail' "$log" | grep -avc WARNING)" \
    --argjson sk "$(grep -aci 'seat kill' "$log")" \
    --arg exit "${GB_EXIT:-unknown}" --arg stop "${GB_STOP_DESC:-}" --argjson ok "${GB_STOP_OK:-0}" \
    --arg nb "$nb" --arg na "$na" --arg delta "$delta" --arg ack "$GB_ACK" --argjson att "${GB_ATT:-0}" --arg gone "$GB_GONE" \
    '. + [{life:$n,log:$log,backend_line:$bk,backend_gb10:($bk|contains("OmegaGb10Backend")),reservation_line:$rv,
           reservation_line_present:($rv|length>0),fallback_lines:$fb,nvrm_log_lines:$nvl,alloc_failure_lines:$af,
           rm_failure_lines:$rm,seat_kill_lines:$sk,daemon_exit:$exit,stop:$stop,stop_clean:($ok==1),shutdown_requests_sent:$att,shutdown_acked:$ack,socket_removed:$gone,
           nvrm_kernel_before:$nb,nvrm_kernel_after:$na,
           nvrm_kernel_delta:(if $delta=="unavailable" then "unavailable" else ($delta|tonumber) end)}]')
}
GB_STOP_OK=0; GB_STOP_DESC=""; GB_EXIT=""; GB_UNCLEAN=0; GB_ACK=no; GB_ATT=0; GB_GONE=no; GB_LAST_DELTA=unavailable
GB_STOPPING=0; SIG_PENDING=""; SPAWNING=0; GB_LIVE_PID=""
GB_SIGS="INT TERM HUP QUIT USR1 USR2 PIPE ALRM"
# A GB10 daemon is started in its own session with HUP and INT ignored, so a terminal Ctrl-C, a hangup or a signal to the
# driver's process group cannot reach it. `( trap '' HUP INT; exec setsid CMD )&` keeps $! equal to the daemon's pid:
# setsid only forks when its caller is a process group leader, which a background subshell of a script is not (checked
# by the session test in start_daemon and by selftest-gb).
# GB_LIVE_PID is the pid of a live GB10 daemon: set at spawn (signals are deferred while SPAWNING=1, so the handler never
# sees a half-started daemon), cleared only after the graceful stop's wait has returned. While it is set, nothing in this
# script may signal that pid (safe_kill9 refuses; kill9, the exit trap and the signal handler all route to gb_stop).
gb_spawn() { # logfile cmd...; sets DPID, DAEMON_GB, GB_LIVE_PID
  local log=$1; shift
  SPAWNING=1; DAEMON_GB=1
  ( trap '' HUP INT; exec setsid "$@" >"$log" 2>&1 </dev/null ) &
  eval "${GB_BEFORE_ASSIGN_HOOK:-:}"   # test hook: a signal here is deferred (SPAWNING=1)
  DPID=$!; GB_LIVE_PID=$DPID; SPAWNING=0
  # the exact graceful-stop command, absolute paths, for a human if the driver itself is SIGKILLed (see DEMO-v4-gb10.md)
  printf 'AIEN_RUNTIME_SOCK=%s %s compose shutdown\n' "${SOCK:-}" "${BIN:-aien-cli}" >"${LOGS:-.}/STOP-BY-HAND.txt" 2>/dev/null
  eval "${GB_AFTER_SPAWN_HOOK:-:}"     # test hook: a signal here sees a live GB10 daemon
  [ -z "$SIG_PENDING" ] || gb_stop      # a signal deferred during the spawn: stop gracefully now (exits 130)
}
safe_kill9() { # the only place a CPU daemon is killed; refuses the live GB10 pid
  if [ -n "$GB_LIVE_PID" ] && [ "$1" = "$GB_LIVE_PID" ]; then echo "REFUSING to signal live GB10 daemon $1" >&2; return 1; fi
  kill -9 "$1" 2>/dev/null
}
gb_install_traps() { local s; for s in $GB_SIGS; do trap "gb_on_signal $s" "$s"; done; }
# The driver itself: on any of these signals it stops a live GB10 daemon gracefully (unbounded wait), then exits.
gb_on_signal() {
  echo "driver got $1: stopping a live GB10 daemon gracefully (no timeout), then exiting" >&2
  if [ "$GB_STOPPING" = 1 ] || [ "$SPAWNING" = 1 ]; then SIG_PENDING=$1; return 0; fi
  SIG_PENDING=$1
  if [ -n "$GB_LIVE_PID" ]; then gb_stop; fi
  exit 130
}
gb_stop() { # graceful stop of the GB10 daemon; waits with NO timeout; never signals it
  local rc gone=no
  GB_STOP_OK=0; GB_STOP_DESC=""; GB_ACK=no; GB_ATT=0
  [ -n "${DPID:-}" ] || DPID=$GB_LIVE_PID
  [ -n "${DPID:-}" ] || { GB_STOP_DESC="no daemon"; return 1; }
  GB_STOPPING=1
  while alive "$DPID"; do
    GB_ATT=$((GB_ATT + 1))
    if gb_send_shutdown; then GB_ACK=yes; break; fi
    # The daemon closes its accept loop before it writes the ack (server.rs), so a missing ack is not an error by itself.
    [ $((GB_ATT % 12)) -eq 0 ] && echo "waiting for the GB10 daemon to take the shutdown request (no timeout, no kill)" >&2
    sleep "$GB_POLL"
  done
  while :; do wait "$DPID"; rc=$?; if [ "$rc" -gt 128 ] && alive "$DPID"; then continue; fi; break; done   # a signal to the driver interrupts wait, not the daemon
  GB_EXIT=$rc; DPID=""; DAEMON_GB=0; GB_LIVE_PID=""
  [ ! -e "$SOCK" ] && gone=yes; GB_GONE=$gone
  # Clean = a shutdown request was sent at least once AND exit 0 AND socket removed AND no seat-kill line.
  # The ack is recorded, not required. A kernel-log NVRM delta above 0 also blocks any further GB10 life.
  if [ "$GB_ATT" -ge 1 ] && [ "$rc" -eq 0 ] && [ "$gone" = yes ] && ! grep -aqi 'seat kill' "$DLOG"; then
    GB_STOP_OK=1; GB_STOP_DESC="graceful stop (exit 0, session closed)"
  else
    GB_UNCLEAN=1
    GB_STOP_DESC="graceful stop NOT clean (exit $rc, shutdown requests sent: $GB_ATT, acked: $GB_ACK, socket removed: $gone)"
  fi
  gb_life_record
  case $GB_LAST_DELTA in 0|unavailable) ;; *) GB_UNCLEAN=1; GB_STOP_OK=0; GB_STOP_DESC="$GB_STOP_DESC; NEW NVRM kernel-log lines: $GB_LAST_DELTA" ;; esac
  GB_STOPPING=0
  [ -z "${GB_RESULT_FILE:-}" ] || echo "exit=$GB_EXIT clean=$GB_STOP_OK" >"$GB_RESULT_FILE"
  [ -z "$SIG_PENDING" ] || { echo "driver exiting after signal $SIG_PENDING" >&2; exit 130; }
  [ "$GB_STOP_OK" = 1 ]
}
gb_early_exit() { wait "$DPID" 2>/dev/null; GB_EXIT=$?; GB_STOP_OK=0; GB_STOP_DESC="daemon exited by itself (exit $GB_EXIT), no shutdown request"; GB_UNCLEAN=1; GB_LIVE_PID=""; GB_ATT=0; GB_ACK=no; GB_GONE=no; gb_life_record; }

# Inner half of the signal proof, run as its own session and process-group leader (see selftest_gb).
selftest_gb_sig() { # SIGNAME DIR [none|before|after]: where the signal is sent relative to gb_spawn
  local sig=$1 t=$2 mode=${3:-none}; mkdir -p "$t"; GB_POLL=0.2; GB_RESULT_FILE=$t/result
  gb_install_traps
  kill() { echo "KILL-CALLED $*" >>"$t/kills"; return 0; }   # any kill the script itself issues is recorded; the test sends with `command kill`
  cli() { [ "$1 $2" = "compose shutdown" ] && { echo '{"step": "S7", "shutdown": true}'; : >"$t/go"; }; }
  LOGS=$t; SOCK=$t/s; DN=1; DLOG=$t/d.log; GB_LIVES='[]'; NVRM_BEFORE=$(nvrm_count); : >"$SOCK"
  cat >"$t/fake.sh" <<F
#!/bin/bash
for s in TERM QUIT USR1 USR2 ALRM; do trap "echo got-\$s >>$t/sigs" \$s; done
echo '  Backend: NativeTransformerBackend/OmegaGb10Backend (fake)'
while [ ! -e $t/go ]; do sleep 0.1 & wait \$!; done
sleep 1; rm -f $SOCK; exit 0
F
  chmod +x "$t/fake.sh"
  case $mode in
    before) GB_BEFORE_ASSIGN_HOOK="sleep 0.5; command kill -s $sig -- -$$" ;;   # between the background launch and DPID=$!
    after) GB_AFTER_SPAWN_HOOK="sleep 0.5; command kill -s $sig -- -$$" ;;      # right after the spawn, before start_daemon's checks
  esac
  gb_spawn "$DLOG" "$t/fake.sh"
  sleep 0.5
  local sid pg; sid=$(sed 's/.*) //' /proc/$DPID/stat | awk '{print $4}'); pg=$(sed 's/.*) //' /proc/$DPID/stat | awk '{print $3}')
  echo "driver pid $$ pgid $(sed 's/.*) //' /proc/$$/stat | awk '{print $3}'); daemon pid $DPID (\$! matches: $([ -d /proc/$DPID ] && echo yes)) session $sid pgrp $pg" >"$t/where"
  # signal the driver's whole process group; the handler must stop the daemon gracefully and the driver must get its exit code
  command kill -s "$sig" -- "-$$"
  sleep 3; echo "handler did not exit the driver" >"$t/result"; exit 1
}
# Control-flow proof without a chip: fake daemons, with `kill` shadowed so any call is recorded.
selftest_gb() {
  local t rc=0; t=$(mktemp -d); GB_POLL=0.2
  kill() { echo "KILL-CALLED $*" >>"$t/kills"; return 0; }
  cli() { [ -e "$t/fifo" ] || return 1; if [ "$1 $2" = "compose shutdown" ]; then
            [ -e "$t/lostack" ] && { [ -e "$t/sent" ] || { : >"$t/sent"; echo down >"$t/fifo" & }; return 1; }
            [ -e "$t/refuse" ] && { rm -f "$t/refuse"; return 1; }; echo '{"step": "S7", "shutdown": true}'; echo down >"$t/fifo" & fi; }
  LOGS=$t; SOCK=$t/s; DN=1; DLOG=$t/d.log; GB_LIVES='[]'; NVRM_BEFORE=$(nvrm_count)
  printf '  Backend: NativeTransformerBackend/OmegaGb10Backend (x)\n  GB10_SERVING_RESERVATION reserved bytes=1 (x)\n' >"$DLOG"
  run_fake() { # exitcode: fake daemon that exits with that code a second after the shutdown request
    rm -f "$t/fifo"; mkfifo "$t/fifo"; : >"$SOCK"
    ( read -r _ <"$t/fifo"; sleep 1; [ "$1" = 0 ] && rm -f "$SOCK"; exit "$1" ) & DPID=$!; DAEMON_GB=1; GB_LIVE_PID=$DPID
  }
  run_fake 0; touch "$t/refuse"; gb_stop; [ "$GB_STOP_OK" = 1 ] && [ "$GB_EXIT" = 0 ] || { echo "clean stop control failed: $GB_STOP_DESC"; rc=1; }
  run_fake 3; gb_stop; [ "$GB_STOP_OK" = 0 ] && [ "$GB_UNCLEAN" = 1 ] || { echo "unclean stop control failed: $GB_STOP_DESC"; rc=1; }
  GB_UNCLEAN=0; run_fake 0; touch "$t/lostack"; gb_stop; rm -f "$t/lostack" "$t/sent"
  [ "$GB_STOP_OK" = 1 ] && [ "$GB_ACK" = no ] && [ "$GB_ATT" -ge 1 ] || { echo "ack lost but daemon exit 0 + socket removed should be clean: $GB_STOP_DESC"; rc=1; }
  DPID=""; GB_LIVE_PID=""; gb_stop; [ "$GB_STOPPING" = 0 ] || { echo "GB_STOPPING stuck after a no-daemon stop"; rc=1; }
  [ "$(printf "%s" "$GB_LIVES" | jq length)" = 3 ] || { echo "life records: $GB_LIVES"; rc=1; }
  [ ! -s "$t/kills" ] || { echo "a kill was called: $(cat "$t/kills")"; rc=1; }
  # Signals sent to the driver's process group: the GB10 daemon (own session, HUP/INT ignored) must not receive them, the
  # script must issue no kill, and the driver must still collect exit 0 after the graceful stop. Also with the signal
  # delivered inside gb_spawn (before DPID is assigned: deferred) and right after it (daemon live, before any later check).
  local s m r
  for s in $GB_SIGS; do
    setsid -w bash "$0" selftest-gb-sig "$s" "$t/sig-$s" none 2>"$t/sig-$s.err"
    r=$(cat "$t/sig-$s/result" 2>/dev/null)
    if [ "$r" = "exit=0 clean=1" ] && [ ! -s "$t/sig-$s/sigs" ] && [ ! -s "$t/sig-$s/kills" ] && grep -q "graceful" "$t/sig-$s.err"; then
      echo "  group $s: daemon received nothing, no kill called; driver stopped it gracefully, $r"
    else
      echo "group $s FAILED: result='$r' daemon saw: $(cat "$t/sig-$s/sigs" 2>/dev/null) kills: $(cat "$t/sig-$s/kills" 2>/dev/null)"; rc=1
    fi
  done
  echo "  last case: $(cat "$t/sig-TERM/where")"
  for m in before after; do for s in TERM INT; do
    setsid -w bash "$0" selftest-gb-sig "$s" "$t/h-$m-$s" "$m" 2>"$t/h-$m-$s.err"
    r=$(cat "$t/h-$m-$s/result" 2>/dev/null)
    if [ "$r" = "exit=0 clean=1" ] && [ ! -s "$t/h-$m-$s/sigs" ] && [ ! -s "$t/h-$m-$s/kills" ]; then
      echo "  $s sent $m the spawn completed: graceful stop, $r, no kill called"
    else
      echo "hook $m/$s FAILED: result='$r' kills: $(cat "$t/h-$m-$s/kills" 2>/dev/null)"; rc=1
    fi
  done; done
  setsid -w bash -c 'bash -c "trap \"echo got-TERM >>'"$t"'/ctl\" TERM; sleep 3 & wait" & sleep 0.5; command kill -s TERM -- -$$; wait' 2>/dev/null; sleep 0.3
  grep -q got-TERM "$t/ctl" 2>/dev/null && echo "  control: an ordinary child in the group DID receive TERM (the test can see a leak)" || { echo "control failed: ordinary child did not get TERM"; rc=1; }
  unset -f kill cli; rm -rf "$t"
  [ $rc -eq 0 ] && echo "selftest-gb ok (graceful stop waits for exit 0; unclean exit flagged; refused request retried; no kill called; lost ack still clean; group INT TERM HUP QUIT USR1 USR2 PIPE ALRM never reach the GB10 daemon, also when sent during the spawn)"
  return $rc
}


case "${1:-}" in
  check) check_doc "${2:?file}"; exit 0 ;;
  selftest) selftest; exit $? ;;
  selftest-gb) selftest_gb; exit $? ;;
  selftest-gb-sig) selftest_gb_sig "$2" "$3" "${4:-none}"; exit $? ;;
  run) ;;
  *) echo "usage: $0 run | check FILE | selftest | selftest-gb" >&2; exit 2 ;;
esac

# ---- setup -------------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOB=${CLAUDE_JOB_DIR:?CLAUDE_JOB_DIR must be set}
D=$JOB/tmp/demo
LIB=$JOB/tmp/librx_compose.a
BIN=$D/target/release/aien-cli
MA=$HOME/models/SmolLM2-1.7B-Instruct-31b70e2e869a
MB=$HOME/models/qwen3-4b-instruct-2507-cdbee75
OUT=$REPO/docs/campaigns/allen-e2e
V=${DEMO_VERSION:-v1}
case $V in v1|v2|v3|v4-gb10) ;; *) echo "DEMO_VERSION must be v1, v2, v3 or v4-gb10" >&2; exit 2 ;; esac
# v4-gb10 (docs/campaigns/allen-e2e/DEMO-v4-gb10.md): v2 rules; M-A is Qwen3-4B on the GB10 (sealed Q277
# binary), M-B is SmolLM2 on the CPU reference. Everything GB10-specific is behind GB=1.
HCNOTE=""; GB=0; LABEL=real-CPU; MAL="M-A"; MABK="CPU-reference"; GBIN=""; DAEMON_GB=0; IS_GB=0; GB_LIVES='[]'; NVRM_BEFORE=""; DLOG=""
if [ "$V" = v4-gb10 ]; then
  HCNOTE="; hard-crash recovery on GB10: NOT TESTED (chip safety rule)"
  GB=1; LABEL=real-GB10; MAL="M-A (GB10)"; MABK="GB10, native Omega engine"
  MA=$HOME/models/qwen3-4b-instruct-2507-cdbee75
  MB=$HOME/models/SmolLM2-1.7B-Instruct-31b70e2e869a
fi
# v3 is v2 unchanged (DEMO-v3.md): every v2 rule below applies to v3 too.
RECEIPTS=${DEMO_RECEIPTS:-$OUT/receipts-$V.jsonl}
ART=${DEMO_ART:-$OUT/artifacts-$V}
RUN=$D/run
SOCK=$RUN/s
GOAL_A="Write garden.md in the workspace, a garden plan."
GOAL_B="Write garden-b.md in the workspace, a garden plan."
if [ "$V" != v1 ]; then
  # v2 change 1: the request states C2 and C4 (never C3, which must come from memory).
  FMT=" Start with a Markdown heading line that begins with \"# \". Use at most 200 words."
  GOAL_A="$GOAL_A$FMT"; GOAL_B="$GOAL_B$FMT"
fi
NWORK="The garden plan must mention tomatoes."
NPERS="$CANARY"
G1="Keep the garden plan short."

[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 2; }
for t in jq iconv sha256sum setsid; do command -v $t >/dev/null || { echo "need $t" >&2; exit 2; }; done

# Quiet flag: wait for it to clear, never touch it.
QUIET=$HOME/workspace/.spark-quiet
if [ "$GB" = 1 ]; then
  # v4-gb10 runs INSIDE a quietlock hold given by the coordinator (never taken or touched here).
  [ -s "$QUIET" ] || { echo "v4-gb10 must run inside a quietlock hold (quiet flag not held)" >&2; exit 3; }
else
while [ -s "$QUIET" ]; do echo "quiet flag set; waiting"; sleep 30; done
fi

# The tree must be clean (apart from the outputs) so the receipt commit is the code that ran.
DIRTY=$(git -C "$REPO" status --porcelain -- . ":!docs/campaigns/allen-e2e/receipts-$V.jsonl" ":!docs/campaigns/allen-e2e/artifacts-$V" ":!docs/campaigns/allen-e2e/RESULT-$V.md" | wc -l)
if [ "$DIRTY" -ne 0 ] && [ -z "${DEMO_ALLOW_DIRTY:-}" ]; then
  echo "working tree has $DIRTY uncommitted changes; commit them first" >&2; exit 2
fi
COMMIT=$(git -C "$REPO" rev-parse HEAD)
# v2 change 3: name the stack. It is main only if every file this branch changes against
# origin/main is a demo-only file (this driver, the demo docs, the test-only subject fixture, crumbs).
git -C "$REPO" fetch -q origin main 2>/dev/null
MAIN=$(git -C "$REPO" rev-parse origin/main)
NONDEMO=$(git -C "$REPO" diff --name-only "$MAIN" HEAD | grep -vE "^(scripts/allen_e2e_demo\.sh|docs/campaigns/allen-e2e/|crates/aien-allen/tests/e2e_demo_subject\.rs$|(.*/)?\.crumb$)" | wc -l)
if git -C "$REPO" merge-base --is-ancestor "$MAIN" HEAD && [ "$NONDEMO" -eq 0 ]; then
  STACK="main at ${MAIN:0:12} plus demo-only files"
else
  STACK="candidate stack at ${COMMIT:0:12}, not main"
fi

if [ "$GB" = 1 ]; then
  # The sealed qualified candidate (sc#277 attempt 3): sovereign-core 6bbe2ec2, built with the GPU lib.
  # Refuse unless it is exactly the sealed file and this tree differs from 6bbe2ec2 only by demo files.
  Q277=${DEMO_GB10_SEAL_DIR:-$HOME/workspace/hive/Q277}
  GB_SHA_WANT=152c0aecce662f618bf683c8854d6de56a7075e0461c2433570f4c15b68571a5
  GB_SRC=6bbe2ec269768c7c9b94b9484c757ca45f55f564
  GBIN=$Q277/bin/aien-cli-main
  GB_SEAL=$(awk '$2=="bin/aien-cli-main"{print $1}' "$Q277/SHA256SUMS" 2>/dev/null)
  GB_SHA=$(sha256sum "$GBIN" 2>/dev/null | cut -d' ' -f1)
  [ -n "$GB_SHA" ] && [ "$GB_SHA" = "$GB_SHA_WANT" ] && [ "$GB_SEAL" = "$GB_SHA_WANT" ] || { echo "sealed GB10 binary missing or sha256 mismatch ($GB_SHA vs $GB_SHA_WANT): refusing" >&2; exit 2; }
  GB_NONDEMO=$(git -C "$REPO" diff --name-only "$GB_SRC" HEAD | grep -vE "^(scripts/allen_e2e_demo\.sh|docs/campaigns/allen-e2e/|crates/aien-allen/tests/e2e_demo_subject\.rs$|(.*/)?\.crumb$)" | wc -l)
  git -C "$REPO" merge-base --is-ancestor "$GB_SRC" HEAD && [ "$GB_NONDEMO" -eq 0 ] || { echo "tree is not 6bbe2ec2 plus demo-only files: the sealed binary would not be this code; refusing" >&2; exit 2; }
  for v in AIEN_OMEGA_SPIN_US AIEN_OMEGA_CTA_BUDGET AIEN_COMPOSE_BUDGET_MS; do
    [ -z "$(printenv $v)" ] || { echo "$v must be unset for the GB10 candidate" >&2; exit 3; }
  done
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')" ] || { echo "another GPU process is present; refusing" >&2; exit 3; }
fi

# Build with the real compose library (no omega dir, no GPU lib).
(
  cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_FORCE_CPU_STUB \
    AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" cargo build --release -p aien-cli
) >"$D/build-run.log" 2>&1 || { echo "build failed, see $D/build-run.log" >&2; exit 2; }
grep -q "building the stub" "$D/build-run.log" && { echo "stub compose build; refusing" >&2; exit 2; }
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)

# Fresh state. An older run directory is moved aside, never deleted.
[ -e "$RUN" ] && mv "$RUN" "$RUN.old-$(date +%s)"
mkdir -p "$RUN"/{state,compose,prov,ws,home,logs} "$ART"
: >"$RECEIPTS"
LOGS=$RUN/logs
WS=$RUN/ws

E=(env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR -u AIEN_OMEGA_GPU_LIB -u AIEN_REQUIRE_BLACKWELL
   -u AIEN_GPU_BACKEND -u AIEN_MODEL_DIR -u AIEN_MODEL_PATH -u AIEN_TOKENIZER_PATH -u AIEN_FORCE_CPU_STUB
   -u AIEN_FAULT_HOLD -u AIEN_FAULT_HOLD_FILE -u AIEN_ALLEN_SUBJECT -u AIEN_ALLEN_ADOPT
   -u AIEN_COMPOSE_MAX_TOKENS -u AIEN_COMPOSE_DOC_MAX_TOKENS
   HOME="$RUN/home" NO_COLOR=1 AIEN_RUNTIME_SOCK="$SOCK" AIEN_RUNTIME_STATE_DIR="$RUN/state"
   AIEN_COMPOSE_DIR="$RUN/compose" AIEN_PROVENANCE_DIR="$RUN/prov" AIEN_REQUIRE_CHECKPOINT=1
   AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=1 AIEN_COMPOSE_EDIT_BUDGET_MS=599000 AIEN_COMPOSE_DOC_BUDGET_MS=599000)
cli() { "${E[@]}" "$BIN" "$@"; }

# Weights digests. Single file: its sha256. Sharded: sha256 of the sorted
# "<sha256>  <name>" listing of every shard (the daemon itself digests only the
# shard file it is pointed at).
weights_digest() {
  local d=$1
  if [ -f "$d/model.safetensors" ]; then sha256sum "$d/model.safetensors" | cut -d' ' -f1
  else (cd "$d" && sha256sum -- *.safetensors | LC_ALL=C sort -k2 | sha256sum | cut -d' ' -f1); fi
}
WD_A=$(weights_digest "$MA"); WD_B=$(weights_digest "$MB")

# ---- receipts ----------------------------------------------------------------
CUR_DIR=none; CUR_WD=none; CUR_BACKEND="no daemon started"; CUR_MODELLINE=none
CUR_DSHA=none; CUR_TSHA=none; CUR_MAC="no daemon started"; ID0=none
rec() { # step status evidence [extra json]
  local extra=${4:-'{}'}
  jq -nc --arg step "$1" --arg status "$2" --arg ev "$3" --argjson extra "$extra" \
    --arg commit "$COMMIT" --arg bin "$BIN_SHA" --arg backend "$CUR_BACKEND" --arg modelline "$CUR_MODELLINE" \
    --arg mdir "$CUR_DIR" --arg wd "$CUR_WD" --arg dsha "$CUR_DSHA" --arg tsha "$CUR_TSHA" \
    --arg mac "$CUR_MAC" --arg id0 "$ID0" --arg stack "$STACK" --arg label "$LABEL" --arg ver "$V" --arg utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{step:$step,status:$status,label:$label,stack:$stack,spec:("DEMO-"+$ver),evidence:$ev,
      repo_commit:$commit,daemon_sha256:$bin,cli_sha256:$bin,
      backend_line:$backend,model_line:$modelline,model_dir:$mdir,
      model_weights_digest:$wd,daemon_model_sha256:$dsha,tokenizer_sha256:$tsha,
      authorize_mac_line:$mac,allen_logical_agent_id:$id0,utc:$utc} + $extra' >>"$RECEIPTS"
  echo "[$(date -u +%H:%M:%S)] $1 $2: $3" | cut -c1-300
}

# ---- daemon control ----------------------------------------------------------
DN=0; DPID=""; ABORT=""
start_daemon() { # model_dir [VAR=val ...]; sets CUR_* from the daemon log
  local d=$1; shift
  local mp="$d/model.safetensors"; [ -f "$mp" ] || mp="$d/model-00001-of-00003.safetensors"
  local dbin=$BIN; IS_GB=0
  if [ "$GB" = 1 ] && [ "$d" = "$MA" ]; then
    # a GB10 life: the sealed Q277 binary and its qualified env; never start another life after an unclean stop
    [ "$GB_UNCLEAN" = 0 ] || { echo "previous GB10 stop was not clean; not starting another GB10 daemon"; return 1; }
    dbin=$GBIN; IS_GB=1; mp="$d/model.safetensors.index.json"
    set -- AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 AIEN_REQUIRE_BLACKWELL=1 AIEN_KV_CONTEXT_TOKENS=4096 "$@"
    NVRM_BEFORE=$(nvrm_count)
  fi
  DN=$((DN + 1)); DLOG=$LOGS/daemon-$DN.log; rm -f "$SOCK"
  if [ "$IS_GB" = 1 ]; then
    gb_spawn "$DLOG" "${E[@]}" AIEN_MODEL_PATH="$mp" AIEN_TOKENIZER_PATH="$d/tokenizer.json" "$@" "$dbin" daemon
  else
  "${E[@]}" AIEN_MODEL_PATH="$mp" AIEN_TOKENIZER_PATH="$d/tokenizer.json" "$@" "$dbin" daemon >"$DLOG" 2>&1 &
  DPID=$!
  fi
  DAEMON_GB=$IS_GB
  if [ "$IS_GB" = 1 ]; then
    # the GB10 daemon must be the leader of its own session (so group and terminal signals cannot reach it)
    local k sid=""
    for k in $(seq 1 50); do sid=$(sed 's/.*) //' "/proc/$DPID/stat" 2>/dev/null | awk '{print $4}'); [ "$sid" = "$DPID" ] && break; sleep 0.1; done
    [ "$sid" = "$DPID" ] || { echo "GB10 daemon pid $DPID is not its own session leader (session '$sid'); stopping it gracefully"; gb_stop; return 1; }
  fi
  local i
  for i in $(seq 1 600); do
    sleep 2
    alive "$DPID" || { [ "$IS_GB" = 1 ] && gb_early_exit; DPID=""; DAEMON_GB=0; echo "daemon exited early: $(tail -c 400 "$DLOG")"; return 1; }
    if [ -S "$SOCK" ] && grep -q "Replay reconcile:" "$DLOG"; then break; fi
  done
  [ -S "$SOCK" ] || { echo "daemon not serving after 20 min"; return 1; }
  CUR_DIR=$d
  if [ "$d" = "$MA" ]; then CUR_WD=$WD_A; else CUR_WD=$WD_B; fi
  CUR_BACKEND=$(grep -m1 '^  Backend:' "$DLOG" | sed 's/^ *//')
  CUR_MODELLINE=$(grep -m1 '^  Model:' "$DLOG" | sed 's/^ *//')
  CUR_MAC=$(grep -m1 '^Authorize MAC:' "$DLOG")
  CUR_DSHA=$(sed -n 's/.*model_sha256=\([0-9a-f]\{64\}\).*/\1/p' "$DLOG" | head -1)
  CUR_TSHA=$(sed -n 's/.*tokenizer_sha256=\([0-9a-f]\{64\}\).*/\1/p' "$DLOG" | head -1)
  : "${CUR_DSHA:=none}" "${CUR_TSHA:=none}"
  if [ "$IS_GB" = 1 ]; then
    # v4-gb10: the GB10 Backend line and the reservation line are required; else a graceful stop (never a kill).
    CUR_RESV=$(grep -a -m1 'GB10_SERVING_RESERVATION reserved bytes=' "$DLOG" | sed 's/^ *//' | cut -c1-300)
    case $CUR_BACKEND in *OmegaGb10Backend*) ;; *) echo "not the GB10 backend: $CUR_BACKEND"; gb_stop; return 1 ;; esac
    [ -n "$CUR_RESV" ] || { echo "no GB10 reservation line"; gb_stop; return 1; }
    return 0
  fi
  case $CUR_BACKEND in *CPU-reference*) ;; *) echo "not the CPU reference backend: $CUR_BACKEND"; safe_kill9 "$DPID"; wait "$DPID" 2>/dev/null; DPID=""; return 1 ;; esac
  return 0
}
KILL_RC=""
kill9() {
  GB_STOP_DESC=""
  # v4-gb10: a live GB10 daemon is never signalled; stop it through its shutdown path and wait without a timeout.
  if [ "$GB" = 1 ] && [ -n "$GB_LIVE_PID" ]; then gb_stop; local r=$?; KILL_RC=$GB_EXIT; return $r; fi
  [ -n "$DPID" ] || { KILL_RC="no daemon"; return 1; }
  safe_kill9 "$DPID"; wait "$DPID" 2>/dev/null; KILL_RC=$?; DPID=""; rm -f "$SOCK"
  [ "$KILL_RC" -eq 137 ]
}
if [ "$GB" = 1 ]; then
  # exit path: a GB10 daemon still alive is stopped gracefully (unbounded wait), a CPU daemon is killed as in v1-v3
  gb_exit() { if [ -n "$GB_LIVE_PID" ]; then gb_stop; elif [ -n "$DPID" ]; then kill9; fi; }
  trap gb_exit EXIT
  gb_install_traps
else
trap '[ -n "$DPID" ] && kill -9 "$DPID" 2>/dev/null' EXIT
fi

# ---- snapshots used by S5-S7 -------------------------------------------------
snap() { # prefix: profile, memory inspect, goals, effects ledger, document
  local p=$1
  cli allen show | jq -S .result >"$ART/$p-profile.json"
  cli allen memory inspect --owner 1 | jq -S .result.result >"$ART/$p-memory.json"
  cli allen goals list --owner 1 | jq -S .result.result >"$ART/$p-goals.json"
  cli compose effects | jq -S .ledger >"$ART/$p-ledger.json"
  # Every daemon open adds 5 bookkeeping records of kind 65538 (same kind as record 1, no effect, no
  # authorization). So compare the host (effect-class) records, not the raw total.
  cli compose recall | jq -S ".recall.host|length" >"$ART/$p-records.json"
  cli compose recall | jq -S .recall.records_total >"$ART/$p-records-total.json"
}
same() { cmp -s "$ART/$1-$3.json" "$ART/$2-$3.json"; }

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
skip() { [ -n "$ABORT" ] && { rec "$1" NOT_RUN "$ABORT"; return 0; }; return 1; }

# One model task: propose -> authorize (desk MAC) -> execute -> checker. $1 tag, $2 file, $3 goal
model_task() {
  local tag=$1 file=$2 goal=$3
  TASK_STATUS=FAIL; TASK_EV=""; TASK_COMMIT=0; TASK_DONE=0; TASK_JSON='{}'
  cli compose propose --goal "$goal" --workspace "$WS" --context work >"$ART/$tag-propose.json" 2>"$ART/$tag-propose.err"
  local rc=$?
  if [ "$(jq -r '.report.committed // false' "$ART/$tag-propose.json" 2>/dev/null)" != true ]; then
    TASK_EV="propose did not commit (exit $rc): $(jq -r '.report.proposer_error // .error // "no proposal"' "$ART/$tag-propose.json" 2>/dev/null | cut -c1-200); model attempts recorded verbatim in $tag-propose.json. No retry (spec)."
    return
  fi
  TASK_COMMIT=1
  local mem_ctx mem_n
  mem_ctx=$(jq -r '.report.memory.context' "$ART/$tag-propose.json"); mem_n=$(jq -r '.report.memory.items_included' "$ART/$tag-propose.json")
  cli compose authorize --report "$ART/$tag-propose.json" --workspace "$WS" --approver drake --desk 1 >"$ART/$tag-authorize.json" 2>"$ART/$tag-authorize.err"
  if [ "$(jq -r '.ok' "$ART/$tag-authorize.json" 2>/dev/null)" != true ]; then
    TASK_EV="authorize refused: $(jq -c . "$ART/$tag-authorize.json" | cut -c1-250)"; return
  fi
  local grant; grant=$(jq -r '.authorization.id' "$ART/$tag-authorize.json")
  cli compose execute --report "$ART/$tag-propose.json" --workspace "$WS" --authorization "$grant" >"$ART/$tag-execute.json" 2>"$ART/$tag-execute.err"
  local st; st=$(jq -r '.state // "none"' "$ART/$tag-execute.json")
  cli compose effects | jq -S . >"$ART/$tag-effects.json"
  cli compose recall >"$ART/$tag-recall.json"
  local failed; failed=$(check_doc "$WS/$file")
  [ -f "$WS/$file" ] && cp "$WS/$file" "$ART/$file"
  TASK_JSON=$(jq -nc --arg f "$file" --arg failed "$failed" --arg st "$st" --arg mc "$mem_ctx" --argjson mn "${mem_n:-0}" --argjson grant "$grant" \
    '{file:$f,constraints_failed:$failed,effect_state:$st,memory_context:$mc,items_included:$mn,grant_id:$grant}')
  [ "$st" = DONE ] && TASK_DONE=1
  if [ "$st" != DONE ]; then
    case $st in UNRESOLVED|UNCERTAIN) TASK_STATUS=UNRESOLVED ;; esac
    TASK_EV="effect state $st (not DONE); constraints failed: [${failed:-none}]"; return
  fi
  if [ "$mem_ctx" != work ] || [ "${mem_n:-0}" -lt 1 ]; then
    TASK_EV="MemoryReport context=$mem_ctx items_included=$mem_n (need work, >=1); constraints failed: [${failed:-none}]"; return
  fi
  if [ -n "$failed" ]; then
    TASK_EV="commit DONE, MemoryReport work/$mem_n, but document fails: $failed (document in artifacts-$V/$file, recorded verbatim)"; return
  fi
  TASK_STATUS=PASS
  TASK_EV="commit DONE, MemoryReport context=work items_included=$mem_n, C1-C5 hold, $(wc -w <"$WS/$file") words, sha256 $(sha_of "$WS/$file")"
}

# The link chain itself (v2): the commit, grant, intent and ack records of this task each name
# the same generation record and ALLEN agent, and that generation record's model_sha256 is the
# digest the daemon loaded. Read through recall, digest-verified. Sets LINK_MODEL, LINK_ID (0/1)
# and LINK_JSON. Stricter than the v1 text search, which only asked whether the values appear.
record_text() { cli compose recall --ids "$1" | jq -c 'if .recall.cited[0].verified == true then (.recall.cited[0].text|fromjson) else {unverified:true} end' 2>/dev/null || echo '{"unreadable":true}'; }
follow_link() { # tag file msha
  local tag=$1 file=$2 msha=$3 gen commit grant intent ack id r ok_m=1 ok_i=1 rows="[]"
  gen=$(jq -r '.report.generation_record // "none"' "$ART/$tag-propose.json")
  commit=$(jq -r '.report.compose_commit // "none"' "$ART/$tag-propose.json")
  grant=$(jq -r '.authorization.id // "none"' "$ART/$tag-authorize.json")
  intent=$(jq -r --arg f "$file" '[.ledger.intents[]|select(.path==$f)][0].intent // "none"' "$ART/$tag-effects.json")
  ack=$(jq -r --arg f "$file" '[.ledger.intents[]|select(.path==$f)][0].state_record // "none"' "$ART/$tag-effects.json")
  for id in "$commit" "$grant" "$intent" "$ack"; do
    case $id in ''|none|null) ok_m=0; ok_i=0; continue ;; esac
    r=$(record_text "$id")
    [ "$(printf '%s' "$r" | jq -r '.provenance.generation_record|tostring')" = "$gen" ] || ok_m=0
    [ "$(printf '%s' "$r" | jq -r '.provenance.allen_agent // "absent"')" = "$ID0" ] || ok_i=0
    rows=$(printf '%s' "$rows" | jq -c --argjson id "$id" --argjson r "$r" '. + [{record:$id,provenance:($r.provenance // null)}]')
  done
  local g gm="none"
  case $gen in ''|none|null) ok_m=0 ;; *) g=$(record_text "$gen"); gm=$(printf '%s' "$g" | jq -r '.model_sha256 // "none"'); [ "$gm" = "$msha" ] || ok_m=0 ;; esac
  LINK_MODEL=$ok_m; LINK_ID=$ok_i
  LINK_JSON=$(jq -nc --arg gen "$gen" --arg gm "$gm" --argjson rows "$rows" --arg c "$commit" --arg g "$grant" --arg i "$intent" --arg a "$ack" \
    '{generation_record:$gen,generation_record_model_sha256:$gm,chain:{compose_commit:$c,grant:$g,intent:$i,ack:$a},records:$rows}')
}

# Provenance links for one commit: which of the five items appear in what the daemon recorded.
# Sources: the effect ledger, every compose record text (recall), the task report, the authorize
# and execute answers, and the provenance receipts. NOT the daemon boot log (it is not a ledger).
provenance() { # tag file model_sha_expected
  local tag=$1 file=$2 msha=$3 ledger recall psha csha auth pool
  recall=$ART/$tag-recall.json; ledger=$ART/$tag-effects.json
  psha=$(jq -r '.report.proposal_sha256' "$ART/$tag-propose.json")
  csha=$(sha_of "$WS/$file")
  auth=$(jq -r '.authorization.id' "$ART/$tag-authorize.json")
  local l_grant l_prop l_content l_model l_id
  l_grant=$(jq -r --arg f "$file" --argjson a "$auth" '[.ledger.intents[]|select(.path==$f and .authorization==$a)]|length' "$ledger")
  l_content=$(jq -r --arg f "$file" --arg c "$csha" '[.ledger.intents[]|select(.path==$f and .content_sha256==$c)]|length' "$ledger")
  pool=$(cat "$ledger" "$recall" "$ART/$tag-propose.json" "$ART/$tag-authorize.json" "$ART/$tag-execute.json" "$RUN"/prov/*.json 2>/dev/null)
  l_prop=$(printf '%s' "$pool" | grep -c "$psha")
  l_model=$(printf '%s' "$pool" | grep -c "$msha")
  l_id=$(printf '%s' "$pool" | grep -c "$ID0")
  LINK_JSON=null
  if [ "$V" != v1 ]; then
    follow_link "$tag" "$file" "$msha"
    l_model=$LINK_MODEL; l_id=$LINK_ID
  fi
  PROV_JSON=$(jq -nc --argjson g "${l_grant:-0}" --argjson p "$l_prop" --argjson c "${l_content:-0}" --argjson m "$l_model" --argjson i "$l_id" \
    --arg auth "$auth" --arg psha "$psha" --arg csha "$csha" --arg msha "$msha" --argjson link "$LINK_JSON" \
    '{grant_id:$auth,proposal_sha256:$psha,content_sha256:$csha,model_digest_expected:$msha,link_chain:$link,
      ledger_links:{grant_id:($g>0),proposal_sha256:($p>0),content_sha256_equals_disk:($c>0),model_digest:($m>0),logical_agent_id:($i>0)}}')
  [ "${l_grant:-0}" -gt 0 ] && [ "$l_prop" -gt 0 ] && [ "${l_content:-0}" -gt 0 ] && [ "$l_model" -gt 0 ] && [ "$l_id" -gt 0 ]
}

# ============================== S0 ============================================
cli compose desk-key --create 1 >"$ART/S0-deskkey.json" 2>&1   # prints id and path, never the key
if start_daemon "$MA"; then
  ST_ALLEN=$(cli allen status | jq -r .result.identity)
  ALLEN_DIRS=$(ls -d "$RUN"/compose.allen-* "$RUN"/compose/*allen* 2>/dev/null | wc -l)
  if [[ $CUR_MAC == "Authorize MAC: on"* ]] && [ "$ST_ALLEN" = not_engaged ] && [ "$ALLEN_DIRS" -eq 0 ]; then
    rec S0 PASS "daemon up on M-A ($MABK), log: '$CUR_MAC', allen status identity=$ST_ALLEN, 0 ALLEN state directories before S1" "$(if [ "$GB" = 1 ]; then jq -nc --arg r "${CUR_RESV:-}" '{gb10_reservation_line:$r}'; else echo '{}'; fi)"
  else
    rec S0 FAIL "MAC line='$CUR_MAC' identity=$ST_ALLEN allen_dirs=$ALLEN_DIRS"
    ABORT="S0 failed, nothing further can be trusted"
  fi
else
  rec S0 FAIL "daemon did not start on M-A: $(tail -c 300 "$LOGS/daemon-$DN.log" 2>/dev/null | tr '\n' ' ')"
  ABORT="S0 failed: daemon did not start on M-A"
fi

# ============================== S1 ============================================
if ! skip S1; then
  # The compose home exists once the daemon opened it; its record 1 is the lineage a subject must bind.
  LINEAGE=$(cli compose recall --ids 1 | jq -r '.recall.cited[0].digest')
  kill9 || true
  SUBJ=$RUN/subject.bin
  FX=$(cd "$REPO" && env -u AIEN_OMEGA_DIR -u AIEN_OMEGA_COMPOSE_DIR AIEN_OMEGA_COMPOSE_LIB="$LIB" CARGO_TARGET_DIR="$D/target" \
        DEMO_SUBJECT_OUT="$SUBJ" DEMO_LINEAGE_HEX="$LINEAGE" DEMO_SUBJECT_NAME="allen-e2e-demo-$V" \
        cargo test --release -p aien-allen --test e2e_demo_subject -- --ignored --nocapture 2>&1)
  AGENT=$(printf '%s\n' "$FX" | sed -n 's/^AGENT=//p'); AROOT=$(printf '%s\n' "$FX" | sed -n 's/^ROOT=//p')
  if [ -z "$AGENT" ] || [ ! -s "$SUBJ" ]; then
    rec S1 FAIL "could not build the subject fixture: $(printf '%s' "$FX" | tail -c 300)"
    ABORT="S1 failed: no identity"
  elif start_daemon "$MA" AIEN_ALLEN_SUBJECT="$SUBJ" AIEN_ALLEN_ADOPT="$AGENT"; then
    ALOG=$(grep '^ALLEN: engaged' "$LOGS/daemon-$DN.log")
    STJ=$(cli allen status | jq -c .)
    FP=$(printf '%s' "$STJ" | jq -r .result.fingerprint)
    LOGID=$(printf '%s' "$ALOG" | sed -n 's/.*agent=\([0-9a-f]\{64\}\).*/\1/p')
    NLINES=$(printf '%s\n' "$ALOG" | grep -c .)
    if [ "$(printf '%s' "$STJ" | jq -r .result.identity)" = engaged ] && [ "$LOGID" = "$AGENT" ] && [ "$FP" = "${AGENT:0:8}" ] && [ "$NLINES" -eq 1 ]; then
      ID0=$AGENT
      rec S1 PASS "ID0=$ID0 (daemon log 'ALLEN: engaged agent=...', allen status fingerprint $FP); AgentRoot $AROOT; one identity" \
        "$(jq -nc --arg root "$AROOT" --arg lineage "$LINEAGE" '{agent_root:$root,cortex_lineage_record1:$lineage,subject_kind:"host-built fixture (test-only encoder), adopted once by AIEN_ALLEN_ADOPT; real subjects come only from AIENOS cs_provision (NOT_RUN)"}')"
    else
      rec S1 FAIL "status/log disagree: identity=$(printf '%s' "$STJ" | jq -r .result.identity) fp=$FP log=$LOGID lines=$NLINES"
      ABORT="S1 failed: identity not engaged"
    fi
  else
    rec S1 FAIL "daemon refused or did not start with the subject: $(tail -c 300 "$LOGS/daemon-$DN.log" | tr '\n' ' ')"
    ABORT="S1 failed: identity not engaged"
  fi
fi

# ============================== S2 ============================================
if ! skip S2; then
  cli allen set --expect 0 --plain-language 1 >"$ART/S2-set.json"
  cli allen memory put --context work --text "$NWORK" >"$ART/S2-put-work.json"
  cli allen memory put --context personal --text "$NPERS" >"$ART/S2-put-pers.json"
  cli allen goals add --context work --text "$G1" >"$ART/S2-goal.json"
  snap S2
  OK_PROF=$(jq -r --arg id "$ID0" '.profile.persona.plain_language==true and .profile.agent==$id and .revision==1' "$ART/S2-profile.json")
  OK_MEM=$(jq -r --arg a "$NWORK" --arg b "$NPERS" --arg g "$G1" \
    '(map(select(.state=="live"))|length)==3 and (map(select(.scope=="work" and .kind=="fact" and .text==$a))|length)==1 and (map(select(.scope=="personal" and .kind=="fact" and .text==$b))|length)==1 and (map(select(.scope=="work" and .kind=="goal" and .text==$g))|length)==1' "$ART/S2-memory.json")
  OK_GOALS=$(jq -r --arg g "$G1" '(.goals|length)==1 and .goals[0].text==$g and .goals[0].scope=="work" and .goals[0].state=="open"' "$ART/S2-goals.json")
  if [ "$OK_PROF" = true ] && [ "$OK_MEM" = true ] && [ "$OK_GOALS" = true ]; then
    rec S2 PASS "profile revision 1 plain-language=1 under ID0; memory inspect shows exactly N-work (work), N-pers (personal), G1 (work goal); goals list shows exactly G1"
  else
    rec S2 FAIL "profile_ok=$OK_PROF memory_ok=$OK_MEM goals_ok=$OK_GOALS (see artifacts S2-*.json)"
  fi
  NPERS_ITEM=$(jq -r --arg b "$NPERS" '.[]|select(.text==$b)|.item' "$ART/S2-memory.json")
fi

# ============================== S3 ============================================
S3_OK=0
if ! skip S3; then
  model_task S3 garden.md "$GOAL_A"
  rec S3 "$TASK_STATUS" "$TASK_EV" "$TASK_JSON"
  S3_OK=$TASK_DONE
fi

# ============================== S3-red ========================================
if ! skip S3-red; then
  printf '# Garden plan\n\nPlant carrots and beans along the south fence.\nNote: %s\n' "$CANARY" >"$ART/red-garden.md"
  RED=$(check_doc "$ART/red-garden.md")
  printf "# Garden plan\n\nPlant tomatoes in May.\n" >"$ART/good-control.md"; GOOD=$(check_doc "$ART/good-control.md")
  if [ "$RED" = "C3 C5" ]; then
    rec S3-red PASS "checker on the hand-written document (breaks C3 and C5) reports exactly: $RED; positive control (good document) reports: [${GOOD:-none}]" "$(jq -nc --arg r "$RED" '{checker_failed:$r}')"
  else
    rec S3-red FAIL "checker reported '$RED', expected exactly 'C3 C5'" "$(jq -nc --arg r "$RED" '{checker_failed:$r}')"
  fi
fi

# ============================== S4 ============================================
if ! skip S4; then
  if [ "$S3_OK" != 1 ]; then
    rec S4 NOT_RUN "S3 produced no commit, so there is no commit to trace"
  elif provenance S3 garden.md "$CUR_DSHA"; then
    rec S4 PASS "ledger links grant id, proposal sha256, content sha256 (= garden.md on disk), M-A model digest and ID0" "$PROV_JSON"
  else
    MISS=$(printf '%s' "$PROV_JSON" | jq -r '.ledger_links|to_entries|map(select(.value==false).key)|join(",")')
    rec S4 FAIL "missing links in what the daemon recorded for the S3 commit: $MISS. Present: the rest$([ "$V" = v1 ] && echo ". The model digest and ID0 appear in the daemon boot log and the ALLEN profile/memory stores, not in the compose or effect ledger records of the commit")" "$PROV_JSON"
  fi
fi

# ============================== S5 ============================================
if ! skip S5; then
  snap S5a
  [ -f "$WS/garden.md" ] && { G_SHA_BEFORE=$(sha_of "$WS/garden.md"); G_STAT_BEFORE=$(stat -c '%i %Y.%y' "$WS/garden.md"); } || { G_SHA_BEFORE=none; G_STAT_BEFORE=none; }
  if kill9; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=$KILL_RC"; fi
  [ -n "$GB_STOP_DESC" ] && KILLED=$GB_STOP_DESC
  if start_daemon "$MA" AIEN_ALLEN_SUBJECT="$RUN/subject.bin"; then
    snap S5b
    G_SHA_AFTER=none; G_STAT_AFTER=none
    [ -f "$WS/garden.md" ] && { G_SHA_AFTER=$(sha_of "$WS/garden.md"); G_STAT_AFTER=$(stat -c '%i %Y.%y' "$WS/garden.md"); }
    REPL=$(grep -m1 '^Replay reconcile:' "$LOGS/daemon-$DN.log")
    FPA=$(cli allen status | jq -r .result.fingerprint)
    ALOG2=$(grep -c "^ALLEN: engaged agent=$ID0" "$LOGS/daemon-$DN.log")
    CHK=""
    [ "$FPA" = "${ID0:0:8}" ] && [ "$ALOG2" = 1 ] || CHK="$CHK identity"
    same S5a S5b profile || CHK="$CHK profile"
    same S5a S5b memory || CHK="$CHK notes"
    same S5a S5b goals || CHK="$CHK goals"
    same S5a S5b ledger || CHK="$CHK ledger-changed(re-executed?)"
    same S5a S5b records || CHK="$CHK host-record-count"
    [ "$S3_OK" = 1 ] || CHK="$CHK no-S3-commit"
    [ "$(jq -r '[.intents[]|select(.path=="garden.md" and .state=="DONE")]|length' "$ART/S5b-ledger.json")" = 1 ] || CHK="$CHK S3-not-DONE"
    [ "$G_SHA_BEFORE" != none ] && [ "$G_SHA_BEFORE" = "$G_SHA_AFTER" ] && [ "$G_STAT_BEFORE" = "$G_STAT_AFTER" ] || CHK="$CHK garden.md"
    if [ -z "$CHK" ]; then
      rec S5 PASS "$KILLED then restart on $MAL: ID0 same, profile/notes/G1 identical, S3 intent still DONE and ledger+record count unchanged (not re-executed), garden.md same sha256, inode and mtime; $REPL$HCNOTE"
    elif [ "$S3_OK" != 1 ]; then
      rec S5 NOT_RUN "no S3 commit existed to check; restart itself: identity/profile/notes/goals problems=[${CHK# }]"
    else
      rec S5 FAIL "differences after restart:${CHK}; $REPL"
    fi
  else
    rec S5 FAIL "$KILLED; daemon did not restart on M-A: $(tail -c 300 "$LOGS/daemon-$DN.log" | tr '\n' ' ')"
    ABORT="S5: daemon would not restart"
  fi
fi

# ============================== S6 ============================================
if ! skip S6; then
  cli allen memory forget --context personal --item "${NPERS_ITEM:-none}" >"$ART/S6-forget.json"
  snap S6
  cli allen memory recall --context personal >"$ART/S6-recall-personal.json"
  LIVE=$(jq -r --arg b "$NPERS" '[.[]|select(.text==$b or .state=="live" and .scope=="personal")]|length' "$ART/S6-memory.json")
  SHOWN=$(jq -r --arg i "${NPERS_ITEM:-none}" '[.[]|select(.item==$i)|.state]|join(",")' "$ART/S6-memory.json")
  RITEMS=$(jq -r '.result.result.items|length' "$ART/S6-recall-personal.json")
  DISK=$(grep -rl "$NPERS" "$RUN/compose" "$RUN/compose.allen-memory" "$RUN/compose.allen-profile" "$RUN/state" "$RUN/prov" "$LOGS" 2>/dev/null | wc -l)
  if [ "$LIVE" -eq 0 ] && [ "$RITEMS" -eq 0 ] && [ "$(jq -r '.ok' "$ART/S6-forget.json")" = true ]; then
    rec S6 PASS "forgot N-pers: inspect no longer shows its text (its row remains as state '${SHOWN:-absent}' with no text), recall --context personal returns 0 items; files under daemon state still containing the canary text: $DISK" \
      "$(jq -nc --argjson d "$DISK" '{canary_files_remaining_in_daemon_state:$d}')"
  else
    rec S6 FAIL "after forget: live/visible=$LIVE recall_items=$RITEMS forget_ok=$(jq -r '.ok' "$ART/S6-forget.json")"
  fi
fi

# ============================== S7 ============================================
[ "$GB" = 1 ] && LABEL=real-CPU   # v4-gb10: S0-S6 real-GB10, S7-S8 real-CPU
if ! skip S7; then
  OLD_BACKEND=$CUR_BACKEND; OLD_MODELLINE=$CUR_MODELLINE; OLD_DSHA=$CUR_DSHA; OLD_TSHA=$CUR_TSHA
  if kill9; then KILLED="kill -9 (exit 137)"; else KILLED="kill rc=$KILL_RC"; fi
  [ -n "$GB_STOP_DESC" ] && KILLED=$GB_STOP_DESC
  if start_daemon "$MB" AIEN_ALLEN_SUBJECT="$RUN/subject.bin"; then
    snap S7
    FPB=$(cli allen status | jq -r .result.fingerprint)
    CHK=""
    if [ "$V" != v1 ]; then
      # v2 change 2: the Model: line must change; the Backend: line is recorded, not required to change.
      [ "$CUR_MODELLINE" != "$OLD_MODELLINE" ] && [ "$CUR_MODELLINE" != none ] || CHK="$CHK model-line-identical"
    else
      [ "$CUR_BACKEND" != "$OLD_BACKEND" ] || CHK="$CHK backend-line-identical"
    fi
    # v4-gb10: model AND backend change (GB10 -> CPU reference); both lines must differ.
    if [ "$GB" = 1 ]; then [ "$CUR_BACKEND" != "$OLD_BACKEND" ] || CHK="$CHK backend-line-identical"; fi
    [ "$CUR_DSHA" != "$OLD_DSHA" ] && [ "$CUR_DSHA" != none ] || CHK="$CHK model-digest-not-changed"
    [ "$FPB" = "${ID0:0:8}" ] || CHK="$CHK identity"
    same S6 S7 profile || CHK="$CHK profile"
    same S6 S7 memory || CHK="$CHK notes"
    same S6 S7 goals || CHK="$CHK goals"
    cli allen memory recall --context personal >"$ART/S7-recall-personal.json"
    [ "$(jq -r '.result.result.items|length' "$ART/S7-recall-personal.json")" = 0 ] || CHK="$CHK N-pers-visible"
    EXTRA=$(jq -nc --arg ob "$OLD_BACKEND" --arg nb "$CUR_BACKEND" --arg om "$OLD_MODELLINE" --arg nm "$CUR_MODELLINE" --arg od "$OLD_DSHA" --arg nd "$CUR_DSHA" \
      '{before:{backend_line:$ob,model_line:$om,daemon_model_sha256:$od},after:{backend_line:$nb,model_line:$nm,daemon_model_sha256:$nd}}')
    if [ -z "$CHK" ]; then
      rec S7 PASS "$KILLED then start on M-B: $([ "$V" != v1 ] && echo "Model: line$([ "$GB" = 1 ] && echo " and Backend: line")" || echo "backend line") and model digest changed (${OLD_DSHA:0:12} -> ${CUR_DSHA:0:12}); ID0 same; profile, N-work, G1 unchanged; N-pers still absent" "$EXTRA"
    else
      rec S7 FAIL "criteria not met:${CHK}. Backend line before: '$OLD_BACKEND'; after: '$CUR_BACKEND'. Model line and digest changed: $([ "$CUR_DSHA" != "$OLD_DSHA" ] && echo yes || echo no)" "$EXTRA"
    fi
  else
    rec S7 FAIL "$KILLED; daemon did not start on M-B: $(tail -c 400 "$LOGS/daemon-$DN.log" | tr '\n' ' ')"
    ABORT="S7: M-B daemon did not start"
  fi
fi

# ============================== S8 ============================================
if ! skip S8; then
  model_task S8 garden-b.md "$GOAL_B"
  S8_STATUS=$TASK_STATUS; S8_EV=$TASK_EV
  if [ "$TASK_COMMIT" = 1 ] && [ -f "$WS/garden-b.md" ]; then
    if provenance S8 garden-b.md "$CUR_DSHA"; then
      PROV_NOTE="provenance links all present (incl. M-B digest and ID0)"
    else
      MISS=$(printf '%s' "$PROV_JSON" | jq -r '.ledger_links|to_entries|map(select(.value==false).key)|join(",")')
      PROV_NOTE="provenance links missing: $MISS (M-B digest expected in the provenance)"
      [ "$S8_STATUS" = PASS ] && S8_STATUS=FAIL
    fi
    S8_EV="$S8_EV; $PROV_NOTE"
    TASK_JSON=$(printf '%s' "$TASK_JSON" | jq -c --argjson p "$PROV_JSON" '. + {provenance:$p}')
  fi
  rec S8 "$S8_STATUS" "$S8_EV" "$TASK_JSON"
fi

[ -n "$DPID" ] && kill9
if [ "$GB" = 1 ]; then
  # One extra receipt for the GB10 daemon lives: the per-life PASS rule of DEMO-v4-gb10.md.
  LABEL=real-GB10
  GB_VERDICT=$(printf '%s' "$GB_LIVES" | jq -r 'if length==0 then "FAIL"
    elif all(.[]; .backend_gb10 and .reservation_line_present and .alloc_failure_lines==0 and .rm_failure_lines==0 and .fallback_lines==0 and .nvrm_log_lines==0 and .seat_kill_lines==0 and .stop_clean) | not then "FAIL"
    elif any(.[]; .nvrm_kernel_delta=="unavailable") then "UNKNOWN"
    elif all(.[]; .nvrm_kernel_delta==0) then "PASS" else "FAIL" end')
  rec GB10-lives "$GB_VERDICT" "per GB10 daemon life: GB10 Backend line, reservation line, 0 allocation/RM failures, 0 fallback lines, 0 NVRM log lines (daemon log and kernel log delta), no seat-kill line, graceful stop (exit 0, shutdown acked, socket removed). Sealed binary sha256 $GB_SHA. UNKNOWN means the kernel log was unreadable. hard-crash recovery on GB10: NOT TESTED (chip safety rule)" \
    "$(jq -nc --argjson l "$GB_LIVES" --arg sha "$GB_SHA" --arg src "$GB_SRC" '{gb10_lives:$l,gb10_binary_sha256:$sha,gb10_binary_source_commit:$src,hard_crash_recovery_on_gb10:"NOT TESTED (chip safety rule)"}')"
fi
cp "$LOGS"/daemon-*.log "$ART/" 2>/dev/null
echo "receipts: $RECEIPTS"
