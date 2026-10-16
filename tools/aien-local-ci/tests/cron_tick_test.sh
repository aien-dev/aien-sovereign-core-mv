#!/usr/bin/env bash
# cron_tick_test.sh: offline proofs for tools/aien-local-ci/cron-tick.sh (power saving: QUIET-SINCE, WAKE, PAUSE).
# Fakes: aien-local-ci (logs "start") and pipeline.sh (logs "round"); a throwaway state directory.
#   bash tools/aien-local-ci/tests/cron_tick_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TICK=$HERE/../cron-tick.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }
export HOME=$T/home AIEN_STATE=$T/state AIEN_CI_LIB=$T/lib
mkdir -p "$HOME/.local/bin" "$AIEN_STATE" "$AIEN_CI_LIB"
LOG=$T/calls.log
printf '#!/bin/bash\necho "$*" >> %s\n' "$LOG" > "$HOME/.local/bin/aien-local-ci"
printf '#!/bin/bash\necho "round $*" >> %s\n' "$LOG" > "$AIEN_CI_LIB/pipeline.sh"
chmod +x "$HOME/.local/bin/aien-local-ci" "$AIEN_CI_LIB/pipeline.sh"
S=$AIEN_STATE
now=$(date +%s)
ago() { touch -d "@$(( now - $1 ))" "$2"; }
tick() { : > "$LOG"; bash "$TICK" > "$T/out" 2>&1 || true; }
ran() { grep -q '^round round$' "$LOG"; }
started() { grep -q '^start --github$' "$LOG"; }
reset() { rm -f "$S"/QUIET-SINCE "$S"/WAKE "$S"/PAUSE "$S"/.paused-seen "$S"/cron-last-round "$S"/worker.heartbeat; }
quiet() { echo "$(( now - $1 )) 2026-10-16T00:00:00Z" > "$S/QUIET-SINCE"; }  # last activity $1 s ago
beat() { : > "$S/worker.heartbeat"; ago "$1" "$S/worker.heartbeat"; }
last() { : > "$S/cron-last-round"; ago "$1" "$S/cron-last-round"; }

reset; tick
{ ran && started; } && ok "no markers: keepalive and round" || bad "no markers"

reset; : > "$S/PAUSE"; tick
{ ! ran && ! started; } && ok "PAUSE: nothing runs, the worker is not started" || bad "PAUSE"
[[ -e $S/.paused-seen ]] && ok "PAUSE leaves .paused-seen" || bad ".paused-seen missing"

rm -f "$S/PAUSE"; quiet 3000; beat 10; last 60; tick
{ ran && grep -q 'wake: PAUSE removed' "$T/out"; } && ok "PAUSE removal runs the round even while quiet" || bad "PAUSE removal"
[[ ! -e $S/.paused-seen ]] && ok ".paused-seen consumed" || bad ".paused-seen kept"

reset; quiet 100; beat 10; last 60; tick
ran && ok "QUIET-SINCE younger than QUIET_AFTER: full cadence" || bad "young QUIET-SINCE skipped"

reset; quiet 3000; beat 10; last 60; tick
{ ! ran && started && grep -q 'skip quiet' "$T/out"; } && ok "quiet: round skipped, keepalive kept" || bad "quiet skip"

reset; quiet 3000; beat 10; last 700; tick
ran && ok "quiet: one round every 10 min" || bad "10-min fallback"

reset; quiet 3000; beat 10; last 60; : > "$S/WAKE"; tick
{ ran && grep -q 'wake: WAKE' "$T/out"; } && ok "WAKE newer than the last round runs it" || bad "WAKE"
tick
! ran && ok "the same WAKE wakes once" || bad "WAKE woke twice"
[[ -e $S/WAKE ]] && ok "WAKE is never removed by cron-tick" || bad "WAKE removed"

reset; quiet 3000; beat 5000; last 60; tick
ran && ok "stale heartbeat: QUIET-SINCE ignored" || bad "stale heartbeat honoured"

reset; quiet 3000; last 60; tick
ran && ok "no heartbeat: QUIET-SINCE ignored" || bad "missing heartbeat honoured"

reset; echo "garbage" > "$S/QUIET-SINCE"; beat 10; last 60; tick
ran && ok "unreadable QUIET-SINCE: full cadence" || bad "garbage honoured"

reset; quiet 3000; beat 10; tick
ran && ok "no cron-last-round: runs" || bad "first round skipped"
[[ -e $S/cron-last-round ]] && ok "cron-last-round written" || bad "cron-last-round missing"
grep -q 'round rc=0 child_cpu user=' "$T/out" && ok "round CPU logged" || bad "round CPU not logged"

echo "cron_tick_test: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
