#!/bin/bash
# AIEN Spark pipeline tick (cron, every 2 min; no systemd by rule).
# 1) keeps the continuous worker (--github, all enrolled repos) alive: `aien-local-ci start` is a no-op when it runs.
# 2) runs one pipeline round: publish signed bundles, rerun stale checks, merge green PRs labelled ready-to-merge.
# Stop everything: touch ~/.local/state/aien-local-ci/PAUSE ; aien-local-ci stop
#
# Power saving (2026-10-16). Markers in $S (see tools/aien-local-ci/README.md, "Power saving"):
#   QUIET-SINCE  written by the worker: "<epoch> <utc>" of the last activity it saw; absent = not quiet.
#   WAKE         touched by anyone to ask for full cadence now; never removed by a reader.
#   PAUSE        unchanged: present = this script exits at once (nothing runs, the worker is not restarted).
# While QUIET-SINCE is at least QUIET_AFTER old (and the worker heartbeat is fresh), the round runs only every
# QUIET_ROUND seconds; WAKE newer than the last round, or PAUSE just removed, runs it at once. Anything unreadable
# runs the round (fail open = full cadence). The keepalive below is never skipped.
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
S=${AIEN_STATE:-$HOME/.local/state/aien-local-ci}
LIB=${AIEN_CI_LIB:-$HOME/.local/lib/aien-local-ci}
QUIET_AFTER=${AIEN_QUIET_AFTER:-600}     # seconds of no activity before rounds thin out
QUIET_ROUND=${AIEN_QUIET_ROUND:-600}     # while quiet, one round this often
BEAT_MAX=${AIEN_BEAT_MAX:-1200}          # heartbeat older than this: the worker is not trusted to be watching
if [ -e "$S/PAUSE" ]; then touch "$S/.paused-seen" 2>/dev/null; exit 0; fi
nice -n 10 aien-local-ci start --github >/dev/null

now=$(date +%s)
age() { local m; m=$(stat -c %Y "$1" 2>/dev/null) || { echo -1; return; }; echo $(( now - m )); }
why=""
if [ -e "$S/.paused-seen" ]; then rm -f "$S/.paused-seen"; why="PAUSE removed"
elif [ -e "$S/WAKE" ] && { [ ! -e "$S/cron-last-round" ] || [ "$S/WAKE" -nt "$S/cron-last-round" ]; }; then why="WAKE"
fi
if [ -z "$why" ]; then
  since=$(awk 'NR==1{print $1}' "$S/QUIET-SINCE" 2>/dev/null)
  beat=$(age "$S/worker.heartbeat")
  last=$(age "$S/cron-last-round")
  if [[ $since =~ ^[0-9]+$ ]] && (( now - since >= QUIET_AFTER )) && (( beat >= 0 && beat < BEAT_MAX )) \
     && (( last >= 0 && last < QUIET_ROUND )); then
    echo "$(date -u +%FT%TZ) skip quiet since=$since last_round_age=${last}s"
    exit 0
  fi
fi
[ -n "$why" ] && echo "$(date -u +%FT%TZ) wake: $why"
touch "$S/cron-last-round"
nice -n 10 /bin/bash "$LIB/pipeline.sh" round
rc=$?
# The round's own CPU (this shell's children), so the idle cost is measured, not guessed.
cpu=$(times | awk 'NR==2{for(i=1;i<=2;i++){split($i,a,"m"); sub("s","",a[2]); t[i]=a[1]*60+a[2]} printf "user=%.2fs sys=%.2fs", t[1], t[2]}')
echo "$(date -u +%FT%TZ) round rc=$rc child_cpu $cpu"
exit $rc
