#!/bin/sh
# Ctrl-C replay for core#688 review finding 1.
# Starts aien-browser-mirror as leader of a new session/process group (as a
# terminal foreground job or a harness timeout group would) with SIGINT reset
# to its default action (a non-interactive sh starts background jobs with
# SIGINT ignored), sends SIGINT to that group, and counts this run's Chrome
# processes that remain once the caller is dead.
# usage: sigint-replay.sh BIN
BIN=$1
setsid perl -e '$SIG{INT}="DEFAULT"; exec @ARGV' "$BIN" --url "data:text/html,hi" \
  --eval "new Promise(r=>setTimeout(r,60000))" >/dev/null 2>&1 &
leader=$!
# The tag and profile name are aien-browser-mirror-<caller pid>-<nanos>.
TAG="aien-browser-mirror-$leader-"
owned() {
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$p" = "$leader" ] && continue
    e=$(tr '\0' '\n' 2>/dev/null <"$d/environ")
    c=$(tr '\0' ' ' 2>/dev/null <"$d/cmdline")
    case "$e
$c" in
      *"$TAG"*) echo "$p" ;;
    esac
  done
}
i=0
until [ -n "$(owned)" ] || [ $i -ge 100 ]; do sleep 0.2; i=$((i+1)); done
sleep 2
echo "caller pid $leader, $(grep -E '^SigIgn' /proc/$leader/status)"
echo "owned processes while running: $(owned | wc -l)"
/bin/kill -s INT -- "-$leader"
sleep 4
echo "caller state 4 s after SIGINT to its group: $(awk '/^State/{print $2}' /proc/$leader/status 2>/dev/null || echo gone)"
left=$(owned)
echo "owned processes left: $(echo "$left" | grep -c .)"
for p in $left; do printf '%s %s\n' "$p" "$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-120)"; done
# cleanup: only the processes counted above
[ -n "$left" ] && kill -KILL $left 2>/dev/null
wait "$leader" 2>/dev/null
echo "caller exit status (wait): $?"
sleep 1
echo "owned after cleanup: $(owned | wc -l)"
exit 0
