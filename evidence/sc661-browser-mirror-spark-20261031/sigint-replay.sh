#!/bin/sh
# Ctrl-C replay for core#688 review finding 1.
# Starts aien-browser-mirror as leader of its own process group (as a terminal
# foreground job or a harness timeout group would), sends SIGINT to that group,
# then counts surviving processes that carry this run's Chrome profile/tag.
# usage: t688-sigint.sh BIN
BIN=$1
PAT='aien-browser-mirror-[0-9]'
before=$(pgrep -fc "$PAT")
setsid sh -c 'exec "$0" --url "data:text/html,hi" --eval "new Promise(r=>setTimeout(r,60000))"' "$BIN" >/tmp/prfix/t688-sigint.out 2>&1 &
leader=$!
i=0
until [ "$(pgrep -fc "$PAT")" -gt "$before" ] || [ $i -ge 100 ]; do sleep 0.2; i=$((i+1)); done
sleep 2
echo "chrome tagged processes while running: $(($(pgrep -fc "$PAT") - before))"
/bin/kill -s INT -- "-$leader"
sleep 4
left=$(($(pgrep -fc "$PAT") - before))
echo "leader alive after SIGINT: $(kill -0 "$leader" 2>/dev/null && echo yes || echo no)"
echo "tagged processes left 4 s after SIGINT to the caller's group: $left"
pgrep -fa "$PAT" | cut -c1-160
# cleanup of this run's leftovers only (pattern carries the profile/tag prefix)
pkill -KILL -f "$PAT" 2>/dev/null
/bin/kill -s KILL -- "-$leader" 2>/dev/null
sleep 1
echo "after cleanup: $(pgrep -fc "$PAT")"
exit 0
