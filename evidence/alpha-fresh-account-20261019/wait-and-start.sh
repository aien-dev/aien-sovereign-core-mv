#!/bin/bash -l
# Wait (every 5 min, up to 2 h) for 40 GiB MemAvailable, then start the guide's server on port 18391.
for i in $(seq 0 20); do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  echo "$(date -u +%FT%TZ) try $i MemAvailable_kB $a load $(cut -d' ' -f1-3 /proc/loadavg)" >> ~/mem-wait.log
  if [ "$a" -ge $((49600*1024)) ]; then
    date +%s > ~/server.start
    PORT=18391 nice -n 10 aien-serve-27b > ~/serve-start.out 2>&1; echo "rc $?" >> ~/serve-start.out
    exit 0
  fi
  [ $i -lt 24 ] && sleep 300
done
echo GAVE_UP >> ~/mem-wait.log
