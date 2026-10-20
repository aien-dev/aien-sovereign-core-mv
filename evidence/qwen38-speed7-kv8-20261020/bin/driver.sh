#!/bin/bash
# driver.sh: for L131, takes a place in the spark-window line (--wait, first come first served) and, in each window,
# runs every phase not done yet whose memory need MemAvailable covers right now (kernel, model, speed, long, in order).
# A phase that does not fit waits for the next window; the window closes as soon as nothing more fits.
# Detached (setsid nohup). Before taking a place in line it waits, holding nothing, for the smallest pending need.
set -u; cd $HOME/workspace/speed7-work; LOG=runs/driver.log; D=runs/done
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
room() { [ $(awk '/MemAvailable/{print $2}' /proc/meminfo) -ge $(($1 * 1024 * 1024)) ]; }
# name GiB script
# phases: runs/phases.txt, one "name GiB script" per line, read again each round (so phases can be added while in line)
mkdir -p $D
pending() { while read -r n g s; do [ -n "$n" ] && [ ! -f $D/$n ] && echo "$n $g $s"; done < runs/phases.txt; }
while [ -n "$(pending)" ]; do
  min=$(pending | awk 'NR==1||$2<m{m=$2} END{print m}')
  say "pending: $(pending | awk '{printf "%s(%s GiB) ", $1, $2}'); waiting for ${min} GiB, then in line"
  until room "$min"; do sleep 10; done
  id=$(spark-window open l131-measure --ledger L131 --lease 150 --wait 2>> $LOG | tail -1)
  [ -n "$id" ] || { say "open failed, retry in 30 s"; sleep 30; continue; }
  say "window $id open"; ledger note L131 "measure: window $id open" > /dev/null 2>&1
  ran=0
  while read -r name mem script; do
    if room "$mem"; then
      say "run $name in $id"; spark-window run "$script" >> $LOG 2>&1 && touch $D/$name; ran=1
      say "end $name in $id"
    else
      say "skip $name in $id: MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB < $mem"
    fi
  done < <(pending)
  spark-window close "$id" >> $LOG 2>&1
  say "window $id closed"; ledger note L131 "measure: window $id closed; done: $(ls $D | tr '\n' ' ')" > /dev/null 2>&1
  [ $ran = 1 ] || sleep 300
done
say "DONE all phases"
