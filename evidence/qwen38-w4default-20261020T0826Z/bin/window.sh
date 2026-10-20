#!/bin/bash
# Waits for the measurement window to be free, opens it for L125, runs session.sh inside it, closes it.
W=$HOME/workspace/speed1-work; log=$W/window.log
echo "$(date -u +%FT%TZ) waiting for a free window" >> $log
while :; do
  spark-window open l125-w4default --ledger L125 --lease 150 > $W/open.last 2>&1; rc=$?; [ $rc -ne 3 ] && cat $W/open.last >> $log
  [ $rc -eq 0 ] && break
  [ $rc -ne 3 ] && { echo "$(date -u +%FT%TZ) open failed rc=$rc" >> $log; exit 1; }
  sleep 5
done
id=$(spark-window id); echo "$(date -u +%FT%TZ) window $id held" >> $log
ledger note L125 "MEASURING start (window $id)"
spark-window run bash $W/bin/session.sh >> $log 2>&1
spark-window close >> $log 2>&1
ledger note L125 "MEASURING end (window $id closed)"
echo "$(date -u +%FT%TZ) done" >> $log
