#!/bin/bash
# L310 one-row per-launch timeline on the live pins. Run ONLY inside a granted exclusive window, e.g.
#   spark-window run ~/workspace/investigations/2026-10-24-q36-1row-timeline/run.sh
# Baseline first (live defaults), then ONE control variant with a known byte change (AIEN_Q36_EXACT_HEAD=0).
# The binary is pinned by sha; the harness itself refuses when MemAvailable < model + 12 GiB (q36_levers KEEP_FREE).
set -u
D=~/workspace/investigations/2026-10-24-q36-1row-timeline
BIN=~/workspace/targets/l310/livepins/release/q36_levers
WANT=852e380012739ef665ce79322a775dc67575a0014bbb82d0cd602b3180c87ce5
MODEL=~/models/qwen3.6-35b-a3b-fp8-95a723d0
G=~/workspace/sc-l310-ceiling/evidence/q36-host-20261020
GOLD="$G/gen-p1.json $G/gen-p2.json $G/gen-p3.json $G/gen-p4.json"
R=$D/runs/$(date -u +%Y%m%dT%H%M%SZ)-repeat; mkdir -p "$R"
sha=$(sha256sum "$BIN" | cut -d' ' -f1); [ "$sha" = "$WANT" ] || { echo "binary sha $sha != pinned $WANT" | tee "$R/ABORT"; exit 2; }
quietlock check >/dev/null 2>&1 || { echo "quiet flag held: not running" | tee "$R/ABORT"; exit 3; }
{ date -u +%FT%TZ; cat /proc/loadavg; awk '/MemAvailable/{printf "MemAvailable %.1f GiB\n",$2/1048576}' /proc/meminfo; ps -eo pid,etime,rss,args | grep '[a]ien-serve-http' | cut -c1-160; nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader 2>&1; } > "$R/ground.txt"
export Q36_LEVERS_STAMPS=40,41,42,43,44,300,301,302,303,304,1000,1001,1002,1003,1004 Q36_LEVERS_POS=1100
T0=$(date +%s)
# baseline: every switch at its live default (exact head ON, CN rows ON, 1 row)
( time "$BIN" "$MODEL" "$R/baseline.json" $GOLD ) > "$R/baseline.log" 2>&1; echo "baseline rc=$? elapsed_s=$(( $(date +%s) - T0 ))" | tee "$R/baseline.rc"
echo "done $(date -u +%FT%TZ)" >> "$R/ground.txt"
