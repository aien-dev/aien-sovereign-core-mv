#!/bin/sh
# jtok.sh RUN_DIR : wall joules per token over exactly the decode span of a qwen38_ab run (out.json "decode_unix",
# "steps"), integrated from the UPS log ~/yardsticks/sensors/<local date>.csv (col 1 unix s, col 3 wall W; samples
# every ~2 s in 10 W steps, so read it as coarse). Prints: span_s samples mean_W joules tokens J_per_token.
set -eu; d=$1
s=$(jq -r '.decode_unix|"\(.[0]),\(.[1])"' "$d/out.json")
t0=${s%,*}; t1=${s#*,}; n=$(grep -o '"steps": *[0-9]*' "$d/out.json" | grep -o '[0-9]*$')
f0=$HOME/yardsticks/sensors/$(date -d @${t0%.*} +%F).csv; f1=$HOME/yardsticks/sensors/$(date -d @${t1%.*} +%F).csv
cat $f0 $( [ "$f1" != "$f0" ] && echo $f1 ) | awk -F, -v t0=$t0 -v t1=$t1 -v n=$n '
  { t[NR]=$1; w[NR]=$3 } END {
    e=0; k=0; sw=0
    for (i=1;i<NR;i++) { a=t[i]; b=t[i+1]; if (b<=t0 || a>=t1) continue
      lo=(a>t0)?a:t0; hi=(b<t1)?b:t1; e+=w[i]*(hi-lo); k++; sw+=w[i] }
    if (k==0) { print "UNMEASURED: no wall samples in the span"; exit 1 }
    printf "span_s %.2f samples %d mean_W %.1f joules %.1f tokens %d J_per_token %.3f\n", t1-t0, k, sw/k, e, n, e/n }'
