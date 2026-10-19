#!/bin/bash
# g4/run.sh: graph chip test, FP8 then MXFP4 then MXFP4 + head, flock per run
cd $(dirname $0); sha256sum gpu_q38_graph_test > ground.txt; grep MemAvailable /proc/meminfo >> ground.txt
i=0; for a in "" "--w4" "--w4 --w4-head" "--fp8-head"; do i=$((i+1)); echo "== $a" > g$i.log; flock /tmp/aien-gb10.lock nice ./gpu_q38_graph_test --reps 20 --tl-reps 11 $a >> g$i.log 2>&1; echo "rc=$?" >> g$i.log; done; echo DONE > done
