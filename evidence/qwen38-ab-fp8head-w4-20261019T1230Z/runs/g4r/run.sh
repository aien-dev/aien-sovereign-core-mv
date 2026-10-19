#!/bin/bash
# g4r/run.sh: 3 interleaved rounds of the top MXFP4 shapes, graph timeline (2 layers --w4), flock per run
cd $(dirname $0); until grep -q DONE ../g4s/sweep.log; do sleep 20; done
sha256sum gpu_q38_graph_test > ground.txt
for r in 1 2 3; do for a in "--w4-shape 16,2,16 --w4-coal 0" "--w4-shape 16,2,8 --w4-coal 0" "--w4-shape 16,2,8" "--w4-shape 16,2,16" "--w4-shape 16,4,4 --w4-coal 0"; do
  n="r$r-$(echo $a | tr -d ' -' )"; flock /tmp/aien-gb10.lock nice ./gpu_q38_graph_test --w4 --reps 10 --tl-reps 11 $a > $n.log 2>&1; echo "rc=$?" >> $n.log
  echo "$r | $a | $(grep -E 'timeline (mix|o_proj|gate_up|down) ' $n.log | awk '{s+=$5} END{print s}') us | $(grep -o '[0-9]* checks, [0-9]* failed' $n.log)" >> table.txt
done; done; echo DONE >> table.txt
