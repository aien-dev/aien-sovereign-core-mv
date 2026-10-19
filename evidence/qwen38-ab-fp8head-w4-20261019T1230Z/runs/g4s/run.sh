#!/bin/bash
# g4s/run.sh: MXFP4 shape sweep timed in the decode graph (2 layers, --w4), flock per run; waits for g4 to finish
cd $(dirname $0); until [ -f ../g4/done ]; do sleep 20; done
sha256sum gpu_q38_graph_test > ground.txt
i=0; for a in "--w4-shape 16,2,16" "--w4-shape 16,2,8" "--w4-shape 16,4,8" "--w4-shape 16,4,4" "--w4-shape 16,1,16" "--w4-shape 16,1,32" "--w4-shape 8,2,16" "--w4-shape 8,4,8" "--w4-ctas 768" "--w4-ctas 192" "--w4-coal 0" "--w4-sched 0"; do
  i=$((i+1)); echo "== $a" > s$i.log; flock /tmp/aien-gb10.lock nice ./gpu_q38_graph_test --w4 --reps 10 --tl-reps 11 $a >> s$i.log 2>&1; echo "rc=$?" >> s$i.log
  { echo "== $a"; grep -E "timeline (mix|o_proj|o |gate_up|down) |PASS|FAIL" s$i.log | grep -v "^ok"; } >> sweep.log
done; echo DONE >> sweep.log
