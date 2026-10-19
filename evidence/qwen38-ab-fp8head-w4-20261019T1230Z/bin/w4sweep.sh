#!/bin/bash
# w4sweep.sh TAG "ARGS1" "ARGS2" ... : one chip run per knob set, flock per run
T=$1; shift; D=$HOME/workspace/ab-w4-work/runs/w4sweep-$T; mkdir -p $D; cp $HOME/workspace/targets/ab-w4/omega-b1/gpu_w4_api_test $D/; cd $D; sha256sum gpu_w4_api_test > ground.txt
i=0; for a in "$@"; do i=$((i+1)); echo "== $a" >> sweep.log; flock /tmp/aien-gb10.lock nice ./gpu_w4_api_test --chip --reps 15 $a > run$i.log 2>&1; echo "rc=$?" >> run$i.log; grep "^  w4\|PASS\|FAIL" run$i.log >> sweep.log; done; echo DONE >> sweep.log
