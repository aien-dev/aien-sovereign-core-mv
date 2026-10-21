#!/bin/bash
# L355 gen-p4 near-tie check (w4-results.md open item). One variable: the code under the same tool (q36_levers, FP8, overlap
# off = omega default, no env), base sc 0915000b + lib 678df626 vs this branch's build (pre-merge, lib 979f928a). gen-p4 + gen-p1.
set -u
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; E=$HOME/workspace/sc-l355/evidence/l355-w4/p4check; G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
unset AIEN_Q36_W4_STORE AIEN_OMEGA_Q36_OVERLAP_AB
for arm in base mine; do for try in 1 2 3 4 5 6; do
  B=$HOME/workspace/targets/l355/sc-base/release/q36_levers; [ $arm = mine ] && B=$HOME/workspace/targets/l355/sc/release/q36_levers
  echo "$(date -u +%FT%TZ) $arm bin $(sha256sum $B | cut -c1-16) avail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> $E/run.out
  env -u AIEN_COORD_KEY Q36_EXTRA_NEED_GB=50 q36-extra flock /tmp/aien-gb10.lock nice -n 5 $B $M $E/levers-$arm.json $G/gen-p4.json $G/gen-p1.json > $E/levers-$arm.log.txt 2>&1
  rc=$?; echo "$(date -u +%FT%TZ) $arm try $try exit $rc $(tail -1 $E/levers-$arm.log.txt | cut -c1-200)" >> $E/run.out
  grep -q refusing $E/levers-$arm.log.txt || break; sleep 300; done; done
