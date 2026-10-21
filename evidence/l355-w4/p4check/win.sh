#!/bin/bash
# L355 gen-p4 check, windowed (WL=5; the full tool's memory margin cannot be met beside the 8-row 18091): FP8 gen-p4 through
# qwen38_omega_full, base (sc 0915000b + lib 678df626) vs mine (sc pre-merge + lib 979f928a). Same golden, same windows.
set -u
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; E=$HOME/workspace/sc-l355/evidence/l355-w4/p4check; G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
unset AIEN_Q36_W4_STORE AIEN_OMEGA_Q36_OVERLAP_AB
for arm in base mine; do
  B=$HOME/workspace/targets/l355/sc-base/release/qwen38_omega_full; [ $arm = mine ] && B=$HOME/workspace/targets/l355/sc/release/qwen38_omega_full
  for try in 1 2 3 4 5; do
    echo "$(date -u +%FT%TZ) win-$arm try $try bin $(sha256sum $B | cut -c1-16) avail_kb $(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> $E/run.out
    env -u AIEN_COORD_KEY flock /tmp/aien-gb10.lock nice -n 5 $B $M 5 $G/gen-p4.json $E/win-$arm.json > $E/win-$arm.log.txt 2>&1; rc=$?
    echo "$(date -u +%FT%TZ) win-$arm exit $rc $(grep '^QWEN38_OMEGA_FULL' $E/win-$arm.log.txt | cut -c1-260)" >> $E/run.out
    [ $rc = 2 ] || break; sleep 60
  done
done
