#!/bin/sh
R=$HOME/workspace/sc-cascade/evidence/qwen38-cascade-suite-20261019T2130Z/run-suite.sh
for s in A-mtp B-merge C-mutants D-tree; do $R $s; ~/workspace/targets/sc-cascade/publish.sh; done
echo ALLDONE $(date -u +%FT%TZ) >> $HOME/workspace/sc-cascade/evidence/qwen38-cascade-suite-20261019T2130Z/progress.txt
