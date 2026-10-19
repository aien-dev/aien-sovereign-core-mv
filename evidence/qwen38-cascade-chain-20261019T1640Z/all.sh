#!/bin/sh
R=$HOME/workspace/sc-cascade/evidence/qwen38-cascade-chain-20261019T1640Z/run.sh
for p in codelong prose codeedit quote; do $R mtp-$p --prompt $p --n 192 --k 0,1,4,8 --source mtp; done
for p in codeedit quote codelong prose; do $R merge-$p --prompt $p --n 192 --k 0,8 --source merge; done
for p in codeedit quote; do $R ngramfirst-$p --prompt $p --n 192 --k 0,8 --source ngram-first; done
for mu in extra norollback rowshift; do $R mutant-$mu-prose --prompt prose --n 96 --k 0,8 --source mtp --mutant $mu; done
$R mutant-extra-codeedit-merge --prompt codeedit --n 96 --k 0,8 --source merge --mutant extra
echo ALLDONE $(date -u +%FT%TZ) >> evidence/qwen38-cascade-chain-20261019T1640Z/progress.txt
