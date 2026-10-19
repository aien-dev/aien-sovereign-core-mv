#!/bin/sh
# Waits for the chain batch to finish, then runs the tree batch (each run takes the locks itself).
C=$HOME/workspace/sc-cascade/evidence/qwen38-cascade-chain-20261019T1650Z
R=$HOME/workspace/sc-cascade/evidence/qwen38-cascade-tree-20261019T1650Z/run.sh
G=$HOME/handoffs/qwen38-ref/gen-chat-max-24.json
while ! grep -q ALLDONE $C/progress.txt 2>/dev/null; do sleep 60; done
$R golden-chain --prompt golden --golden $G --n 64 --k 0,8
$R golden-tree-332 --prompt golden --golden $G --n 64 --k 0,8 --tree 3,3,2
for p in codelong prose codeedit quote; do
  $R tree-213-$p --prompt $p --n 192 --k 0,8 --tree 2,1,3
  $R tree-332-$p --prompt $p --n 192 --k 0,8 --tree 3,3,2
done
for p in codelong prose; do $R tree-412-$p --prompt $p --n 192 --k 0,8 --tree 4,1,2; done
for mu in treenorebuild treenokv extra norollback; do $R mutant-$mu-tree-prose --prompt prose --n 96 --k 0,8 --tree 3,3,2 --mutant $mu; done
echo ALLDONE $(date -u +%FT%TZ) >> evidence/qwen38-cascade-tree-20261019T1650Z/progress.txt
