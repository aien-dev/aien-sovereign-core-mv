#!/bin/sh
cd $HOME/workspace/sc-tvwire/evidence/qwen38-cascade-tvgemm-20261019
sh run.sh tree-codelong --prompt codelong --n 192 --k 0,4,8 --source mtp --verify tree
sh run.sh exact-codelong --prompt codelong --n 192 --k 0,4,8 --source mtp --verify exact
