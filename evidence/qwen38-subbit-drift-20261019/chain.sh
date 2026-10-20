#!/bin/bash
# Runs the coded variants one after another once the reference logits exist (L134).
# Usage: chain.sh <variant> [<variant> ...]   (variant names double as out-dir names, ':' -> '_')
cd "$(dirname "$0")"
B=./qwen38_subbit.ref-build   # frozen copy of the build used for the reference
while ! [ -f ref/summary.json ]; do sleep 30; done
for v in "$@"; do
  d=out-$(echo "$v" | tr ':' '_')
  echo "$(date -u +%T) start $v" >> chain.log
  nice -n 5 $B ~/models/qwen3.8-27b-fp8-017b9c7a ~/workspace/ab-w4-work/seq "$d" "$v" --ref-dir ref --codebooks cb > "$d.log" 2>&1
  echo "$(date -u +%T) done $v rc=$?" >> chain.log
done
