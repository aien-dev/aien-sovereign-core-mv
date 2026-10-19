#!/usr/bin/env bash
# prints the I64 values of a one-tensor safetensors file as a JSON array
f=$1
hl=$(od -An -tu8 -N8 "$f" | tr -d ' ')
od -An -td8 -j $((8 + hl)) -v "$f" | tr -s ' \n' '\n' | grep -v '^$' | jq -s -c .
