#!/bin/bash
# L371: requests inside a spark-window. Usage: run-docs.sh PORT OUTDIR doc:mode... (mode readers|plain)
set -uo pipefail
P=$1; O=$2; shift 2; H=$(cd "$(dirname "$0")"; pwd); mkdir -p "$O"
M=qwen3.6-35b-a3b-fp8
curl -s "http://127.0.0.1:$P/v1/models" > "$O/models.json"
for dm in "$@"; do
  d=${dm%%:*}; m=${dm##*:}; id=$M; [ "$m" = readers ] && id=$M-readers
  "$H/ask.sh" "$P" "$id" "$d" "$O/$d-$m.json"
done
