#!/bin/bash
# L375: the chip gate inside its own spark-window (gb10 + quiet exclusive: the gate reports times),
# first come first served behind the windows already in line. Usage: run-window.sh <bin dir>
set -uo pipefail
cd "$(dirname "$0")"
out=$(spark-window open l375-prefix --ledger L375 --lease 60 --wait 2>&1)
id=$(grep -oE '[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}' <<<"$out" | tail -1)
[ -n "$id" ] || { echo "no window: $out"; exit 1; }
echo "$(date -u +%FT%TZ) window $id open"
spark-window run ./gate.sh "$1"
spark-window close "$id"
echo "$(date -u +%FT%TZ) window $id closed"
