#!/bin/bash
# Summarise one q36_levers json: identity, unstamped bands, per-kind means over the stamped positions.
# Numbers here are raw sums/means from the json (jq); the figures quoted in the report are re-done in aien-calc + Wolfram.
J=$1; [ -f "$J" ] || { echo "no json $J"; exit 1; }
echo "file: $J"; jq -r '"all_identical=\(.all_identical) goldens=\([.goldens[].identical]|map(tostring)|join(",")) env=\(.env|join(" "))"' "$J"
echo "BANDS (unstamped tokens, per token):"; jq -r '.bands[] | "  pos [\(.lo),\(.hi)) tokens=\(.tokens) chip_ms=\(.chip_ms|.*1000|round/1000) graph_total_ms=\(.graph_total_ms|.*1000|round/1000) wall_ms=\(.wall_ms|.*1000|round/1000)"' "$J"
echo "STAMPED positions:"; jq -r '.timelines[] | "  pos \(.pos) launches=\(.launches) span_us=\(.span_ns/1000|round) chip_us=\(.chip_ns/1000|round) wall_us=\(.wall_ns/1000|round) bytes=\(.bytes|round)"' "$J"
echo "PER KIND, mean over stamped positions (ns and bytes per token; gap273 = ns - bytes/273e9*1e9):"
jq -r '[.timelines[].kinds[]] | group_by(.kind) | map({kind: .[0].kind, n: length, launches: (map(.launches)|add/length), ns: (map(.ns)|add/length), bytes: (map(.bytes)|add/length)}) | sort_by(-.ns)[] | "  \(.kind)\tlaunches=\(.launches|round)\tmean_ns=\(.ns|round)\tmean_bytes=\(.bytes|round)\tbyte_ns@273=\(.bytes/273e9*1e9|round)\tgap_ns=\((.ns - .bytes/273e9*1e9)|round)"' "$J"
jq -r '[.timelines[]] | "TOTAL stamped mean: span_us=\((map(.span_ns)|add/length)/1000|round) chip_us=\((map(.chip_ns)|add/length)/1000|round) bytes=\((map(.bytes)|add/length)|round) launches=\((map(.launches)|add/length)|round)"' "$J"
