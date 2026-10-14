#!/usr/bin/env bash
# OWN-30B W-L2h: full-model gate for the resident decode chain (omega AIEN_OMEGA_DGRAPH_RESIDENT=1).
#
#   scripts/own30b-resident-gate.sh [evidence-dir]
#
# Runs ONLY in a pause window: loads the 30 GB model four times (one at a time, under /tmp/aien-gb10.lock through
# scripts/own30b-chip-run.sh). It never stops a server; it refuses when less than 70 GB is available.
# Runs, back to back and interleaved, all with the one-list decode on (OWN30B_DECODE_GRAPH=1):
#   off-a  resident OFF      on-a  resident ON (grid ${AIEN_OMEGA_DGRAPH_RESIDENT_GRID:-128})
#   off-b  resident OFF      on-b  resident ON
# each = own30b_check golden tokens + parity, prompts p1..p4, 16 new tokens (the RC flags).
# BASELINE=1 adds a first run with the one-list decode off (the live :18091 path).
# Verdict: TOKENS PASS needs SUMMARY tokens=PASS and IDENTICAL on p1..p4 in every run; SPEED prints the decode tok/s
# per prompt (ON minus OFF). GO only if tokens PASS and ON is faster than OFF on all four prompts in both pairs.
set -uo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
ev=${1:-$repo/docs/inference/evidence/own30b-L2h-resident-$(date -u +%Y%m%dT%H%MZ)}
omega=${AIEN_OMEGA_DIR:-$HOME/workspace/omega-own30b-decode}
physics=${AIEN_PHYSICS_DIR:-$HOME/workspace/physics-own30b-k}
golden=$repo/docs/inference/evidence/own30b-f-20261013T1745Z/golden.json
args=(--prompts 1,2,3,4 --new 16 --layers 0,23,47 --weights fp8 --attn fused --router device --lm-head gemv --prefill auto)
avail=$(free -g | awk '/^Mem:/ {print $7}')
[ "${avail:-0}" -ge 70 ] || { echo "RESIDENT_GATE: REFUSED only ${avail} GB available (needs 70: pause :18091 first)"; exit 2; }
mkdir -p "$ev"
( cd "$repo" && AIEN_OMEGA_DIR="$omega" AIEN_PHYSICS_DIR="$physics" AIEN_REQUIRE_NATIVE_LIBS=1 \
    cargo build -j4 --release -p aien-inference-abi --bin own30b_check ) > "$ev/build.log" 2>&1 \
  || { echo "RESIDENT_GATE: FAIL build (see $ev/build.log)"; exit 1; }
bin=$ev/own30b_check
cp "$repo/target/release/own30b_check" "$bin"
{ echo "sc $(git -C "$repo" rev-parse HEAD) omega $(git -C "$omega" rev-parse HEAD) omega.lock $(cat "$repo/omega.lock")"
  echo "args ${args[*]}"; } > "$ev/gate.txt"
export AIEN_OMEGA_SPIN_US=${AIEN_OMEGA_SPIN_US:-2000}
run() { # name, resident (0|1), graph (0|1)
  local n=$1
  if [ "$3" = 1 ]; then export OWN30B_DECODE_GRAPH=1; else unset OWN30B_DECODE_GRAPH; fi
  if [ "$2" = 1 ]; then export AIEN_OMEGA_DGRAPH_RESIDENT=1; else unset AIEN_OMEGA_DGRAPH_RESIDENT; fi
  echo "$n: OWN30B_DECODE_GRAPH=${OWN30B_DECODE_GRAPH:-} AIEN_OMEGA_DGRAPH_RESIDENT=${AIEN_OMEGA_DGRAPH_RESIDENT:-} GRID=${AIEN_OMEGA_DGRAPH_RESIDENT_GRID:-128}" >> "$ev/gate.txt"
  OWN30B_WHISPER="OWN-30B W-L2h resident gate: $n (30 GB load), ~5 min" "$repo/scripts/own30b-chip-run.sh" "$ev" "$n" "$bin" "$golden" "${args[@]}"
}
[ "${BASELINE:-0}" = 1 ] && run base 0 0
run off-a 0 1; run on-a 1 1; run off-b 0 1; run on-b 1 1
tok() { grep -E "^TOKENS p$2" "$ev/$1.raw" | sed -E 's/.* = ([0-9.]+) tok\/s.*/\1/'; }
tokens=PASS
for n in off-a on-a off-b on-b; do
  grep -q 'SUMMARY.*tokens=PASS' "$ev/$n.raw" || tokens=FAIL
  [ "$(grep -cE '^TOKENS p[1-4].*IDENTICAL' "$ev/$n.raw")" = 4 ] || tokens=FAIL
done
grep -q 'per graph token launches 0.0' "$ev/on-a.raw" && tokens=FAIL # the one-list decode must have run
go=GO
{ echo "prompt  off-a  on-a  off-b  on-b (decode tok/s)"
  for p in 1 2 3 4; do
    a=$(tok off-a $p); b=$(tok on-a $p); c=$(tok off-b $p); d=$(tok on-b $p)
    echo "p$p  $a  $b  $c  $d"
    awk -v a="$a" -v b="$b" -v c="$c" -v d="$d" 'BEGIN { exit !(b > a && d > c) }' || go=NO-GO
  done
  grep -h 'DECODE GRAPH' "$ev"/on-a.raw | head -1
} | tee "$ev/speed.txt"
[ "$tokens" = PASS ] || go=NO-GO
( cd "$ev" && find . -type f ! -name SHA256SUMS ! -name own30b_check -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "RESIDENT_GATE: tokens=$tokens speed=$go evidence=$ev"
[ "$tokens" = PASS ]
