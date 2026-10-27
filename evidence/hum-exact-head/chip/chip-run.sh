#!/bin/bash
# chip-run.sh: L314 chip receipt (2026-10-21). The spark-window line is stuck (L361), so this takes the chip the other
# allowed way: quiet-measure (builds drained, fast cores) + flock /tmp/aien-gb10.lock (the lock 18091 takes per request).
# gpu_exact_head_test --chip: full bf16 head list vs two-pass list on all 1010 hidden records (token identical,
# candidate logits bit-identical, timing), then --mutant 1 (XBOUND without the bound) must lose tokens.
set -u
# 12:25Z: the first run (11:39Z) hung 28 min holding the chip lock: the test created its command lists while holding the
# session lock, and cmdlist_create takes that lock itself (not recursive). Fixed in omega; every run is now capped at 300 s.
D=$(cd "$(dirname "$0")" && pwd)
T=$HOME/workspace/targets/hum-exact-head
X=$T/omega-stroke/gpu_exact_head_test   # omega hum/exact-head-stroke 27290d1d, physics.lock 6d7cf0d4
mem=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$mem" -ge 8 ] || { echo "MemAvailable ${mem} GB < 8 GB, not loading"; exit 1; }
{ date -u +%FT%TZ; sha256sum $X $T/hidden.bin $T/chip/*; } > $D/inputs.txt
quiet-measure flock /tmp/aien-gb10.lock timeout 300 $X --chip $T/chip $T/hidden.bin --out $D/chip-receipt.json > $D/chip.txt 2>&1; echo "chip rc $?" >> $D/chip.txt
quiet-measure flock /tmp/aien-gb10.lock timeout 300 $X --chip $T/chip $T/hidden.bin --limit 200 --mutant 1 --out $D/chip-mutant1.json > $D/chip-mutant1.txt 2>&1; echo "mutant rc $?" >> $D/chip-mutant1.txt
echo DONE > $D/chip.done
