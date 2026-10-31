#!/bin/bash
# chip.sh: L314 chip receipt. Waits for the export, then a chip window (first come, first served), then
# gpu_exact_head_test --chip on all 1010 hidden records (identity + timing) and --mutant 1 (must lose tokens).
set -u
E=$HOME/workspace/sc-hum-exact-head/evidence/hum-exact-head
T=$HOME/workspace/targets/hum-exact-head
X=$T/omega/gpu_exact_head_test
cd $E
until grep -q "export rc" export.txt 2>/dev/null; do sleep 30; done
grep -q "export rc 0" export.txt || { echo "export failed"; exit 1; }
mem=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
[ "$mem" -ge 8 ] || { echo "MemAvailable ${mem} GB < 8 GB, not loading"; exit 1; }
ledger note L314 "measure: in line for a window (chip receipt, ~1.6 GB device, short)" >/dev/null 2>&1
spark-window open l314-exact-head --ledger L314 --lease 30 --wait || { echo "no window"; exit 1; }
spark-window run quiet-measure $X --chip $T/chip $T/hidden.bin --out chip-receipt.json > chip.txt 2>&1; echo "chip rc $?" >> chip.txt
spark-window run quiet-measure $X --chip $T/chip $T/hidden.bin --limit 200 --mutant 1 --out chip-mutant1.json > chip-mutant1.txt 2>&1; echo "mutant rc $?" >> chip-mutant1.txt
spark-window close
echo DONE > chip.done
