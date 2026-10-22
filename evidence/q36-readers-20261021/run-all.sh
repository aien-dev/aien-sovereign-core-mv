#!/bin/bash
# L371, 18091 down for tests (coordinator 13:59Z): A = 8 rows x 20480: readers on all four docs + plain on the two that fit;
# B = 1 row x 122880: plain on the 54k log and the 117k logs (one token per beat, the path the readers replace).
H=$(cd "$(dirname "$0")"; pwd)
"$H/run-chip.sh" all8x20480 8 20480 design:readers design:plain code:readers code:plain log:readers big:readers
LEASE=180 ASK_MAX_TIME=9000 "$H/run-chip.sh" plain1x122880 1 122880 log:plain big:plain
ledger note L371 "chip runs finished: $H/chip-all8x20480, $H/chip-plain1x122880 (run-all.log)"
