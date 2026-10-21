#!/bin/bash
# L354: the stroke build through the same gate: 8 slots (real + row swap), then 32 slots (real). Results in run-gate.result.
cd "$(dirname "$0")"; ./run-gate.sh 829a2dfd 8; mv run-gate.result run-gate-829a2dfd-8.result
sed -i 's/^run swap40-\$SLOTS 1 AIEN_Q36_TEST_ROW_SWAP=40$/[ "$SLOTS" = 8 ] \&\& run swap40-$SLOTS 1 AIEN_Q36_TEST_ROW_SWAP=40/' run-gate.sh
./run-gate.sh 829a2dfd 32; mv run-gate.result run-gate-829a2dfd-32.result
