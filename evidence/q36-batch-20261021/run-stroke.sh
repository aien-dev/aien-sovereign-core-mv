#!/bin/bash
# L354: a stroke build through the same gate: 8 slots (real + row swap), then 32 slots (real only, SWAP=0).
# Usage: run-stroke.sh [sha8, default 13f06473 = strokes + exact head + levers]. Results in run-gate-<sha>-<slots>.result.
cd "$(dirname "$0")"; SHA=${1:-13f06473}
./run-gate.sh $SHA 8; mv run-gate.result run-gate-$SHA-8.result
SWAP=0 ./run-gate.sh $SHA 32; mv run-gate.result run-gate-$SHA-32.result
