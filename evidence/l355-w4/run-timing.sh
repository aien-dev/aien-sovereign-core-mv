#!/bin/bash
cd $HOME/workspace/sc-l355/evidence/l355-w4
spark-window open l355-w4 --ledger L355 --gb10 none --wait > runs/window.out 2>&1 || { echo "window refused: $(tail -2 runs/window.out)" >> timing.log.txt; exit 3; }
spark-window run ./timing.sh >> runs/timing.out 2>&1
spark-window close >> runs/window.out 2>&1
