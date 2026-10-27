#!/bin/bash
# speed.sh TAG: two 64/576 pairs, print wall seconds
cd $CLAUDE_JOB_DIR/tmp; for r in 1 2; do for n in 64 576; do s=$(date +%s.%N); curl -s http://127.0.0.1:18091/v1/chat/completions -H 'content-type: application/json' -d @d$n.json > /dev/null; e=$(date +%s.%N); echo "$1 r$r $n $(echo "$e - $s" | bc)"; done; done
