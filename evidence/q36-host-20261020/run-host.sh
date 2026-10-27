#!/bin/bash
cd /home/drakestapleton/workspace/sc-q36/evidence/q36-host-20261020
for p in p1 p2 p3 p4; do nice -n 19 /home/drakestapleton/workspace/targets/q36/sc/release/qwen38_check generate /home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0 $p.txt 32 gen-$p.json > gen-$p.log 2>&1; done
nice -n 19 /home/drakestapleton/workspace/targets/q36/sc/release/qwen38_check score /home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0 passage-readme.txt 400 score-readme.json > score-readme.log 2>&1
echo DONE > host.done
