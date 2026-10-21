#!/bin/bash
# L355 NV16 arm (w4-gate.md "NV16 arm"): go/no-go (b) on the written store, then control + w4n goldens + held-out, verdict, then
# the scale control w4ns (reported, not gated).
set -u
D=$HOME/workspace/sc-l355/evidence/l355-w4; cd $D; S=$(cat nv16-store-path.txt)
export BIN=$HOME/workspace/targets/l355/sc-merged/release/q36_w4_drift
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision
log() { echo "$(date -u +%FT%TZ) [nv16 chain] $*" >> runs/run.out; }
until [ -s runs/nv16-v1.json ]; do pgrep -f "q36_w4_store write .*nv16" > /dev/null || { log "store write ended without output"; exit 1; }; sleep 30; done
rr=$(jq -r .rel_rms runs/nv16-v1.json)
go=$(awk -v r=$rr 'BEGIN { print (r * r <= 0.0063333) ? "GO" : "NO-GO" }')
jq -n --arg rr "$rr" --arg go "$go" '{nv16_rel_rms: ($rr|tonumber), nv16_sq: (($rr|tonumber) * ($rr|tonumber)), best_rel_rms: 0.112546, limit_sq: 0.0063333, limit_calc: "calc:466b6d16", verdict: $go}' > nv16-gonogo-b.json
log "go/no-go (b): rel_rms $rr -> $go"
[ "$go" = GO ] || exit 0
quiet-build nice -n 10 env -u AIEN_COORD_KEY $BIN forced $M $R/drift-p1.json 512 fp8 runs/forced-fp8m-p1.json > runs/forced-fp8m-p1.log.txt 2>&1 &
STORE=$S ./run-arm.sh w4n w4e "" &
./heldout.sh w4n $S &
wait
ctl=$(jq -n --slurpfile a runs/forced-fp8m-p1.json --slurpfile b runs/forced-fp8-p1.json '($a[0].argmax == $b[0].argmax) and ($a[0].nll == $b[0].nll) and ($a[0].margin == $b[0].margin)')
log "control (this build fp8 p1 == forced-fp8-p1): $ctl"
./verdict-gptq.sh w4n w4e > verdict-w4n.json 2> verdict-w4n.err.txt
jq --argjson c "$ctl" '. + {control_this_build: $c} | if ($c|not) then .verdict = "VOID (this build fp8 control)" else . end' verdict-w4n.json > verdict-w4n.tmp && mv verdict-w4n.tmp verdict-w4n.json
log "verdict $(jq -r .verdict verdict-w4n.json)"
STORE=$S ./run-arm.sh w4ns w4e-smut ""
./verdict.sh w4ns > verdict-w4ns.json 2>/dev/null
log "scale control w4ns: R1 $(jq -r .R1_nll_ratio verdict-w4ns.json) (reported, not gated)"
