#!/bin/bash
# L355 mixed-in-expert arms (w4-gate.md "NV16 arm" item 4): m-gu first (nv16 gate|up, FP8 down), m-dn only if m-gu REJECTs.
# Every full-35B load runs under q36-extra (L366), one at a time: control, five goldens + AGPL score, held-out h1..h5,
# then verdict-gptq.sh <arm> w4e (A + B + C).
set -u
D=$HOME/workspace/sc-l355/evidence/l355-w4; cd $D; S=$(cat nv16-store-path.txt); O=$D/runs
B=$HOME/workspace/targets/l355/sc-merged/release/q36_w4_drift; M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0
R=$HOME/workspace/sc-hum-state-precision/evidence/hum-state-precision; G=$HOME/workspace/sc-l355/evidence/q36-host-20261020
export Q36_EXTRA_WAIT=14400
log() { echo "$(date -u +%FT%TZ) [mixed chain] $*" >> $O/run.out; }
x() { local out=$1; shift; env -u AIEN_COORD_KEY q36-extra quiet-build nice -n 10 "$@" > $out.log.txt 2>&1; local rc=$?; log "$(basename $out) exit $rc"; return $rc; }
x $O/forced-fp8m2-p1.json $B forced $M $R/drift-p1.json 512 fp8 $O/forced-fp8m2-p1.json
ctl=$(jq -n --slurpfile a $O/forced-fp8m2-p1.json --slurpfile b $O/forced-fp8-p1.json '($a[0].argmax == $b[0].argmax) and ($a[0].nll == $b[0].nll) and ($a[0].margin == $b[0].margin)')
log "control (this build fp8 p1 == forced-fp8-p1): $ctl"
[ "$ctl" = true ] || { log "control failed: stop"; exit 1; }
arm() {
  local name=$1 drift=$2
  export AIEN_Q36_W4_STORE=$S AIEN_Q36_W4_FP8_LAYERS=
  log "arm $name ($drift) start"
  for p in p1 p2 p3 p4 chat-max-24; do x $O/forced-$name-$p.json $B forced $M $R/drift-$p.json 512 $drift $O/forced-$name-$p.json; done
  x $O/score-$name.json $B score $M $G/passage-agpl.txt 400 $drift $O/score-$name.json
  for i in 1 2 3 4 5; do x $O/forced-$name-h$i.json $B forced $M $O/forced-fp8-h$i.json 512 $drift $O/forced-$name-h$i.json; done
  ./verdict-gptq.sh $name w4e > verdict-$name.json 2> verdict-$name.err.txt
  jq '. + {control_this_build: true}' verdict-$name.json > verdict-$name.tmp && mv verdict-$name.tmp verdict-$name.json
  log "arm $name verdict $(jq -r .verdict verdict-$name.json)"
}
arm w4mgu w4e-mgu
case "$(jq -r .verdict verdict-w4mgu.json)" in ACCEPT*) log "m-gu ACCEPT: m-dn not run (order frozen)"; exit 0 ;; esac
arm w4mdn w4e-mdn
