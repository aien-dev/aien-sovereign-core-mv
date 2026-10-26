#!/bin/bash
# L355 low-rank arm (docs/inference/q36/w4-gate.md "Low-rank arm"): every step that loads the full 35B on the host.
# Runs ONLY inside an exclusive window Drake authorised (a second full 35B load beside the live one OOM-killed 18091 on
# 2026-10-21). Host CPU only: no GPU, no q36-extra, no service touched. Resumable: a step whose output exists is kept, so
# a window that ends early is continued by the next one.
#
#   As a Q27 window hook (same outage as the 27B qualification; q27-window.sh stops and restores 18091):
#     ln -s <this file> /home/drakestapleton/handoffs/Q27/hooks.d/50-w4-lowrank
#     Q27_MAX_OUTAGE_MIN=<minutes> Q27_AUTH='<who/when>' q27-window.sh --live
#   (--dry-run: this hook only checks its inputs and prints the plan.) Steps that would end past Q27_DEADLINE_EPOCH are
#   left NOT_RUN for the next window.
#   Standalone, with 18091 already stopped by the window procedure:
#     W4C_AUTH='<who/when authorised>' run-lowrank-window.sh
#
# Order: control -> calibration (hdump) -> calibrated screen; NO-GO on both screens stops (diagnostics still run);
# GO -> FP8 and m-dn on fresh h6..h10 -> the arm on the goldens, AGPL and h6..h10 -> verdict -> diagnostics.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
D=$HOME/workspace/sc-l355/evidence/l355-w4; O=$D/runs              # m-dn's runs, the FP8 references, verdict.sh's W4_DIR
E=$HOME/handoffs/W4Correct                                          # this lane's evidence
M=$HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; MX=$M.mxfp4; S=$MX/q36-nv16-v1.store; HD=$MX/hdn
T=$HOME/workspace/targets/w4c/sc/release; B=$T/q36_w4_drift; ST=$T/q36_w4_store
R=$E/l313-ref; G=$HOME/workspace/sc-l355/evidence/q36-host-20261020; CAL=$HOME/handoffs/l355/calib/calib-v1.txt
export L313_REF=$R W4_DIR=$D
NEED_MIB=60000   # one forced run maps the 35B (~35 GB); hdump holds ~22 GB of H on top
log() { echo "$(date -u +%FT%TZ) [w4c window] $*" | tee -a $E/window.log; }

# ---- inputs (also the dry run)
miss=0
for f in $B $ST $S $CAL $R/drift-p1.json $R/drift-chat-max-24.json $G/passage-agpl.txt $O/forced-fp8-p1.json $O/forced-fp8-h1.json \
  $O/forced-w4mdn-p1.json $O/forced-w4mdn-chat-max-24.json $E/screen-plain.json $HERE/verdict-gptq.sh $HERE/heldout/h6.json; do
  [ -s $f ] || { echo "missing input: $f"; miss=1; }
done
[ $miss = 0 ] || exit 2
if [ "${Q27_MODE:-}" = --dry-run ]; then
  echo "w4c: inputs present; plan: control, hdump 32768 tokens, calibrated screen, then (GO) 20 forced runs + score + verdict, then routes"
  exit 0
fi

# ---- authorisation and the live 35B
if [ "${Q27_MODE:-}" = --live ]; then AUTH="q27-window.sh --live run ${Q27_RUN:-?}"
else AUTH=${W4C_AUTH:-}; [ -n "$AUTH" ] || { echo "w4c: needs W4C_AUTH='<who/when authorised>' or a Q27 --live hook" >&2; exit 2; }; fi
if ss -ltn 2>/dev/null | grep -q ':18091 '; then echo "w4c: 18091 is listening; the live 35B must be stopped by the window first" >&2; exit 2; fi
log "start ($AUTH); build $(sha256sum $B | cut -c1-16) / $(sha256sum $ST | cut -c1-16); head $(git -C $HERE rev-parse --short HEAD)"

left_s() { [ -n "${Q27_DEADLINE_EPOCH:-}" ] && echo $(( Q27_DEADLINE_EPOCH - $(date +%s) )) || echo 999999; }
mem_ok() { local a; a=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo); [ $a -ge $NEED_MIB ] || { log "MemAvailable $a MiB < $NEED_MIB: stop"; exit 3; }; }
# x <minutes estimate> <out> <cmd...>: kept when the output exists; skipped NOT_RUN when it would end past the deadline
x() {
  local min=$1 out=$2; shift 2
  [ -s $out ] && jq -e . $out > /dev/null 2>&1 && { log "$(basename $out) kept"; return 0; }
  [ $(left_s) -gt $((min * 60)) ] || { log "$(basename $out) NOT_RUN (window deadline)"; return 9; }
  mem_ok
  nice -n 10 "$@" > $out.tmp 2> $out.log.txt; local rc=$?
  [ $rc = 0 ] && mv $out.tmp $out || rm -f $out.tmp
  log "$(basename $out) exit $rc"; return $rc
}
# drift runs write their own output file: f <out> <args after the binary...>; kept when <out> is complete JSON
f() {
  local out=$1; shift
  [ -s $out ] && jq -e '.nll // .score // .' $out > /dev/null 2>&1 && { log "$(basename $out) kept"; return 0; }
  [ $(left_s) -gt 1200 ] || { log "$(basename $out) NOT_RUN (window deadline)"; return 9; }
  mem_ok
  rm -f $out; nice -n 10 $B "$@" > $out.log.txt 2>&1; local rc=$?
  log "$(basename $out) exit $rc"; [ $rc = 0 ] && [ -s $out ]
}

# ---- 1. exactness control with this build (must equal forced-fp8-p1.json)
f $O/forced-fp8w4c-p1.json forced $M $R/drift-p1.json 512 fp8 $O/forced-fp8w4c-p1.json || exit 4
ctl=$(jq -n --slurpfile a $O/forced-fp8w4c-p1.json --slurpfile b $O/forced-fp8-p1.json '($a[0].argmax == $b[0].argmax) and ($a[0].nll == $b[0].nll) and ($a[0].margin == $b[0].margin)')
log "control (this build fp8 p1 == forced-fp8-p1): $ctl"; [ "$ctl" = true ] || exit 4

# ---- 2. calibration and the calibrated screen (b-C)
mkdir -p $HD
# disk floor (Main, 2026-10-26): / keeps >= 150 GiB free; hdump writes two H files of 40 x 257 x 512^2 f32 (21.6 GB) and
# lowrank the calibrated sidecar (0.84 GB)
[ -s $HD/calib-v1-32768-b.hdn ] || [ $(df -B1 --output=avail / | tail -1) -ge $(( (150 + 24) * 1024 * 1024 * 1024 )) ] \
  || { log "disk: less than the 150 GiB floor + 24 GiB free: stop"; exit 5; }
if ! [ -s $HD/calib-v1-32768-b.hdn ]; then
  x 90 $E/hdump.json $ST hdump $M $CAL 32768 $HD/calib-v1-32768 || exit 5
fi
x 60 $E/screen-cal.json $ST lowrank $M $S 16 $MX/q36-lr16-cal.lr $HD/calib-v1-32768-a.hdn $HD/calib-v1-32768-b.hdn || exit 5
if [ "$(jq -r .verdict $E/screen-cal.json)" = GO ]; then LR=$MX/q36-lr16-cal.lr; FIT=calibrated
elif [ "$(jq -r .verdict $E/screen-plain.json)" = GO ]; then LR=$MX/q36-lr16-plain.lr; FIT=plain
else LR=""; FIT=none; fi
log "screens: calibrated $(jq -c '[.verdict,.gate_ratio]' $E/screen-cal.json), plain $(jq -c '[.verdict,.gate_ratio]' $E/screen-plain.json) -> arm fit $FIT"

if [ -n "$LR" ]; then
  # ---- 3. fresh held-out h6..h10: FP8 greedy reference, then m-dn (the paired base) forced on its ids
  for i in 6 7 8 9 10; do f $O/forced-fp8-h$i.json forced $M $HERE/heldout/h$i.json 512 fp8 $O/forced-fp8-h$i.json || exit 6; done
  export AIEN_Q36_W4_STORE=$S AIEN_Q36_W4_FP8_LAYERS=
  for i in 6 7 8 9 10; do f $O/forced-w4mdn-h$i.json forced $M $O/forced-fp8-h$i.json 512 w4e-mdn $O/forced-w4mdn-h$i.json || exit 6; done
  # ---- 4. the arm: goldens, AGPL score, h6..h10
  export AIEN_Q36_W4_LOWRANK=$LR
  for p in p1 p2 p3 p4 chat-max-24; do f $O/forced-w4mdnlr-$p.json forced $M $R/drift-$p.json 512 w4e-mdn-lr $O/forced-w4mdnlr-$p.json || exit 7; done
  f $O/score-w4mdnlr.json score $M $G/passage-agpl.txt 400 w4e-mdn-lr $O/score-w4mdnlr.json || exit 7
  for i in 6 7 8 9 10; do f $O/forced-w4mdnlr-h$i.json forced $M $O/forced-fp8-h$i.json 512 w4e-mdn-lr $O/forced-w4mdnlr-h$i.json || exit 7; done
  unset AIEN_Q36_W4_LOWRANK
  # ---- 5. verdict (A goldens + B' h6..h10 + C' paired vs w4mdn)
  $HERE/verdict-gptq.sh w4mdnlr w4mdn 6 > $E/verdict-w4mdnlr.json 2> $E/verdict-w4mdnlr.err.txt
  jq --arg fit $FIT --arg lr "$LR" --arg auth "$AUTH" '. + {control_this_build: true, fit: $fit, sidecar: $lr, window: $auth}' \
    $E/verdict-w4mdnlr.json > $E/verdict-w4mdnlr.tmp && mv $E/verdict-w4mdnlr.tmp $E/verdict-w4mdnlr.json
  log "verdict w4mdnlr: $(jq -c '[.verdict,.A.R1_nll_ratio,.A.R2_confident,.B.R1_nll_ratio,.B.R2_confident,.C.diff_ci95]' $E/verdict-w4mdnlr.json)"
else
  log "both screens NO-GO: no arm (w4-gate.md Low-rank arm item 2)"
fi

# ---- 6. diagnostics (never gated): routes on h1 and the chat golden, FP8 and m-dn
export AIEN_L355_ROUTES=1 AIEN_Q36_W4_STORE=$S AIEN_Q36_W4_FP8_LAYERS=
f $O/routes-fp8-h1.json forced $M $O/forced-fp8-h1.json 512 fp8 $O/routes-fp8-h1.json || exit 8
f $O/routes-w4mdn-h1.json forced $M $O/forced-fp8-h1.json 512 w4e-mdn $O/routes-w4mdn-h1.json || exit 8
f $O/routes-fp8-chat.json forced $M $R/drift-chat-max-24.json 512 fp8 $O/routes-fp8-chat.json || exit 8
f $O/routes-w4mdn-chat.json forced $M $R/drift-chat-max-24.json 512 w4e-mdn $O/routes-w4mdn-chat.json || exit 8
log "done"
