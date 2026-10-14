#!/usr/bin/env bash
# OWN-30B W-L1I: full-model drift split (PREPARED, NOT RUN). RUN ONLY IN A PAUSE WINDOW: the live :18091 server must be
# stopped first (this script refuses while anything listens there). It never starts or stops :18091.
#
#   ~/workspace/sc-own30b-a16b/scripts/own30b-l1i-drift-gate.sh          lens 1,512,2048, all variants below
#   LENS=512 ~/workspace/sc-own30b-a16b/scripts/own30b-l1i-drift-gate.sh  one length (~8 min)
#
# What it answers: the 3-layer run (docs/inference/evidence/own30b-l1i-drift3-20261014T0510Z) split the W8A16 drift at
# layers 0-2; whether the same split holds through 48 layers, and whether the max-over-layers number (0.10 at 512, 0.14
# at 2048 in the RC2 gate) is driven by the sources at all or by routing flips that ANY tiny difference sets off.
# One 30 GB load, then per length: OFF, ULP (1-ulp floor), HATT32 (host attention floor), HQ, HP, HQP, MUTQS (mutant),
# A16, A16QKV, FULL, FIXQ, FIXQKV (forecasts), ON16, ON8 (served path: --devhost). ~30 min for all three lengths
# (2048: 8 host-attention runs x ~2.5 min + 4 chip runs x ~1 min; 512: ~5 min; 1: <1 min) [INFERRED from the 3-layer
# timings x 16].
#
# Binary: ~/handoffs/l1i-bin-16cf49e/own30b_drift_split (sc own30b/a16b 16cf49e, omega own30b/a16b = rc2 848a999b,
# archive sha256 1baabd41...; the binary that produced the 3-layer evidence), checked against its SHA256SUMS first.
# Holds /tmp/aien-gb10.lock for its whole life.
#
# Pre-registered reading (fixed 2026-10-14 before the full-model run; same rules as the 3-layer plan, plus):
#   R1 floor  layer 0: ULP and HATT32 <= 0.1 x ON16                      (instrument can say no)
#   R2 teeth  MUTQS attention-op error >= 10 x HQP's every layer          (FAILED on 3 layers at 6-15x: read as information)
#   R3 repro  layer 0: relerr(ON16,FULL) <= 0.1 x relerr(ON16,OFF); also read the same-route line
#   S1 shares at layer 0 and per-token medians per layer: A16 (and A16QKV), HQ, HP, HATT32
#   S2 saturation: the first layer where ULP's (or HATT32's) curve reaches half of ON16's. If HATT32 (a 1e-6 seed)
#      ends within 2x of ON16's max_l, the max-over-layers drift is routing chaos: no rounding fix lowers it materially.
#   S3 forecasts: FIXQ / FIXQKV max_l and per-token medians vs FULL. A fix is worth building only if its forecast lowers
#      the per-token median at the last layer by >= 30% AND first-step logits top-1 agree with OFF at every length.
set -uo pipefail
SC=$HOME/workspace/sc-own30b-a16b
BIN=$HOME/handoffs/l1i-bin-16cf49e
CKPT=${OWN30B_CHECKPOINT:-$HOME/models/qwen3-30b-a3b-instruct-2507-fp8}
LENS=${LENS:-1,512,2048}
VARIANTS=${VARIANTS:-OFF,ULP,HATT32,HQ,HP,HQP,MUTQS,A16,A16QKV,FULL,FIXQ,FIXQKV,ON16,ON8}
EV=$SC/docs/inference/evidence/own30b-l1i-drift48-$(date -u +%Y%m%dT%H%MZ)
LOCK=/tmp/aien-gb10.lock

die() { echo "L1I_GATE: REFUSED $*"; exit 2; }
listening() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
avail_gb() { free -g | awk '/^Mem:/ {print $7}'; }

listening 18091 && die "something listens on :18091 (the live server): stop it first (pause window)"
[ "$(avail_gb)" -ge 70 ] || die "only $(avail_gb) GB available (needs 70: one 30 GB model at a time)"
( cd "$BIN" && sha256sum -c --quiet SHA256SUMS ) || die "binary digest in $BIN does not match SHA256SUMS"
[ -f "$CKPT/config.json" ] || die "no checkpoint at $CKPT"
grep -q '"num_hidden_layers": *48' "$CKPT/config.json" || die "$CKPT is not the 48-layer model"
mkdir -p "$EV"
{
	echo "OWN-30B W-L1I full-model drift split $(date -u +%FT%TZ)"
	echo "sc $(git -C "$SC" rev-parse HEAD) omega $(git -C "$HOME/workspace/omega-own30b-a16b" rev-parse HEAD)"
	echo "binary:"; cat "$BIN/SHA256SUMS"
	echo "checkpoint $CKPT lens $LENS variants $VARIANTS"
	echo "cmd flock $LOCK env AIEN_OMEGA_SPIN_US=2000 $BIN/own30b_drift_split $CKPT $EV --lens $LENS --variants $VARIANTS --devhost --max-context 4096"
	free -g | head -2
} > "$EV/MANIFEST"
echo "evidence $EV (about 30 min)"
flock "$LOCK" env AIEN_OMEGA_SPIN_US=2000 "$BIN/own30b_drift_split" "$CKPT" "$EV" --lens "$LENS" --variants "$VARIANTS" --devhost --max-context 4096 \
	> "$EV/stdout.txt" 2> "$EV/stderr.txt"
rc=$?
echo "$rc" > "$EV/exit_code.txt"
grep -E "R1 |R2 |R3 |same-route|SHARES|DEPTH|DRIFT|logits cosine" "$EV/stdout.txt" > "$EV/SUMMARY.txt"
( cd "$EV" && sha256sum $(find . -type f ! -name SHA256SUMS | sort) > SHA256SUMS )
cat "$EV/SUMMARY.txt"
echo "L1I_GATE: exit $rc | evidence $EV"
exit "$rc"
