#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Checks om/dsv4/fp4dq.om, the DeepSeek-V4-Flash routed-expert decoder, against reference outputs
# made by DeepSeek's own decoder (inference/convert.py FP4_TABLE x torch float8_e8m0fnu scale,
# checkpoint revision 60d8d70770c6776ff598c94bb586a859a38244f1, torch 2.14.1+cpu).
# The reference outputs are recorded as sha256 values below; torch is not needed to run this.
#   edge:  every one of the 16 codes under every one of the 256 scale bytes (subnormal, NaN, overflow).
#   shard: six real expert matrices of layer 5 cut from model-00007-of-00046.safetensors, if
#          DSV4_SHARD7 points at that file (sha256 2d782c46d6d2...); otherwise NOT_RUN.
# Also: refusal exit codes and three mutants that must change the output.
# Usage: OMEGA_OMC=<seed omc> [DSV4_SHARD7=<path>] bash om/dsv4/run.sh
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
OMC=${OMEGA_OMC:?set OMEGA_OMC to the seed compiler (omega-mv seed/out/omc)}
D=$ROOT/om/dsv4
O=$ROOT/target/om-dsv4
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
rm -rf "$O"; mkdir -p "$O"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
build() { "$OMC" < "$1" > "$2" && chmod +x "$2"; }
build "$D/fp4dq.om" "$O/fp4dq" || { echo "FAIL build fp4dq.om"; exit 1; }
H() { sha256sum | cut -d' ' -f1; }
EDGE=6a35e83e299a76416e32c650443e490806ec5658e0c75df657f6b38e4b76076d

# edge: all 4096 (code, scale) pairs
g=$("$O/fp4dq" < "$D/fixtures/edge.in" | H)
[ "$g" = "$EDGE" ] && pass "edge 16 codes x 256 scales" || fail "edge (got $g)"

# refusals
le8() { local v=$1 i; for i in 0 1 2 3 4 5 6 7; do printf "\\x$(printf %02x $(( (v >> (8*i)) & 255 )))"; done; }
refuse() {  # label want-exit
    "$O/fp4dq" < "$O/case" > /dev/null 2>&1; r=$?
    [ $r = "$2" ] && pass "refuse $1 (exit $r)" || fail "refuse $1 (want $2, got $r)"
}
: > "$O/case"; refuse "empty input" 5
head -c 15 "$D/fixtures/edge.in" > "$O/case"; refuse "short header" 5
{ le8 1; le8 31; } > "$O/case"; refuse "cols not a multiple of 32" 4
{ le8 0; le8 32; } > "$O/case"; refuse "zero rows" 4
{ le8 1; le8 32; head -c 16 /dev/zero; } > "$O/case"; refuse "missing scale byte" 5
{ le8 1; le8 32; head -c 18 /dev/zero; } > "$O/case"; refuse "trailing byte" 5
head -c 16777217 /dev/zero > "$O/case"; refuse "input over 16 MiB" 3

# mutants: each must change the edge output
mutant() {  # label sed-expression
    sed "$2" "$D/fp4dq.om" > "$O/m.om"
    cmp -s "$O/m.om" "$D/fp4dq.om" && { fail "mutant $1 (sed matched nothing)"; return; }
    build "$O/m.om" "$O/m" || { fail "mutant $1 (does not build)"; return; }
    g=$("$O/m" < "$D/fixtures/edge.in" | H)
    [ "$g" != "$EDGE" ] && pass "mutant $1 caught" || fail "mutant $1 survived"
}
mutant "high nibble first" 's/^c = and shr x mul and j 1 4 15$/c = and shr x mul sub 1 and j 1 4 15/'
mutant "code 8 as -0" '/^if eq m 0$/{n;s/^ret 0$/ret shl and shr c 3 1 31/}'
mutant "NaN bits 0x7fc00000" 's/^ret 2143289345$/ret 2143289344/'

# shard: real layer-5 expert matrices, base = 8 + header length 172656
if [ -n "${DSV4_SHARD7:-}" ] && [ -r "$DSV4_SHARD7" ]; then
    B=172664
    while read -r name rows cols w0 w1 s0 s1 want; do
        { le8 "$rows"; le8 "$cols"
          tail -c +$((B + w0 + 1)) "$DSV4_SHARD7" | head -c $((w1 - w0))
          tail -c +$((B + s0 + 1)) "$DSV4_SHARD7" | head -c $((s1 - s0)); } > "$O/case"
        g=$("$O/fp4dq" < "$O/case" | H)
        [ "$g" = "$want" ] && pass "shard7 $name identical to reference" || fail "shard7 $name (got $g)"
    done < "$D/fixtures/shard7.txt"
else
    echo "NOT_RUN shard7 (set DSV4_SHARD7 to model-00007-of-00046.safetensors)"
fi

[ $fails = 0 ] && echo "ALL PASS" || echo "$fails FAILED"
exit $fails
