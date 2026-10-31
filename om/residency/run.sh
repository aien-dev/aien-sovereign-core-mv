#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host test of the seeded .om expert residency simulator (om/residency/residency.om).
# Expected outputs in fixtures/*.want were computed by fixtures/oracle.wl, an independent LRU,
# Belady (OPT) and hinted-LRU (hlru, hcold) implementation in the Wolfram Language, not by
# residency.om:
#   real-p4-L4T64: first 64 decode steps and first 4 layers of
#     evidence/l355-w4/routes/forced-fp8-p4.json (Qwen3.6-35B-A3B FP8, top-8 of 256 experts).
#   syn-E40L3K4T200: SeedRandom[630]; RandomSample[Range[0,39],4] per step and layer, 200 steps.
#   syn-E40L3K4T200H3: the same trace with H=3 hints per row; SeedRandom[633] and per row r of the
#     trace {r[[1]], RandomInteger[39], r[[2]]}, so two hints are used and one is mostly wasted.
# Also checks every refusal exit code, the L*E <= 65536 bound of the shared-cache (glru/gopt/hlru/
# hcold) section, two hint invariants for each of hlru and hcold (H=0 replays glru; hints equal to
# the trace prefetch every glru miss and leave no demand miss), the C >= K+H gate of hinted lines,
# and that ten mutants are caught.
# Usage: OMEGA_OMC=<seed omc> bash om/residency/run.sh [TRACE.txt ...]
# Extra TRACE arguments are replayed after the checks and their output printed (not compared).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# omega-mv main 0b77e6ea294aa3d69ccbdb86b9c14d133abea689, seed/out/omc
PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
OMC=${OMEGA_OMC:?set OMEGA_OMC to the seed compiler (omega-mv seed/out/omc)}
D=$ROOT/om/residency
O=$ROOT/target/om-residency
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
rm -rf "$O"; mkdir -p "$O"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
build() { "$OMC" < "$1" > "$2" 2> "$2.err" && chmod +x "$2"; }
build "$D/residency.om" "$O/sim" || { echo "FAIL build residency.om: $(cat "$O/sim.err")"; exit 1; }

for f in real-p4-L4T64 syn-E40L3K4T200 syn-E40L3K4T200H3; do
    "$O/sim" < "$D/fixtures/$f.txt" > "$O/$f.out"; rc=$?
    if [ $rc = 0 ] && cmp -s "$O/$f.out" "$D/fixtures/$f.want"; then pass "$f matches the Wolfram oracle"
    else fail "$f (exit $rc)"; diff "$D/fixtures/$f.want" "$O/$f.out" | head -5; fi
done

refuse() {
    printf '%b' "$2" | "$O/sim" > /dev/null; rc=$?
    if [ $rc = "$3" ]; then pass "refuse $1 (exit $rc)"; else fail "refuse $1: exit $rc, want $3"; fi
}
refuse "id equal to E" '4 1 1 2\n0 4\n' 7
refuse "repeated id in one step" '4 1 2 2\n0 0 1 2\n' 8
refuse "K above 16" '64 1 17 2\n' 4
refuse "E zero" '0 1 1 2\n' 4
refuse "one step" '4 1 1 1\n0\n' 4
refuse "missing ids" '4 1 1 3\n0 1\n' 5
refuse "letter in a number" '4 1 1 2\n0 1x\n' 5
refuse "trailing number" '4 1 1 2\n0 1\n0\n7\n' 6
refuse "empty input" '' 5
refuse "T*L above 1000000" '4 500001 1 2\n' 4
refuse "H above 16" '4 1 1 2\n0 1\n17\n' 4
refuse "hint id equal to E" '4 1 1 2\n0 1\n1 0 4\n' 7
refuse "missing hint ids" '4 1 1 2\n0 1\n1 0\n' 5
{ echo "1 1 1 524289"; yes 0 | head -n 524289; echo 16; } | "$O/sim" > /dev/null; rc=$?
if [ $rc = 4 ]; then pass "refuse T*L*H above 8388608 (exit 4)"; else fail "T*L*H above 8388608: exit $rc, want 4"; fi

# T*L = 1000000 rows: one for loop makes exactly 1000000 passes and must finish (no trap 4).
{ echo "1 500000 1 2"; yes 0 | head -n 1000000; } | "$O/sim" > "$O/r.out"; rc=$?
if [ $rc = 0 ] && [ "$(cat "$O/r.out")" = "trace E=1 L=500000 K=1 T=2" ]; then pass "T*L = 1000000 rows read"
else fail "T*L = 1000000 rows: exit $rc"; fi

# Shared-cache section boundary: L*E = 65536 is replayed (8 glru lines), L*E = 66560 is not.
bound() {
    { echo "1024 $1 1 2"; for i in $(seq 1 $((2 * $1))); do echo 0; done; } | "$O/sim" > "$O/b.out"; rc=$?
    n=$(grep -c '^glru ' "$O/b.out")
    if [ $rc = 0 ] && [ "$n" = "$2" ]; then pass "E=1024 L=$1: $n glru lines"
    else fail "E=1024 L=$1: exit $rc, $n glru lines, want $2"; fi
}
bound 64 8
bound 65 0
{ echo "1024 65 1 2"; for i in $(seq 1 130); do echo 0; done; echo 0; } | "$O/sim" > /dev/null; rc=$?
if [ $rc = 4 ]; then pass "refuse hints with L*E = 66560 (exit 4)"; else fail "hints with L*E = 66560: exit $rc, want 4"; fi

# hlru lines need C >= K+H: E=4 L=1 K=2 gives 6 glru lines (C >= 2), 2 hlru lines with H=1 (C = 3)
# and none with H=2.
gate() {
    printf "4 1 2 2\n0 1 2 3\n$1\n$2\n" | "$O/sim" > "$O/g.out"; rc=$?
    n=$(grep -c '^hlru ' "$O/g.out")
    if [ $rc = 0 ] && [ "$n" = "$3" ]; then pass "C >= K+H gate, H=$1: $n hlru lines"
    else fail "C >= K+H gate, H=$1: exit $rc, $n hlru lines, want $3"; fi
}
gate 1 '0 1' 2
gate 2 '0 1 2 3' 0

# Hint invariants on the synthetic trace (T*L*K = 2400 demands), for both hint modes.
syn=$D/fixtures/syn-E40L3K4T200.txt
"$O/sim" < "$syn" | grep '^glru ' > "$O/glru.out"
{ cat "$syn"; echo 0; } | "$O/sim" > "$O/h0all.out"
{ cat "$syn"; echo 4; tail -n +2 "$syn"; } | "$O/sim" > "$O/hpall.out"
for m in hlru hcold; do
    grep "^$m " "$O/h0all.out" \
        | sed "s/^$m \(.*\) hint=0 \(.*\) warm_pf=0 warm_used=0 warm_waste=0 /glru \1 \2 /" > "$O/h0.out"
    if [ -s "$O/h0.out" ] && cmp -s "$O/h0.out" "$O/glru.out"; then pass "H=0 $m equals glru"
    else fail "H=0 $m differs from glru"; diff "$O/glru.out" "$O/h0.out" | head -5; fi
    grep "^$m " "$O/hpall.out" > "$O/hp.out"
    sed "s/^glru \(cap=[0-9]* pct=[0-9]*\) hits=[0-9]* miss=[0-9]* warm_miss=\([0-9]*\) /$m \1 hint=4 hits=2400 miss=0 warm_miss=0 warm_pf=\2 warm_used=\2 warm_waste=0 /" \
        "$O/glru.out" > "$O/hpw.out"
    if [ -s "$O/hp.out" ] && cmp -s "$O/hp.out" "$O/hpw.out"; then pass "perfect $m hints prefetch every glru miss, no demand miss"
    else fail "perfect $m hints"; diff "$O/hpw.out" "$O/hp.out" | head -5; fi
done

# Mutants: each must build, run to completion on its fixture and change its output.
mutant() {
    f=${3:-syn-E40L3K4T200}
    sed "$2" "$D/residency.om" > "$O/m.om"
    cmp -s "$O/m.om" "$D/residency.om" && { fail "mutant $1 did not change the source"; return; }
    build "$O/m.om" "$O/m" || { fail "mutant $1 did not build"; return; }
    "$O/m" < "$D/fixtures/$f.txt" > "$O/m.out"; rc=$?
    if [ $rc != 0 ]; then fail "mutant $1 did not complete (exit $rc)"
    elif cmp -s "$O/m.out" "$D/fixtures/$f.want"; then fail "mutant $1 not caught"
    else pass "mutant $1 caught"; fi
}
mutant "LRU evicts most recent" 's/^ret sub add t 2 get stp e$/ret get stp e/'
mutant "OPT evicts nearest next use" 's/^ret get nu e$/ret sub get hd 3 get nu e/'
mutant "shared cache ignores the layer" 's/^globst s mul rem s l0 e0$/globst s 0/'
mutant "hints ignored" 's/^pfstep t c cd$/put ac 7 0/' syn-E40L3K4T200H3
mutant "hints ignore the layer" 's/^globh s mul rem s l0 e0$/globh s 0/' syn-E40L3K4T200H3
mutant "prefetch evicts by next use" 's/^evict s 0$/evict s 1/' syn-E40L3K4T200H3
mutant "wasted prefetch not counted" 's/^put ac 6 add get ac 6 1$/put ac 6 get ac 6/' syn-E40L3K4T200H3
mutant "hcold prefetch inserted hot" 's/^put stp v 1$/put stp v add s 1/' syn-E40L3K4T200H3
mutant "hcold hint marks use" 's/^if eq cd 0$/if eq cd cd/' syn-E40L3K4T200H3
mutant "hint protection dropped" 's/^put prt v add s 1$/put prt v 0/' syn-E40L3K4T200H3

echo "residency.om $(sha256sum < "$D/residency.om" | cut -d' ' -f1)"
echo "sim binary $(sha256sum < "$O/sim" | cut -d' ' -f1)"
echo "omc $got"
for t in "$@"; do echo "== $t"; "$O/sim" < "$t"; echo "exit $?"; done
[ $fails = 0 ] && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
