#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Parity of the seeded .om effect world check (om/effects/world.om) with the Rust reference world
# check (aien-runtime effects::world_state + effects::reconcile_text; not the confined production
# path confined_world_state): same stdout bytes and exit code
# on every fixture, refusals on inputs Rust's u64 parse refuses, and two mutants caught.
# Usage: OMEGA_OMC=<seed omc> bash om/effects/run.sh
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PIN=734f0a39319fbfa930bcc62cb30dbf06d47d78e126372b74f5baf478e6126b61
OMC=${OMEGA_OMC:?set OMEGA_OMC to the seed compiler (omega-mv seed/out/omc)}
O=$ROOT/target/om-effects
got=$(sha256sum < "$OMC" | cut -d' ' -f1)
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
rm -rf "$O"; mkdir -p "$O/fx"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
build() { cat "$ROOT/om/vendor/sha256.om" "$1" | "$OMC" > "$2" && chmod +x "$2"; }
build "$ROOT/om/effects/world.om" "$O/world" || { echo "FAIL build world.om"; exit 1; }
(cd "$ROOT" && cargo build --offline -q -p aien-runtime --example om_world_reference) || { echo "FAIL build reference"; exit 1; }
REF=$ROOT/target/debug/examples/om_world_reference
OMW=$O/world

# Fixtures (relative to $O/fx, so paths in error messages are short and stable).
cd "$O/fx"
printf abc > abc
printf 'prior bytes' > prior
: > empty
head -c 1048579 /dev/urandom > big
mkdir dir
ln -s abc link
ln -s loop loop
mkfifo fifo
printf secret > noread; chmod 000 noread
WEIRD=$'we"ird\\na\nme\001'
printf w > "$WEIRD"
mkdir "$WEIRD.d"
LONG=$(printf 'n%.0s' $(seq 300))
H() { sha256sum < "$1" | cut -d' ' -f1; }
A=$(H abc); P=$(H prior); E=$(H empty); B=$(H big); W=$(H "$WEIRD")
UP=$(printf '%s' "$A" | tr a-f A-F)
CTL=$(printf 'c%b' '\001\002\003\004\005\006\007\010\011\012\013\014\015\016\017\020\021\022\023\024\025\026\027\030\031\032\033\034\035\036\037\177')
n=0
same() {
    n=$((n + 1)); label=$1; want=$2; shift 2
    "$REF" "$@" > "$O/r$n.out" 2> /dev/null; rr=$?
    "$OMW" "$@" > "$O/w$n.out" 2> /dev/null; wr=$?
    st=$(grep -o '"state":"[A-Z_]*"' "$O/w$n.out" | cut -d'"' -f4)
    if [ $rr = 0 ] && [ $wr = 0 ] && cmp -s "$O/r$n.out" "$O/w$n.out" && [ "$st" = "$want" ]; then
        pass "$label: $st, identical bytes"
    else fail "$label (want $want, got $st; rust exit $rr, om exit $wr; $O/r$n.out vs $O/w$n.out)"; fi
}
same "content written" DONE abc "$A" - 3 2 'reconcile@start'
same "absent, no prior" NOT_DONE absent "$A" - 3 2 'reconcile@start'
same "prior bytes still there" NOT_DONE prior "$A" "$P" 3 2 x
same "absent, prior named" UNRESOLVED absent "$A" "$P" 3 2 x
same "other bytes, no prior" UNRESOLVED prior "$A" - 3 2 x
same "content equals prior" DONE abc "$A" "$A" 3 2 x
same "uppercase content digest" UNRESOLVED abc "$UP" - 3 2 x
same "empty file" DONE empty "$E" - 3 2 x
same "1 MiB + 3 bytes" DONE big "$B" - 3 2 x
same "directory" UNRESOLVED dir "$A" - 3 2 x
same "symlink to a file" UNRESOLVED link "$A" - 3 2 x
same "fifo" UNRESOLVED fifo "$A" - 3 2 x
same "unreadable file (or read by root: other bytes)" UNRESOLVED noread "$A" - 3 2 x
same "path through a file (ENOTDIR)" UNRESOLVED abc/x "$A" - 3 2 x
same "symlink loop (ELOOP)" UNRESOLVED loop/x "$A" - 3 2 x
same "300-byte name (ENAMETOOLONG)" UNRESOLVED "$LONG" "$A" - 3 2 x
same "empty path" NOT_DONE "" "$A" - 3 2 x
same "quote, backslash, newline, 0x01 in the name" DONE "$WEIRD" "$W" - 3 2 x
same "that name inside disk_error" UNRESOLVED "$WEIRD.d" "$A" - 3 2 x
same "every control byte and 0x7f in by" DONE abc "$A" - 3 2 "$CTL"
same "quote, backslash, slash, UTF-8 in by" DONE abc "$A" - 3 2 'q"b\s/é ✓'
same "u64 parse: + and leading zeros" DONE abc "$A" - +7 007 x
same "largest u64 intent, authorization 0" DONE abc "$A" - 18446744073709551615 0 x
same "empty by" DONE abc "$A" - 3 2 ""

# Refusals: Rust's parse refuses these (the reference panics); world exits 2 with no output.
refuse() {
    "$REF" "$@" > /dev/null 2>&1; rr=$?
    "$OMW" "$@" > "$O/ref.out" 2> /dev/null; wr=$?
    if [ $rr != 0 ] && [ $wr = 2 ] && [ ! -s "$O/ref.out" ]; then pass "both refuse intent '$4'"
    else fail "refusal intent '$4' (rust exit $rr, om exit $wr)"; fi
}
refuse abc "$A" - 18446744073709551616 2 x
refuse abc "$A" - -1 2 x
refuse abc "$A" - + 2 x
refuse abc "$A" - "" 2 x
"$OMW" abc "$A" - 3 2 > "$O/ref.out" 2> /dev/null; wr=$?
if [ $wr = 2 ] && [ ! -s "$O/ref.out" ]; then pass "world refuses five arguments"; else fail "five arguments (exit $wr)"; fi

# Mutants: one byte of the record and one decision; each must run to completion (exit 0 on both
# fixtures) and make a fixture's output differ. A mutant that refuses or crashes is not a catch.
mutant() {
    sed "$2" "$ROOT/om/effects/world.om" > "$O/m.om"
    cmp -s "$O/m.om" "$ROOT/om/effects/world.om" && { fail "mutant $1 did not change the source"; return; }
    build "$O/m.om" "$O/m" || { fail "mutant $1 did not build"; return; }
    "$REF" abc "$A" - 3 2 x > "$O/mr.out"; "$O/m" abc "$A" - 3 2 x > "$O/mw.out"; m1=$?
    "$REF" absent "$A" - 3 2 x >> "$O/mr.out"; "$O/m" absent "$A" - 3 2 x >> "$O/mw.out"; m2=$?
    if [ $m1 != 0 ] || [ $m2 != 0 ]; then fail "mutant $1 did not complete (exit $m1, $m2)"
    elif cmp -s "$O/mr.out" "$O/mw.out"; then fail "mutant $1 not caught"; else pass "mutant $1 caught"; fi
}
mutant "key by -> bx" 's/,\\"by\\":/,\\"bx\\":/'
mutant "absent with no prior -> UNRESOLVED" '/^if eq pnone 1$/{n;s/^ret 1$/ret 2/}'

chmod 600 "$O/fx/noread"
echo "world.om $(sha256sum < "$ROOT/om/effects/world.om" | cut -d' ' -f1)"
echo "sha256.om $(sha256sum < "$ROOT/om/vendor/sha256.om" | cut -d' ' -f1)"
echo "world binary $(sha256sum < "$OMW" | cut -d' ' -f1)"
echo "omc $got"
if [ "$fails" -ne 0 ]; then echo "OM-EFFECT-WORLD FAIL $fails"; exit 1; fi
echo "OM-EFFECT-WORLD PASS ($n fixtures)"
