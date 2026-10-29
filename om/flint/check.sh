#!/usr/bin/env bash
# Flint P0 manifest gate (Core #802). Builds om/flint/manifest.om with the pinned omc, then:
#  1. runs it on a synthetic Flint-shaped checkpoint built by test/gen.py from config.json alone
#     (must exit 0, print the exact summary and a TSV byte-equal to the referee's);
#  2. runs every refusal fixture (each must exit with its own code, print "refused <code> ..."
#     and write nothing to stdout);
#  3. rebuilds the reader with each guard removed and requires that guard's fixture to stop
#     producing its code, so every refusal above is proven to come from the guard named for it.
# Usage: OMC=/path/to/omega-mv/seed/out/omc bash om/flint/check.sh   (OMEGA_OMC is accepted too)
# Optional real-checkpoint arms, each NOT_RUN when unset:
#   FLINT_SRC=dir     the pinned source capture (config.json, model.safetensors.index.json,
#                     headers/<shard>.header.json, shards.lfs-sha256): sparse shards rebuilt from
#                     the exact published header bytes and sizes.
#   FLINT_SHARDS=dir  the 15 downloaded shards (config and index come from FLINT_SRC).
#   With both set, the forward-operator arm runs ops.om against test/opref.py fixtures;
#   FLINT_PY is the referee's interpreter (torch + transformers a672eb5f), default python3.
# Needs: python3 (fixture referees only), sha256sum, cmp, timeout. Host only; no network, no GPU.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
OMC=${OMC:-${OMEGA_OMC:-}}
: "${OMC:?set OMC to the seed compiler (omega-mv seed/out/omc)}"
PIN=62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b
CFG_PIN=a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7
IDX_PIN=e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311
TSV_PIN=60ff4dbdaa1c3632c934c6935aa60ed5b2def4e15f0ca5083ddc7dd660f083f5
SUMMARY='flint-manifest ok tensors 36734 params 39785287456 routed 37849399296 shards 15 bytes 79570574912'
sha() { sha256sum < "$1" | cut -d' ' -f1; }
got=$(sha "$OMC")
[ "$got" = "$PIN" ] || { echo "FAIL omc sha256 $got, pinned $PIN"; exit 1; }
[ "$(sha "$here/test/config.json")" = "$CFG_PIN" ] || { echo "FAIL test/config.json is not the pinned config"; exit 1; }
b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
pass=0
fail=0
ok() { pass=$((pass + 1)); echo "ok   $*"; }
bad() { fail=$((fail + 1)); echo "FAIL $*"; }

build() { # out source
  { printf 'om2\n'; cat "$root/om/vendor/json.om" "$2"; } | "$OMC" > "$1" 2> "$1.err" || { echo "compile: $(cat "$1.err")"; return 1; }
  chmod 700 "$1"
}
gen() { rm -rf "$b/fx"; python3 "$here/test/gen.py" "$here/test/config.json" "$b/fx" "$1" < /dev/null; }
run() { # binary; sets rc, leaves out.tsv and err in $b/fx
  (cd "$b/fx" && timeout 60 "$1" config.json index.json shards > out.tsv 2> err < /dev/null)
  rc=$?
}

build "$b/fm" "$here/manifest.om" || exit 1

# 1. synthetic checkpoint must load
gen base || exit 1
run "$b/fm"
if [ "$rc" = 0 ] && cmp -s "$b/fx/out.tsv" "$b/fx/expected.tsv" && [ "$(cat "$b/fx/err")" = "$SUMMARY" ]; then
  ok "base: 36734 tensors, TSV equals referee, summary exact"
else
  bad "base: rc $rc, $(head -c 200 "$b/fx/err")"
fi
for args in "" "a" "a b" "a b c d"; do
  # shellcheck disable=SC2086
  "$b/fm" $args > "$b/u.out" 2> /dev/null < /dev/null
  r=$?
  if [ "$r" = 2 ] && [ ! -s "$b/u.out" ]; then ok "usage ($args): 2"; else bad "usage ($args): $r"; fi
done

# 2. refusals: fixture, required exit code
FIX='cfg-json 10
cfg-field 11
cfg-unknown 11
cfg-dup 11
cfg-missing 11
cfg-softmax 11
cfg-window 11
cfg-tie 11
idx-size 20
idx-syntax 21
idx-shard-name 22
idx-unknown 23
idx-dup 24
idx-total 25
idx-missing 26
shard-absent 3
shard-symlink 4
shard-dir 4
shard-short 30
hdr-len-be 31
hdr-len-big 31
hdr-json 32
hdr-entry 33
hdr-meta 34
hdr-dtype 35
hdr-shape 36
hdr-rank 36
hdr-off-len 37
hdr-off-overflow 37
hdr-truncated 37
hdr-overlap 38
hdr-gap 38
hdr-trailing 38
hdr-dup 39
hdr-mismatch 40
hdr-missing 41
hdr-unknown 23
shard-cap 42'
while read -r m want; do
  gen "$m" || { bad "$m: fixture generator failed"; continue; }
  run "$b/fm"
  if [ "$rc" = "$want" ] && [ ! -s "$b/fx/out.tsv" ] && grep -q "^refused $want " "$b/fx/err"; then
    ok "$m: $want"
  else
    bad "$m: want $want, got $rc ($(head -c 120 "$b/fx/err"))"
  fi
done <<< "$FIX"

# 3. guard mutants: the reader with one guard removed must stop producing that guard's code
s=$(cat "$here/manifest.om")
mutant() { # name want fixture; reads $b/mut.om
  build "$b/mut" "$b/mut.om" || { bad "mutant $1: does not compile"; return; }
  gen "$3" || { bad "mutant $1: fixture generator failed"; return; }
  run "$b/mut"
  if [ "$rc" = "$2" ]; then bad "mutant $1: still exits $2 on $3, so the guard is not what refuses it"; else ok "mutant $1: $3 now exits $rc, not $2"; fi
}
GUARD='3 shard-absent
4 shard-symlink
10 cfg-json
11 cfg-softmax
20 idx-size
21 idx-syntax
22 idx-shard-name
23 idx-unknown
24 idx-dup
25 idx-total
26 idx-missing
30 shard-short
31 hdr-len-be
32 hdr-json
33 hdr-entry
34 hdr-meta
35 hdr-dtype
36 hdr-shape
37 hdr-off-overflow
38 hdr-overlap
39 hdr-dup
40 hdr-mismatch
41 hdr-missing
42 shard-cap'
nl=$'\n'
while read -r code fx; do
  t=${s//"fm_die $code tr$code$nl"/"put gst 14 get gst 14$nl"}
  [ "$t" != "$s" ] || { bad "mutant die-$code: no guard text found"; continue; }
  printf '%s\n' "$t" > "$b/mut.om"
  mutant "die-$code" "$code" "$fx"
done <<< "$GUARD"
t=${s//"fopen gpth 0 n 557056 0"/"fopen gpth 0 n 524288 0"}
if [ "$t" = "$s" ]; then bad "mutant no-nofollow: open flags not found"; else
  printf '%s\n' "$t" > "$b/mut.om"
  mutant no-nofollow 4 shard-symlink
fi

# 4. real checkpoint arms
if [ -n "${FLINT_SRC:-}" ]; then
  if [ "$(sha "$FLINT_SRC/config.json")" != "$CFG_PIN" ] || [ "$(sha "$FLINT_SRC/model.safetensors.index.json")" != "$IDX_PIN" ]; then
    bad "FLINT_SRC: config or index is not the pinned revision 2a247be0"
  else
    mkdir -p "$b/hdr"
    python3 - "$FLINT_SRC" "$b/hdr" < /dev/null <<'PY'
import os, struct, sys
src, out = sys.argv[1:3]
for line in open(src + "/shards.lfs-sha256"):
    _, name, size = line.split()
    h = open(src + "/headers/" + name + ".header.json", "rb").read()
    with open(out + "/" + name, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        f.truncate(int(size))
PY
    timeout 60 "$b/fm" "$FLINT_SRC/config.json" "$FLINT_SRC/model.safetensors.index.json" "$b/hdr" > "$b/hdr.tsv" 2> "$b/hdr.err" < /dev/null
    r=$?
    if [ "$r" = 0 ] && [ "$(sha "$b/hdr.tsv")" = "$TSV_PIN" ] && [ "$(cat "$b/hdr.err")" = "$SUMMARY" ]; then
      ok "FLINT_SRC: published headers and sizes, TSV sha $TSV_PIN"
    else
      bad "FLINT_SRC: rc $r, $(head -c 200 "$b/hdr.err")"
    fi
  fi
else
  echo "NOT_RUN FLINT_SRC (pinned header capture of RecursiveSI/flint 2a247be0)"
fi
if [ -n "${FLINT_SHARDS:-}" ] && [ -n "${FLINT_SRC:-}" ]; then
  timeout 600 "$b/fm" "$FLINT_SRC/config.json" "$FLINT_SRC/model.safetensors.index.json" "$FLINT_SHARDS" > "$b/real.tsv" 2> "$b/real.err" < /dev/null
  r=$?
  if [ "$r" = 0 ] && [ "$(sha "$b/real.tsv")" = "$TSV_PIN" ] && [ "$(cat "$b/real.err")" = "$SUMMARY" ]; then
    ok "FLINT_SHARDS: downloaded checkpoint, TSV sha $TSV_PIN"
  else
    bad "FLINT_SHARDS: rc $r, $(head -c 200 "$b/real.err")"
  fi
else
  echo "NOT_RUN FLINT_SHARDS (the 15 downloaded shards; needs FLINT_SRC too)"
fi

# 5. forward operators (Core #802): ops.om recomputes, from the real BF16 shard weights, the
#    outputs that test/opref.py takes from the pinned reference modules, and must land within twice
#    the reference's own f32 error bound; each mutant must turn its operator's verdict to FAIL
#    (exit 1, the comparison, not a trap or refusal). FLINT_PY runs the referee (torch and the
#    transformers a672eb5f checkout); default python3.
OPS='rmsnorm rmsnorm
qkv qkv-2
qkv qkv-5
attn attn-2
attn attn-5
router router
shared shared
expert expert
moe moe'
MUT='softmax-router|sc = om_sig l|sc = om_exp l|router router
shared-gate-sigmoid0|v = add get gacm o get gx3 o|v = add get gacm o mul 0.5f64 get gx3 o|moe moe
no-sink|att_sm n get gsk h get gst 4|att_sm n -1.0e30f64 get gst 4|attn attn-2
rope-global|rp = fl_swa L|rp = true|qkv qkv-5
expert-permuted|nm_u ex|nm_u sub 255 ex|expert expert
window-514|const FWIN ulong 513|const FWIN ulong 514|attn attn-2'
opr() { # binary op fixture; sets rc and line
  line=$(timeout 300 "$1" "$2" "$b/opfx/$3.bin" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
  rc=$?
}
if [ -n "${FLINT_SHARDS:-}" ] && [ -n "${FLINT_SRC:-}" ] && [ -s "$b/real.tsv" ]; then
  if ! build "$b/ops" "$here/ops.om"; then bad "ops: ops.om does not compile"
  elif ! ${FLINT_PY:-python3} "$here/test/opref.py" "$FLINT_SRC" "$FLINT_SHARDS" "$b/opfx" > "$b/opref.out" 2>&1 < /dev/null; then
    bad "ops: opref.py failed: $(head -c 300 "$b/opref.out")"
  else
    echo "$(cat "$b/opref.out"), fixtures sha256 $(cat "$b"/opfx/*.bin | sha256sum | cut -d' ' -f1)"
    line=$(timeout 60 "$b/ops" selftest 2>&1 < /dev/null)
    rc=$?
    if [ "$rc" = 0 ] && [[ $line == *" PASS" ]]; then ok "PASS $line"; else bad "selftest rc $rc: $line"; fi
    while read -r op fx; do
      opr "$b/ops" "$op" "$fx"
      if [ "$rc" = 0 ] && [[ $line == *" PASS" ]]; then ok "PASS $fx: $line"; else bad "$fx rc $rc: $line"; fi
    done <<< "$OPS"
    opr "$b/ops" router router-swap
    if [ "$rc" = 1 ] && [[ $line == *" FAIL" ]]; then ok "CAUGHT control router-swap (reference top-6 swapped): $line"; else bad "control router-swap rc $rc: $line"; fi
    opr "$b/ops" router router-tie
    if [ "$rc" = 0 ] && [[ $line == *" PASS" ]] && [[ $line != *" ties 0 "* ]]; then ok "PASS router-tie (rank-6/7 near tie reported, not failed): $line"; else bad "control router-tie rc $rc: $line"; fi
    s=$(cat "$here/ops.om")
    while IFS='|' read -r name from to target; do
      n=$(grep -cF -- "$from" "$here/ops.om")
      [ "$n" = 1 ] || { bad "mutant $name: site found $n times"; continue; }
      printf '%s\n' "${s/"$from"/"$to"}" > "$b/opmut.om"
      build "$b/opmut" "$b/opmut.om" || { bad "mutant $name: does not compile"; continue; }
      opr "$b/opmut" $target
      if [ "$rc" = 1 ] && [[ $line == *" FAIL" ]]; then ok "CAUGHT mutant $name on ${target#* }: $line"; else bad "mutant $name not caught on ${target#* }: rc $rc $line"; fi
    done <<< "$MUT"
  fi
else
  echo "NOT_RUN ops (forward operators; needs FLINT_SRC, FLINT_SHARDS and the FLINT_SHARDS arm's TSV)"
fi

n=$((pass + fail))
if [ "$fail" = 0 ]; then echo "OM-FLINT-MANIFEST PASS ($n)"; else echo "OM-FLINT-MANIFEST FAIL ($fail of $n)"; exit 1; fi
