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
#   FLINT_FORWARD=1 additionally runs the full 49-layer, two-token chained forward and mutants.
#   FLINT_GEN=1 runs greedy 16-token generation from a chat-template prompt against test/genref.py.
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
#    transformers a672eb5f checkout); default python3. The fixtures, concatenated in FXS order,
#    must hash to FX_PIN: recorded with torch 2.14.0 (written into every fixture header) on the
#    Spark host CPU, so another torch, transformers file or CPU result shows up as a FAIL here.
FX_PIN=2d838552a1ee0b726ee1b5cfb2a13a101f2520d522cbc4cf55dfcfe110287b7a
FXS='rmsnorm qkv-2 qkv-5 attn-2 attn-5 router router-swap router-tie shared expert moe'
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
    fxsha=$(for f in $FXS; do cat "$b/opfx/$f.bin" || echo missing; done | sha256sum | cut -d' ' -f1)
    cat "$b/opref.out"
    if [ "$fxsha" = "$FX_PIN" ]; then ok "fixtures sha256 $fxsha (pinned)"; else bad "fixtures sha256 $fxsha, pinned $FX_PIN"; fi
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

# Full chained forward is opt-in because it streams the real checkpoint and runs all 49 layers.
# No intermediate reference tensor is fed back into the native computation.
if [ "${FLINT_FORWARD:-0}" = 1 ]; then
  if [ -z "${FLINT_SRC:-}" ] || [ -z "${FLINT_SHARDS:-}" ] || [ ! -s "$b/real.tsv" ]; then
    bad "forward requested without FLINT_SRC, FLINT_SHARDS and admitted real.tsv"
  elif ! ${FLINT_PY:-python3} "$here/test/fwdref.py" "$FLINT_SRC" "$FLINT_SHARDS" "$b/forward.fx" > "$b/fwdref.out" 2>&1 < /dev/null; then
    bad "forward referee: $(cat "$b/fwdref.out")"
  elif [ "$(sha "$b/forward.fx")" != b5e4bae28af3c1fa0c8f7454c496e72da1388e17f51bb3ec9e9b0387aac97031 ]; then
    bad "forward fixture pin mismatch: $(sha "$b/forward.fx")"
  else
    cat "$b/fwdref.out"
    line=$(timeout 600 "$b/ops" forward "$b/forward.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
    rc=$?
    echo "$line"
    if [ "$rc" = 0 ] && [[ $line == *"forward layer 49 "* ]] && [[ $line == *" PASS" ]]; then ok "chained forward, all layers and vocabulary logits"; else bad "forward rc $rc"; fi
    # KV-cache decode: one row per chunk, earlier K/V only from the cache, same fixture and budget
    dline=$(timeout 900 "$b/ops" decode "$b/forward.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
    rc=$?
    dlast=${dline##*$'\n'}; de=${dlast##*elems }; fe=${line##*elems }
    echo "$dlast"
    if [ "$rc" = 0 ] && [[ $dlast == "flint-op decode layer 49 "* ]] && [[ $dlast == *" PASS" ]] && [ "${de%% *}" = "${fe%% *}" ]; then ok "KV-cache decode passes the same forward reference within budget, same element count"; else bad "decode rc $rc"; fi
    # cache bound: with FKVP 1 the two-row decode must be refused (7), never overrun the next layer
    [ "$(grep -c '^const FKVP ulong 32$' "$here/ops.om")" = 1 ] || bad "kv guard: FKVP site"
    { s=$(cat "$here/ops.om"); printf '%s\n' "${s/const FKVP ulong 32/const FKVP ulong 1}"; } > "$b/kvg.om"
    if build "$b/kvg" "$b/kvg.om"; then
      gline=$(timeout 900 "$b/kvg" decode "$b/forward.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null); rc=$?
      if [ "$rc" = 7 ] && [[ $gline == *"refused 7 kv-cache-bound"* ]]; then ok "KV cache bound refuses P+n > FKVP (rc 7)"; else bad "kv guard rc $rc: ${gline##*$'\n'}"; fi
    else bad "kv guard compile"; fi
    s=$(cat "$here/ops.om")
    while IFS='|' read -r name op from to; do
      [ "$(grep -cF -- "$from" "$here/ops.om")" = 1 ] || { bad "forward mutant $name site"; continue; }
      printf '%s\n' "${s/"$from"/"$to"}" > "$b/fwdmut.om"
      build "$b/fwdmut" "$b/fwdmut.om" || { bad "forward mutant $name compile"; continue; }
      line=$(timeout 900 "$b/fwdmut" "$op" "$b/forward.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
      rc=$?
      if [ "$rc" = 1 ] && [[ $line == *" FAIL" ]]; then ok "CAUGHT $op mutant $name: ${line##*$'\n'}"; else bad "$op mutant $name rc $rc: $line"; fi
    done <<'FORWARD_MUTANTS'
wrong-token|forward|id = get gtok add get gfs 2 t|id = add get gtok add get gfs 2 t 1
skip-final-norm|forward|put gfh o get gy i|put gfh o get gv i
kv-slot-overwrite|decode|c = mul add add mul L FKVP P j 512|c = mul add mul L FKVP j 512
FORWARD_MUTANTS
  fi
else
  echo "NOT_RUN chained forward (set FLINT_FORWARD=1)"
fi

# 6. greedy generation (Core #834 item 5), opt-in: 24 one-row host forwards and 16 vocabulary passes.
#    test/genref.py renders "What is 2+2?" with the pinned chat template and greedy-generates 16
#    tokens with the pinned reference in f32 (KV cache); the fixture holds the prompt IDs and each
#    step's reference top-1/top-2 IDs and logits. ops.om generate feeds its own argmax tokens back
#    through its KV cache and must meet test/TOLERANCE.md (greedy generation parity) on every step.
#    Controls: the K/V append skipped for layer 0 must FAIL (rc 1, the comparison); with FKVP 8 the
#    request must be refused at position 8 (rc 7), never overrun the cache.
GEN_PIN=d5d3daecc777532638758402839a0bdfcdc9455ea913cd48c4941390c69c95e9
genr() { # binary; sets rc, gline, glast
  gline=$(timeout 3600 "$1" generate "$b/gen.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
  rc=$?
  glast=${gline##*$'\n'}
}
if [ "${FLINT_GEN:-0}" = 1 ]; then
  if [ -z "${FLINT_SRC:-}" ] || [ -z "${FLINT_SHARDS:-}" ] || [ ! -s "$b/real.tsv" ] || [ ! -x "$b/ops" ]; then
    bad "generate requested without FLINT_SRC, FLINT_SHARDS, admitted real.tsv and a compiled ops.om"
  elif ! ${FLINT_PY:-python3} "$here/test/genref.py" "$FLINT_SRC" "$FLINT_SHARDS" "$b/gen.fx" > "$b/genref.out" 2>&1 < /dev/null; then
    bad "generate referee: $(tail -c 300 "$b/genref.out")"
  elif [ "$(sha "$b/gen.fx")" != "$GEN_PIN" ]; then
    bad "generate fixture pin mismatch: $(sha "$b/gen.fx")"
  else
    cat "$b/genref.out"
    genr "$b/ops"
    echo "$gline"
    if [ "$rc" = 0 ] && [[ $gline == *"flint-gen matched 16 of 16"* ]] && [[ $glast == "flint-op generate layer 49 "* ]] && [[ $glast == *" PASS" ]]; then ok "greedy generation, 16 of 16 tokens match the reference"; else bad "generate rc $rc: $glast"; fi
    s=$(cat "$here/ops.om")
    from='kvn = get gfs 0'
    if [ "$(grep -cF -- "$from" "$here/ops.om")" != 1 ]; then bad "generate mutant skip-kv-append-L0 site"
    else
      printf '%s\n' "${s/"$from"/kvn = mul get gfs 0 sub 1 div 1 add L 1}" > "$b/genmut.om"
      if build "$b/genmut" "$b/genmut.om"; then
        genr "$b/genmut"
        if [ "$rc" = 1 ] && [[ $glast == *" FAIL" ]]; then ok "CAUGHT generate mutant skip-kv-append-L0: $glast"; else bad "generate mutant skip-kv-append-L0 rc $rc: $glast"; fi
      else bad "generate mutant compile"; fi
    fi
    printf '%s\n' "${s/const FKVP ulong 32/const FKVP ulong 8}" > "$b/genkv.om"
    if build "$b/genkv" "$b/genkv.om"; then
      genr "$b/genkv"
      if [ "$rc" = 7 ] && [[ $gline == *"refused 7 kv-cache-bound"* ]]; then ok "generate with FKVP 8 refused at the cache bound (rc 7)"; else bad "generate kv guard rc $rc: $glast"; fi
    else bad "generate kv guard compile"; fi
  fi
else
  echo "NOT_RUN greedy generation (set FLINT_GEN=1)"
fi

n=$((pass + fail))
if [ "$fail" = 0 ]; then echo "OM-FLINT-MANIFEST PASS ($n)"; else echo "OM-FLINT-MANIFEST FAIL ($fail of $n)"; exit 1; fi
