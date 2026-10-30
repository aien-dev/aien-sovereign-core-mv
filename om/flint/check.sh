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

# 4b. tokenizer and control-token identity (#834): tok.om reads the pinned tokenizer.json,
#     tokenizer_config.json and generation_config.json and checks every record of
#     test/tok-fixture.txt (vocab size, BOS/EOS/pad roles, 19 control tokens by id, special flag
#     and bytes, the decode of 5 reference-encoded texts back to their exact bytes, the encode of
#     440 UTF-8 texts (ASCII and CJK, accented Latin, combining marks, emoji, other digits and
#     spaces; 400 of them from the seeded corpora) to the reference's exact ids, the piece ends of
#     the 258 of them without added tokens, and the reference's class of every code point).
#     test/tokref.py (FLINT_PY: tokenizers 0.23.2, transformers a672eb5f) must reproduce the
#     committed fixture byte for byte. Every fixture, input and source mutant must turn its named
#     record to FAIL (exit 1); every refusal must come from the guard named for it.
#     The Unicode class table in tok.om is generated by test/ucdgen.py from UCD 16.0.0 (the
#     reference's Unicode version): the block must hash to UCD_PIN, and with FLINT_UCD (the
#     directory holding UnicodeData.txt and PropList.txt of 16.0.0) ucdgen.py must reproduce it.
UCD_PIN=e56ba64df874bee73f2ff576ef498dc1283d02e9c258f9a1d945a9a605fd1b39
sed -n '/^note ======== unicode classes/,/^note ======== end of generated unicode classes/p' "$here/tok.om" > "$b/ucdblk.txt"
if [ "$(sha "$b/ucdblk.txt")" = "$UCD_PIN" ]; then ok "tok: generated Unicode class block, sha $UCD_PIN"
else bad "tok: Unicode class block in tok.om does not hash to UCD_PIN"; fi
if [ -n "${FLINT_UCD:-}" ]; then
  if ${FLINT_PY:-python3} "$here/test/ucdgen.py" "$FLINT_UCD" > "$b/ucdgen.txt" 2>&1 < /dev/null && cmp -s "$b/ucdgen.txt" "$b/ucdblk.txt"; then
    ok "tok: test/ucdgen.py reproduces the Unicode class block from UCD 16.0.0"
  else bad "tok: test/ucdgen.py does not reproduce the Unicode class block: $(head -c 200 "$b/ucdgen.txt")"; fi
else
  echo "NOT_RUN ucdgen (regenerating the Unicode class block; needs FLINT_UCD)"
fi
TOKJ_PIN=a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68
TOKC_PIN=159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf
GENC_PIN=a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3
TPL_PIN=7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466
TOK_SUM='flint-tok summary vocab 150000 added 144 records 729 pass 729 fail 0 PASS'
TOK_MUT='eos-id#x#s/^role eos 150001$/role eos 150002/#role eos
think-id#x#s/^ctl think-open 150006 /ctl think-open 150007 /#ctl think-open
think-special#x#s/^ctl think-close 150007 0 /ctl think-close 150007 1 /#ctl think-close
chat-id#x#s/^case chat 41 150002 /case chat 41 150003 /#case chat
ascii-byte#x#s/ 48656c6c6f2c2077/ 49656c6c6f2c2077/#case ascii
vocab-size#x#s/^size 150000 144$/size 150000 143/#size
json-think-swap#j#s/"content": "<think>"/"content": "@T@"/; s/"content": "<\/think>"/"content": "<think>"/; s/"content": "@T@"/"content": "<\/think>"/#ctl think-open
json-vocab-swap#j#s/"Hello": 9479,/"Hello": 1571,/; s/"Ġworld": 1571,/"Ġworld": 9479,/#case ascii
cfg-eos#c#s/"eos_token": "<|end_of_text|>"/"eos_token": "<|user|>"/#role eos
gen-eos#g#s/150005,/150005/; /^ *150018$/d#role geneos
enc-id#x#s/^enc hello 1 9479 /enc hello 1 9480 /#enc hello
pre-end#x#s/^pre hello 1 5 /pre hello 1 4 /#pre hello
ucd-class#x#s/^ucd unicode-16.0.0 8013 61396b35/ucd unicode-16.0.0 8013 62396b35/#ucd unicode-16.0.0'
TOK_FIX='json-size#j#d#50
json-syntax#j#$s/}$//#51
json-decoder#j#s/"type": "ByteLevel"/"type": "Metaspace"/#52
json-vocab-dup#j#s/"Hello": 9479,/"Hello": 11,/#53
json-vocab-char#j#s/"Hello": 9479,/"\\u4e00": 9479,/#54
json-added-gap#j#s/"id": 150006,/"id": 150099,/#55
cfg-json#c#$s/}$//#56
gen-num#g#s/"pad_token_id": 150001/"pad_token_id": -1/#57
fixture-record#x#$a bogus x#58
json-pretok-regex#j#s/p{N}{1,3}/p{N}{1,4}/#52
enc-bad-cont#x#$a enc badcont 1 0 2 c328#81
enc-overlong#x#$a enc overlong 1 0 2 c0af#81
enc-surrogate#x#$a enc surrogate 1 0 3 eda080#81
enc-past-max#x#$a enc pastmax 1 0 4 f4908080#81
enc-truncated#x#$a enc truncated 1 0 2 e381#81'
if [ -n "${FLINT_SRC:-}" ]; then
  tj=$FLINT_SRC/tokenizer.json tc=$FLINT_SRC/tokenizer_config.json tg=$FLINT_SRC/generation_config.json
  tfx=$here/test/tok-fixture.txt
  tok() { # binary tokjson tokcfg gencfg fixture; sets rc, leaves $b/tok.out and $b/tok.err
    timeout 60 "$1" "$2" "$3" "$4" "$5" > "$b/tok.out" 2> "$b/tok.err" < /dev/null
    rc=$?
  }
  tokmut() { # kind sed: writes the four inputs with one of them edited to $b/tm.{j,c,g,x}
    cp "$tj" "$b/tm.j"; cp "$tc" "$b/tm.c"; cp "$tg" "$b/tm.g"; cp "$tfx" "$b/tm.x"
    if [ "$2" = d ]; then : > "$b/tm.$1"; else sed -i "$2" "$b/tm.$1"; fi
    ! cmp -s "$b/tm.$1" "$( case $1 in j) echo "$tj";; c) echo "$tc";; g) echo "$tg";; x) echo "$tfx";; esac )"
  }
  if [ "$(sha "$tj")" != "$TOKJ_PIN" ] || [ "$(sha "$tc")" != "$TOKC_PIN" ] || [ "$(sha "$tg")" != "$GENC_PIN" ] || [ "$(sha "$FLINT_SRC/chat_template.jinja")" != "$TPL_PIN" ]; then
    bad "tok: tokenizer files in FLINT_SRC are not the pinned revision 2a247be0"
  elif ! build "$b/tok" "$here/tok.om"; then bad "tok: tok.om does not compile"
  else
    trj="$b/tokraw.json"
    sed 's/"content": "<|reserved_150143|>"/"content": "<|reserved 150143\\t|>"/' "$tj" > "$trj"
    if ${FLINT_PY:-python3} "$here/test/tokref.py" "$FLINT_SRC" "$b/tokfx.txt" "$trj" "$b/tokfxr.txt" > "$b/tokref.out" 2>&1 < /dev/null && cmp -s "$b/tokfx.txt" "$tfx" && cmp -s "$b/tokfxr.txt" "$here/test/tok-fixture-raw.txt"; then
      ok "tok: test/tokref.py reproduces test/tok-fixture.txt and test/tok-fixture-raw.txt"
    else
      bad "tok: test/tokref.py does not reproduce the fixtures: $(head -c 300 "$b/tokref.out")"
    fi
    tok "$b/tok" "$trj" "$tc" "$tg" "$here/test/tok-fixture-raw.txt"
    if [ "$rc" = 0 ] && grep -qx "flint-tok case raw-added PASS" "$b/tok.out"; then ok "tok: raw-added decode (token outside the ByteLevel alphabet) PASS"
    else bad "tok: raw-added decode rc $rc, $(tail -n 1 "$b/tok.out")"; fi
    tok "$b/tok" "$tj" "$tc" "$tg" "$tfx"
    if [ "$rc" = 0 ] && [ "$(tail -n 1 "$b/tok.out")" = "$TOK_SUM" ] && [ ! -s "$b/tok.err" ]; then
      while read -r line; do ok "$line"; done < <(grep -v '^flint-tok summary ' "$b/tok.out")
    else
      bad "tok: rc $rc, $(tail -n 1 "$b/tok.out") $(head -c 200 "$b/tok.err")"
    fi
    while IFS='#' read -r m k e want; do
      tokmut "$k" "$e" || { bad "tok mutant $m: edit changed nothing"; continue; }
      tok "$b/tok" "$b/tm.j" "$b/tm.c" "$b/tm.g" "$b/tm.x"
      if [ "$rc" = 1 ] && grep -qx "flint-tok $want FAIL" "$b/tok.out"; then ok "tok mutant $m: $want FAIL"
      else bad "tok mutant $m: want $want FAIL, rc $rc, $(grep -m1 ' FAIL$' "$b/tok.out") $(head -c 120 "$b/tok.err")"; fi
    done <<< "$TOK_MUT"
    "$b/tok" "$tj" "$tc" "$tg" > "$b/tok.out" 2> /dev/null < /dev/null
    r=$?
    if [ "$r" = 2 ] && [ ! -s "$b/tok.out" ]; then ok "tok usage: 2"; else bad "tok usage: $r"; fi
    tok "$b/tok" "$b/absent.json" "$tc" "$tg" "$tfx"
    if [ "$rc" = 3 ] && [ ! -s "$b/tok.out" ]; then ok "tok absent tokenizer.json: 3"; else bad "tok absent tokenizer.json: $rc"; fi
    tok "$b/tok" "$b" "$tc" "$tg" "$tfx"
    if [ "$rc" = 4 ] && [ ! -s "$b/tok.out" ]; then ok "tok directory as tokenizer.json: 4"; else bad "tok directory as tokenizer.json: $rc"; fi
    st=$(cat "$here/tok.om")
    { printf 'enc capacity 1 0 16385 61'; printf '2061%.0s' $(seq 8192); echo; } > "$b/tokcap.txt"
    TOK_FIX="$TOK_FIX"$'\n'"enc-capacity#x#\$r $b/tokcap.txt#59"
    while IFS='#' read -r m k e want; do
      tokmut "$k" "$e" || { bad "tok refusal $m: edit changed nothing"; continue; }
      tok "$b/tok" "$b/tm.j" "$b/tm.c" "$b/tm.g" "$b/tm.x"
      if [ "$rc" = "$want" ] && [ ! -s "$b/tok.out" ] && grep -q "^refused $want " "$b/tok.err"; then ok "tok refusal $m: $want"
      else bad "tok refusal $m: want $want, got $rc ($(head -c 120 "$b/tok.err"))"; continue; fi
      t=${st//"tk_die $want tr$want$nl"/"put gs 63 get gs 63$nl"}
      if [ "$t" = "$st" ]; then bad "tok mutant die-$want: no guard text found"; continue; fi
      printf '%s\n' "$t" > "$b/tmut.om"
      if ! build "$b/tmut" "$b/tmut.om"; then bad "tok mutant die-$want: does not compile"; continue; fi
      tok "$b/tmut" "$b/tm.j" "$b/tm.c" "$b/tm.g" "$b/tm.x"
      if [ "$rc" = "$want" ]; then bad "tok mutant die-$want: still exits $want on $m"; else ok "tok mutant die-$want: $m now exits $rc, not $want"; fi
    done <<< "$TOK_FIX"
    TOK_SRC='byte-map|ret add k 94|ret add k 93|case utf8|pin
special-flag|b = eq ulong get gsp id add sp 1|b = eq ulong get gsp id 1|ctl begin-text|pin
raw-bytes|tk_dput ulong get gpool add o i|tk_dput ulong get gpool add o 0|case raw-added|raw
digit-split|const NDIG ulong 3|const NDIG ulong 4|enc digits|pin
no-lookahead|r = te_a5 p e w|r = p|enc ws-word|pin
bpe-max-rank|if or eq bi NOINT lt r br|if or eq bi NOINT gt r br|enc hello|pin
contraction-3|ret add l 3|ret l|enc contract|pin
a1-no-x|r = te_a1 p e 0|r = NOINT|enc contract|pin
utf8-cp|c = add mul c 64 sub x 128|c = add mul c 64 sub x 127|enc cjk|pin
ws-class|ret eq k 10|ret eq k 0|enc r025|pin
no-mark-w|ret or eq k 2 and ge k 4 le k 6|ret or eq k 2 and ge k 4 le k 5|pre decomp|pin
no-mark-u|ret or eq k 1 and ge k 3 le k 6|ret or eq k 1 and ge k 3 le k 5|pre u031|pin
cjk-kata|and ge c 12448 le c 12543|and ge c 12544 le c 12543|pre cjk-kata|pin
a1-no-backtrack|q = sub u j|q = u|pre decomp|pin
long-s-fold|if eq c 383|if eq c 384|pre long-s|pin
ucd-table|text ucd "a9k5|text ucd "a9a5|ucd unicode-16.0.0|pin'
    while IFS='|' read -r m from to want fx; do
      [ "$(grep -cF "$from" "$here/tok.om")" = 1 ] || { bad "tok mutant $m: site not found exactly once"; continue; }
      printf '%s\n' "${st/"$from"/"$to"}" > "$b/tmut.om"
      if ! build "$b/tmut" "$b/tmut.om"; then bad "tok mutant $m: does not compile"; continue; fi
      if [ "$fx" = raw ]; then tok "$b/tmut" "$trj" "$tc" "$tg" "$here/test/tok-fixture-raw.txt"
      else tok "$b/tmut" "$tj" "$tc" "$tg" "$tfx"; fi
      if [ "$rc" = 1 ] && grep -qx "flint-tok $want FAIL" "$b/tok.out"; then ok "tok mutant $m: $want FAIL"
      else bad "tok mutant $m: want $want FAIL, rc $rc, $(head -c 120 "$b/tok.err")"; fi
    done <<< "$TOK_SRC"
  fi
else
  echo "NOT_RUN tok (tokenizer identity; needs FLINT_SRC)"
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
    s=$(cat "$here/ops.om")
    while IFS='|' read -r name from to; do
      [ "$(grep -cF -- "$from" "$here/ops.om")" = 1 ] || { bad "forward mutant $name site"; continue; }
      printf '%s\n' "${s/"$from"/"$to"}" > "$b/fwdmut.om"
      build "$b/fwdmut" "$b/fwdmut.om" || { bad "forward mutant $name compile"; continue; }
      line=$(timeout 600 "$b/fwdmut" forward "$b/forward.fx" "$b/real.tsv" "$FLINT_SHARDS" 2>&1 < /dev/null)
      rc=$?
      if [ "$rc" = 1 ] && [[ $line == *" FAIL" ]]; then ok "CAUGHT forward mutant $name: ${line##*$'\n'}"; else bad "forward mutant $name rc $rc: $line"; fi
    done <<'FORWARD_MUTANTS'
wrong-token|id = fx_w add 8 t|id = add fx_w add 8 t 1
skip-final-norm|put gfh o get gy i|put gfh o get gv i
FORWARD_MUTANTS
  fi
else
  echo "NOT_RUN chained forward (set FLINT_FORWARD=1)"
fi

n=$((pass + fail))
if [ "$fail" = 0 ]; then echo "OM-FLINT-MANIFEST PASS ($n)"; else echo "OM-FLINT-MANIFEST FAIL ($fail of $n)"; exit 1; fi
