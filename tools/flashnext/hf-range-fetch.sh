#!/usr/bin/env bash
# hf-range-fetch.sh: fetch single tensors (or a slice along dim 0) of a Hugging Face safetensors checkpoint by
# HTTP range request, without downloading the shards. Written for the Qwen3.8-Flash-Next port (disk blocker:
# the full checkpoint may not be downloaded yet), usable for any safetensors repo.
#
#   hf-range-fetch.sh REPO REV HEADERS_DIR OUT_DIR SPEC_FILE
#
# HEADERS_DIR holds one JSON header per shard, named <shard>.json (the safetensors header, as fetched earlier).
# SPEC_FILE lines (blank and # lines skipped):
#   ALIAS SOURCE_NAME                 whole tensor
#   ALIAS SOURCE_NAME FIRST COUNT     rows FIRST .. FIRST+COUNT-1 along dim 0 (shape [COUNT, rest...];
#                                     COUNT 1 drops dim 0 when the source has more than one dim)
# Each tensor lands as OUT_DIR/<ALIAS>.safetensors (one tensor per file, alias as its name) and
# OUT_DIR/model.safetensors.index.json maps every alias to its file. OUT_DIR/SOURCE.tsv records repo, revision,
# shard, source name, byte range, and the sha256 of the bytes fetched, so a run can be repeated and checked.
# The header offset of each shard is read from the shard itself (first 8 bytes), never assumed.
set -uo pipefail
REPO=$1 REV=$2 HDIR=$3 OUT=$4 SPEC=$5
TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}
AUTH=()
[ -r "$TOKEN_FILE" ] && AUTH=(-H "Authorization: Bearer $(cat "$TOKEN_FILE")")
mkdir -p "$OUT"
BASE="https://huggingface.co/$REPO/resolve/$REV"
declare -A HLEN

le64() { # print 8 little-endian bytes of $1
    local v=$1 i
    for i in 0 1 2 3 4 5 6 7; do printf "\\x$(printf %02x $(( (v >> (8 * i)) & 255 )))"; done
}
dsize() {
    case $1 in BF16|F16|I16|U16) echo 2;; F32|I32|U32) echo 4;; F64|I64|U64) echo 8;; U8|I8|F8_E4M3|F8_E5M2|BOOL) echo 1;;
    *) echo 0;; esac
}
shard_of() { # shard file holding tensor $1
    grep -l -F "\"$1\":" "$HDIR"/*.json | head -n 1 | xargs -r basename | sed 's/\.json$//'
}
header_len() {
    local shard=$1
    if [ -z "${HLEN[$shard]:-}" ]; then
        local hex
        hex=$(curl -sfL "${AUTH[@]}" -r 0-7 "$BASE/$shard" | od -An -tx1 | tr -d ' \n')
        [ ${#hex} -eq 16 ] || { echo "FAIL: header length of $shard" >&2; return 1; }
        local v=0 i
        for i in 7 6 5 4 3 2 1 0; do v=$(( (v << 8) | 16#${hex:$((2 * i)):2} )); done
        HLEN[$shard]=$v
    fi
    echo "${HLEN[$shard]}"
}

[ -f "$OUT/SOURCE.tsv" ] || printf 'alias\trepo\trevision\tshard\tsource\tbyte_first\tbyte_last\tdtype\tshape\tsha256\n' > "$OUT/SOURCE.tsv"
fails=0
while read -r alias src first count; do
    [ -z "${alias:-}" ] && continue
    case $alias in \#*) continue;; esac
    if [ -s "$OUT/$alias.safetensors" ]; then echo "have $alias"; continue; fi
    shard=$(shard_of "$src")
    [ -n "$shard" ] || { echo "FAIL: $src not in any header" >&2; fails=$((fails + 1)); continue; }
    meta=$(jq -c --arg k "$src" '.[$k]' "$HDIR/$shard.json")
    dtype=$(jq -r .dtype <<<"$meta"); shape=$(jq -c .shape <<<"$meta")
    o0=$(jq -r '.data_offsets[0]' <<<"$meta"); o1=$(jq -r '.data_offsets[1]' <<<"$meta")
    es=$(dsize "$dtype")
    [ "$es" -gt 0 ] || { echo "FAIL: dtype $dtype" >&2; fails=$((fails + 1)); continue; }
    if [ -n "${first:-}" ]; then
        d0=$(jq -r '.[0]' <<<"$shape"); nd=$(jq -r length <<<"$shape")
        rowel=$(jq -r '.[1:] | reduce .[] as $x (1; . * $x)' <<<"$shape")
        [ $((first + count)) -le "$d0" ] || { echo "FAIL: $src rows $first+$count > $d0" >&2; fails=$((fails + 1)); continue; }
        a=$((o0 + first * rowel * es)); b=$((a + count * rowel * es))
        if [ "$count" -eq 1 ] && [ "$nd" -gt 1 ]; then oshape=$(jq -c '.[1:]' <<<"$shape"); else oshape=$(jq -c --argjson c "$count" '[$c] + .[1:]' <<<"$shape"); fi
    else
        a=$o0; b=$o1; oshape=$shape
    fi
    hl=$(header_len "$shard") || { fails=$((fails + 1)); continue; }
    abs0=$((8 + hl + a)); abs1=$((8 + hl + b - 1)); n=$((b - a))
    tmp="$OUT/.$alias.part"
    curl -sfL "${AUTH[@]}" --retry 3 -r "$abs0-$abs1" "$BASE/$shard" -o "$tmp"
    got=$(stat -c %s "$tmp" 2>/dev/null || echo 0)
    [ "$got" -eq "$n" ] || { echo "FAIL: $alias got $got of $n bytes" >&2; rm -f "$tmp"; fails=$((fails + 1)); continue; }
    sha=$(sha256sum "$tmp" | cut -d' ' -f1)
    hdr=$(jq -cn --arg k "$alias" --arg d "$dtype" --argjson s "$oshape" --argjson n "$n" '{($k): {dtype: $d, shape: $s, data_offsets: [0, $n]}}')
    while [ $(( ${#hdr} % 8 )) -ne 0 ]; do hdr="$hdr "; done
    { le64 ${#hdr}; printf '%s' "$hdr"; cat "$tmp"; } > "$OUT/$alias.safetensors"
    rm -f "$tmp"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$alias" "$REPO" "$REV" "$shard" "$src" "$abs0" "$abs1" "$dtype" "$oshape" "$sha" >> "$OUT/SOURCE.tsv"
    echo "got $alias $dtype $oshape $n B"
done < "$SPEC"
# index over every tensor file in OUT_DIR
( cd "$OUT" && for f in *.safetensors; do printf '%s\n' "${f%.safetensors}"; done ) \
  | jq -R . | jq -s '{metadata: {}, weight_map: (map({(.): (. + ".safetensors")}) | add)}' > "$OUT/model.safetensors.index.json"
echo "fails=$fails"
[ "$fails" -eq 0 ]
