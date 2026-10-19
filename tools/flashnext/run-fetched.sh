#!/usr/bin/env bash
# run-fetched.sh: runs `flashnext_check run` on HTTP-range-fetched tensors and fetches whatever it reports missing
# (routed experts chosen by the real router, n-gram table rows of the real token history), until it passes.
#   REPO=... REV=... HEADERS=... run-fetched.sh BIN CONFIG_JSON DIR LAYERS stored|mxfp4 OUT_DIR ID...
# Every fetch goes through hf-range-fetch.sh (SOURCE.tsv in DIR records each tensor's byte range and sha256).
set -u
BIN=$1 CFG=$2 DIR=$3 LAYERS=$4 FMT=$5 OUT=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
SHARD_ROWS=2500012
for round in $(seq 1 16); do
    res=$("$BIN" run "$CFG" "$DIR" "$LAYERS" "$FMT" "$OUT" "$@" 2>&1); rc=$?
    echo "round $round rc $rc: $(printf '%s' "$res" | head -c 300)"
    [ $rc -eq 0 ] && exit 0
    [ $rc -eq 3 ] || exit $rc
    spec=$(mktemp)
    printf '%s\n' "$res" | grep '^MISSING' | while read -r line; do
        case $line in
        *"missing experts: layer "*)
            L=$(sed -E 's/.*layer ([0-9]+):.*/\1/' <<<"$line"); ids=${line##*: }
            for e in $ids; do for m in gate up down; do for s in weight weight_scale weight_scale_2 input_scale; do
                n=model.language_model.layers.$L.mlp.experts.$e.${m}_proj.$s; echo "$n $n"
            done; done; done;;
        *"missing ngram rows: "*)
            for id in ${line##*: }; do
                echo "ngram_row.$id model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_$((id / SHARD_ROWS)).weight $((id % SHARD_ROWS)) 1"
            done;;
        esac
    done > "$spec"
    echo "  fetching $(wc -l < "$spec") tensors"
    "$HERE/hf-range-fetch.sh" "$REPO" "$REV" "$HEADERS" "$DIR" "$spec" | tail -1
    rm -f "$spec"
done
echo "gave up after 16 rounds"; exit 1
