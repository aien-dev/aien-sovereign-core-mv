#!/usr/bin/env bash
# spec of every non-expert tensor of layers 0..N-1 (plus the final mixer and embedding rows) in a headers dir
# usage: mkstage1.sh HEADERS_DIR N "TOKEN_IDS" > spec
H=$1 N=$2 IDS=$3
cat "$H"/*.json | jq -r 'to_entries[] | select(.key != "__metadata__") | "\(.key) \(.value.shape | reduce .[] as $x (1; . * $x)) \(.value.dtype)"' |
while read -r k n d; do
  case $k in
    model.language_model.hyper_connection_mixer.*) echo "$k $k";;
    model.language_model.layers.*)
      L=${k#model.language_model.layers.}; L=${L%%.*}
      [ "$L" -lt "$N" ] || continue
      case $k in *.mlp.experts.*|*ngram_embedding.shard_*) continue;; esac
      echo "$k $k";;
  esac
done | sort -u
for t in $IDS; do echo "embed_row.$t model.language_model.embed_tokens.weight $t 1"; done | sort -u
