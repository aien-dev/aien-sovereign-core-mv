#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DRA-65 chip window (Main runs it inside a quiet exclusive spark-window; see README.md "Window").
# The script never starts or stops a server: Main starts each server as README.md lists, then runs
#   window-dra65.sh golden PORT OUT green|red   32-prompt set + long prompts vs the pinned goldens
#   window-dra65.sh time   PORT OUT              TTFT and end-to-end tok/s, 512 and ~8k prompts, 3 runs each
# Exit: golden 0 = verdict as expected (green: 35/35 identical; red: at least one differs), 1 = not as
# expected, 2 = no verdict (server unreachable, unparsable replies). time: 0 = receipts written, 2 = none.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
MODE=${1:?golden|time}; PORT=${2:?port}; OUT=${3:?out dir}; mkdir -p "$OUT"
URL=http://127.0.0.1:$PORT
MODEL=$(curl -s -m 10 "$URL/v1/models" | jq -r '.data[0].id // empty')
[ -n "$MODEL" ] || { echo "window-dra65: no model on $URL" >&2; exit 2; }
DEF=$(jq -c .defaults "$HERE/set32.json")
body() { # <user text> <max_tokens> [stream]
  jq -cn --arg m "$MODEL" --arg u "$1" --argjson n "$2" --argjson d "$DEF" --argjson s "${3:-false}" \
    '$d + {model:$m, messages:[{role:"user",content:$u}], max_tokens:$n} + (if $s then {stream:true, stream_options:{include_usage:true}} else {} end)'
}
ask() { curl -s -m 1800 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
# pins: the set, the prompt files and the long goldens must hash as committed
(cd "$REPO" && sha256sum -c --quiet "$HERE/SHA256SUMS") || { echo "window-dra65: pinned inputs changed" >&2; exit 2; }

case $MODE in
golden)
  EXPECT=${4:?green|red}; same=0; diff=0; bad=0
  for i in $(seq 0 31); do
    u=$(jq -r ".prompts[$i].user" "$HERE/set32.json"); n=$(jq -r ".prompts[$i].max_tokens" "$HERE/set32.json")
    ask "$(body "$u" "$n")" > "$OUT/p$i.raw"
    got=$(jq -c '.usage.aien_token_ids // empty' "$OUT/p$i.raw" 2>/dev/null); want=$(jq -c ".prompts[$i].golden_ids" "$HERE/set32.json")
    if [ -z "$got" ]; then bad=$((bad+1)); v=NO_IDS; elif [ "$got" = "$want" ]; then same=$((same+1)); v=SAME; else diff=$((diff+1)); v=DIFF; fi
    jq -cn --arg id "p$i" --arg v $v --argjson g "${got:-null}" --argjson w "$want" '{id:$id, verdict:$v, got:$g, golden:$w}' >> "$OUT/golden.jsonl"
  done
  for k in 0 1 2; do
    f=$(jq -r ".prompts[$k].file" "$HERE/long-goldens.json"); name=$(jq -r ".prompts[$k].name" "$HERE/long-goldens.json")
    ask "$(body "$(cat "$REPO/$f")" 24)" > "$OUT/$name.raw"
    got=$(jq -c '.usage.aien_token_ids // empty' "$OUT/$name.raw" 2>/dev/null); want=$(jq -c ".prompts[$k].golden_ids" "$HERE/long-goldens.json")
    if [ -z "$got" ]; then bad=$((bad+1)); v=NO_IDS; elif [ "$got" = "$want" ]; then same=$((same+1)); v=SAME; else diff=$((diff+1)); v=DIFF; fi
    jq -cn --arg id "$name" --arg v $v --argjson g "${got:-null}" --argjson w "$want" '{id:$id, verdict:$v, got:$g, golden:$w}' >> "$OUT/golden.jsonl"
  done
  if [ $bad -gt 0 ]; then verdict=INCONCLUSIVE; rc=2
  elif [ "$EXPECT" = green ] && [ $same -eq 35 ]; then verdict=PASS; rc=0
  elif [ "$EXPECT" = red ] && [ $diff -gt 0 ]; then verdict=PASS; rc=0
  else verdict=FAIL; rc=1; fi
  jq -cn --arg e "$EXPECT" --arg v $verdict --argjson s $same --argjson d $diff --argjson b $bad --arg port "$PORT" --arg at "$(date -u +%FT%TZ)" \
    '{case:"golden", expect:$e, verdict:$v, identical:$s, differ:$d, no_ids:$b, of:35, port:$port, at:$at}' | tee "$OUT/golden-receipt.json"
  exit $rc ;;
time)
  # One streamed request per run: TTFT = first content delta (client clock), e2e tok/s = completion
  # tokens / (last byte - first content delta). Medians go through Wolfram afterwards (README).
  for name in long-512 long-8192; do
    f=$HERE/$name.txt; [ -f "$f" ] || f=$REPO/evidence/merge-train/l369/$name.txt
    for run in 1 2 3; do
      o=$OUT/$name-r$run; t0=$(date +%s.%N); first=""
      while IFS= read -r line; do
        case $line in data:*'"content":"'?*) [ -z "$first" ] && first=$(date +%s.%N);; esac
        printf '%s\n' "$line" >> "$o.sse"
      done < <(curl -s -N -m 1800 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body "$(cat "$f")" 256 true)")
      t1=$(date +%s.%N)
      usage=$(grep '"usage"' "$o.sse" | grep -v '"usage":null' | tail -n 1 | sed 's/^data: //' | jq -c '.usage // null' 2>/dev/null)
      jq -cn --arg n "$name" --argjson r $run --arg t0 "$t0" --arg tf "${first:-}" --arg t1 "$t1" --argjson u "${usage:-null}" \
        '{prompt:$n, run:$r, t0:$t0, t_first:$tf, t_end:$t1, usage:$u}' | tee -a "$OUT/time.jsonl"
    done
  done
  exit 0 ;;
*) echo "window-dra65: mode golden|time" >&2; exit 2 ;;
esac
