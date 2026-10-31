#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DRA-65 chip window. Main runs each step as `spark-window run bash window-dra65.sh ...` inside a
# `--gb10 exclusive --quiet exclusive` window; anywhere else it refuses with exit 3 (NOT_RUN): it needs
# SPARK_WINDOW (the window id) and AIEN_GB10_LOCK_HELD=1, both set by `spark-window run`. It never takes
# /tmp/aien-gb10.lock itself (the window holds it) and never starts or stops a server: Main starts each
# server as README.md lists (~33 s per arm), then runs
#   window-dra65.sh plan     (no window needed: the steps with their estimated durations)
#   window-dra65.sh golden PORT OUT red   BIN_SHA256 LABEL
#   window-dra65.sh golden PORT OUT green BIN_SHA256 LABEL RED_RECEIPT RED_RECEIPT [...]
#       36 cases (32-prompt set, 3 long prompts, turn 2 restoring the long-2048 snapshot) vs the pinned goldens.
#       green refuses to run unless it is handed at least two red receipts (verdict PASS, distinct labels)
#       of the same server binary sha256: red before green.
#   window-dra65.sh time PORT OUT BIN_SHA256 LABEL
#       TTFT and end-to-end tok/s, ~512 and 8210-token prompts, 3 runs each.
# Per case: SAME (ids equal), DIFF (a flip inside the common length, or more ids than the golden),
# SHORT (a strict prefix of the golden: no flip shown), NO_IDS (HTTP error, missing or empty ids).
# golden red:   PASS = at least one DIFF and no NO_IDS/SHORT; NO_IDS/SHORT -> INCONCLUSIVE; else FAIL.
# golden green: PASS = all SAME; any NO_IDS -> INCONCLUSIVE; else FAIL.
# Exit 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE or refused, 3 = NOT_RUN (not inside a chip window). time: 0
# only if every run has HTTP 200, a first content delta and usage with completion_tokens > 0; else 2.
# Every receipt line carries the window id and the binary sha256.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
die() { echo "window-dra65: $*" >&2; exit 2; }
if [ "${1:-}" = plan ]; then
  # MODELED (Wolfram, README "Window"): decode 0.0201 s/token, chunked prompts 250 tok/s (8210: 196.8),
  # stepped prompts 0.0201 s/token; upper bounds at max_tokens; plus ~33 s server start per arm.
  cat <<'EOF'
step arm                                   mode    est_s  (incl. 33 s start)
1    single NEW AIEN_Q36_TEST_MUTANT=4     red     120
2    single NEW AIEN_OMEGA_Q36_PF_MUTANT=2 red     120
3    single NEW PF_MUTANT=2 + CHUNK=0      green   289    (every prompt token stepped)
4    single NEW plain                      green   120
5    batch NEW --batch-slots 8             green   120
6a   time single NEW                       time    195
6b   time single NEW CHUNK=0               time    590
6c   time batch NEW                        time    195
6d   time batch NEW AIEN_Q36_PREFILL_MIN=1 time    195
20-min leases: L1 = steps 1-5 (769 s), L2 = 6a + 6b (785 s), L3 = 6c + 6d (390 s).
EOF
  exit 0
fi
if [ -z "${SPARK_WINDOW:-}" ] || [ "${AIEN_GB10_LOCK_HELD:-}" != 1 ]; then
  echo "window-dra65: NOT_RUN: run inside a --gb10 exclusive window via 'spark-window run' (SPARK_WINDOW='${SPARK_WINDOW:-}', AIEN_GB10_LOCK_HELD='${AIEN_GB10_LOCK_HELD:-}')" >&2
  exit 3
fi
WIN=$SPARK_WINDOW
MODE=${1:-}; PORT=${2:-}; OUT=${3:-}
[ -n "$MODE" ] && [ -n "$PORT" ] && [ -n "$OUT" ] || die "usage: see the header"
URL=http://127.0.0.1:$PORT
(cd "$REPO" && sha256sum -c --quiet "$HERE/SHA256SUMS") || die "pinned inputs changed"
mkdir -p "$OUT"
MODEL=$(curl -s -m 10 "$URL/v1/models" | jq -r '.data[0].id // empty' 2>/dev/null)
[ -n "$MODEL" ] || die "no model on $URL"
DEF=$(jq -c .defaults "$HERE/set32.json")
SET=$HERE/set32.json; LONG=$HERE/long-goldens.json
req() { # <messages json> <max_tokens> [stream]
  jq -cn --arg m "$MODEL" --argjson msg "$1" --argjson n "$2" --argjson d "$DEF" --argjson s "${3:-false}" \
    '$d + {model:$m, messages:$msg, max_tokens:$n} + (if $s then {stream:true, stream_options:{include_usage:true}} else {} end)'
}
user() { jq -cn --arg u "$1" '[{role:"user",content:$u}]'; }
# ask <messages> <max_tokens> <raw file>: prints the HTTP code
ask() { curl -s -m 1800 -o "$3" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$(req "$1" "$2")"; }
judge() { # <http code> <raw file> <golden ids json>: prints SAME|DIFF|SHORT|NO_IDS
  local got; [ "$1" = 200 ] || { echo NO_IDS; return; }
  got=$(jq -c '.usage.aien_token_ids | select(type == "array" and length > 0)' "$2" 2>/dev/null)
  [ -n "$got" ] || { echo NO_IDS; return; }
  jq -rn --argjson g "$got" --argjson w "$3" '
    ([$g, $w] | map(length) | min) as $m
    | if $g == $w then "SAME"
      elif ([range(0; $m) | select($g[.] != $w[.])] | length) > 0 then "DIFF"
      elif ($g | length) > ($w | length) then "DIFF"
      else "SHORT" end'
}

case $MODE in
golden)
  EXPECT=${4:-}; BIN=${5:-}; LABEL=${6:-}
  case $EXPECT in red|green) ;; *) die "golden needs red|green";; esac
  [[ $BIN =~ ^[0-9a-f]{64}$ ]] || die "BIN_SHA256 must be the server binary's sha256"
  [ -n "$LABEL" ] || die "LABEL missing"
  if [ "$EXPECT" = green ]; then
    shift 6; reds=$(for r in "$@"; do jq -r --arg b "$BIN" \
      'select(.case == "golden" and .expect == "red" and .verdict == "PASS" and .bin_sha256 == $b) | .label' "$r" 2>/dev/null; done | sort -u | wc -l)
    [ "$reds" -ge 2 ] || die "green refused: needs >= 2 red PASS receipts (distinct labels) for binary $BIN, got $reds"
  fi
  : > "$OUT/golden.jsonl"; declare -A n=([SAME]=0 [DIFF]=0 [SHORT]=0 [NO_IDS]=0)
  rec() { jq -cn --arg id "$1" --arg v "$2" --arg h "$3" --argjson w "$4" --arg win "$WIN" --arg b "$BIN" \
    --slurpfile r <(jq -c . "$5" 2>/dev/null || echo null) \
    '{id:$id, verdict:$v, http:$h, got:($r[0].usage.aien_token_ids // null), golden:$w, window:$win, bin_sha256:$b}' >> "$OUT/golden.jsonl"; }
  for i in $(seq 0 31); do
    u=$(jq -r ".prompts[$i].user" "$SET"); m=$(jq -r ".prompts[$i].max_tokens" "$SET"); w=$(jq -c ".prompts[$i].golden_ids" "$SET")
    h=$(ask "$(user "$u")" "$m" "$OUT/p$i.raw"); v=$(judge "$h" "$OUT/p$i.raw" "$w"); n[$v]=$((n[$v]+1)); rec "p$i" "$v" "$h" "$w" "$OUT/p$i.raw"
  done
  # turn2 right after long-2048: the door restores the snapshot it took at long-2048's answer header.
  for name in long-256 long-8192 long-2048 turn2; do
    c=$(jq -c --arg n "$name" '.prompts[] | select(.name == $n)' "$LONG"); w=$(jq -c .golden_ids <<<"$c"); m=$(jq -r .max_tokens <<<"$c")
    if [ "$name" = turn2 ]; then msg=$(jq -c . "$REPO/$(jq -r .messages <<<"$c")"); else msg=$(user "$(cat "$REPO/$(jq -r .file <<<"$c")")"); fi
    h=$(ask "$msg" "$m" "$OUT/$name.raw"); v=$(judge "$h" "$OUT/$name.raw" "$w"); n[$v]=$((n[$v]+1)); rec "$name" "$v" "$h" "$w" "$OUT/$name.raw"
  done
  if [ "$EXPECT" = red ]; then
    if [ $((n[NO_IDS] + n[SHORT])) -gt 0 ]; then verdict=INCONCLUSIVE; rc=2
    elif [ "${n[DIFF]}" -gt 0 ]; then verdict=PASS; rc=0; else verdict=FAIL; rc=1; fi
  else
    if [ "${n[NO_IDS]}" -gt 0 ]; then verdict=INCONCLUSIVE; rc=2
    elif [ "${n[SAME]}" -eq 36 ]; then verdict=PASS; rc=0; else verdict=FAIL; rc=1; fi
  fi
  jq -cn --arg e "$EXPECT" --arg v $verdict --arg b "$BIN" --arg l "$LABEL" --argjson s "${n[SAME]}" --argjson d "${n[DIFF]}" \
    --argjson sh "${n[SHORT]}" --argjson x "${n[NO_IDS]}" --arg port "$PORT" --arg at "$(date -u +%FT%TZ)" --arg win "$WIN" \
    '{case:"golden", expect:$e, verdict:$v, bin_sha256:$b, label:$l, window:$win, same:$s, diff:$d, short:$sh, no_ids:$x, of:36, port:$port, at:$at}' \
    | tee "$OUT/golden-receipt.json"
  exit $rc ;;
time)
  BIN=${4:-}; LABEL=${5:-}
  [[ $BIN =~ ^[0-9a-f]{64}$ ]] || die "BIN_SHA256 must be the server binary's sha256"
  [ -n "$LABEL" ] || die "LABEL missing"
  # TTFT = first content delta - t0 (client clock); e2e tok/s = completion_tokens / (t_end - t_first).
  # Medians of the 3 runs go through Wolfram afterwards (README).
  : > "$OUT/time.jsonl"; ok=1
  for name in long-512 long-8192; do
    f=$HERE/$name.txt; [ -f "$f" ] || f=$REPO/evidence/merge-train/l369/$name.txt
    for run in 1 2 3; do
      o=$OUT/$name-r$run; : > "$o.sse"; t0=$(date +%s.%N); first=""
      while IFS= read -r line; do
        [ -z "$first" ] && [[ $line =~ ^data:.*\"content\":\ *\"[^\"] ]] && first=$(date +%s.%N)
        printf '%s\n' "$line" >> "$o.sse"
      done < <(curl -s -N -m 1800 -w '\nHTTP %{http_code}\n' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
                 -d "$(req "$(user "$(cat "$f")")" 256 true)")
      t1=$(date +%s.%N)
      code=$(grep -o '^HTTP [0-9]*' "$o.sse" | tail -n 1 | cut -c6-)
      usage=$(grep '^data: {' "$o.sse" | sed 's/^data: //' | jq -c 'select(.usage != null) | .usage' 2>/dev/null | tail -n 1)
      good=$(jq -n --arg c "${code:-}" --arg f "$first" --argjson u "${usage:-null}" \
        '$c == "200" and $f != "" and ($u | type) == "object" and (($u.completion_tokens // 0) > 0)')
      [ "$good" = true ] || ok=0
      jq -cn --arg n "$name" --argjson r $run --arg b "$BIN" --arg l "$LABEL" --arg h "${code:-}" --arg t0 "$t0" --arg tf "$first" \
        --arg t1 "$t1" --argjson u "${usage:-null}" --argjson g "$good" --arg win "$WIN" \
        '{prompt:$n, run:$r, bin_sha256:$b, label:$l, window:$win, http:$h, ok:$g, t0:$t0, t_first:$tf, t_end:$t1, usage:$u}' | tee -a "$OUT/time.jsonl"
    done
  done
  if [ $ok = 1 ]; then jq -cn --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" '{case:"time", verdict:"RECORDED", window:$win, bin_sha256:$b, label:$l}'; exit 0; fi
  jq -cn --arg win "$WIN" --arg b "$BIN" --arg l "$LABEL" '{case:"time", verdict:"INCONCLUSIVE", window:$win, bin_sha256:$b, label:$l, why:"a run had no HTTP 200, no first content delta or no usage"}'; exit 2 ;;
*) die "mode golden|time" ;;
esac
