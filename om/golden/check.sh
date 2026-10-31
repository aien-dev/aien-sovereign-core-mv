#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# om/golden/check.sh: host arms for the golden-token harness (Core #853, DRA-52).
# Usage: OMC=/path/to/omega-mv/seed/out/omc bash om/golden/check.sh
# Every engine reply here comes from fake.om on a random 127.0.0.1 port and every record it makes is
# labelled FAKE. No live engine, model or GPU is touched. Exit 0 all arms as expected, 1 an arm
# failed, 3 omc missing or not pinned (NOT_RUN).
set -u
here=$(cd "$(dirname "$0")" && pwd)
fx=$here/fixtures
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
bash "$here/build.sh" "$t/golden" || exit $?
bash "$here/build.sh" "$t/fake" "$here/fake.om" || exit $?
G=$t/golden
fail=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1: $2"; fail=1; }
# arm NAME WANT_RC WANT_GREP cmd...: exit code must be WANT_RC; stdout+stderr must contain WANT_GREP
arm() {
  local name=$1 want=$2 pat=$3; shift 3
  "$@" > "$t/out" 2>&1; local rc=$?
  if [ "$rc" != "$want" ]; then bad "$name" "exit $rc, want $want: $(cat "$t/out")"; return; fi
  if [ -n "$pat" ] && ! grep -qxF -- "$pat" "$t/out"; then bad "$name" "no line '$pat' in: $(cat "$t/out")"; return; fi
  ok "$name (exit $rc${pat:+, '$pat'})"
}

echo "== comparator, synthetic fixtures"
arm cmp-match 0 'golden-cmp prompts=2 match=2 diverge=0' "$G" cmp "$fx/synthetic-ref.json" "$fx/synthetic-match.json"
arm cmp-diverge-1-token 1 'DIVERGE p1 index=1 ref=20 cand=21' "$G" cmp "$fx/synthetic-ref.json" "$fx/synthetic-diverge.json"
arm cmp-header-engine 3 "NOTCOMPARABLE engine_sha256 ref=\"$(printf 'e%.0s' $(seq 64))\" cand=\"$(printf 'f%.0s' $(seq 64))\"" "$G" cmp "$fx/synthetic-ref.json" "$fx/synthetic-header.json"
sed 's/"model_rev":"0000000000000000000000000000000000000000"/"model_rev":"1111111111111111111111111111111111111111"/' "$fx/synthetic-match.json" > "$t/rev.json"
arm cmp-header-model-rev 3 'NOTCOMPARABLE model_rev ref="0000000000000000000000000000000000000000" cand="1111111111111111111111111111111111111111"' "$G" cmp "$fx/synthetic-ref.json" "$t/rev.json"
sed 's/"ids":\[10,20,30\]/"ids":[10,20]/' "$fx/synthetic-match.json" > "$t/short.json"
arm cmp-prefix-ends-early 1 'DIVERGE p1 index=2 ref=30 cand=end' "$G" cmp "$fx/synthetic-ref.json" "$t/short.json"
head -c 100 "$fx/synthetic-match.json" > "$t/trunc.json"
arm cmp-malformed-json 4 'golden: refused 4 json' "$G" cmp "$fx/synthetic-ref.json" "$t/trunc.json"
sed 's/"temperature":0,/"temperature":0.7,/' "$fx/synthetic-match.json" > "$t/temp.json"
arm cmp-not-greedy-record 4 'golden: refused 4 record-schema' "$G" cmp "$fx/synthetic-ref.json" "$t/temp.json"
sed 's/"label":"SYNTHETIC",/"label":"SYNTHETIC","extra":1,/' "$fx/synthetic-match.json" > "$t/extra.json"
arm cmp-unknown-key 4 'golden: refused 4 record-schema' "$G" cmp "$fx/synthetic-ref.json" "$t/extra.json"
arm usage 2 '' "$G" cmp "$fx/synthetic-ref.json"

echo "== recorder against fake.om (FAKE replies, random 127.0.0.1 port)"
# canned DIR K STATUS BODY: one full HTTP/1.1 response for connection K
canned() {
  mkdir -p "$1"
  printf 'HTTP/1.1 %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' \
    "$3" "${#4}" "$4" > "$1/$2.http"
}
reply() { printf '{"id":"FAKE","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"FAKE"},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6,"aien_token_ids":[%s],"aien_sampling":{"mode":"%s","temperature":0.0}}}' "$1" "$2"; }
# serve NAME N: start fake for N connections on canned dir $t/NAME.c, capture to $t/NAME.cap; sets port
serve() {
  mkdir -p "$t/$1.cap"
  timeout 60 "$t/fake" "$t/$1.port" "$t/$1.c" "$t/$1.cap" "$2" 2> "$t/$1.fakeerr" &
  for _ in $(seq 100); do [ -s "$t/$1.port" ] && break; sleep 0.1; done
  port=$(cat "$t/$1.port")
}
eng=$(printf 'e%.0s' $(seq 64))
rev=0000000000000000000000000000000000000000
printf 'Authorization: Bearer synthetic-test-token\n' > "$t/hdr"
rec() { # rec NAME OUT
  GOLDEN=$G bash "$here/record.sh" --url "http://127.0.0.1:$port" --prompts "$fx/synthetic-prompts.json" \
    --model-repo synthetic/not-a-model --model-rev $rev --max-tokens 8 --engine-sha256 "$eng" \
    --host-id fake-host --label FAKE --out "$2" --header-file "$t/hdr" --timeout 20
}
canned "$t/a.c" 0 '200 OK' "$(reply 1,2,3,4,5 greedy)"
canned "$t/a.c" 1 '200 OK' "$(reply 10,20,30 greedy)"
serve a 2
before=$(date -u +%s)
arm rec-fake-success 0 '' rec a "$t/a.json"
after=$(date -u +%s)
wait
if [ -s "$t/a.json" ]; then
  grep -qxF '{"id":"p0","ids":[1,2,3,4,5]},' "$t/a.json" && grep -qxF '{"id":"p1","ids":[10,20,30]}' "$t/a.json" \
    && ok "rec-ids-taken-from-usage" || bad rec-ids-taken-from-usage "$(cat "$t/a.json")"
  grep -qF '"label":"FAKE"' "$t/a.json" && ok "rec-record-says-FAKE" || bad rec-record-says-FAKE "no FAKE label"
  ps=$(sha256sum "$fx/synthetic-prompts.json" | cut -c1-64)
  grep -qF "\"prompts_sha256\":\"$ps\"" "$t/a.json" && ok "rec-prompts-sha256 = sha256sum ($ps)" || bad rec-prompts-sha256 "want $ps"
  utc=$(grep -o '"utc":"[^"]*"' "$t/a.json" | cut -d'"' -f4); us=$(date -u -d "$utc" +%s)
  [ "$us" -ge "$before" ] && [ "$us" -le "$after" ] && ok "rec-utc $utc within [$before,$after]" || bad rec-utc "$utc not within [$before,$after]"
  "$G" req "$fx/synthetic-prompts.json" synthetic/not-a-model 8 "$(mkdir "$t/rq" && echo "$t/rq")" > /dev/null
  for i in 0 1; do
    tail -c "$(stat -c %s "$t/rq/$i.json")" "$t/a.cap/$i.http" | cmp -s - "$t/rq/$i.json" \
      && ok "rec-request-body-$i byte-equal to golden req" || bad "rec-request-body-$i" "differs"
  done
  grep -qaF 'Authorization: Bearer synthetic-test-token' "$t/a.cap/0.http" && ok "rec-header-file-sent" || bad rec-header-file-sent "missing"
fi
canned "$t/b.c" 0 '200 OK' "$(reply 1,2,3,4,5 greedy)"
canned "$t/b.c" 1 '200 OK' "$(reply 10,20,31 greedy)"
serve b 2
arm rec-fake-rerecord 0 '' rec b "$t/b.json"
wait
arm rec-then-cmp-divergence 1 'DIVERGE p1 index=2 ref=30 cand=31' "$G" cmp "$t/a.json" "$t/b.json"
canned "$t/c.c" 0 '200 OK' "$(reply 1,2,3,4,5 greedy)"
canned "$t/c.c" 1 '200 OK' "$(reply 10,20,30 sampled)"
serve c 2
arm rec-sampled-reply-refused 5 'golden: refused 5 not-greedy' rec c "$t/c.json"
wait
[ ! -e "$t/c.json" ] && ok "rec-refusal-leaves-no-record" || bad rec-refusal-leaves-no-record "$t/c.json written"
canned "$t/d.c" 0 '200 OK' '{"choices":[],"usage":{"prompt_tokens":3}}'
canned "$t/d.c" 1 '200 OK' "$(reply 10,20,30 greedy)"
serve d 2
arm rec-missing-token-ids-refused 5 'golden: refused 5 no-usage-aien_token_ids' rec d "$t/d.json"
wait
canned "$t/e.c" 0 '500 Internal Server Error' '{"error":{"message":"FAKE"}}'
serve e 1
arm rec-http-500-transport 6 'record: refused 6 transport prompt 0' rec e "$t/e.json"
wait
arm rec-existing-out-refused 2 "record: refused 2 $t/a.json exists" rec x "$t/a.json"

echo "== negative controls: mutants of golden.om must fail their arm"
mutant() { # mutant NAME FROM TO: replace exactly one source line, build, set M
  awk -v f="$2" -v r="$3" '$0 == f { $0 = r; n++ } { print } END { if (n != 1) exit 9 }' "$here/golden.om" > "$t/$1.om" \
    || { bad "$1" "line '$2' not unique"; return 1; }
  bash "$here/build.sh" "$t/$1" "$t/$1.om" || { bad "$1" "mutant did not build"; return 1; }
  M=$t/$1
}
mflip() { # mflip NAME CHECK-CMD...: the arm must now fail
  "${@:2}" > "$t/mout" 2>&1 && bad "$1" "mutant survived" || ok "$1 killed"
}
if mutant M1-ignore-header 'if gt bad 0' 'if gt bad 99'; then
  mflip M1-ignore-header bash -c '"$0" cmp "$1" "$2"; [ $? = 3 ]' "$M" "$fx/synthetic-ref.json" "$fx/synthetic-header.json"
fi
if mutant M2-index-off-by-one 'ret k' 'ret add k 1'; then
  mflip M2-index-off-by-one bash -c '"$0" cmp "$1" "$2" | grep -qxF "DIVERGE p1 index=1 ref=20 cand=21"' "$M" "$fx/synthetic-ref.json" "$fx/synthetic-diverge.json"
fi
if mutant M3-accept-sampled 'die 5 tsampled' 'ret'; then
  canned "$t/m.c" 0 '200 OK' "$(reply 1,2,3,4,5 greedy)"
  canned "$t/m.c" 1 '200 OK' "$(reply 10,20,30 sampled)"
  serve m 2
  G0=$G; G=$M
  m3() { rec m "$t/m.json" > /dev/null 2>&1; [ $? = 5 ]; }
  mflip M3-accept-sampled m3
  G=$G0
  wait
fi
[ "$fail" = 0 ] && echo "golden check: all arms as expected" || echo "golden check: FAILED"
exit "$fail"
