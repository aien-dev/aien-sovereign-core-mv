#!/usr/bin/env bash
# om/timeline/gate.sh: SC-STAMPS-TIMELINE (DRA-75). Builds om/timeline/timeline.om with the pinned
# seeded omc and checks it on the historical AIEN_STROKE_STAMPS=1 logs of the stroke gate:
#   1. five stamps logs (l2, l1, v2, b of evidence/motor-chip-20261021, l7 of
#      evidence/dra75-ladder-20261031/historical) must exit 0 with TSV byte-identical to
#      om/timeline/fixtures/*.tsv; the fixtures come from each run's own json
#      (timeline_last_stroke.per_kind, ns rounded to the microsecond), not from this parser;
#   2. the script log of the l2 ladder (heads 1, 2, 4 and 8 in one file) must refuse without N
#      (exit 7) and give the l2 fixture with N 8;
#   3. negative controls, each must refuse with its own code and print nothing on stdout:
#      a misspelled row (4), a kind out of its 8-wide column (4), a row with one launch more (5),
#      a row 1.000 ms longer (5), the total line removed (6), two blocks (6), a duplicate row (8),
#      two rows swapped (8), N 4 against a heads-8 log (7), a log without a timeline (6).
# The shell only sequences and compares bytes; every judgment about a log is made by the .om program.
# Compiler: AIEN_OMC (OMC and OMEGA_OMC are accepted too) must hash to omega.pin, else NOT_RUN.
# Exit 0 PASS, 1 FAIL, 77 NOT_RUN (with an AIEN-OBS not_run_reason line), as om/expert-bytes/gate.sh.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
not_run() { echo "timeline gate: NOT_RUN $1: $2" >&2; obs not_run_reason "\"$1\""; exit 77; }

omc=${AIEN_OMC:-${OMC:-${OMEGA_OMC:-}}}
[ -n "$omc" ] || not_run omc_missing "set AIEN_OMC to omega-mv seed/out/omc at the commit in om/timeline/omega.pin"
[ -x "$omc" ] || not_run omc_missing "$omc is not executable"
pin=$(sed -n 's/^omc_sha256: //p' "$here/omega.pin")
got=$(sha256sum < "$omc" | cut -d' ' -f1)
[ "$got" = "$pin" ] || not_run omc_unpinned "omc sha256 $got, pinned $pin"

b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
{ printf 'om2\n'; cat "$here/timeline.om"; } | "$omc" > "$b/tl" 2> "$b/tl.err" \
  || { echo "FAIL compile: $(cat "$b/tl.err")"; exit 1; }
chmod 700 "$b/tl"
obs omc_sha256 "\"$got\""
obs timeline_sha256 "\"$(sha256sum < "$b/tl" | cut -d' ' -f1)\""

E=$root/evidence/motor-chip-20261021
H=$root/evidence/dra75-ladder-20261031/historical
F=$here/fixtures
fail=0
ok() { # name input want_tsv [N]
  local rc
  timeout 30 "$b/tl" ${4:+"$4"} < "$2" > "$b/out" 2> "$b/err"; rc=$?
  if [ "$rc" = 0 ] && cmp -s "$b/out" "$3"; then echo "ok   $1: exit 0, TSV equal to ${3##*/}"; return; fi
  echo "FAIL $1: exit $rc, $(cat "$b/err")"; diff "$3" "$b/out" | head -5; fail=1
}
refuse() { # name input want_rc [N]
  local rc
  timeout 30 "$b/tl" ${4:+"$4"} < "$2" > "$b/out" 2> "$b/err"; rc=$?
  if [ "$rc" = "$3" ] && [ ! -s "$b/out" ]; then echo "ok   $1: exit $rc, $(cat "$b/err")"; return; fi
  echo "FAIL $1: exit $rc (want $3), stdout $(wc -c < "$b/out") bytes, $(cat "$b/err")"; fail=1
}

for r in l2 l1 v2 b; do ok "stroke8$r-stamps" "$E/stroke8$r-stamps.log" "$F/stroke8$r-stamps.tsv"; done
ok stroke8l7-stamps "$H/stroke8l7-stamps.log" "$F/stroke8l7-stamps.tsv"
refuse ladder-log-no-N "$H/motor-stroke8l2-gate.log" 7
ok ladder-log-N8 "$H/motor-stroke8l2-gate.log" "$F/stroke8l2-stamps.tsv" 8

src=$E/stroke8l2-stamps.log
m=$b/m.log
sed 's/^\(timeline:      gdn   240\) launches/\1 launchs/' "$src" > "$m"; refuse control-misspelled "$m" 4
sed 's/^timeline:      gdn /timeline:     gdn  /' "$src" > "$m"; refuse control-column "$m" 4
sed 's/^\(timeline:  gate_up \)   40 launches/\1   41 launches/' "$src" > "$m"; refuse control-launches "$m" 5
sed 's/^\(timeline:     down    40 launches \)   15\.858/\1   16.858/' "$src" > "$m"; refuse control-ms "$m" 5
grep -v '^timeline: total ' "$src" > "$m"; refuse control-no-total "$m" 6
cat "$src" "$src" > "$m"; refuse control-two-blocks "$m" 6
sed '/^timeline:  gate_up /p' "$src" > "$m"; refuse control-duplicate "$m" 8
sed '/^timeline:  gate_up /{h;d};/^timeline:     down /G' "$src" > "$m"; refuse control-order "$m" 8
refuse control-wrong-N "$src" 7 4
refuse control-no-timeline "$H/stroke8l6-stamps.log" 6
if [ "$fail" = 0 ]; then obs verdict '"PASS"'; echo "SC-STAMPS-TIMELINE PASS"; exit 0; fi
obs verdict '"FAIL"'
echo "SC-STAMPS-TIMELINE FAIL"
exit 1
