#!/usr/bin/env bash
# om/prefill-logits/gate.sh: SC-PREFILL-LOGITS, the G1 logits judge of P-DRA-4 Lane P
# (aien-dev/aien-sovereign-core-mv#877, Linear P-DRA-4 Lane P G1; DRA-82/DRA-83). Builds om/prefill-logits/prefill-logits.om
# with the pinned seeded omc and runs it on synthetic fixtures (om/prefill-logits/mkfix.py, test equipment):
#   1. same       identical pair                          must exit 0, verdict PASS, max_abs_d_u 0;
#   2. noise      ref + N(0, 0.05)                        must exit 0, verdict PASS (bound not over-strict);
#   3. edge       one logit |d| = 1.0 exactly = TAU       must exit 0, verdict PASS (TAU inclusive);
#   4. over       one logit |d| = 1.0 + 2^-23             must exit 1, verdict FAIL (red mutant, top-1 kept);
#   5. flip       top-1 flips, A gap 0.2 > BAND 0.1       must exit 1, verdict FAIL;
#   5b. tie       top-1 flips, A gap 0.05 < BAND 0.1      must exit 0, verdict PASS (written near-tie band);
#   6. nan        one NaN in B                            must exit 1, verdict FAIL, nonfinite_b 1;
#   7. other      a different prompt's logits             must exit 1, verdict FAIL (negative control);
#   8. short      993276-byte file                        must exit 3 (io);
#   9. tau0       judge rebuilt with TAU 0.0 (source mutant): noise must turn FAIL, same must stay PASS;
#  10. band0      judge rebuilt with BAND 0.0 (source mutant): tie must turn FAIL.
# With two arguments (gate.sh A.f32 B.f32) the self-test runs first and must pass; then the real pair is
# judged and its verdict is the gate's verdict.
# The shell only sequences; every judgment about the logits is made by the .om program.
# Compiler: AIEN_OMC (OMC and OMEGA_OMC are accepted too) must hash to omega.pin, else NOT_RUN;
# python3 (fixtures only) missing is NOT_RUN.
# Exit 0 PASS, 1 FAIL, 77 NOT_RUN (with an AIEN-OBS not_run_reason line).
set -u
here=$(cd "$(dirname "$0")" && pwd)
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
not_run() { echo "prefill-logits gate: NOT_RUN $1: $2" >&2; obs not_run_reason "\"$1\""; exit 77; }

[ $# = 0 ] || [ $# = 2 ] || { echo "usage: gate.sh [A.f32 B.f32]" >&2; exit 2; }
omc=${AIEN_OMC:-${OMC:-${OMEGA_OMC:-}}}
[ -n "$omc" ] || not_run omc_missing "set AIEN_OMC to omega-mv seed/out/omc at the commit in om/prefill-logits/omega.pin"
[ -x "$omc" ] || not_run omc_missing "$omc is not executable"
pin=$(sed -n 's/^omc_sha256: //p' "$here/omega.pin")
got=$(sha256sum < "$omc" | cut -d' ' -f1)
[ "$got" = "$pin" ] || not_run omc_unpinned "omc sha256 $got, pinned $pin"
command -v python3 > /dev/null || not_run python3_missing "fixtures need python3"

b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
build() { # source out
  { printf 'om2\n'; cat "$1"; } | "$omc" > "$2" 2> "$2.err" || { echo "FAIL compile $1: $(cat "$2.err")"; exit 1; }
  chmod 700 "$2"
}
build "$here/prefill-logits.om" "$b/pl"
sed 's/^ret 1\.0f64$/ret 0.0f64/' "$here/prefill-logits.om" > "$b/tau0.om"
cmp -s "$here/prefill-logits.om" "$b/tau0.om" && { echo "FAIL tau0: source mutant did not apply"; exit 1; }
build "$b/tau0.om" "$b/pl0"
sed 's/^ret 0\.1f64$/ret 0.0f64/' "$here/prefill-logits.om" > "$b/band0.om"
cmp -s "$here/prefill-logits.om" "$b/band0.om" && { echo "FAIL band0: source mutant did not apply"; exit 1; }
build "$b/band0.om" "$b/plb0"
obs omc_sha256 "\"$got\""
obs check_sha256 "\"$(sha256sum < "$b/pl" | cut -d' ' -f1)\""
python3 "$here/mkfix.py" "$b" || { echo "FAIL fixtures"; exit 1; }

fail=0
line=
arm() { # name judge A B want_rc want_verdict [required substring]
  line=$(timeout 120 "$2" "$3" "$4" 2>&1 < /dev/null)
  rc=$?
  echo "$line"
  if [ "$rc" = "$5" ] && { [ "$6" = - ] || [[ $line == "prefill-logits verdict $6 "* ]]; } \
     && { [ -z "${7:-}" ] || [[ "$line " == *" $7 "* ]]; }; then
    echo "ok   $1: exit $rc, verdict $6${7:+, $7}"
    return 0
  fi
  echo "FAIL $1: exit $rc, want $5 and verdict $6${7:+ and $7}"
  fail=1
  return 1
}
f=$b
arm same     "$b/pl"  "$f/ref.f32"  "$f/same.f32"  0 PASS "max_abs_d_u 0"
arm noise    "$b/pl"  "$f/ref.f32"  "$f/noise.f32" 0 PASS "top1_b 1234"
arm edge     "$b/pl"  "$f/ref.f32"  "$f/edge.f32"  0 PASS "max_abs_d_u 1000000"
arm over     "$b/pl"  "$f/ref.f32"  "$f/over.f32"  1 FAIL "max_at 5000"
arm flip     "$b/pl"  "$f/refc.f32" "$f/flipc.f32" 1 FAIL "top1_b 77"
arm tie      "$b/pl"  "$f/refn.f32" "$f/flipn.f32" 0 PASS "flip_gap_a_u 62500"
arm nan      "$b/pl"  "$f/ref.f32"  "$f/nan.f32"   1 FAIL "nonfinite_b 1"
arm other    "$b/pl"  "$f/ref.f32"  "$f/other.f32" 1 FAIL "top1_b 42"
arm short    "$b/pl"  "$f/ref.f32"  "$f/short.f32" 3 -
arm tau0-noise "$b/pl0" "$f/ref.f32" "$f/noise.f32" 1 FAIL "tau_u 0"
arm tau0-same  "$b/pl0" "$f/ref.f32" "$f/same.f32"  0 PASS "tau_u 0"
arm band0-tie  "$b/plb0" "$f/refn.f32" "$f/flipn.f32" 1 FAIL "top1_b 77"
if [ "$fail" != 0 ]; then obs verdict '"FAIL"'; echo "SC-PREFILL-LOGITS FAIL (self-test)"; exit 1; fi
echo "SC-PREFILL-LOGITS self-test PASS (12 arms)"

if [ $# = 2 ]; then
  obs a_sha256 "\"$(sha256sum < "$1" | cut -d' ' -f1)\""
  obs b_sha256 "\"$(sha256sum < "$2" | cut -d' ' -f1)\""
  line=$(timeout 120 "$b/pl" "$1" "$2" 2>&1 < /dev/null)
  rc=$?
  echo "$line"
  obs judge_rc "$rc"
  if [ "$rc" = 0 ]; then obs verdict '"PASS"'; echo "SC-PREFILL-LOGITS PASS"; exit 0; fi
  obs verdict '"FAIL"'
  echo "SC-PREFILL-LOGITS FAIL (judge exit $rc)"
  exit 1
fi
obs verdict '"PASS"'
echo "SC-PREFILL-LOGITS PASS"
exit 0
