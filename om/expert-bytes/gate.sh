#!/usr/bin/env bash
# om/expert-bytes/gate.sh: SC-EXPERT-BYTES (DRA-63, aien-dev/aien-sovereign-core-mv#855).
# Builds om/expert-bytes/expert-bytes.om with the pinned seeded omc and runs it on the
# Qwen3.6-35B-A3B-FP8 checkpoint headers (no tensor data is read):
#   1. width 512, BF16 scales: must exit 0 with the contract figures 3146112 B per expert
#      (3145728 B FP8 weight + 384 B BF16 scales);
#   2. negative control, width 768: must exit 1 with verdict FAIL;
#   3. negative control, FP32 scales (the old 768 B scale figure): must exit 1 with verdict FAIL.
# The shell only sequences; every judgment about the header is made by the .om program.
# Compiler: AIEN_OMC (OMC and OMEGA_OMC are accepted too) must hash to omega.pin, else NOT_RUN.
# Model: MODEL_DIR, else $AIEN_MODELS_DIR/qwen3.6-35b-a3b-fp8-95a723d0, else
# $HOME/models/qwen3.6-35b-a3b-fp8-95a723d0; missing is NOT_RUN.
# Exit 0 PASS, 1 FAIL, 77 NOT_RUN (with an AIEN-OBS not_run_reason line).
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
MODEL=qwen3.6-35b-a3b-fp8-95a723d0
CONTRACT='expect_expert 3146112 expect_weight 3145728 expect_scale 384 '
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
not_run() { echo "expert-bytes gate: NOT_RUN $1: $2" >&2; obs not_run_reason "\"$1\""; exit 77; }

omc=${AIEN_OMC:-${OMC:-${OMEGA_OMC:-}}}
[ -n "$omc" ] || not_run omc_missing "set AIEN_OMC to omega-mv seed/out/omc at the commit in om/expert-bytes/omega.pin"
[ -x "$omc" ] || not_run omc_missing "$omc is not executable"
pin=$(sed -n 's/^omc_sha256: //p' "$here/omega.pin")
got=$(sha256sum < "$omc" | cut -d' ' -f1)
[ "$got" = "$pin" ] || not_run omc_unpinned "omc sha256 $got, pinned $pin"

if [ -n "${MODEL_DIR:-}" ]; then dir=$MODEL_DIR
elif [ -n "${AIEN_MODELS_DIR:-}" ]; then dir=$AIEN_MODELS_DIR/$MODEL
else dir=$HOME/models/$MODEL
fi
[ -f "$dir/layers-0.safetensors" ] || not_run model_missing "$dir/layers-0.safetensors"

b=$(mktemp -d)
trap 'rm -rf "$b"' EXIT
{ printf 'om2\n'; cat "$root/om/vendor/json.om" "$here/expert-bytes.om"; } | "$omc" > "$b/eb" 2> "$b/eb.err" \
  || { echo "FAIL compile: $(cat "$b/eb.err")"; exit 1; }
chmod 700 "$b/eb"
obs omc_sha256 "\"$got\""
obs check_sha256 "\"$(sha256sum < "$b/eb" | cut -d' ' -f1)\""
obs model_dir "\"$dir\""

fail=0
arm() { # name width scale_bytes want_rc want_verdict
  line=$(timeout 120 "$b/eb" "$dir" "$2" "$3" 2>&1 < /dev/null)
  rc=$?
  echo "$line"
  if [ "$rc" = "$4" ] && [[ $line == "expert-bytes verdict $5 "* ]]; then
    echo "ok   $1: exit $rc, verdict $5"
    return 0
  fi
  echo "FAIL $1: exit $rc, want $4 and verdict $5"
  fail=1
  return 1
}
if arm real 512 2 0 PASS; then
  if [[ $line != *" $CONTRACT"* ]]; then echo "FAIL real: contract figures ($CONTRACT) not printed"; fail=1; fi
fi
arm control-width-768 768 2 1 FAIL
arm control-fp32-scales 512 4 1 FAIL
if [ "$fail" = 0 ]; then obs verdict '"PASS"'; echo "SC-EXPERT-BYTES PASS"; exit 0; fi
obs verdict '"FAIL"'
echo "SC-EXPERT-BYTES FAIL"
exit 1
