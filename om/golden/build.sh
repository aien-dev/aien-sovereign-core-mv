#!/usr/bin/env bash
# Build om/golden/golden.om (or another source given as $2) with the pinned seeded omc.
# Usage: OMC=/path/to/omega-mv/seed/out/omc bash om/golden/build.sh OUT [SOURCE]
# Exit 0 built, 3 omc missing or not the pinned compiler (NOT_RUN), 1 compile error.
set -u
PIN=62775d2a391cfef7cde64e387655caa9809a267093112910ec76bfa568f3712b # omega-mv bf54b19f seed/out/omc
root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?usage: OMC=... build.sh OUT [SOURCE]}
src=${2:-$root/om/golden/golden.om}
OMC=${OMC:-${OMEGA_OMC:-}}
[ -n "$OMC" ] && [ -x "$OMC" ] || { echo "NOT_RUN: set OMC to omega-mv bf54b19f seed/out/omc" >&2; exit 3; }
got=$(sha256sum "$OMC" | cut -c1-64)
[ "$got" = "$PIN" ] || { echo "NOT_RUN: omc sha256 $got, pinned $PIN" >&2; exit 3; }
{ printf 'om2\n'; cat "$root/om/vendor/json.om" "$root/om/vendor/sha256.om" "$src"; } | "$OMC" > "$out" 2> "$out.err" \
  || { echo "compile $src: $(cat "$out.err")" >&2; rm -f "$out" "$out.err"; exit 1; }
rm -f "$out.err"
chmod 700 "$out"
