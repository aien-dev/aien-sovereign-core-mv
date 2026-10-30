#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Headstart M6 (#538): held A/B samples of provisional preparation, flag off
# vs flag on, over the shapes of crates/aien-runtime/tests/headstart_ab.rs.
# Measurement only: this script never edits the checkout, Cargo.toml, CI or
# the toolchain. It builds the opt-in test target once, then runs ONE sample
# per process, each under `quietlock hold` (owner HS_M6_OWNER, at most
# HS_M6_HOLD_MINUTES), so no other build or test on the host runs during a
# timed sample.
#
# Usage:
#   scripts/bench/headstart-m6-ab.sh SAMPLES OUT_DIR
# Per sample index i (1..SAMPLES) and shape: the arms run in alternating
# order (odd i: off then on, tags A B; even i: on then off), then an A/A pair
# (off twice, tags AA1 AA2) unless HS_M6_AA=0.
#
# Environment (defaults in brackets):
#   HS_M6_BUILD [linked]   linked: the real librx_compose (World commits);
#                          stub: no World commit, every line says STUB and is
#                          an overhead and correctness check, not a result.
#   HS_M6_OMEGA, HS_M6_PHYSICS, HS_M6_AIENOS
#                          checkouts for the linked build, pinned exactly as
#                          scripts/test-linked-compose.sh checks them (omega
#                          HEAD = omega.lock, physics = omega physics.lock,
#                          aienos holds omega aienos.lock).
#   HS_M6_SHAPES [chain parallel failure]
#   HS_M6_THREADS [4]      worker threads = M2 max_concurrency, both arms.
#   HS_M6_AA [1]           run the A/A pair.
#   HS_M6_OWNER [hs-m6], HS_M6_HOLD_MINUTES [5], HS_M6_REASON
#   HS_M6_TARGET_DIR [<repo>/target/hs-m6-<build>]  separate target dir, so
#                          a linked build never replaces the stub in target/.
#   HS_M6_PLATFORM [unlabelled]  e.g. SPARK or CI-RUNNER; copied into the
#                          manifest. Numbers from one platform say nothing
#                          about another.
# Output in OUT_DIR: samples.jsonl (one JSON line per sample, schema
# headstart-m6-ab/v1), logs/, manifest.json, summary.txt (medians), and
# failures.txt when a sample failed its correctness gates (exit 1).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "headstart-m6-ab: $*" >&2; exit 2; }
SAMPLES=${1:?usage: headstart-m6-ab.sh SAMPLES OUT_DIR}
OUT=${2:?usage: headstart-m6-ab.sh SAMPLES OUT_DIR}
case "$SAMPLES" in '' | *[!0-9]*) die "SAMPLES must be a positive integer" ;; esac
[ "$SAMPLES" -ge 1 ] || die "SAMPLES must be at least 1"
BUILD=${HS_M6_BUILD:-linked}
SHAPES=${HS_M6_SHAPES:-chain parallel failure}
THREADS=${HS_M6_THREADS:-4}
AA=${HS_M6_AA:-1}
OWNER=${HS_M6_OWNER:-hs-m6}
MINUTES=${HS_M6_HOLD_MINUTES:-5}
REASON=${HS_M6_REASON:-headstart M6 A/B timing sample (#538)}
PLATFORM=${HS_M6_PLATFORM:-unlabelled}
QL=${HS_M6_QUIETLOCK_BIN:-quietlock}
# Team rule: per-sample holds of at most 5 minutes.
case "$MINUTES" in [1-5]) ;; *) echo "headstart-m6-ab: HS_M6_HOLD_MINUTES must be 1 to 5 (team rule)" >&2; exit 2 ;; esac
export CARGO_TARGET_DIR=${HS_M6_TARGET_DIR:-$ROOT/target/hs-m6-$BUILD}
command -v "$QL" >/dev/null 2>&1 || die "quietlock not found ($QL)"
for s in $SHAPES; do
  case "$s" in chain | parallel | failure) ;; *) die "unknown shape $s" ;; esac
done
mkdir -p "$OUT/logs"
[ ! -e "$OUT/samples.jsonl" ] || die "$OUT/samples.jsonl exists: use a new OUT_DIR"

unset AIEN_CUDAPP_DIR AIEN_OMEGA_DIR AIEN_OMEGA_GPU_LIB AIEN_OMEGA_COMPOSE_LIB AIEN_OMEGA_COMPOSE_SHA AIEN_DEV_FALLBACK
PINS="{}"
case "$BUILD" in
  linked)
    OMW=$(cd "${HS_M6_OMEGA:?HS_M6_OMEGA (omega checkout at omega.lock)}" && pwd)
    PHW=$(cd "${HS_M6_PHYSICS:?HS_M6_PHYSICS (physics checkout)}" && pwd)
    AO=$(cd "${HS_M6_AIENOS:?HS_M6_AIENOS (aienos checkout)}" && pwd)
    pin=$(tr -d '[:space:]' <"$ROOT/omega.lock")
    [ "$(git -C "$OMW" rev-parse HEAD)" = "$pin" ] || die "omega HEAD != omega.lock $pin"
    [ "$(git -C "$PHW" rev-parse HEAD)" = "$(tr -d '[:space:]' <"$OMW/physics.lock")" ] || die "physics HEAD != omega physics.lock"
    [ "$(git -C "$AO" rev-parse HEAD)" = "$(tr -d '[:space:]' <"$OMW/aienos.lock")" ] || die "aienos HEAD != omega aienos.lock"
    unset AIEN_FORCE_CPU_STUB
    export AIEN_OMEGA_COMPOSE_DIR="$OMW" AIEN_PHYSICS_DIR="$PHW" AIEN_AIENOS_LOCK_REPO="$AO"
    PINS="{\"omega\":\"$pin\",\"physics\":\"$(git -C "$PHW" rev-parse HEAD)\",\"aienos\":\"$(git -C "$AO" rev-parse HEAD)\"}"
    ;;
  stub)
    unset AIEN_OMEGA_COMPOSE_DIR AIEN_PHYSICS_DIR AIEN_AIENOS_LOCK_REPO
    export AIEN_FORCE_CPU_STUB=1
    ;;
  *) die "HS_M6_BUILD must be linked or stub" ;;
esac

# Wait while another session holds the quiet flag, then run under
# `quietlock run`; exit 75 (taken meanwhile) waits and tries again.
quiet_run() {
  local rc
  while :; do
    until "$QL" check >/dev/null 2>&1; do sleep 20; done
    set +e; "$QL" run -- "$@"; rc=$?; set -e
    [ "$rc" -eq 75 ] || return "$rc"
  done
}

echo "headstart-m6-ab: building the test target ($BUILD, CARGO_TARGET_DIR=$CARGO_TARGET_DIR)"
cd "$ROOT"
quiet_run cargo test -p aien-runtime --features provisional-prep --test headstart_ab \
  --no-run --message-format=json -j 8 >"$OUT/logs/build.json" 2>"$OUT/logs/build.err" \
  || { tail -20 "$OUT/logs/build.err" >&2; die "build failed"; }
BIN=$(grep -o '"executable":"[^"]*headstart_ab[^"]*"' "$OUT/logs/build.json" | tail -1 | sed 's/^"executable":"//; s/"$//')
[ -x "$BIN" ] || die "test executable not found in the build output"
if [ "$BUILD" = linked ] && grep -q "building the stub" "$OUT/logs/build.err" "$OUT/logs/build.json"; then
  die "a stub was built: the composition library is not linked"
fi

cat >"$OUT/manifest.json" <<EOF
{"schema":"headstart-m6-ab-manifest/v1","issue":"aien-spark/aien-sovereign-core#538",
 "git_head":"$(git rev-parse HEAD)","git_dirty":$( [ -z "$(git status --porcelain --untracked-files=no)" ] && echo false || echo true ),
 "rustc":"$(rustc -V)","build":"$BUILD","pins":$PINS,"platform":"$PLATFORM",
 "host":"$(hostname)","arch":"$(uname -m)","kernel":"$(uname -r)","nproc":$(nproc),
 "samples":$SAMPLES,"shapes":"$SHAPES","threads":$THREADS,"aa":$AA,
 "quiet_hold":{"owner":"$OWNER","max_minutes":$MINUTES},
 "binary":"$BIN","started":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

FAILED=0
WANT=$( [ "$BUILD" = linked ] && echo LINKED || echo STUB )
# One held sample: env for the test, one process, appended to samples.jsonl.
held_sample() {
  local i=$1 shape=$2 arm=$3 tag=$4 log rc before after line
  log="$OUT/logs/s$(printf %03d "$i")-$shape-$tag-$arm.log"
  before=$(wc -l 2>/dev/null <"$OUT/samples.jsonl" || echo 0)
  while :; do
    until "$QL" check >/dev/null 2>&1; do sleep 20; done
    set +e
    HS_M6_ARM=$arm HS_M6_SHAPE=$shape HS_M6_SAMPLE=$i HS_M6_TAG=$tag HS_M6_THREADS=$THREADS \
      HS_M6_OUT="$OUT/samples.jsonl" \
      "$QL" hold --owner "$OWNER" --minutes "$MINUTES" --reason "$REASON" -- \
      "$BIN" --ignored --exact headstart_ab_sample --nocapture --test-threads=1 >"$log" 2>&1
    rc=$?
    set -e
    # 75 counts as a refusal only with quietlock's own line (as lanes.sh checks).
    [ "$rc" -eq 75 ] && tail -n 1 "$log" | grep -q "^quietlock: QUIETLOCK_REFUSED " || break
    echo "headstart-m6-ab: quiet flag taken by someone else, waiting" >&2
  done
  # Exactly one new line, and it must be this sample (index, shape, arm,
  # tag) from the expected build: a linked run must never contain a STUB
  # line (a cached stub build would otherwise pass unnoticed).
  if [ "$rc" -eq 0 ]; then
    after=$(wc -l 2>/dev/null <"$OUT/samples.jsonl" || echo 0)
    line=$(tail -n 1 "$OUT/samples.jsonl" 2>/dev/null || true)
    if [ "$after" -ne $((before + 1)) ]; then
      rc=3
    else
      for kv in "\"build\":\"$WANT\"" "\"sample\":$i" "\"shape\":\"$shape\"" "\"arm\":\"$arm\"" "\"tag\":\"$tag\""; do
        printf "%s\n" "$line" | grep -qE "$kv([,}])" || rc=3
      done
    fi
  fi
  if [ "$rc" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "sample $i $shape $arm $tag: exit $rc (see $log)" >>"$OUT/failures.txt"
  fi
  echo "headstart-m6-ab: sample $i $shape $tag $arm exit $rc"
}

for i in $(seq 1 "$SAMPLES"); do
  for shape in $SHAPES; do
    if [ $((i % 2)) -eq 1 ]; then
      held_sample "$i" "$shape" off A; held_sample "$i" "$shape" on B
    else
      held_sample "$i" "$shape" on B; held_sample "$i" "$shape" off A
    fi
    if [ "$AA" = 1 ]; then
      held_sample "$i" "$shape" off AA1; held_sample "$i" "$shape" off AA2
    fi
  done
done

# Medians of wall_ms per shape and tag (observed values only).
field() { grep -o "\"$1\":[^,}]*" | head -1 | sed "s/^\"$1\"://; s/\"//g"; }
{
  echo "build=$BUILD platform=$PLATFORM samples=$SAMPLES threads=$THREADS (wall_ms medians; tag A=off B=on AA1/AA2=off)"
  for shape in $SHAPES; do
    for tag in A B AA1 AA2; do
      vals=$(grep "\"shape\":\"$shape\"" "$OUT/samples.jsonl" 2>/dev/null | grep "\"tag\":\"$tag\"" | while read -r l; do printf '%s\n' "$l" | field wall_ms; done | sort -g || true)
      n=$(printf '%s\n' "$vals" | grep -c . || true)
      [ "$n" -gt 0 ] || continue
      med=$(printf '%s\n' "$vals" | awk -v n="$n" 'NR == int((n + 1) / 2) { a = $1 } NR == int(n / 2) + 1 { b = $1 } END { printf "%.3f", (a + b) / 2 }')
      echo "$shape $tag n=$n median_wall_ms=$med$( [ "$shape" = failure ] && echo " (not a timing comparison: the on arm waits on purpose)")"
    done
  done
} >"$OUT/summary.txt"
cat "$OUT/summary.txt"
sed -i "s/}\$/,\"finished\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"failed_samples\":$FAILED}/" "$OUT/manifest.json"
if [ "$FAILED" -gt 0 ]; then
  echo "headstart-m6-ab: $FAILED sample(s) failed (failures.txt)" >&2
  exit 1
fi
