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

# SMOLLM2-Q qualification (copied from cand4-qualification): the `cpu-fault` test binary for Q2 (ACCEPTANCE-SMOLLM2-Q.md section 5).
# Same recipe as NEXT-PHASE-2 (ACCEPTANCE-v2 section 4, receipt e401fffe build_env):
# aien-cli release build with the composition archive linked, NO GPU archive, cargo
# feature `fault-hold`; built with scripts/repro-build.sh (the release flags) from a
# clean checkout of the frozen sovereign-core commit. Shell only.
# Usage:
#   build-cpu-fault.sh SC_CHECKOUT OMEGA_DIR PHYSICS_DIR AIENOS_REPO OUT_DIR
# SC_CHECKOUT must be clean; OMEGA_DIR HEAD must equal SC_CHECKOUT/omega.lock.
# Writes OUT_DIR/aien-cli-cpu-fault, OUT_DIR/build-cpu-fault.json, OUT_DIR/build.log.
set -u
SCW=${1:?SC_CHECKOUT} OMW=${2:?OMEGA_DIR} PHW=${3:?PHYSICS_DIR} AO=${4:?AIENOS_REPO} OUT=${5:?OUT_DIR}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
SCW=$(cd "$SCW" && pwd); OMW=$(cd "$OMW" && pwd); PHW=$(cd "$PHW" && pwd); AO=$(cd "$AO" && pwd)
die() { echo "build-cpu-fault: $*" >&2; exit 3; }
[ -z "$(git -C "$SCW" status --porcelain --untracked-files=no)" ] || die "sovereign-core checkout not clean"
pin=$(tr -d '[:space:]' <"$SCW/omega.lock")
[ "$(git -C "$OMW" rev-parse HEAD)" = "$pin" ] || die "omega HEAD $(git -C "$OMW" rev-parse HEAD) != omega.lock $pin"
[ -z "$(git -C "$OMW" status --porcelain --untracked-files=no)" ] || die "omega checkout not clean"
T=$OUT/target
export CARGO_HOME=${CARGO_HOME:-$OUT/cargo-home}
(cd "$SCW" && cargo fetch --locked) >"$OUT/fetch.log" 2>&1 || die "cargo fetch failed (see $OUT/fetch.log)"
(cd "$SCW" && env -u AIEN_DEV_FALLBACK -u AIEN_FORCE_CPU_STUB -u AIEN_OMEGA_DIR -u AIEN_OMEGA_GPU_LIB \
   -u AIEN_OMEGA_COMPOSE_LIB -u AIEN_OMEGA_COMPOSE_SHA \
   AIEN_OMEGA_COMPOSE_DIR="$OMW" AIEN_PHYSICS_DIR="$PHW" AIEN_AIENOS_LOCK_REPO="$AO" CARGO_TARGET_DIR="$T" \
   bash scripts/repro-build.sh --offline -vv -p aien-cli --features aien-cli/fault-hold) >"$OUT/build.log" 2>&1
rc=$?
[ $rc = 0 ] || die "cargo build failed rc=$rc (see $OUT/build.log)"
cp "$T/release/aien-cli" "$OUT/aien-cli-cpu-fault"
B=$OUT/aien-cli-cpu-fault
cnt() { strings -a "$B" | grep -c -- "$1"; }
lib=$(find "$T/release/build" -path '*omega-compose-build/librx_compose.a' | head -1)
jq -n --arg sc "$(git -C "$SCW" rev-parse HEAD)" --arg om "$pin" --arg ph "$(git -C "$PHW" rev-parse HEAD)" \
  --arg sha "$(sha256sum "$B" | cut -d' ' -f1)" --arg lib "${lib:+$(sha256sum "$lib" | cut -d' ' -f1)}" \
  --argjson c1 "$(cnt 'compose.candidate.0')" --argjson c2 "$(cnt 'compose.commit')" \
  --argjson g "$(cnt 'NVIDIA_DGX_SPARK_GB10_SM121')" --argjson fh "$(cnt 'AIEN_FAULT_HOLD')" \
  --argjson rp "$(cnt 'reconcile_panic')" --argjson re "$(cnt 'reconcile_error')" \
  --argjson sw "$(grep -c 'building the stub' "$OUT/build.log")" \
  --argjson hc "$(grep -c 'cargo:rustc-cfg=has_omega_compose' "$OUT/build.log")" \
  --argjson hg "$(grep -c 'cargo:rustc-cfg=has_omega_gpu' "$OUT/build.log")" \
  '{name:"cpu-fault", sovereign_core:$sc, omega:$om, physics:$ph, sha256:$sha, librx_compose_sha256:$lib,
    command:"scripts/repro-build.sh --offline -vv -p aien-cli --features aien-cli/fault-hold (AIEN_OMEGA_COMPOSE_DIR set; AIEN_OMEGA_DIR, AIEN_OMEGA_GPU_LIB, AIEN_OMEGA_COMPOSE_LIB, AIEN_OMEGA_COMPOSE_SHA, AIEN_FORCE_CPU_STUB, AIEN_DEV_FALLBACK unset)",
    link_proof:{compose_markers:($c1 + $c2), gpu_markers:$g, fault_hold_strings:$fh, reconcile_panic_strings:$rp,
                reconcile_error_strings:$re, stub_warning_lines:$sw, has_omega_compose_lines:$hc, has_omega_gpu_lines:$hg},
    ok:(($c1 > 0) and ($c2 > 0) and $g == 0 and $fh > 0 and $rp > 0 and $re > 0 and $hc > 0 and $hg == 0)}' \
  >"$OUT/build-cpu-fault.json"
cat "$OUT/build-cpu-fault.json"
jq -e .ok "$OUT/build-cpu-fault.json" >/dev/null
