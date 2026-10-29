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

# abi-qualify.sh: qualify the omega and AIEN native boundary on this machine, in one command.
#
# Builds sovereign-core against the pinned omega, physics and aienos checkouts with
# AIEN_REQUIRE_NATIVE_LIBS=1 (a stub is refused, never a quiet pass), then runs:
#   conformance   aien-abi-conformance unit and C-side tests (synthetic drift cases)
#   identity      scripts/test-native-identity.sh (stub, prebuilt and wrong-commit refusals)
#   compose       aien-omega-compose linked: ABI manifest, native identity, callback safety,
#                 the compose suite (ignored tests included)
#   gpu-host      aien-omega-gpu linked host tests: ABI manifest, native identity, argument
#                 refusals (no chip work)
#   chip          only with --chip: GB10 parity (aien-omega-gpu parity, aien-inference-abi
#                 omega_backend_parity and the crate's linked tests), under the GPU lease
# The build scripts check the C headers, a compiled layout probe and the archive symbols against
# the checked-in ABI manifests and the archive digests against native-artifacts.lock, and the
# identity tests run with AIEN_ABI_QUALIFY=1: canonical CUDApp and compose must report
# pinned-source + abi pass + digest locked, companion libomega_gpu.a reports pinned-source +
# abi none + digest locked, and native_pair_qualifies must be true.
#
# Usage: scripts/abi-qualify.sh [--chip] [--pins DIR] [--out DIR]
#   --pins DIR  pinned checkouts (default target/pins), materialized by scripts/pins.sh --gpu
#   --out DIR   logs and evidence.json (default target/abi-qualification)
#   --chip      also run the GB10 tests. Needs `quietlock check` to return 0 and
#               /tmp/aien-gb10.lock to be free (taken with flock -n, as omega tools/chip_run.sh
#               does); otherwise the chip part is INCOMPLETE. Never killed, never timed out.
# Exit: 0 PASS, 1 FAIL (a check failed), 2 INCOMPLETE (something required could not run).
# Without --chip a PASS covers the host scope only; evidence.json says "scope":"host".
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS="$ROOT/target/pins"
OUT="$ROOT/target/abi-qualification"
CHIP=0
while [ $# -gt 0 ]; do
    case "$1" in
    --chip) CHIP=1 ;;
    --pins) PINS="${2:?--pins DIR}"; shift ;;
    --out) OUT="${2:?--out DIR}"; shift ;;
    *) echo "usage: scripts/abi-qualify.sh [--chip] [--pins DIR] [--out DIR]" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"

STEPS=()
fails=0
incomplete=0
CHIP_STATE="not-requested"
CHIP_REASON=""
note() { echo "abi-qualify: $*"; }
record() { # record <name> <pass|fail|incomplete> <detail>
    STEPS+=("{\"name\":\"$1\",\"status\":\"$2\",\"log\":\"$OUT/$1.log\",\"detail\":\"$3\"}")
    note "$1: $2${3:+ ($3)}"
    case "$2" in fail) fails=$((fails + 1)) ;; incomplete) incomplete=$((incomplete + 1)) ;; esac
}
step() { # step <name> <command...>: pass when the command exits 0
    local name=$1
    shift
    if "$@" >"$OUT/$name.log" 2>&1; then record "$name" pass ""; else record "$name" fail "exit $?"; fi
}
# A cargo test step must also have run at least one test and left nothing ignored.
tests_ran() {
    local log="$OUT/$1.log"
    grep -qE '^test result: ok\. [1-9][0-9]* passed' "$log" &&
        ! grep -qE '^test result:.* [1-9][0-9]* ignored' "$log"
}

# 1. Pins (omega at omega.lock, physics and aienos at omega's locks).
if bash scripts/pins.sh --gpu "$PINS" >"$OUT/pins.log" 2>&1 && [ -f "$PINS/env.sh" ]; then
    record pins pass ""
    # shellcheck disable=SC1091
    . "$PINS/env.sh"
else
    record pins incomplete "scripts/pins.sh --gpu failed: no pinned omega, nothing native can be qualified"
fi
export AIEN_ABI_QUALIFY=1

if [ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = "1" ]; then
    step conformance cargo test -p aien-abi-conformance
    step identity bash scripts/test-native-identity.sh
    step compose cargo test -p aien-omega-compose --test abi_manifest --test callbacks --test compose \
        -- --include-ignored --test-threads=1 --nocapture
    tests_ran compose || record compose-ran fail "no linked compose test ran, or some were ignored"
    step gpu-host cargo test -p aien-omega-gpu -- --test-threads=1 --nocapture
fi

# 2. What was linked: canonical CUDApp qualifies, companion is admitted (abi none, digest locked), pair qualifies.
IDS=$(cat "$OUT"/compose.log "$OUT"/gpu-host.log 2>/dev/null | sed -n 's/.*AIEN_NATIVE_IDENTITY \({.*}\).*/\1/p' | sort -u)
PAIR_OK=$(cat "$OUT"/gpu-host.log 2>/dev/null | sed -n 's/.*AIEN_NATIVE_PAIR_QUALIFIES \(true\|false\).*/\1/p' | tail -1)
if printf '%s\n' "$IDS" | grep -F '"component":"librx_compose.a"' | grep -qF '"qualifies":true'; then
    record "identity-librx_compose.a" pass ""
elif [ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = "1" ]; then
    record "identity-librx_compose.a" fail "no qualifying compose identity reported"
else
    record "identity-librx_compose.a" incomplete "not built"
fi
if printf '%s\n' "$IDS" | grep -F '"component":"libcudapp.a"' | grep -qF '"qualifies":true'; then
    record "identity-libcudapp.a" pass ""
elif [ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = "1" ]; then
    record "identity-libcudapp.a" fail "no qualifying canonical CUDApp identity reported"
else
    record "identity-libcudapp.a" incomplete "not built"
fi
if printf '%s\n' "$IDS" | grep -F '"component":"libomega_gpu.a"' | grep -F '"link":"pinned-source"' | grep -F '"abi":"none"' | grep -qF '"digest":"locked"'; then
    record "identity-libomega_gpu.a" pass ""
elif [ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = "1" ]; then
    record "identity-libomega_gpu.a" fail "companion not admitted: requires pinned-source, abi none, and locked digest"
else
    record "identity-libomega_gpu.a" incomplete "not built"
fi
if [ "$PAIR_OK" = "true" ]; then
    record "identity-pair" pass ""
elif [ "${AIEN_REQUIRE_NATIVE_LIBS:-}" = "1" ]; then
    record "identity-pair" fail "native_pair_qualifies returned false or missing"
else
    record "identity-pair" incomplete "not built"
fi

# 3. GB10 chip tests, only on request, only when the machine is free.
if [ "$CHIP" = 1 ] && [ "$fails" -eq 0 ] && [ "$incomplete" -eq 0 ]; then
    if ! command -v quietlock >/dev/null; then
        CHIP_STATE=incomplete CHIP_REASON="quietlock not installed"
    elif ! quietlock check >"$OUT/quietlock.log" 2>&1; then
        CHIP_STATE=incomplete CHIP_REASON="quietlock check is nonzero: a machine hold is up"
    else
        exec 9>/tmp/aien-gb10.lock
        if ! flock -n 9; then
            CHIP_STATE=incomplete CHIP_REASON="/tmp/aien-gb10.lock is held by another run"
        else
            CHIP_STATE=ran
            step chip-gpu cargo test -p aien-omega-gpu --test parity -- --include-ignored --test-threads=1 --nocapture
            tests_ran chip-gpu || record chip-gpu-ran fail "chip parity did not run"
            step chip-inference cargo test -p aien-inference-abi --test omega_backend_parity -- \
                --include-ignored --skip chip_mixed_matmul_then_elementwise --test-threads=1 --nocapture
            tests_ran chip-inference || record chip-inference-ran fail "chip backend parity did not run"
            step chip-inference-linked cargo test -p aien-inference-abi -- --test-threads=1
            flock -u 9
        fi
        exec 9>&-
    fi
    [ "$CHIP_STATE" = incomplete ] && record chip incomplete "$CHIP_REASON"
elif [ "$CHIP" = 1 ]; then
    CHIP_STATE=incomplete CHIP_REASON="host checks did not pass"
    record chip incomplete "$CHIP_REASON"
fi

# 4. Evidence.
if [ "$fails" -gt 0 ]; then VERDICT=FAIL CODE=1
elif [ "$incomplete" -gt 0 ]; then VERDICT=INCOMPLETE CODE=2
else VERDICT=PASS CODE=0; fi
SCOPE=host
[ "$CHIP" = 1 ] && SCOPE=host+chip
j() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr -d '\n'; }
rev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo unknown; }
dirty=false
[ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=true
{
    printf '{\n"verdict":"%s",\n"scope":"%s",\n"date":"%s",\n' "$VERDICT" "$SCOPE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '"host":"%s",\n"target":"%s",\n' "$(j "$(uname -srm)")" "$(rustc -vV | sed -n 's/^host: //p')"
    printf '"rustc":"%s",\n"cc":"%s",\n"nm":"%s",\n' "$(j "$(rustc -V)")" "$(j "$(${CC:-cc} --version | head -1)")" "$(j "$(nm --version | head -1)")"
    printf '"flags":{"RUSTFLAGS":"%s","CFLAGS":"%s","CC":"%s","AIEN_REQUIRE_NATIVE_LIBS":"%s"},\n' \
        "$(j "${RUSTFLAGS:-}")" "$(j "${CFLAGS:-}")" "$(j "${CC:-cc}")" "${AIEN_REQUIRE_NATIVE_LIBS:-}"
    printf '"commits":{"sovereign_core":"%s","sovereign_core_dirty":%s,"omega_lock":"%s","omega":"%s","cudapp_lock":"%s","cudapp":"%s","physics":"%s","aienos":"%s"},\n' \
        "$(rev "$ROOT")" "$dirty" "$(tr -d '[:space:]' <omega.lock)" "$(rev "$PINS/omega")" \
        "$([ -f "$PINS/omega/cudapp.lock" ] && tr -d '[:space:]' <"$PINS/omega/cudapp.lock" || echo none)" \
        "$(rev "$PINS/cudapp")" "$(rev "$PINS/physics")" "$(rev "$PINS/aienos")"
    printf '"native_artifacts_lock_sha256":"%s",\n' "$(sha256sum native-artifacts.lock | cut -d' ' -f1)"
    printf '"identities":[%s],\n' "$(printf '%s\n' "$IDS" | grep . | paste -sd, -)"
    printf '"chip":{"state":"%s","reason":"%s"},\n' "$CHIP_STATE" "$(j "$CHIP_REASON")"
    printf '"steps":[\n%s\n]\n}\n' "$(printf '%s\n' "${STEPS[@]}" | paste -sd, -)"
} >"$OUT/evidence.json"
note "$VERDICT (scope $SCOPE): evidence $OUT/evidence.json"
exit "$CODE"
