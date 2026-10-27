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

# test-pins.sh: offline self-test of scripts/pins.sh with throwaway git repositories
# (no network). Proves: correct pins pass (--check and a full run against a local mirror),
# a checkout off its pin is refused (exit 3), a malformed lock is refused (exit 3), and
# env.sh carries the three directories and the AIEN_CUDAPP_DIR choice (--gpu places cudapp
# at cudapp.lock).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
chk() { if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; fail=1; fi; }
mk() { # mk NAME [FILE CONTENT]...: a one-commit repo under $T/mirror/NAME, prints its sha
    local d="$T/mirror/$1"; shift; mkdir -p "$d"
    git -C "$d" init -q; while [ $# -ge 2 ]; do printf '%s\n' "$2" > "$d/$1"; shift 2; done
    [ -n "$(ls -A "$d" | grep -v '^.git$')" ] || : > "$d/.keep"
    git -C "$d" add -A; git -C "$d" -c user.name=t -c user.email=t@t commit -q -m pin; git -C "$d" rev-parse HEAD
}
P=$(mk physics); A=$(mk aienos); O=$(mk omega physics.lock "$P" aienos.lock "$A"); C=$(mk cudapp)
mkdir -p "$T/root"; printf '%s\n' "$O" > "$T/root/omega.lock"; printf '%s\n' "$C" > "$T/root/cudapp.lock"
run() { AIEN_PINS_ROOT="$T/root" AIEN_PINS_REMOTE_BASE="$T/mirror" bash "$ROOT/scripts/pins.sh" "$@"; }
echo "test-pins: full run against a local mirror"
chk 'run "$T/pins" >/dev/null' "pins.sh materializes omega, physics, aienos"
chk '[ "$(git -C "$T/pins/omega" rev-parse HEAD)" = "$O" ]' "omega at omega.lock"
chk '[ "$(git -C "$T/pins/physics" rev-parse HEAD)" = "$P" ]' "physics at omega physics.lock"
chk '[ "$(git -C "$T/pins/aienos" rev-parse HEAD)" = "$A" ]' "aienos at omega aienos.lock"
chk 'grep -q "AIEN_OMEGA_COMPOSE_DIR=.*pins/omega" "$T/pins/env.sh" && grep -q "unset AIEN_CUDAPP_DIR" "$T/pins/env.sh" && ! grep -q "export AIEN_OMEGA_DIR" "$T/pins/env.sh"' "env.sh: composition only without --gpu"
chk '[ ! -e "$T/pins/cudapp" ]' "no cudapp checkout without --gpu"
chk 'run --gpu "$T/pins" >/dev/null && grep -q "export AIEN_CUDAPP_DIR=.*pins/cudapp" "$T/pins/env.sh" && grep -q "unset AIEN_OMEGA_DIR" "$T/pins/env.sh"' "env.sh: --gpu sets AIEN_CUDAPP_DIR, never AIEN_OMEGA_DIR"
chk '[ "$(git -C "$T/pins/cudapp" rev-parse HEAD)" = "$C" ]' "cudapp at cudapp.lock with --gpu"
chk 'grep -q "AIEN_REQUIRE_NATIVE_LIBS=1" "$T/pins/env.sh"' "env.sh: stubs refused (AIEN_REQUIRE_NATIVE_LIBS=1)"
chk 'run --check "$T/pins" >/dev/null' "--check passes when every clone is at its pin"
chk 'run --gpu --check "$T/pins" >/dev/null' "--gpu --check passes when cudapp is at its pin"
echo "test-pins: refusals"
git -C "$T/pins/physics" -c user.name=t -c user.email=t@t commit -q --allow-empty -m drift
rc=0; run --check "$T/pins" >/dev/null 2>&1 || rc=$?
chk '[ "$rc" -eq 3 ]' "--check refuses a checkout off its pin (exit 3, got $rc)"
chk 'run "$T/pins" >/dev/null && [ "$(git -C "$T/pins/physics" rev-parse HEAD)" = "$P" ]' "a full run moves it back to the pin"
git -C "$T/pins/cudapp" -c user.name=t -c user.email=t@t commit -q --allow-empty -m drift
rc=0; run --gpu --check "$T/pins" >/dev/null 2>&1 || rc=$?
chk '[ "$rc" -eq 3 ]' "--gpu --check refuses cudapp off cudapp.lock (exit 3, got $rc)"
printf 'not-a-sha\n' > "$T/root/omega.lock"
rc=0; run "$T/pins" >/dev/null 2>&1 || rc=$?
chk '[ "$rc" -eq 3 ]' "malformed omega.lock refused (exit 3, got $rc)"
rc=0; run --bogus >/dev/null 2>&1 || rc=$?
chk '[ "$rc" -eq 2 ]' "unknown flag is a usage error (exit 2, got $rc)"
[ "$fail" -eq 0 ] && echo "test-pins: PASS" || { echo "test-pins: FAIL"; exit 1; }
