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

# test-gb10-model-dir.sh (SC-GB10-MODEL-DIR): where SC-GB10-SERVE-CHAT finds its checkpoint, tested
# without a GPU (sc#760 review). Each case gives tests/gates/step-home.sh step_home (what run.sh runs)
# a read-only original HOME, then runs the real scripts/gb10-serve-chat.sh in the HOME and environment
# step_home leaves. Proves: (1) a read-only HOME with a models/ mount and a distinct writable per-step
# HOME resolves the checkpoint under the mount; (2) a missing mount, config.json or tokenizer.json is
# NOT_RUN model_missing (exit 77) before the memory check, the build and any GPU start; (3) an explicit
# MODEL_DIR is receipted as model_dir_source "override", also when the mount holds the default.
# Negative controls: the mount checks fail against the pre-#760 resolution ($HOME/models, the fresh
# HOME) and against a step_home that drops the models/ mount (the pre-#760 run.sh).
#
# Nothing here can build or start a server: the gate always gets a pins env.sh that does not exist
# (it fails on that right after the memory check), and a cargo stub first on PATH records any call,
# which fails the test. No GPU, no /tmp/aien-gb10.lock, no port is touched; /proc/meminfo is only read
# by the gate's own memory check, whose outcome no assertion depends on.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T=$(mktemp -d); trap 'chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"' EXIT
# run.sh exports AIEN_MODELS_DIR in the sandbox; every case sets its own environment.
unset MODEL_DIR AIEN_MODELS_DIR AIEN_GB10_LOCK_HELD MIRI_SYSROOT
M=qwen3.8-27b-fp8-017b9c7a
fail=0 bad=0 checks=0
chk() { checks=$((checks + 1)); if eval "$1"; then echo "  ok   $2"; else echo "  FAIL $2"; bad=1; fi; }
ov() { sed -n "s/^AIEN-OBS $2 //p" "$1"; }   # ov OUT NAME: the gate's observation NAME

mkdir -p "$T/stub"
printf '#!/bin/sh\n: > "%s/cargo-ran"\nexit 97\n' "$T" > "$T/stub/cargo"; chmod +x "$T/stub/cargo"

ckpt() {  # ckpt DIR [no-config|no-tokenizer]: a checkpoint directory the gate accepts, or not
    mkdir -p "$1"
    [ "${2:-}" = no-config ] || echo '{}' > "$1/config.json"
    [ "${2:-}" = no-tokenizer ] || echo '{}' > "$1/tokenizer.json"
}
rohome() {  # rohome [none|no-config|no-tokenizer]: a new read-only original HOME with models/$M, prints it
    local h; h=$(mktemp -d "$T/ro-home.XXXXXX")
    [ "${1:-}" = none ] || ckpt "$h/models/$M" "${1:-}"
    chmod -R a-w "$h"; echo "$h"
}
probe=$(rohome)
if [ -w "$probe" ]; then
    echo "test-gb10-model-dir: NOT_RUN: a read-only directory is writable here (root?); step_home cannot be exercised"
    exit 77
fi

step() {  # step ROOTDIR HOMEDIR OUT [VAR=VALUE]...: step_home from ROOTDIR under HOME=HOMEDIR, then the gate
    local root=$1 home=$2 out=$3; shift 3
    rm -f "$T/cargo-ran"; rc=0
    (
        export HOME="$home"
        for kv in "$@"; do export "${kv?}"; done
        # shellcheck source=tests/gates/step-home.sh
        . "$root/tests/gates/step-home.sh"
        step_home
        printf '%s\n' "$HOME" > "$out.home"
        if [ -w "$HOME" ]; then echo writable >> "$out.home"; fi
        PATH="$T/stub:$PATH" bash "$root/scripts/gb10-serve-chat.sh" "$T/no-pins/env.sh" > "$out" 2> "$out.err"
    ) || rc=$?
    fresh=$(head -n 1 "$out.home")
}
# Nothing past the memory check ran: no cargo call, no build, chip-lock or server observation.
no_build() {
    [ ! -e "$T/cargo-ran" ] && [ -z "$(ov "$1" build_rc)" ] && [ -z "$(ov "$1" chip_lock)" ] \
        && [ -z "$(ov "$1" serve_rc)" ]
}

case_mount() {  # case_mount ROOTDIR: (1), returns 1 when any check fails
    bad=0; local h out=$T/mount
    h=$(rohome)
    step "$1" "$h" "$out"
    chk '[ "$fresh" != "$h" ] && grep -qx writable "$out.home"' "step_home swapped the read-only HOME for a distinct writable one"
    chk '[ "$(ov "$out" model_dir)" = "\"$h/models/$M\"" ]' "model_dir is the read-only home's models/$M (got $(ov "$out" model_dir))"
    chk '[ "$(ov "$out" model_dir_source)" = "\"models-mount\"" ]' "model_dir_source is models-mount"
    chk '! grep -q model_missing "$out" "$out.err"' "no model_missing"
    chk '[ -n "$(ov "$out" mem_available_mib)" ]' "the gate went on to its memory check"
    chk '{ [ "$rc" -eq 77 ] && grep -q "NOT_RUN memory_short" "$out.err"; } || { [ "$rc" -eq 1 ] && grep -q "pins env file .* is missing" "$out.err"; }' \
        "it stopped at memory_short or the missing pins env.sh (rc $rc)"
    chk 'no_build "$out"' "no cargo call, no build, no chip lock, no server start"
    return "$bad"
}

case_missing() {  # case_missing VARIANT WANT_SOURCE: (2)
    bad=0; local h out=$T/missing-$1 want=$2
    h=$(rohome "$1")
    step "$ROOT" "$h" "$out"
    chk '[ "$rc" -eq 77 ] && [ "$(ov "$out" not_run_reason)" = "\"model_missing\"" ] && grep -q "NOT_RUN model_missing" "$out.err"' \
        "$1: NOT_RUN model_missing, exit 77 (rc $rc)"
    chk '[ "$(ov "$out" model_dir_source)" = "\"$want\"" ]' "$1: model_dir_source is $want"
    chk '[ -z "$(ov "$out" mem_available_mib)" ] && no_build "$out"' "$1: before the memory check, the build and any server start"
    return "$bad"
}

echo "test-gb10-model-dir: (1) read-only HOME with a models/ mount"
case_mount "$ROOT" || fail=1

echo "test-gb10-model-dir: (2) missing mount, config.json or tokenizer.json"
case_missing none home || fail=1
chk '[ "$(ov "$T/missing-none" model_dir)" = "\"$fresh/models/$M\"" ]' "none: without a mount the gate looks in the fresh HOME and says so"
[ "$bad" -eq 0 ] || fail=1
case_missing no-config models-mount || fail=1
case_missing no-tokenizer models-mount || fail=1

echo "test-gb10-model-dir: (3) an explicit MODEL_DIR stays visible in the receipt"
bad=0
h=$(rohome); ckpt "$T/override/$M"
step "$ROOT" "$h" "$T/ovr" "MODEL_DIR=$T/override/$M"
chk '[ "$(ov "$T/ovr" model_dir)" = "\"$T/override/$M\"" ] && [ "$(ov "$T/ovr" model_dir_source)" = "\"override\"" ]' \
    "override beside a populated mount: model_dir is the override, model_dir_source override"
chk '! grep -q model_missing "$T/ovr" "$T/ovr.err" && no_build "$T/ovr"' "override: accepted, still no build"
step "$ROOT" "$h" "$T/ovr-missing" "MODEL_DIR=$T/nowhere/$M"
chk '[ "$rc" -eq 77 ] && [ "$(ov "$T/ovr-missing" not_run_reason)" = "\"model_missing\"" ]' \
    "missing override: NOT_RUN model_missing, the mount's checkpoint is not substituted (rc $rc)"
chk '[ "$(ov "$T/ovr-missing" model_dir)" = "\"$T/nowhere/$M\"" ] && [ "$(ov "$T/ovr-missing" model_dir_source)" = "\"override\"" ]' \
    "missing override: receipt names the override"
wh="$T/writable-home"; ckpt "$wh/models/$M"
step "$ROOT" "$wh" "$T/wh"
chk '[ "$fresh" = "$wh" ] && [ "$(ov "$T/wh" model_dir)" = "\"$wh/models/$M\"" ] && [ "$(ov "$T/wh" model_dir_source)" = "\"home\"" ]' \
    "writable HOME (hosted, developer): kept, model_dir_source home"
[ "$bad" -eq 0 ] || fail=1

echo "test-gb10-model-dir: negative controls (each must make (1) fail)"
broken() {  # broken NAME APPEND: a copy of the gate at $T/NAME whose step-home.sh ends with APPEND
    local d="$T/$1"
    mkdir -p "$d/scripts" "$d/tests/gates" \
        && cp "$ROOT/scripts/gb10-serve-chat.sh" "$d/scripts/" \
        && cp "$ROOT/tests/gates/step-home.sh" "$d/tests/gates/" \
        && printf '%s\n' "$2" >> "$d/tests/gates/step-home.sh"
}
# A control passes only when the copy was made, the copied gate ran and reported model_missing, and
# the reason (1) fails is the one under test: the checkpoint resolved outside the mount. Any other
# failure (a broken copy, a gate that never ran) fails the control.
control() {  # control NAME APPEND LABEL
    checks=$((checks + 1))
    if ! broken "$1" "$2"; then
        echo "  FAIL control $3: could not make the broken copy under $T/$1"; fail=1; return
    fi
    if (case_mount "$T/$1") > "$T/control.log" 2>&1; then
        echo "  FAIL control $3: the mount checks passed"; fail=1
    elif grep -qE "FAIL model_dir is the read-only home's models/$M \(got \"/.+/models/$M\"\)" "$T/control.log" \
        && grep -q "FAIL no model_missing" "$T/control.log"; then
        echo "  ok   control $3: the mount checks fail:"; grep FAIL "$T/control.log" | sed 's/^ */         /'
    else
        echo "  FAIL control $3: (1) failed, but not because the checkpoint resolved outside the mount:"
        sed 's/^ */         /' "$T/control.log"; fail=1
    fi
}
# shellcheck disable=SC2016
control pre760 'gb10_model_dir() { MODEL_DIR=${MODEL_DIR:-$HOME/models/$GB10_MODEL}; MODEL_DIR_SOURCE=home; }' \
    "gate resolves \$HOME/models (pre-#760, the fresh HOME)"
control nomount 'step_home_models() { :; }' "step_home drops the models/ mount (pre-#760 run.sh)"

if [ "$fail" -eq 0 ]; then echo "test-gb10-model-dir: PASS ($checks checks)"; else echo "test-gb10-model-dir: FAIL"; exit 1; fi
