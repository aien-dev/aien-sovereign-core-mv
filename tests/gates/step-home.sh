# shellcheck shell=bash
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

# tests/gates/step-home.sh: sourced, defines functions only. The per-step HOME of tests/gates/run.sh
# and the checkpoint location it must not hide, in one place so scripts/gb10-serve-chat.sh and its
# host test scripts/test-gb10-model-dir.sh run the same code as the gate (sc#760 review).

GB10_MODEL=qwen3.8-27b-fp8-017b9c7a

# step_home: GitHub runners give every job a writable HOME; the sandbox's HOME (/var/lib/aien-run) is
# read-only on purpose. Tools that write under HOME (Chrome's crash database in aien-cli's browser_mirror
# test, which otherwise dies with SIGTRAP) get a fresh one for this step. The sandbox's git settings come
# along: its .gitconfig trusts the operator-owned checkout (safe.directory), and without it git refuses
# the tree ("not inside a git repository"). CARGO_HOME and RUSTUP_HOME are set by the sandbox and do not
# follow HOME. A writable HOME is left as it is.
step_home() {
  local ro_home
  if [ -z "${HOME:-}" ] || [ -w "$HOME" ]; then return 0; fi
  ro_home="$HOME"
  HOME="$(mktemp -d)"; export HOME
  trap 'rm -rf "$HOME"' EXIT
  if [ -f "$ro_home/.gitconfig" ]; then cp "$ro_home/.gitconfig" "$HOME/.gitconfig"; fi
  # The miri sysroot that install.sh prebuilt under the read-only HOME (#600) stays the one miri uses;
  # the fresh HOME would otherwise send `cargo miri setup` off to rebuild it offline, which cannot work.
  if [ -z "${MIRI_SYSROOT:-}" ] && [ -d "$ro_home/.cache/miri/lib" ]; then
    MIRI_SYSROOT="$ro_home/.cache/miri"; export MIRI_SYSROOT
  fi
  step_home_models "$ro_home"
}

# step_home_models RO_HOME: the operator's checkpoints are mounted read-only at the sandbox home's
# models/ (aien-run-exec); a model gate looks there, not in the fresh HOME (2026-10-25:
# SC-GB10-SERVE-CHAT reported model_missing under /tmp/tmp.*/models on the first Alpha judge of #751).
step_home_models() {
  if [ -z "${AIEN_MODELS_DIR:-}" ] && [ -d "$1/models" ]; then
    AIEN_MODELS_DIR="$1/models"; export AIEN_MODELS_DIR
  fi
}

# gb10_model_dir: sets MODEL_DIR to the Qwen3.8-27B checkpoint SC-GB10-SERVE-CHAT serves and
# MODEL_DIR_SOURCE to where that came from, for the receipt: override (MODEL_DIR was set by the
# caller), models-mount (AIEN_MODELS_DIR, which step_home sets), home ($HOME/models).
gb10_model_dir() {
  if [ -n "${MODEL_DIR:-}" ]; then
    MODEL_DIR_SOURCE=override
  elif [ -n "${AIEN_MODELS_DIR:-}" ]; then
    MODEL_DIR="$AIEN_MODELS_DIR/$GB10_MODEL"; MODEL_DIR_SOURCE=models-mount
  else
    MODEL_DIR="$HOME/models/$GB10_MODEL"; MODEL_DIR_SOURCE=home
  fi
}
