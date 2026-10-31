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

# Stage 7: upstream unit tests for the patches (after the timing runs).
# PATH changes are local to this process only.
set -uo pipefail
E=~/workspace/headstart-exp
H=$E/headstart
L=$E/logs
S=$H/rustc/build/host/stage1/bin
J=${J:-12}
res() { echo "$1 rc=$2 wall_s=$3" | tee -a "$L/03-results.txt"; }
t0=$(date +%s)
(cd "$H/rustc" && nice -n 10 ./x test tests/run-make/early-metadata tests/ui/rmeta -j "$J") > "$L/07-x-test-early-metadata.log" 2>&1
res x-test-early-metadata-ui-rmeta $? $(( $(date +%s) - t0 ))
t0=$(date +%s)
(cd "$H/cargo" && RUSTC=$(rustup which --toolchain 1.98.0 rustc) PATH=$S:$PATH nice -n 10 cargo +1.98.0 test --test testsuite --no-run -j "$J") > "$L/07-cargo-testsuite-build.log" 2>&1
rc=$?
res cargo-testsuite-build $rc $(( $(date +%s) - t0 ))
if [ "$rc" -eq 0 ]; then
  t0=$(date +%s)
  bin=$(ls -t "$H"/cargo/target/debug/deps/testsuite-* | grep -v '\.d$' | head -1)
  (cd "$H/cargo" && PATH=$S:$PATH "$bin" headstart) > "$L/07-cargo-testsuite-headstart.log" 2>&1
  res cargo-testsuite-headstart $? $(( $(date +%s) - t0 ))
fi
echo UNIT_DONE
