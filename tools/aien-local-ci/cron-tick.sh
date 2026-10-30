#!/bin/bash
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

# AIEN Spark pipeline tick (cron, every 5 min; no systemd by rule). Installed copy: ~/.local/lib/aien-local-ci/.
#   */5 * * * * $HOME/.local/lib/aien-local-ci/cron-tick.sh >> $HOME/.local/state/aien-local-ci/cron-tick.log 2>&1
# Spark qualification is the Alpha stage, not a merge gate (Drake 2026-10-24: "We must not limit our development
# process by these tests"). So:
# 1) crumb drift guard: crumb-doctor keeps the trusted sovereign-core work tree on origin/main (the workers read the
#    sibling map from it) and, at most every 15 min, raises a CRUMB-DRIFT alert when the crumb tools disagree.
# 2) keeps the worker alive WITHOUT --github: it judges only trigger files. `aien-alpha REPO` writes one for main's
#    head with a full-scope run; pull requests are never judged on the Spark. `start` is a no-op while it runs.
# 3) rebase train: moves ready-to-merge PRs that conflict only in generated .crumb files onto main.
# 4) one pipeline round: publish signed bundles, merge ready-to-merge PRs that merge cleanly, tag Alpha builds.
# With $S/KEEP-RUNNING present, desk-loop.sh runs 3) and 4) continuously instead.
# Stop everything: touch ~/.local/state/aien-local-ci/PAUSE ; aien-local-ci stop
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
S=$HOME/.local/state/aien-local-ci
L=$HOME/.local/lib/aien-local-ci
export GH_DESK_KEY=$(cat "$S/DESK.key" 2>/dev/null)
[ -e "$S/PAUSE" ] && exit 0
crumb-doctor --heal-only >/dev/null 2>&1 || true
if [ ! -e "$S/crumb-doctor.last" ] || [ -n "$(find "$S/crumb-doctor.last" -mmin +14 2>/dev/null)" ]; then
  ( flock -n 9 || exit 0
    nice -n 10 crumb-doctor --heal --alert > "$S/crumb-doctor.last.tmp" 2>&1
    mv -f "$S/crumb-doctor.last.tmp" "$S/crumb-doctor.last"
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$(tail -1 "$S/crumb-doctor.last")" >> "$S/crumb-doctor.log" ) 9>"$S/crumb-doctor.lock" &
fi
nice -n 10 aien-local-ci start >/dev/null
if [ -e "$S/KEEP-RUNNING" ] && flock -n "$S/desk-loop.lock" true 2>/dev/null; then
  nohup nice -n 10 /bin/bash "$L/desk-loop.sh" >> "$S/desk-loop.log" 2>&1 &
else
  nice -n 10 /bin/bash "$L/rebase-train.sh" || echo "cron-tick: rebase-train.sh exit $?"
  nice -n 10 /bin/bash "$L/pipeline.sh" round
fi
