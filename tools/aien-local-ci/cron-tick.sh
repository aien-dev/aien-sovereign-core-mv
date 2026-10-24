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
# 1) keeps four worker lanes alive (--github): main (aienos, synapse, interplane), small (the small repos), big (omega,
#    sovereign-core) and core (sovereign-core), so one long job cannot starve the rest. `start` is a no-op while a
#    lane runs.
# 2) rebase train: moves ready-to-merge PRs that conflict only in .crumb files (or are behind) onto main.
# 3) runs one pipeline round: publish signed bundles, rerun stale checks, merge green PRs labelled ready-to-merge.
# Every 5 min, not 2: the 2-min round plus the worker's polling used up the 5000/h GitHub API limit (2026-10-18).
# No gh call here; pipeline.sh and rebase-train.sh treat the gh guard's exit 98 as "try next round".
# Stop everything: touch ~/.local/state/aien-local-ci/PAUSE ; aien-local-ci stop ; aien-local-ci stop --lane small ; aien-local-ci stop --lane big ; aien-local-ci stop --lane core
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
S=$HOME/.local/state/aien-local-ci
L=$HOME/.local/lib/aien-local-ci
[ -e "$S/PAUSE" ] && exit 0
nice -n 10 aien-local-ci start --github >/dev/null
nice -n 10 aien-local-ci start --github --lane small --repos physics,aien-protocols,aegis-runtime,spark-rsi,aien-architecture,benchmarks >/dev/null
nice -n 10 aien-local-ci start --github --lane big --repos omega,aien-sovereign-core >/dev/null
nice -n 10 aien-local-ci start --github --lane core --repos aien-sovereign-core >/dev/null
nice -n 10 /bin/bash "$L/rebase-train.sh" || echo "cron-tick: rebase-train.sh exit $?"
nice -n 10 /bin/bash "$L/pipeline.sh" round
