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

# Start the held cold A/B rerun detached from the agent's shell, so a tool
# time limit cannot stop it mid-sample. Writes its exit code to a marker file.
E=~/workspace/headstart-exp
rm -f "$E/logs/12-ab-held.rc"
setsid nohup bash -c "HS_AB_JOBS='' '$E/bin/05-measure.sh' ab > '$E/logs/05-measure-ab-held.log' 2>&1; echo \$? > '$E/logs/12-ab-held.rc'" > /dev/null 2>&1 < /dev/null &
echo "started pid $!"
