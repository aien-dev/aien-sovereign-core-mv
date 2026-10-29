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

p=$1; R=/home/drakestapleton/workspace/oq3-v3-records
date -u +%T > $R/hold-p$p.start
quietlock hold --owner 031756-oq3-v3-p$p --minutes 20 --reason "OPEN-MODEL-QWEN3 v3 part $p" -- $R/run-part.sh $p > $R/hold-p$p.log 2>&1
echo $? > $R/hold-p$p.rc
date -u +%T > $R/hold-p$p.end
