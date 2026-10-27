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

# OPEN-MODEL-QWEN3 v8 gate G6: dry run of part 1 on GB10 in its own hold, then g6-gate.sh over D1 D2 D3 (SPEC-v8.md
# Section 5 item 7). Runs AFTER the freeze, as v6 did (frozen-v8.json FROZEN at the wrap commit): the -PIN and -GR rows
# compare the build and the generation record against the frozen pins and model hash, so a draft can only fail them (attempt 2,
# 2026-10-11, evidence-v8/attempt-2-g6-fail/). A dry run is never a verdict. Start detached ONLY on the coordinator's go for the hold:
#   setsid nohup bash run-dry.sh <freeze commit> <oq3-v8-build-<commit> folder> > <log> 2>&1 < /dev/null &
set -u
WRAP_COMMIT=${1:?freeze commit}; export OQ3_BUILD=${2:?build folder}
HERE=$(cd "$(dirname "$0")" && pwd)
V=/home/drakestapleton/workspace/oq3-v8-real; R=$V/records
[ ! -e $V/dry-runs ] || { echo "run-dry: $V/dry-runs exists already (one dry run per freeze candidate; move it aside to rerun)"; exit 2; }
[ -x "$OQ3_BUILD/target/release/aien-cli" ] || { echo "run-dry: no aien-cli in $OQ3_BUILD"; exit 2; }
quietlock check || { echo "run-dry: quiet flag held, not starting"; exit 2; }
mkdir -p $R
[ -d $V/wrap ] || git clone -q https://github.com/aien-dev/aien-sovereign-core $V/wrap || { echo "run-dry: wrapper clone failed"; exit 4; }
git -C $V/wrap fetch -q origin && git -C $V/wrap checkout -q "$WRAP_COMMIT" || { echo "run-dry: checkout $WRAP_COMMIT failed"; exit 4; }
Q=$V/wrap/docs/campaigns/open-model-qwen3
[ "$(jq -r .status $Q/frozen-v8.json | cut -c1-6)" = FROZEN ] || { echo "run-dry: frozen-v8.json at $WRAP_COMMIT is not FROZEN; freeze first (the -PIN and -GR rows need the pins and the model hash)"; exit 2; }
( cd $V/wrap/docs/campaigns/open-model-qwen3 && bash test-v8.sh > $R/test-v8.out 2>&1 ) || { echo "run-dry: test-v8.sh failed at $WRAP_COMMIT"; exit 2; }
cp "$HERE/run-part.sh" $V/run-part.sh; chmod +x $V/run-part.sh; sha256sum $V/run-part.sh "$0" > $R/launcher-dry.sha256
date -u +%FT%TZ > $R/hold-dry-p1.start
# quietlock: holds over 20 minutes need Drake's approval token (attempt 1 at 22:10:23Z was refused with 40); on overrun the flag is kept and the command is never killed.
quietlock hold --owner laneB-oq3-v8-dry-p1 --minutes 20 --reason "OPEN-MODEL-QWEN3 v8 gate G6 dry run part 1, Qwen3-8B (not scored, sc#384)" -- $V/run-part.sh dry 1 > $R/hold-dry-p1.log 2>&1
echo $? > $R/hold-dry-p1.rc; date -u +%FT%TZ > $R/hold-dry-p1.end
Q=$V/wrap/docs/campaigns/open-model-qwen3
bash $Q/g6-gate.sh $R/dry-out/oq3-v8-DRYRUN-results.jsonl $Q/../scoring/declarations/oq3-v5.decl.json $V/dry-runs D1 D2 D3 > $R/g6-reasons.txt 2>&1
rc=$?; [ $rc = 0 ] && echo "G6 PASS" > $R/g6.txt || echo "G6 FAIL (rc $rc, see g6-reasons.txt)" > $R/g6.txt
bash $Q/v8-checks.sh $V/dry-runs $Q/evidence-v7/run/s3-reports D1 D2 D3 > $R/v8-checks-dry.txt 2>&1; tail -1 $R/v8-checks-dry.txt
cat $R/g6.txt; grep -m1 -o 'CHECKPOINT_SHARDS.*' $V/dry-runs/D1/daemon-1.log 2>/dev/null | cut -c1-200 > $R/model_sha256-line.txt; cat $R/model_sha256-line.txt
