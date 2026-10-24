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

# OPEN-MODEL-QWEN3 v8 REAL qualification run (SPEC-v8.md Sections 3 and 5 item 8). Start detached ONLY on the
# coordinator's go, after run-dry.sh recorded "G6 PASS" and the freeze PR (frozen-v8.json FROZEN, all pins filled) is merged:
#   setsid nohup bash run.sh <sc main commit that merged the freeze PR> <oq3-v8-build-<commit> folder> > <log> 2>&1 < /dev/null &
# Four parts, each in its own 20-minute hold laneB-oq3-v8-pN. quietlock allows up to 20 minutes without Drake's approval token and on overrun
# keeps the flag until the command exits without killing it (quietlock strings, OBSERVED 2026-10-11). The 8B is predicted to run about
# twice the v6 part time (v6 parts took
# 7 to 8 minutes each). Stop if a part's wrapper exits non-zero (part 4 exits 1 on a FAIL verdict: the scorer's verdict, not a crash).
set -u
WRAP_COMMIT=${1:?sc main commit holding the FROZEN frozen-v8.json}; export OQ3_BUILD=${2:?build folder}
V=/home/drakestapleton/workspace/oq3-v8-real; R=$V/records
[ ! -e /home/drakestapleton/workspace/oq3-v8-runs ] || { echo "run: the v8 run base exists already (v8 runs once)"; exit 2; }
[ "$(cat $R/g6.txt 2>/dev/null)" = "G6 PASS" ] || { echo "run: no G6 PASS recorded by run-dry.sh"; exit 2; }
[ "$(tail -1 $R/v8-checks-dry.txt 2>/dev/null)" = "V8 CHECKS PASS" ] || { echo "run: no V8 CHECKS PASS on the dry launches (run-dry.sh)"; exit 2; }
quietlock check || { echo "run: quiet flag held, not starting"; exit 2; }
git -C $V/wrap fetch -q origin && git -C $V/wrap checkout -q "$WRAP_COMMIT" || { echo "run: checkout $WRAP_COMMIT failed"; exit 4; }
grep -q '"status": *"FROZEN' $V/wrap/docs/campaigns/open-model-qwen3/frozen-v8.json || { echo "run: frozen-v8.json at $WRAP_COMMIT is not FROZEN"; exit 2; }
( cd $V/wrap/docs/campaigns/open-model-qwen3 && bash test-v8.sh > $R/test-v8-real.out 2>&1 ) || { echo "run: test-v8.sh failed at $WRAP_COMMIT"; exit 2; }
sha256sum $V/run-part.sh "$0" > $R/launcher-real.sha256
for p in 1 2 3 4; do
  date -u +%FT%TZ > $R/hold-p$p.start
  quietlock hold --owner laneB-oq3-v8-p$p --minutes 20 --reason "OPEN-MODEL-QWEN3 v8 REAL qualification run part $p, Qwen3-8B (frozen-v8 FROZEN, sc#384)" -- $V/run-part.sh real $p > $R/hold-p$p.log 2>&1
  rc=$?; echo $rc > $R/hold-p$p.rc; date -u +%FT%TZ > $R/hold-p$p.end
  [ $rc = 0 ] || { echo "part $p rc=$rc, stopping" > $R/stopped; break; }
done
Q=$V/wrap/docs/campaigns/open-model-qwen3
bash $Q/v8-checks.sh /home/drakestapleton/workspace/oq3-v8-runs $Q/evidence-v7/run/s3-reports D1 D2 D3 D4 D5 D6 E1 E2 N1 N2 R1 > $R/v8-checks.txt 2>&1; tail -1 $R/v8-checks.txt
date -u +%FT%TZ > $R/all.end
