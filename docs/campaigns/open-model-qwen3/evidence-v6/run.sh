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

# OPEN-MODEL-QWEN3 v6 qualification run (QUALIFICATION-v6.md Sections 6 and 7). Start detached ONLY on the coordinator's go:
#   setsid nohup bash run.sh <sc main commit that merged QUALIFICATION-v6.md> > <log> 2>&1 < /dev/null &
# Order: quietlock check; clean clone of the wrapper at that commit; G6 dry part 1 in its own hold (stop on a product failure);
# then real parts 1..4, each in its own 20-minute hold laneB-oq3-v6-pN; stop if a part's wrapper exits non-zero
# (part 4 exits 1 on a FAIL verdict: that is the scorer's verdict, not a crash).
set -u
WRAP_COMMIT=${1:?sc main commit holding QUALIFICATION-v6.md}
HERE=$(cd "$(dirname "$0")" && pwd)
V=/home/drakestapleton/workspace/oq3-v6-real; R=$V/records
[ ! -e /home/drakestapleton/workspace/oq3-v6-runs ] || { echo "run: the v6 run base exists already (v6 runs once)"; exit 2; }
quietlock check || { echo "run: quiet flag held, not starting"; exit 2; }
mkdir -p $R
git clone -q https://github.com/aien-dev/aien-sovereign-core $V/wrap && git -C $V/wrap checkout -q "$WRAP_COMMIT" || { echo "run: wrapper clone failed"; exit 4; }
grep -q '^\*\*Status: FROZEN' $V/wrap/docs/campaigns/open-model-qwen3/QUALIFICATION-v6.md || { echo "run: QUALIFICATION-v6.md at $WRAP_COMMIT is not FROZEN"; exit 2; }
( cd $V/wrap/docs/campaigns/open-model-qwen3 && bash test-v6.sh > $R/test-v6.out 2>&1 ) || { echo "run: test-v6.sh failed at $WRAP_COMMIT"; exit 2; }
cp "$HERE/run-part.sh" $V/run-part.sh; chmod +x $V/run-part.sh; sha256sum $V/run-part.sh "$0" > $R/launcher.sha256
# G6: dry part 1 (never scored)
date -u +%FT%TZ > $R/hold-dry-p1.start
quietlock hold --owner laneB-oq3-v6-dry-p1 --minutes 20 --reason "OPEN-MODEL-QWEN3 v6 gate G6 dry run part 1 (not scored)" -- $V/run-part.sh dry 1 > $R/hold-dry-p1.log 2>&1
echo $? > $R/hold-dry-p1.rc; date -u +%FT%TZ > $R/hold-dry-p1.end
g6=PASS
for id in D1 D2 D3; do
  [ -f $V/dry-runs/$id/run.json ] || g6="FAIL $id no run.json"
  for r in PRE OPR BE PIN; do
    v=$(jq -r --arg row "$id/$id-$r" 'select(.row == $row) | .verdict' $R/dry-out/oq3-v6-DRYRUN-results.jsonl 2>/dev/null | head -1)
    [ "$v" = PASS ] || g6="FAIL $id-$r=${v:-missing}"
  done
done
echo "G6 $g6" > $R/g6.txt
[ "$g6" = PASS ] || { echo "run: gate G6 $g6; real run NOT started" > $R/stopped; exit 3; }
for p in 1 2 3 4; do
  date -u +%FT%TZ > $R/hold-p$p.start
  quietlock hold --owner laneB-oq3-v6-p$p --minutes 20 --reason "OPEN-MODEL-QWEN3 v6 REAL qualification run part $p (QUALIFICATION-v6 FROZEN, sc#384)" -- $V/run-part.sh real $p > $R/hold-p$p.log 2>&1
  rc=$?; echo $rc > $R/hold-p$p.rc; date -u +%FT%TZ > $R/hold-p$p.end
  [ $rc = 0 ] || { echo "part $p rc=$rc, stopping" > $R/stopped; break; }
done
date -u +%FT%TZ > $R/all.end
