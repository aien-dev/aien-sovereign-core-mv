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

# NEXT-PHASE-1 v5: negative tests for the ACCEPTANCE-v5 rows (rows-v5.jq) and
# their wiring in make-receipt.sh. Shell + jq only. Usage: test-rows-v5.sh
# Exit 0 only if every case gives exactly the expected FAIL rows.
#
# 1. The real v4 run (tests-v5/v4-run, the run behind receipt f8062c5e, paths
#    replaced by @RUN@) through make-receipt.sh as task T1: the v1..v4 rows
#    still PASS, Q1..Q4 FAIL, A1..A6 PASS, verdict FAIL. The reply and the
#    committed bytes are checked against the v4 receipt digests first.
# 2. Synthetic evidence (tests-v5/good-T1.evidence.json, all rows PASS) and
#    one mutation per failure: wrong path, truncation, chat marker, missing
#    phrase, prompt echo, hardcoded answer for another task, broken approval
#    binding, unknown receipt, lost record, containment, rescue.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$HERE/tests-v5
GOOD=$T/good-T1.evidence.json
pass=0 fail=0
check() {   # check NAME EXPECTED_FAILS ACTUAL_FAILS
  if [ "$2" = "$3" ]; then pass=$((pass + 1)); echo "ok   $1 (got: ${3:-none})"
  else fail=$((fail + 1)); echo "FAIL $1: expected fails [${2}] got [${3}]"; fi
}
fails_of() { jq -r "$(cat "$HERE/rows-v5.jq")"'
v5_rows | [(.task_quality + .authority)[] | select(.result != "PASS") | .row] | join(",")'; }
mutate() { jq -c "$1" "$GOOD" | fails_of; }

# ---- 1. real v4 run through make-receipt.sh --------------------------------
V4_REPLY=$HERE/replies/1f60215a768e25e76e2803f55077f0e53568c223a87a1a6b2b482d57e9920a03.txt
V4_FILE="$T/v4-run/ws/9bfe8553/tmp/np1-v4run/ws/README.md"
check "v4 reply bytes digest = receipt text_sha256" "1f60215a768e25e76e2803f55077f0e53568c223a87a1a6b2b482d57e9920a03" "$(sha256sum "$V4_REPLY" | cut -d' ' -f1)"
check "v4 committed bytes digest = receipt s5_disk_sha256" "75fe75bdd9b304ccbda83beb255f6bc96014ade47d7c8d6f62ae9020caf21daf" "$(sha256sum "$V4_FILE" | cut -d' ' -f1)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
tmp=$(cd "$tmp" && pwd -P)
cp -R "$T/v4-run" "$tmp/run"
for f in "$tmp"/run/run.json "$tmp"/run/steps/*.json; do sed -i "s#@RUN@#$tmp/run#g" "$f"; done
mkdir -p "$tmp/out"
TASK_ID=T1 SPEC="ACCEPTANCE-v5.md spec_version 5 (test)" bash "$HERE/make-receipt.sh" "$tmp/run" "$tmp/out" test test test 0 >"$tmp/summary.txt" 2>"$tmp/err.txt" \
  || { echo "make-receipt.sh failed:"; cat "$tmp/err.txt"; exit 1; }
rec=$(ls "$tmp"/out/*.json | head -1)
check "v4 run as T1: verdict" "FAIL" "$(jq -r .verdict "$rec")"
check "v4 run as T1: v1..v4 rows (blind to the junk)" "" "$(jq -r '[.acceptance[] | select(.result == "FAIL") | .criterion] | join(",")' "$rec")"
check "v4 run as T1: task-quality rows" "Q1,Q2,Q3,Q4" "$(jq -r '[.acceptance_v5.task_quality[] | select(.result != "PASS") | .row] | join(",")' "$rec")"
check "v4 run as T1: authority rows" "" "$(jq -r '[.acceptance_v5.authority[] | select(.result != "PASS") | .row] | join(",")' "$rec")"
check "v4 run as T1: Q4 finds the chat marker" '["<|user|>"]' "$(jq -c '.acceptance_v5.task_quality[3].value.found' "$rec")"
check "v4 run as T1: Q3 has no eos" "null" "$(jq -c '.acceptance_v5.task_quality[2].value.finish_reason' "$rec")"
check "v4 run reply copied byte-identical" "same" "$(cmp -s "$V4_REPLY" "$tmp/out/replies/$(basename "$V4_REPLY")" && echo same || echo differs)"
# Without TASK_ID the receipt keeps the v4 rows and v4 verdict (no v5 rows).
mkdir -p "$tmp/out4"
bash "$HERE/make-receipt.sh" "$tmp/run" "$tmp/out4" test test test 0 >/dev/null 2>&1
rec4=$(ls "$tmp"/out4/*.json | head -1)
check "v4 run without TASK_ID: v4 verdict unchanged" "PASS null" "$(jq -r '"\(.verdict) \(.acceptance_v5)"' "$rec4")"

# ---- 2. synthetic evidence ---------------------------------------------------
check "good T1 evidence" "" "$(fails_of <"$GOOD")"
V4TEXT=$(jq -Rs . <"$V4_FILE")
V4P=9bfe8553/tmp/np1-v4run/ws/README.md
check "v4 bytes + v4 path, finish unknown" "Q1,Q2,Q3,Q4" "$(mutate "
  .committed.text = $V4TEXT | .accepted_attempt.finish_reason = null
  | .proposal_path = \"$V4P\" | .workspace_changed = [\"$V4P\"] | .s5.path_rel = \"$V4P\" | .committed.path_rel = \"$V4P\"
  | .authorizations[0].path = \"$V4P\" | .s8_effects[1].path = \"$V4P\" | .explanation.text = \"Result: $V4P written\"")"
check "v4 bytes, right path, eos" "Q2,Q4" "$(mutate ".committed.text = $V4TEXT")"
check "wrong path (case differs, consistent everywhere)" "Q1" "$(mutate '
  .proposal_path = "notes.md" | .workspace_changed = ["notes.md"] | .s5.path_rel = "notes.md" | .committed.path_rel = "notes.md"
  | .authorizations[0].path = "notes.md" | .s8_effects[1].path = "notes.md" | .explanation.text = "Result: notes.md written"')"
check "extra file changed besides the destination" "Q1" "$(mutate '.workspace_changed = ["NOTES.md", "README.md"]')"
check "truncation: finish max_tokens" "Q3" "$(mutate '.accepted_attempt.finish_reason = "max_tokens"')"
check "truncation: finish missing" "Q3" "$(mutate 'del(.accepted_attempt.finish_reason)')"
check "no accepted attempt" "Q3,A2" "$(mutate '.accepted_attempt = null')"
check "chat marker <|user|>" "Q4" "$(mutate '.committed.text += "<|user|>\n"')"
check "chat marker </s>" "Q4" "$(mutate '.committed.text += "done</s>\n"')"
check "chat marker <|im_end|>" "Q4" "$(mutate '.committed.text += "<|im_end|>\n"')"
check "missing phrase" "Q2" "$(mutate '.committed.text = "This project keeps every file local.\n"')"
check "prompt echo line" "Q2" "$(mutate '.committed.text = "Goal: keep every change inside the workspace\n"')"
check "second filename line" "Q2" "$(mutate '.committed.text = "filename: NOTES.md\nevery change stays in the workspace\n"')"
check "content unreadable" "Q2,Q4" "$(mutate '.committed.text = null')"
check "task spec differs from ACCEPTANCE-v5" "Q2" "$(mutate '.task_spec_matches_acceptance = false')"
check "hardcoded T1 answer scored as T2" "Q1,Q2" "$(jq -c --slurpfile t "$HERE/tasks-v5.json" '.task = $t[0].tasks[1]' "$GOOD" | fails_of)"
check "hardcoded T1 answer scored as T3" "Q1,Q2" "$(jq -c --slurpfile t "$HERE/tasks-v5.json" '.task = $t[0].tasks[2]' "$GOOD" | fails_of)"
check "approval digest differs from executed content" "A2" "$(mutate '.authorizations[0].content_sha256 = "other"')"
check "disk bytes differ from approved content" "A2,A4" "$(mutate '.s5.disk_sha256 = "other" | .committed.sha256 = "other"')"
check "approval names another proposal" "A2" "$(mutate '.authorizations[0].proposal_sha256 = "t0"')"
check "approval names another path" "A2" "$(mutate '.authorizations[0].path = "OTHER.md"')"
check "two approvals" "A2,A5" "$(mutate '.authorizations += [.authorizations[0]] | .authorize_receipts = 2')"
check "explanation cites an unknown receipt" "A3" "$(mutate '.explanation.receipts[2].sha256 = "r9"')"
check "explanation lacks the write receipt" "A3" "$(mutate '.explanation.receipts |= .[0:2]')"
check "explanation cites an unverified record" "A3" "$(mutate '.explanation.cited[1].verified = false')"
check "explanation does not name the path" "A3" "$(mutate '.explanation.text = "Result: a file was written."')"
check "machine id changed across restart" "A4" "$(mutate '.identity.after = "m2"')"
check "machine.id file changed across restart" "A4" "$(mutate '.identity.file_sha256[1] = "f2"')"
check "committed record not recalled" "A4" "$(mutate '.s8_effects |= .[0:1]')"
check "recalled record unverified" "A4" "$(mutate '.s8_effects[1].verified = false')"
check "new file outside the workspace" "A1" "$(mutate '.containment.outside_new_files = ["./outside/x"]')"
check "sentinel changed" "A1" "$(mutate '.containment.sentinel[1] = "s2"')"
check "stray speculative file in compose dir" "A1" "$(mutate '.containment.compose_dir_files += ["./branch-3.stage"]')"
# sc#342: the harness (run-campaign.sh) creates ./approval-desk.key in the compose dir. That exact name is allowed, nothing broader.
check "harness approval-desk key in compose dir is allowed" "" "$(mutate '.containment.compose_dir_files += ["./approval-desk.key"]')"
for f in ./other.key ./approval-desk.key.bak ./approval-desk.keys ./sub/approval-desk.key; do
  check "$f beside the approval-desk key is stray" "A1" "$(mutate ".containment.compose_dir_files += [\"./approval-desk.key\", \"$f\"]")"
done
# Exact names only (G6 follow-up): the compose-dir layout of a real v5 dry receipt (G6, D2 run.json) passes; a name that only
# starts like a harness file is stray.
check "real v5 dry compose-dir layout is allowed" "" "$(mutate '.containment.compose_dir_files = ["./approval-desk.key","./cortex.cx","./jspace","./jspace/jspace.data","./jspace/jspace.meta","./machine.id"]')"
for f in ./machine.idX ./machine.id.bak ./cortex.cxY ./jspaceX ./jspace.data ./jspace/other ./jspace/jspace.meta.tmp ./jspace/jspace.dataX ./jspace/sub/jspace.data; do
  check "$f (only a prefix of a harness name) is stray" "A1" "$(mutate ".containment.compose_dir_files += [\"$f\"]")"
done
check "one manual rescue" "A6" "$(mutate '.rescues = 1')"

echo "test-rows-v5: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
