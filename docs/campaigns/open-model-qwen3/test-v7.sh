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

# OPEN-MODEL-QWEN3 v7 harness self-test (D1, sc#384). CPU only, no model, no GPU, no network. Shell + jq.
# Proves the two harness corrections: (a) the driver engages ALLEN before S1 and changes nothing else; (b) gate G6 checks every
# product row and goes red on the failure v6 missed. Thresholds and rows are untouched (declaration and task files are not read here
# except to list rows).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE" || exit 2
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok: $1"; }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }
chk() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
DECL=../scoring/declarations/oq3-v5.decl.json G=./g6-gate.sh

chk "all new scripts parse (bash -n)" 'for f in run-campaign-allen.sh build-allen-subject.sh g6-gate.sh run-qwen3-v7.sh; do bash -n $f || exit 1; done'
# ---- driver: the only difference from the frozen driver is the ALLEN block, the mark-evidence path and the allen evidence in run.json
D=$(diff ../next-phase-1/run-campaign.sh run-campaign-allen.sh | grep -c "^<")
chk "driver diff against next-phase-1/run-campaign.sh removes only 5 lines (header comment, mark-evidence path, one find-exclusion line, the two run.json lines); all else is added ALLEN lines (removed: $D)" '[ "$D" = 5 ]'
chk "the ALLEN block runs before the measured window (.mark) and before S1" \
  'a=$(grep -n "ALLEN_LINE=\$(grep" run-campaign-allen.sh | head -1 | cut -d: -f1); m=$(grep -n "^touch \"\$R/.mark\"" run-campaign-allen.sh | cut -d: -f1); s=$(grep -n "^run_step S1" run-campaign-allen.sh | cut -d: -f1); [ "$a" -lt "$m" ] && [ "$m" -lt "$s" ]'
chk "the driver refuses to run without the subject builder" 'out=$(env -u NP1_ALLEN_BUILDER AIEN_BIN=/bin/true AIEN_MODEL_PATH=x AIEN_TOKENIZER_PATH=y NP1_DIR=. bash run-campaign-allen.sh "$T/r" 2>&1); echo "$out" | grep -q NP1_ALLEN_BUILDER || echo "$out" | grep -q "daemon 1 failed"'
chk "the frozen next-phase-1 driver is unchanged by this lane" '! git diff --quiet origin/main -- ../next-phase-1 2>/dev/null || [ -z "$(git status --porcelain ../next-phase-1)" ]'

# ---- wrapper: v7 differs from v6 only where intended
chk "run-qwen3-v7.sh uses the ALLEN driver and keeps the v6 refusals (pinned base, FROZEN status, placeholder pins)" \
  'grep -q "run-campaign-allen.sh" run-qwen3-v7.sh && grep -q "PINNED_BASE=/home/drakestapleton/workspace/oq3-v7-runs" run-qwen3-v7.sh && grep -q "status is not FROZEN" run-qwen3-v7.sh && grep -q "build pins are not frozen" run-qwen3-v7.sh'
chk "frozen-v7.json is FROZEN: status says so, the same eight pin names as v6, every pin 40 or 64 lowercase hex, model_sha256 64 hex, no placeholder left" \
  '[ "$(jq -r .status frozen-v7.json | cut -c1-6)" = FROZEN ] && [ "$(jq -c ".pins | keys" frozen-v7.json)" = "$(jq -c ".pins | keys" frozen-v6.json)" ] && [ "$(jq "[.pins[] | select(test(\"^([0-9a-f]{40}|[0-9a-f]{64})$\") | not)] | length" frozen-v7.json)" = 0 ] && jq -e ".model.model_sha256 | test(\"^[0-9a-f]{64}$\")" frozen-v7.json >/dev/null && [ "$(jq -r "[(.pins[], .model[]) | strings | select(. == \"TO FILL AT FREEZE\")] | length" frozen-v7.json)" = 0 ]'
chk "a real v7 run is refused while frozen-v7.json is a draft" \
  'out=$(OQ3_PART=1 AIEN_MODEL_PATH=x AIEN_TOKENIZER_PATH=y AIEN_BIN=z bash run-qwen3-v7.sh /home/drakestapleton/workspace/oq3-v7-runs-selftest-absent "$T/o" . . /bin/true /bin/true 2>&1); echo "$out" | grep -q "RUN_BASE must be"'
chk "OQ3_FROZEN outside a dry run is refused" 'out=$(OQ3_FROZEN=$HERE/frozen-v6.json OQ3_PART=1 bash run-qwen3-v7.sh a b c d e f 2>&1); echo "$out" | grep -q "allowed in a dry run only"'

# ---- gate G6 on synthetic launches built from the declaration's own rows
mk() { # dir ; builds D1..D3 with every row PASS, S1..S8 ok, ALLEN engaged
  local b=$1; mkdir -p "$b"
  for L in D1 D2 D3; do mkdir -p "$b/$L"
    jq -n '{steps: [range(1;9) | {step: ("S\(.)"), ok: true}], allen: {engaged_log_line: "ALLEN: engaged agent=ab head_sequence=2", fingerprint: "ab"}}' >"$b/$L/run.json"
  done
  jq -c '.rows[] | select(.row | test("^D[123]/")) | {row: .row, verdict: "PASS", rep: 1}' $DECL >"$b/res.jsonl"
}
mk "$T/g"; chk "G6 green: all rows PASS, ALLEN engaged" 'bash $G "$T/g/res.jsonl" $DECL "$T/g" D1 D2 D3 >"$T/o1"; [ $? = 0 ] && grep -q "G6 PASS" "$T/o1"'
mk "$T/a"; jq '(.steps[0].ok) = false | (.allen.engaged_log_line) = ""' "$T/a/D1/run.json" >"$T/a/x" && mv "$T/a/x" "$T/a/D1/run.json"
chk "G6 red (negative control, the v6 failure): S1 refused and no ALLEN line" 'bash $G "$T/a/res.jsonl" $DECL "$T/a" D1 D2 D3 >"$T/o2"; [ $? = 1 ] && grep -q "step S1 not ok" "$T/o2" && grep -q "ALLEN: engaged" "$T/o2"'
mk "$T/b"; sed -i 's|\("row":"D2/v1 Correctness: recall","verdict":"\)PASS|\1FAIL|' "$T/b/res.jsonl"
chk "G6 red: the recall row FAILs although the other four rows v6 checked pass" 'bash $G "$T/b/res.jsonl" $DECL "$T/b" D1 D2 D3 >"$T/o3"; [ $? = 1 ] && grep -q "D2 row .v1 Correctness: recall. = FAIL" "$T/o3"'
mk "$T/c"; grep -v '"row":"D3/D3-MEAS"' "$T/c/res.jsonl" >"$T/c/x" && mv "$T/c/x" "$T/c/res.jsonl"
chk "G6 red: a missing required row is a failure" 'bash $G "$T/c/res.jsonl" $DECL "$T/c" D1 D2 D3 >"$T/o4"; [ $? = 1 ] && grep -q "D3-MEAS. = missing" "$T/o4"'
mk "$T/d"; sed -i 's|\("row":"D1/D1-RQ1","verdict":"\)PASS|\1FAIL|; s|\("row":"D1/D1-CUT","verdict":"\)PASS|\1FAIL|' "$T/d/res.jsonl"
chk "G6 green: model-behaviour rows (RQ, CUT, F, SB) may FAIL" 'bash $G "$T/d/res.jsonl" $DECL "$T/d" D1 D2 D3 >"$T/o5"; [ $? = 0 ]'
mk "$T/e"; jq '(.steps[4].ok) = false' "$T/e/D2/run.json" >"$T/e/x" && mv "$T/e/x" "$T/e/D2/run.json"
sed -i 's|\("row":"D2/Q1","verdict":"\)PASS|\1FAIL|' "$T/e/res.jsonl"
chk "G6 green: a launch whose model reply was refused (S5 not ok) is checked on product rows only" 'bash $G "$T/e/res.jsonl" $DECL "$T/e" D1 D2 D3 >"$T/o6"; [ $? = 0 ] && grep -q "D2 committed=false" "$T/o6"'
mk "$T/f"; sed -i 's|\("row":"D1/D1-BE","verdict":"\)PASS|\1FAIL|' "$T/f/res.jsonl"
chk "G6 red: a product row (BE) FAILs" 'bash $G "$T/f/res.jsonl" $DECL "$T/f" D1 D2 D3 >"$T/o7"; [ $? = 1 ]'
# ---- gate G6 corrections after v7 dry attempt 2 (2026-10-11, evidence-v7/attempt-2-g6-fail/)
mk "$T/h"; jq '(.steps[3].ok) = false | (.steps[4].ok) = false' "$T/h/D1/run.json" >"$T/h/x" && mv "$T/h/x" "$T/h/D1/run.json"
sed -i 's|\("row":"D1/v1 Memory","verdict":"\)PASS|\1FAIL|' "$T/h/res.jsonl"
chk "G6 green: v1 Memory FAILs on a launch that did not commit (no record to digest, v6 class (a))" 'bash $G "$T/h/res.jsonl" $DECL "$T/h" D1 D2 D3 >"$T/o8"; [ $? = 0 ] && grep -q "D1 committed=false" "$T/o8"'
mk "$T/i"; sed -i 's|\("row":"D1/v1 Memory","verdict":"\)PASS|\1FAIL|' "$T/i/res.jsonl"
chk "G6 red: v1 Memory FAILs on a committed launch" 'bash $G "$T/i/res.jsonl" $DECL "$T/i" D1 D2 D3 >"$T/o9"; [ $? = 1 ] && grep -q "D1 row .v1 Memory. = FAIL" "$T/o9"'
mk "$T/j"; sed -i 's|\("row":"D3/D3-DEC","verdict":"\)PASS|\1FAIL|; s|\("row":"D3/D3-MEAS","verdict":"\)PASS|\1FAIL|' "$T/j/res.jsonl"
jq -n '{proposal_attempts: [{attempt: 1, outcome: "refused"}, {attempt: 2, outcome: "timeout", tokens: 0}]}' >"$T/j/D3/s3-report.json"
chk "G6 green with a NOTE: -DEC and -MEAS FAIL only because an attempt ended by timeout (model too slow for the budget)" 'bash $G "$T/j/res.jsonl" $DECL "$T/j" D1 D2 D3 >"$T/o10"; [ $? = 0 ] && grep -q "NOTE D3 attempt 2 ended by timeout" "$T/o10" && grep -q "D3 committed=true .* not_checked_timeout=2" "$T/o10"'
mk "$T/k"; sed -i 's|\("row":"D3/D3-DEC","verdict":"\)PASS|\1FAIL|' "$T/k/res.jsonl"
jq -n '{proposal_attempts: [{attempt: 1, outcome: "parsed"}]}' >"$T/k/D3/s3-report.json"
chk "G6 red: -DEC FAILs with no timeout in the attempts" 'bash $G "$T/k/res.jsonl" $DECL "$T/k" D1 D2 D3 >"$T/o11"; [ $? = 1 ] && grep -q "D3 row .D3-DEC. = FAIL" "$T/o11"'
echo "test-v7: $pass ok, $fail failed"; [ $fail = 0 ]
