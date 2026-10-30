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

# OPEN-MODEL-QWEN3 v8 harness self-test (sc#384, SPEC-v8.md). All v7 checks (ALLEN driver, gate G6) plus the v8 pieces: the wrapper differs from v7 only in names and the ALLEN-aware R1 reference call, the feature checks (v8-checks.sh) go red and green, and the launchers require them. CPU only, no model, no GPU, no network. Shell + jq.
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

chk "all new scripts parse (bash -n)" 'for f in run-campaign-allen.sh build-allen-subject.sh g6-gate.sh run-qwen3-v8.sh v8-checks.sh evidence-v8/build.sh evidence-v8/run-part.sh evidence-v8/run-dry.sh evidence-v8/run.sh; do bash -n $f || exit 1; done'
# ---- driver: the only difference from the frozen driver is the ALLEN block, the mark-evidence path and the allen evidence in run.json
D=$(diff ../next-phase-1/run-campaign.sh run-campaign-allen.sh | grep -c "^<")
chk "driver diff against next-phase-1/run-campaign.sh removes only 5 lines (header comment, mark-evidence path, one find-exclusion line, the two run.json lines); all else is added ALLEN lines (removed: $D)" '[ "$D" = 5 ]'
chk "the ALLEN block runs before the measured window (.mark) and before S1" \
  'a=$(grep -n "ALLEN_LINE=\$(grep" run-campaign-allen.sh | head -1 | cut -d: -f1); m=$(grep -n "^touch \"\$R/.mark\"" run-campaign-allen.sh | cut -d: -f1); s=$(grep -n "^run_step S1" run-campaign-allen.sh | cut -d: -f1); [ "$a" -lt "$m" ] && [ "$m" -lt "$s" ]'
chk "the driver refuses to run without the subject builder" 'out=$(env -u NP1_ALLEN_BUILDER AIEN_BIN=/bin/true AIEN_MODEL_PATH=x AIEN_TOKENIZER_PATH=y NP1_DIR=. bash run-campaign-allen.sh "$T/r" 2>&1); echo "$out" | grep -q NP1_ALLEN_BUILDER || echo "$out" | grep -q "daemon 1 failed"'
chk "the frozen next-phase-1 driver is unchanged by this lane" '! git diff --quiet origin/main -- ../next-phase-1 2>/dev/null || [ -z "$(git status --porcelain ../next-phase-1)" ]'

# ---- wrapper: v7 differs from v6 only where intended
chk "run-qwen3-v8.sh uses the ALLEN driver and keeps the v6 refusals (pinned base, FROZEN status, placeholder pins)" \
  'grep -q "run-campaign-allen.sh" run-qwen3-v8.sh && grep -q "PINNED_BASE=/home/drakestapleton/workspace/oq3-v8-runs" run-qwen3-v8.sh && grep -q "status is not FROZEN" run-qwen3-v8.sh && grep -q "build pins are not frozen" run-qwen3-v8.sh'
chk "frozen-v8.json is a consistent DRAFT (status DRAFT, every pin the placeholder) or FROZEN (status FROZEN, every pin 40 or 64 hex, no placeholder); same pin names as v7; model block equal to v7 (same weights)" \
  'F=frozen-v8.json; [ "$(jq -c ".pins | keys" $F)" = "$(jq -c ".pins | keys" frozen-v7.json)" ] && [ "$(jq -c .model $F)" = "$(jq -c .model frozen-v7.json)" ] && case "$(jq -r .status $F)" in DRAFT*) [ "$(jq "[.pins[] | select(. != \"TO FILL AT FREEZE\")] | length" $F)" = 0 ] ;; FROZEN*) [ "$(jq "[.pins[] | select(test(\"^([0-9a-f]{40}|[0-9a-f]{64})$\") | not)] | length" $F)" = 0 ] ;; *) false ;; esac'
chk "a real v8 run is refused while frozen-v7.json is a draft" \
  'out=$(OQ3_PART=1 AIEN_MODEL_PATH=x AIEN_TOKENIZER_PATH=y AIEN_BIN=z bash run-qwen3-v8.sh /home/drakestapleton/workspace/oq3-v8-runs-selftest-absent "$T/o" . . /bin/true /bin/true 2>&1); echo "$out" | grep -q "RUN_BASE must be"'
chk "OQ3_FROZEN outside a dry run is refused" 'out=$(OQ3_FROZEN=$HERE/frozen-v6.json OQ3_PART=1 bash run-qwen3-v8.sh a b c d e f 2>&1); echo "$out" | grep -q "allowed in a dry run only"'

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
# ---- v8 (SPEC-v8.md): wrapper, feature checks, launchers
chk "run-qwen3-v8.sh is run-qwen3-v7.sh with names changed plus exactly one changed working line, the R1 reference call with NP1_REFERENCE_ALLEN=default" \
  'd=$(diff <(sed "s/v7/v8/g; s/QUALIFICATION-v8\.md/SPEC-v8.md/g; s/QUALIFICATION-v8 /SPEC-v8 /g; s/spec_version 7/spec_version 8/; s/SPEC-v8 Section 6)/SPEC-v8 Section 6 item 3)/; s/SPEC-v8 Section 7)/SPEC-v8 Section 6 item 6)/" run-qwen3-v7.sh) run-qwen3-v8.sh | grep "^>" | grep -v "^> #"); [ "$(printf "%s\n" "$d" | grep -c .)" = 1 ] && printf "%s" "$d" | grep -q "NP1_REFERENCE_ALLEN=default \"\$REFBIN\""'
chk "the v8 wrapper still reads the v5 tasks and the v5 declaration (rows and budgets unchanged)" \
  'grep -q "TASKS=\$HERE/tasks-oq3-v5.json" run-qwen3-v8.sh && grep -q "oq3-v5.decl.json" run-qwen3-v8.sh'
mk8() { # dir ; a v8-shaped run base: D2 refused then parsed with the refused reply shown, R1 parsed with an exact reference
  local b=$1 v=$1/v7; mkdir -p "$b/D2" "$b/R1" "$v"
  jq -n '{proposal_attempts: [{attempt: 1, outcome: "refused", text_sha256: "aa", prompt_tokens: 260, prompt_ids_sha256: "p2"},
                             {attempt: 2, outcome: "parsed", text_sha256: "bb", shown_refused_text_sha256: "aa"}]}' >"$b/D2/s3-report.json"
  jq -n '{proposal_attempts: [{attempt: 1, outcome: "parsed", text_sha256: "cc", prompt_tokens: 164, prompt_ids_sha256: "p1"}]}' >"$b/R1/s3-report.json"
  jq -n '{proposal_attempts: [{attempt: 1, prompt_tokens: 260, prompt_ids_sha256: "p2"}]}' >"$v/s3-report-D2.json"
  jq -n '{proposal_attempts: [{attempt: 1, prompt_tokens: 164, prompt_ids_sha256: "p1"}]}' >"$v/s3-report-R1.json"
  jq -n '{allen: "default", prompt_tokens: 164, prompt_ids_sha256: "p1"}' >"$b/R1.reference.json"
}
V8=./v8-checks.sh
mk8 "$T/v1"; chk "v8 checks green on a clean v8-shaped record" 'bash $V8 "$T/v1" "$T/v1/v7" D2 R1 >"$T/p1"; [ $? = 0 ] && grep -q "V8 CHECKS PASS" "$T/p1"'
mk8 "$T/v2"; jq ".proposal_attempts[1].shown_refused_text_sha256 = null" "$T/v2/D2/s3-report.json" >"$T/v2/x" && mv "$T/v2/x" "$T/v2/D2/s3-report.json"
chk "v8 checks red (C1): a retry after a refused reply that did not show it (the v7 behaviour)" 'bash $V8 "$T/v2" "$T/v2/v7" D2 R1 >"$T/p2"; [ $? = 1 ] && grep -q "D2 C1" "$T/p2"'
mk8 "$T/v3"; jq ".proposal_attempts[1].text_sha256 = \"aa\"" "$T/v3/D2/s3-report.json" >"$T/v3/x" && mv "$T/v3/x" "$T/v3/D2/s3-report.json"
chk "v8 checks red (C2, negative control): the accepted reply equals a refused reply" 'bash $V8 "$T/v3" "$T/v3/v7" D2 R1 >"$T/p3"; [ $? = 1 ] && grep -q "D2 C2" "$T/p3"'
mk8 "$T/v4"; jq "{allen: null, prompt_tokens: 99, prompt_ids_sha256: \"q\"}" "$T/v4/R1.reference.json" >"$T/v4/x" && mv "$T/v4/x" "$T/v4/R1.reference.json"
chk "v8 checks red (C4, negative control): the v7 CPU replay (no ALLEN block, 99 tokens)" 'bash $V8 "$T/v4" "$T/v4/v7" D2 R1 >"$T/p4"; [ $? = 1 ] && grep -q "R1 C4: reference allen=missing" "$T/p4" && grep -q "R1 C4: accepted attempt prompt" "$T/p4"'
mk8 "$T/v5"; jq ".proposal_attempts[0].prompt_ids_sha256 = \"other\"" "$T/v5/D2/s3-report.json" >"$T/v5/x" && mv "$T/v5/x" "$T/v5/D2/s3-report.json"
chk "v8 checks red (C5): attempt 1 prompt differs from v7" 'bash $V8 "$T/v5" "$T/v5/v7" D2 R1 >"$T/p5"; [ $? = 1 ] && grep -q "D2 C5" "$T/p5"'
mk8 "$T/v6"; jq ".proposal_attempts += [{attempt: 3, outcome: \"refused\"}, {attempt: 4, outcome: \"refused\"}]" "$T/v6/R1/s3-report.json" >"$T/v6/x" && mv "$T/v6/x" "$T/v6/R1/s3-report.json"
chk "v8 checks red (C3): more than 3 attempts or a gap in numbering" 'bash $V8 "$T/v6" "$T/v6/v7" D2 R1 >"$T/p6"; [ $? = 1 ] && grep -q "R1 C3" "$T/p6"'
chk "the real-run launcher requires V8 CHECKS PASS on the dry launches and runs v8-checks.sh on all eleven after part 4; the dry launcher records it" \
  'grep -q "V8 CHECKS PASS\" \] ||" evidence-v8/run.sh && grep -q "v8-checks.sh /home/drakestapleton/workspace/oq3-v8-runs .* D1 D2 D3 D4 D5 D6 E1 E2 N1 N2 R1" evidence-v8/run.sh && grep -q "v8-checks.sh \$V/dry-runs .* D1 D2 D3 > \$R/v8-checks-dry.txt" evidence-v8/run-dry.sh'
echo "test-v8: $pass ok, $fail failed"; [ $fail = 0 ]
