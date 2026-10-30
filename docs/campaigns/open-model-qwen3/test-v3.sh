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

# OPEN-MODEL-QWEN3 v3 (ACCEPTANCE-v3.md Section 9, FROZEN): CPU-only checks of the v3 files.
# No model, no GPU, no quietlock. Shell + jq only. Exit 0 only if every check passes.
#   1 declaration is the generated one, 156 rows, ids unique
#   2 every positive task row is a line of ACCEPTANCE-v3.md in the exact format make-receipt.sh greps for
#   3 scorer dry run on the declaration with made-up result lines (all PASS, one missing, one FAIL)
#   4 extra rows on synthetic run directories (made-up replies): positive and named negative cases
#   5 wrapper refusals (exit 2 or 3 before any run)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SC=$HERE/../scoring
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }
chk() { if eval "$2"; then ok; else bad "$1"; fi; }

# ---- 1 declaration
bash "$HERE/gen-decl-oq3-v3.sh" "$T/decl.json"
chk "declaration equals the generated one" 'cmp -s "$T/decl.json" "$SC/declarations/oq3-v3.decl.json"'
chk "156 declared rows" '[ "$(jq ".rows|length" "$SC/declarations/oq3-v3.decl.json")" = 156 ]'
chk "row ids unique" '[ "$(jq ".rows|map(.row)|(length==(unique|length))" "$SC/declarations/oq3-v3.decl.json")" = true ]'
chk "21 extra rows, 135 rows of the v8 shape" '[ "$(jq "[.rows[]|select(.row|test(\"-(SB|RQ|D)\$\") or test(\"^G[0-9]/G[0-9]-F\$\") or . == \"N2/N2-R\")]|length" "$SC/declarations/oq3-v3.decl.json")" = 21 ]'

# ---- 2 task rows in ACCEPTANCE-v3.md
if [ -f "$HERE/ACCEPTANCE-v3.md" ]; then
  while IFS= read -r row; do
    chk "ACCEPTANCE-v3.md has task row: ${row:0:50}" 'grep -Fqx -- "$row" "$HERE/ACCEPTANCE-v3.md"'
  done < <(jq -r '.tasks[] | select(.kind != "negative-boundary" and .kind != "negative-budget") | "| \(.id) | `\(.goal)` | `\(.destination)` | \(.phrases | map("`" + . + "`") | join(", ")) |"' "$HERE/tasks-oq3-v3.json")
  chk "ACCEPTANCE-v3.md says Status: FROZEN" 'grep -q "^\*\*Status: FROZEN\*\*" "$HERE/ACCEPTANCE-v3.md"'
fi
chk "no goal contains a backtick or a pipe" '! jq -r ".tasks[].goal" "$HERE/tasks-oq3-v3.json" | grep -q "[\`|]"'
chk "no goal of v8 or the diagnostic is reused" '! jq -r ".tasks[].goal" "$HERE/tasks-oq3-v3.json" | grep -Fxf <(jq -r ".tasks[].goal" "$HERE/../next-phase-1/tasks-v8.json") | grep -q .'

# ---- 3 scorer dry run (made-up result lines)
jq -r '.rows[].row' "$SC/declarations/oq3-v3.decl.json" | jq -R -c '{row: ., rep: 1, verdict: "PASS", receipt: "made-up"}' >"$T/res-all.jsonl"
bash "$SC/score-rows.sh" "$SC/declarations/oq3-v3.decl.json" "$T/res-all.jsonl" >"$T/s1.json"; rc=$?
chk "scorer: all 156 made-up PASS lines give PASS" '[ $rc = 0 ] && [ "$(jq -r .verdict "$T/s1.json")" = PASS ]'
sed '5d' "$T/res-all.jsonl" >"$T/res-missing.jsonl"
bash "$SC/score-rows.sh" "$SC/declarations/oq3-v3.decl.json" "$T/res-missing.jsonl" >"$T/s2.json"
chk "scorer: one missing row is not PASS" '[ "$(jq -r .verdict "$T/s2.json")" != PASS ]'
sed '7s/"PASS"/"FAIL"/' "$T/res-all.jsonl" >"$T/res-fail.jsonl"
bash "$SC/score-rows.sh" "$SC/declarations/oq3-v3.decl.json" "$T/res-fail.jsonl" >"$T/s3.json"
chk "scorer: one FAIL line gives FAIL" '[ "$(jq -r .verdict "$T/s3.json")" = FAIL ]'
{ cat "$T/res-all.jsonl"; echo '{"row":"G1/G1-SB","rep":1,"verdict":"PASS"}'; } >"$T/res-extra.jsonl"
bash "$SC/score-rows.sh" "$SC/declarations/oq3-v3.decl.json" "$T/res-extra.jsonl" >"$T/s4.json"
chk "scorer: a duplicate result line is not PASS" '[ "$(jq -r .verdict "$T/s4.json")" != PASS ]'

# ---- 4 extra rows on synthetic run directories
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
# mk_run DIR ID DEST CONTENT REPLY [ATTEMPTS_JSON] [RECOGNIZED_JSON]: a made-up launch directory.
mk_run() {
  local d=$1 id=$2 dest=$3 content=$4 reply=$5 att=${6:-} rec=${7:-'[]'} ws
  mkdir -p "$d/ws/$(dirname "$dest")" "$d/steps"; ws=$(cd "$d/ws" && pwd -P)
  printf '%s\n' "$content" >"$d/ws/$dest"
  local h; h=$(sha256sum "$d/ws/$dest" | cut -d' ' -f1)
  [ -n "$att" ] || att=$(jq -n --arg r "$reply" '[{attempt: 1, outcome: "parsed", reason: null, tokens: 400, ms: 42000, finish_reason: "eos", text: $r, text_sha256: "x"}]')
  jq -n --argjson a "$att" --arg h "$h" --argjson rec "$rec" '{proposal_attempts: $a, proposal_content_sha256: $h, requirements_recognized: $rec}' >"$d/s3-report.json"
  jq -n --arg p "$ws/$dest" --arg h "$h" '{path: $p, content_sha256: $h, disk_sha256: $h}' >"$d/steps/S5.json"
  jq -n --arg h "$h" '{authorizations: [{id: 1, verified: true, text: ({content_sha256: $h} | tojson)}]}' >"$d/steps/S8.json"
}
rowres() { jq -r --arg r "$2" 'select(.row == $r) | .verdict' <<<"$1"; }
run_rows() { bash "$HERE/v3-rows.sh" "$1" "$2" "$HERE/tasks-oq3-v3.json" "$T" "made-up.json"; }

DOC=$'# Pipeline\n\n## Building\n```bash\nmake build\n```\nThis builds it.\n\n## Testing\n```bash\nmake test\n```\nThis tests it.\n\n## Packaging\n```bash\nmake dist\n```\nThis packs it.\n\n## Summary\nThe pipeline builds, tests and packs the project. It is small and easy to read.'
REPLY=$'filename: docs/PIPELINE.md\n```markdown\n'"$DOC"$'\n```'
RECG1='["at least 18 non-empty lines"]'
mk_run "$T/g1" G1 docs/PIPELINE.md "$DOC" "$REPLY" "" "$RECG1"
out=$(run_rows "$T/g1" G1)
for r in SB F RQ D; do chk "G1 good launch: G1-$r PASS" '[ "$(rowres "$out" "G1/G1-$r")" = PASS ]'; done

# truncated at the first inner closing fence (the DIAGNOSTIC-LONG-1 D5 failure)
TRUNC=$'# Pipeline\n\n## Building\n```bash\nmake build'
mk_run "$T/g1t" G1 docs/PIPELINE.md "$TRUNC" "$REPLY" "" "$RECG1"
out=$(run_rows "$T/g1t" G1)
chk "G1 truncated at an inner fence: G1-F FAIL" '[ "$(rowres "$out" G1/G1-F)" = FAIL ]'
chk "G1 truncated: G1-RQ FAIL (too few lines)" '[ "$(rowres "$out" G1/G1-RQ)" = FAIL ]'
chk "G1 truncated: G1-SB still PASS (the bytes saved are the bytes approved)" '[ "$(rowres "$out" G1/G1-SB)" = PASS ]'

# balanced fences but nothing after the last example
NOTAIL=$'# Pipeline\n\n## Building\n```bash\nmake build\n```\n## Testing\n```bash\nmake test\n```\n## Packaging\n```bash\nmake dist\n```'
NOTAILR=$'filename: docs/PIPELINE.md\n```markdown\n'"$NOTAIL"$'\n```'
mk_run "$T/g1n" G1 docs/PIPELINE.md "$NOTAIL" "$NOTAILR" "" "$RECG1"
out=$(run_rows "$T/g1n" G1)
chk "G1 no text after the last example: G1-F FAIL" '[ "$(rowres "$out" G1/G1-F)" = FAIL ]'


# merged-parser outer-fence rule (sovereign-core 8f3e8c8): text after the outer close is not part of the file;
# an opener without any bare fence is dropped and the body kept; an unfenced reply is taken whole.
REPLYT="$REPLY"$'\nHope this helps.'
mk_run "$T/g1x" G1 docs/PIPELINE.md "$DOC" "$REPLYT" "" "$RECG1"
out=$(run_rows "$T/g1x" G1)
chk "G1 prose after the outer close is not in the file: G1-F PASS" '[ "$(rowres "$out" G1/G1-F)" = PASS ]'
REPLYU=$'filename: docs/PIPELINE.md\n'"$DOC"
mk_run "$T/g1u" G1 docs/PIPELINE.md "$DOC" "$REPLYU" "" "$RECG1"
out=$(run_rows "$T/g1u" G1)
chk "G1 unfenced reply taken whole: G1-F PASS" '[ "$(rowres "$out" G1/G1-F)" = PASS ]'
PLAIN=$'# Pipeline\n\n## Building\nRun make build.\n\n## Summary\nIt builds.'
REPLYO=$'FILENAME: docs/PIPELINE.md\n\n```md\n'"$PLAIN"
mk_run "$T/g1o" G1 docs/PIPELINE.md "$PLAIN" "$REPLYO" "" "$RECG1"
out=$(run_rows "$T/g1o" G1)
chk "G1 opener with no close: reading equals the saved file (opener dropped, body whole)" '[ "$(jq -r ".rows[]|select(.row==\"G1-F\")|.value.saved_equals_reply_document" "$T/$(jq -r "select(.row==\"G1/G1-F\")|.extra_rows_file" <<<"$out")")" = true ]'
# saved bytes differ from approved bytes
mk_run "$T/g1b" G1 docs/PIPELINE.md "$DOC" "$REPLY" "" "$RECG1"
jq '.authorizations[0].text = ({content_sha256: "0000"} | tojson)' "$T/g1b/steps/S8.json" >"$T/g1b/x" && mv "$T/g1b/x" "$T/g1b/steps/S8.json"
out=$(run_rows "$T/g1b" G1)
chk "G1 approved digest differs from the file: G1-SB FAIL" '[ "$(rowres "$out" G1/G1-SB)" = FAIL ]'
mk_run "$T/g1c" G1 docs/PIPELINE.md "$DOC" "$REPLY" "" "$RECG1"
printf 'changed after approval\n' >"$T/g1c/ws/docs/PIPELINE.md"
out=$(run_rows "$T/g1c" G1)
chk "G1 file changed after approval: G1-SB FAIL" '[ "$(rowres "$out" G1/G1-SB)" = FAIL ]'

# outcome-based RQ (ACCEPTANCE-v3 Section 12): runtime recognition is information, not a condition
mk_run "$T/g1r" G1 docs/PIPELINE.md "$DOC" "$REPLY" "" '[]'
out=$(run_rows "$T/g1r" G1)
chk "G1 requirement not recognized by the runtime but met by the saved bytes: G1-RQ PASS" '[ "$(rowres "$out" G1/G1-RQ)" = PASS ]'
chk "G1 recognition is recorded as information in the side-file" '[ "$(jq -c ".rows[]|select(.row==\"G1-RQ\")|.value.recognized_by_runtime" "$T/$(jq -r "select(.row==\"G1/G1-RQ\")|.extra_rows_file" <<<"$out")")" = "[]" ]'

# deadline: attempts over the shared budget, and a timeout reason
ATT=$(jq -n --arg r "$REPLY" '[{attempt: 1, outcome: "refused", reason: "model proposal exceeded 30000 ms", tokens: 300, ms: 61000, finish_reason: "eos"},
                              {attempt: 2, outcome: "parsed", reason: null, tokens: 400, ms: 60000, finish_reason: "eos", text: $r}]')
mk_run "$T/g1d" G1 docs/PIPELINE.md "$DOC" "$REPLY" "$ATT" "$RECG1"
out=$(run_rows "$T/g1d" G1)
chk "G1 attempts total 121000 ms over a 120000 ms deadline, one timed out: G1-D FAIL" '[ "$(rowres "$out" G1/G1-D)" = FAIL ]'
ATT2=$(jq '.[0].ms = 30000 | .[1].ms = 60000' <<<"$ATT")
mk_run "$T/g1e" G1 docs/PIPELINE.md "$DOC" "$REPLY" "$ATT2" "$RECG1"
out=$(run_rows "$T/g1e" G1)
chk "G1 within the deadline but an attempt ended by timeout: G1-D FAIL" '[ "$(rowres "$out" G1/G1-D)" = FAIL ]'
ATT3=$(jq '.[0].reason = "unmet requirement: at least 18 non-empty lines, found 9" | .[0].ms = 30000' <<<"$ATT")
mk_run "$T/g1f" G1 docs/PIPELINE.md "$DOC" "$REPLY" "$ATT3" "$RECG1"
out=$(run_rows "$T/g1f" G1)
chk "G1 first attempt refused for an unmet requirement, second accepted, within 120 s: G1-D PASS" '[ "$(rowres "$out" G1/G1-D)" = PASS ]'

# G3: outcome-based RQ, 36 lines, six named headings
G3=$'# Onboarding\n## Welcome\na\nb\n## Accounts and access\na\nb\n## Tools to install\na\nb\n## Your first week\na\nb\n## Who to ask\na\nb\n## Glossary\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns'
G3R=$'filename: docs/ONBOARDING.md\n'"$G3"
mk_run "$T/g3" G3 docs/ONBOARDING.md "$G3" "$G3R" "" '["at least 35 non-empty lines","at least 6 headings"]'
out=$(run_rows "$T/g3" G3)
chk "G3 unfenced reply taken whole: G3-F PASS" '[ "$(rowres "$out" G3/G3-F)" = PASS ]'
chk "G3 36 lines, 6 named headings, recognition recorded: G3-RQ PASS" '[ "$(rowres "$out" G3/G3-RQ)" = PASS ]'
mk_run "$T/g3b" G3 docs/ONBOARDING.md "$G3" "$G3R" "" '[]'
out=$(run_rows "$T/g3b" G3)
chk "G3 36 lines, 6 named headings, recognition EMPTY (the product's real behaviour): G3-RQ PASS" '[ "$(rowres "$out" G3/G3-RQ)" = PASS ]'
G3S=$(printf '%s\n' "$G3" | head -n 30)
mk_run "$T/g3c" G3 docs/ONBOARDING.md "$G3S" $'filename: docs/ONBOARDING.md\n'"$G3S" "" '[]'
out=$(run_rows "$T/g3c" G3)
chk "G3 30 lines (all six headings present): G3-RQ FAIL" '[ "$(rowres "$out" G3/G3-RQ)" = FAIL ]'
G3F=$'# Onboarding\n```markdown\n## Welcome\na\nb\n## Accounts and access\na\nb\n## Tools to install\na\nb\n## Your first week\na\nb\n## Who to ask\na\nb\n## Glossary\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\n```\nEnd of file.'
mk_run "$T/g3d" G3 docs/ONBOARDING.md "$G3F" $'filename: docs/ONBOARDING.md\n'"$G3F" "" '[]'
out=$(run_rows "$T/g3d" G3)
chk "G3 headings only inside a code fence (36+ lines): G3-RQ FAIL" '[ "$(rowres "$out" G3/G3-RQ)" = FAIL ]'
G3M=$(printf '%s\n' "$G3" | sed 's/^## Glossary$/Glossary/')
mk_run "$T/g3e" G3 docs/ONBOARDING.md "$G3M" $'filename: docs/ONBOARDING.md\n'"$G3M" "" '[]'
out=$(run_rows "$T/g3e" G3)
chk "G3 one named section missing as a heading: G3-RQ FAIL" '[ "$(rowres "$out" G3/G3-RQ)" = FAIL ]'
# nothing saved: no file, no authorization
mkdir -p "$T/g3n/ws" "$T/g3n/steps"; echo '{"proposal_attempts":[],"requirements_recognized":["at least 35 non-empty lines","at least 6 headings"]}' >"$T/g3n/s3-report.json"
out=$(run_rows "$T/g3n" G3)
chk "G3 nothing saved (even if recognized): G3-RQ FAIL" '[ "$(rowres "$out" G3/G3-RQ)" = FAIL ]'

# negatives
NAT=$(jq -n '[{attempt: 1, outcome: "refused", reason: "reply cut at the token limit after 16 tokens (finish_reason max_tokens); a cut reply is never a proposal", tokens: 16, ms: 2100, finish_reason: "max_tokens"}]')
mkdir -p "$T/n2/steps" "$T/n2/ws"; jq -n --argjson a "$NAT" '{proposal_attempts: $a}' >"$T/n2/s3-report.json"
out=$(run_rows "$T/n2" N2)
chk "N2 token-limit cut: N2-R PASS" '[ "$(rowres "$out" N2/N2-R)" = PASS ]'
chk "N2 token-limit cut: N2-D PASS" '[ "$(rowres "$out" N2/N2-D)" = PASS ]'
jq -n --argjson a "$(jq '.[0].reason = "model proposal exceeded 120000 ms" | .[0].finish_reason = "eos"' <<<"$NAT")" '{proposal_attempts: $a}' >"$T/n2/s3-report.json"
out=$(run_rows "$T/n2" N2)
chk "N2 a timeout is not a token-limit cut: N2-R FAIL" '[ "$(rowres "$out" N2/N2-R)" = FAIL ]'

# edit launch
E1S=$'# Roadmap\n\n## Planned\n- ship search\n- add dark mode\n\n## Shipped\n- public beta'
mk_run "$T/e1" E1 ROADMAP.md "$E1S" $'filename: ROADMAP.md\n'"$E1S" "$(jq -n '[{attempt: 1, outcome: "parsed", reason: null, tokens: 40, ms: 4000, finish_reason: "eos", text: "x"}]')"
out=$(run_rows "$T/e1" E1)
chk "E1 edit: E1-SB and E1-D PASS" '[ "$(rowres "$out" E1/E1-SB)" = PASS ] && [ "$(rowres "$out" E1/E1-D)" = PASS ]'
mk_run "$T/e1b" E1 ROADMAP.md "$E1S" x "$(jq -n '[{attempt: 1, outcome: "parsed", reason: null, tokens: 40, ms: 29500, finish_reason: "eos", text: "x"}]')"
out=$(run_rows "$T/e1b" E1)
chk "E1 edit over 29000 ms: E1-D FAIL" '[ "$(rowres "$out" E1/E1-D)" = FAIL ]'
chk "row count: G1 gives 4 extra rows, E1 gives 2, N2 gives 2" '[ "$(run_rows "$T/g1" G1 | wc -l)" = 4 ] && [ "$(run_rows "$T/e1" E1 | wc -l)" = 2 ] && [ "$(run_rows "$T/n2" N2 | wc -l)" = 2 ]'

# ---- 5 wrapper refusals (CPU, before any run)
W=$HERE/run-qwen3-v3.sh
PINNED=/home/drakestapleton/workspace/oq3-v3-runs
if [ -e "$PINNED" ]; then bad "pinned run path already exists, wrapper checks skipped"; else
  : >"$T/x"; chmod +x "$T/x"
  base=(env -i PATH="$PATH" HOME="$HOME")
  okenv=(AIEN_BIN="$T/x" AIEN_MODEL_PATH="$T/m.json" AIEN_TOKENIZER_PATH="$T/t.json" AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1)
  wr() { local b=$1; shift; "${base[@]}" "$@" bash "$W" "$b" "$T/out" 0000 1111 2222 "$T/x" "$T/x" >/dev/null 2>"$T/w.err"; echo $?; }
  chk "wrapper refuses a missing OQ3_PART" '[ "$(wr "$T/b")" != 0 ]'
  chk "wrapper refuses part 4 (exit 2)" '[ "$(wr "$T/b" OQ3_PART=4)" = 2 ]'
  chk "wrapper refuses a run base other than the pinned path (exit 2)" '[ "$(wr "$T/b" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses commit arguments that are not the frozen ones (exit 2), pinned path not created" '[ "$(wr "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "must equal the frozen commits" "$T/w.err"'
  chk "ACCEPTANCE-v3.md Section 7 build identity equals the wrapper FROZEN_* lines" '
    for kv in "sovereign-core = $(sed -n "s/^FROZEN_SC_COMMIT=//p" "$W")" "omega.lock     = $(sed -n "s/^FROZEN_OMEGA_COMMIT=//p" "$W")" "aien-cli       sha256 $(sed -n "s/^FROZEN_AIEN_CLI_SHA256=//p" "$W")" "np1_reference  sha256 $(sed -n "s/^FROZEN_NP1_REFERENCE_SHA256=//p" "$W")" "np1_edit_merge sha256 $(sed -n "s/^FROZEN_NP1_EDIT_MERGE_SHA256=//p" "$W")"; do
      grep -q -- "^$kv" "$HERE/ACCEPTANCE-v3.md" || { echo "missing: $kv"; false; break; }
    done'
  chk "wrapper refuses a binary whose sha256 differs (the three chk lines are present)" 'grep -q "chk \"\$AIEN_BIN\" \"\$FROZEN_AIEN_CLI_SHA256\"" "$W" && grep -q "chk \"\$REFBIN\" \"\$FROZEN_NP1_REFERENCE_SHA256\"" "$W" && grep -q "chk \"\$MERGEBIN\" \"\$FROZEN_NP1_EDIT_MERGE_SHA256\"" "$W"'
  chk "wrapper refuses changed dry-run task shape (exit 2)" '
    jq ".tasks[0].max_tokens = 512" "$HERE/tasks-oq3-v3.json" >"$T/dry-bad.json"
    [ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$T/dry-bad.json" "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses a retired AIEN_COMPOSE_BUDGET_MS (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v3.json" AIEN_COMPOSE_BUDGET_MS=29000 "${okenv[@]}")" = 3 ] && [ ! -e "$T/b" ]'
  chk "wrapper refuses a preset AIEN_COMPOSE_DOC_MAX_TOKENS (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v3.json" AIEN_COMPOSE_DOC_MAX_TOKENS=1024 "${okenv[@]}")" = 3 ]'
  chk "wrapper refuses a wrong KV context (exit 3)" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v3.json" "${okenv[@]}" AIEN_KV_CONTEXT_TOKENS=2048)" = 3 ]'
  chk "wrapper refuses a model file with a wrong digest (exit 3), base not created" '[ "$(wr "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v3.json" "${okenv[@]}")" = 3 ] && [ ! -e "$T/b" ]'
fi

# ---- 6 the unchanged receipt builder, staged as the wrapper stages it, yields exactly the declared v8-shape rows
NP1=$(cd "$HERE/../next-phase-1" && pwd); STAGE=$T/stage; mkdir -p "$STAGE" "$T/rec"
for f in "$NP1"/*; do n=$(basename "$f"); [ "$n" = tasks-v8.json ] && continue; ln -s "$f" "$STAGE/$n"; done
cp "$HERE/tasks-oq3-v3.json" "$STAGE/tasks-v8.json"; ln -s "$HERE/seed-v3" "$STAGE/seed-v3"
for id in $(jq -r '.tasks[].id' "$HERE/tasks-oq3-v3.json"); do
  R=$T/stg-$id; mkdir -p "$R/ws" "$R/steps" "$R/prov"; echo '{"steps":[],"daemon":[],"containment":{}}' >"$R/run.json"
  seed=$(jq -r --arg i "$id" '.tasks[]|select(.id==$i)|.seed//empty' "$HERE/tasks-oq3-v3.json"); [ -n "$seed" ] && cp -R "$HERE/$seed/." "$R/ws/"
  e=(); [ "${id#N}" = "$id" ] && e=(TASK_ID="$id" TASK_SPEC="$HERE/tasks-oq3-v3.json" TASK_ACCEPTANCE=/dev/null)
  env "${e[@]}" V6_ROWS=rows-v8.jq V6_TASK="$id" V6_MAX_TOKENS=1 V8_MERGE=null bash "$STAGE/make-receipt.sh" "$R" "$T/rec" sc omc omg 0 n >/dev/null 2>&1
  rec=$(ls -t "$T/rec"/*.json | head -1)
  xs="SB|D"; [ "$(jq -r --arg i "$id" ".tasks[]|select(.id==\$i)|.kind" "$HERE/tasks-oq3-v3.json")" = long ] && xs="SB|F|RQ|D"
  bash "$NP1/v8-results.sh" "$id" "$rec" | jq -r .row | sort >"$T/$id.got"
  jq -r --arg i "$id" '.rows[].row | select(startswith($i + "/"))' "$SC/declarations/oq3-v3.decl.json" | grep -vE "^$id/$id-($xs)\$" | grep -v "^N2/N2-R\$" | sort >"$T/$id.want"
  chk "launch $id: receipt rows == declared v8-shape rows" 'cmp -s "$T/$id.got" "$T/$id.want"'
done

echo "test-v3: $pass passed, $fail failed"
[ $fail = 0 ]
