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

# OPEN-MODEL-QWEN3 v6 self-test (QUALIFICATION-v6.md Section 7, CPU only, no model, no GPU). Shell + jq.
# v6 reuses every v5 campaign file byte for byte; this test proves that, proves the v6 wrapper differs from the v5
# wrapper only in names, and checks the v6 wrapper's refusals. The v5 self-test (test-v5.sh) still runs the row
# red and green cases; after the v5 run it reports exactly one expected failure (its pinned run path exists).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE" || exit 2
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { pass=$((pass + 1)); echo "ok: $1"; }
bad() { fail=$((fail + 1)); echo "FAIL: $1"; }
chk() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
FROZEN=$HERE/frozen-v6.json W=$HERE/run-qwen3-v6.sh PINNED=/home/drakestapleton/workspace/oq3-v6-runs

# ---- 1 every v5 campaign file is unchanged (evidence-v5/campaign-files.sha256, written at the v5 freeze)
chk "every v5 campaign file (wrapper, tasks, declaration, rows, generator, self-test, frozen values, driver, receipt builder, scorer, seeds, self-test fixtures) equals its v5 freeze sha256" \
  'sha256sum --quiet -c evidence-v5/campaign-files.sha256 >/dev/null 2>&1'
chk "the declaration v6 scores against is oq3-v5.decl.json with QUALIFICATION_ROWS=334" \
  '[ "$(jq ".rows | length" ../scoring/declarations/oq3-v5.decl.json)" = 334 ] && grep -q "DECL=\$HERE/../scoring/declarations/oq3-v5.decl.json" "$W"'

# ---- 2 the v6 wrapper is the v5 wrapper with names changed only
sed -e 's#FROZEN=\$HERE/frozen-v5.json#FROZEN=$HERE/frozen-v6.json#' -e 's#TASKS=\$HERE/tasks-oq3-v5.json TAG=oq3-v5$#TASKS=$HERE/tasks-oq3-v5.json TAG=oq3-v6#' \
  -e 's#TAG=oq3-v5-DRYRUN#TAG=oq3-v6-DRYRUN#' -e 's#PINNED_BASE=/home/drakestapleton/workspace/oq3-v5-runs#PINNED_BASE=/home/drakestapleton/workspace/oq3-v6-runs#' \
  -e 's#run-qwen3-v5:#run-qwen3-v6:#g' -e 's#(the v5 run is fresh and happens once)#(the v6 run is fresh and happens once)#' \
  -e 's#are not frozen yet (frozen-v5.json, ACCEPTANCE-v5 Section 7)#are not frozen yet (frozen-v6.json, QUALIFICATION-v6 Section 6)#' \
  -e 's#frozen-v5.json status is not FROZEN (ACCEPTANCE-v5 Section 9, gate G8)#frozen-v6.json status is not FROZEN (QUALIFICATION-v6 Section 7)#' \
  -e 's#differ from frozen-v5.json:#differ from frozen-v6.json:#' \
  -e 's#SPEC="open-model-qwen3/ACCEPTANCE-v5.md spec_version 5#SPEC="open-model-qwen3/QUALIFICATION-v6.md (rows of ACCEPTANCE-v5.md) spec_version 6#' \
  -e 's#"OPEN-MODEL-QWEN3 v5\${OQ3_DRY_TASKS:+ DRY RUN} launch \$id"#"OPEN-MODEL-QWEN3 v6${OQ3_DRY_TASKS:+ DRY RUN} launch $id"#' \
  -e 's#echo "OPEN-MODEL-QWEN3 v5\${OQ3_DRY_TASKS#echo "OPEN-MODEL-QWEN3 v6${OQ3_DRY_TASKS#' run-qwen3-v5.sh | sed '1,3d' >"$T/expect.sh"
chk "run-qwen3-v6.sh = run-qwen3-v5.sh with only the listed name changes and a new header (lines after the header identical)" \
  'cmp -s "$T/expect.sh" <(sed "1,8d" "$W")'
chk "the v6 wrapper still requires the GB10 path (AIEN_REQUIRE_BLACKWELL=1) and the v5 tasks, seeds and row tooling" \
  'grep -q "need AIEN_REQUIRE_BLACKWELL 1" "$W" && grep -q "TASKS=\$HERE/tasks-oq3-v5.json TAG=oq3-v6" "$W" && grep -q "seed-v5" "$W" && grep -q "v5-rows.sh\" \"\$BASE/\$id\" \"\$id\" \"\$TASKS\" \"\$OUT\"" "$W"'

# ---- 3 frozen-v6.json: model values identical to v5, pins filled (40 or 64 lowercase hex)
chk "frozen-v6.json model block == frozen-v5.json model block (same weights, same digests)" '[ "$(jq -c .model frozen-v6.json)" = "$(jq -c .model frozen-v5.json)" ]'
chk "frozen-v6.json has the same eight pin names as v5" '[ "$(jq -c ".pins | keys" frozen-v6.json)" = "$(jq -c ".pins | keys" frozen-v5.json)" ]'
chk "every v6 pin is filled with 40 or 64 lowercase hex" '[ "$(jq "[.pins[] | select(test(\"^([0-9a-f]{40}|[0-9a-f]{64})$\") | not)] | length" frozen-v6.json)" = 0 ]'

# ---- 4 v6 wrapper refusals (CPU, before any run); the pinned path is never created
if [ -e "$PINNED" ]; then bad "pinned run path already exists, wrapper checks skipped"; else
  : >"$T/x"; chmod +x "$T/x"
  base=(env -i PATH="$PATH" HOME="$HOME")
  okenv=(AIEN_BIN="$T/x" AIEN_MODEL_PATH="$T/m.json" AIEN_TOKENIZER_PATH="$T/t.json" AIEN_KV_CONTEXT_TOKENS=4096 AIEN_REQUIRE_BLACKWELL=1 AIEN_REQUIRE_CHECKPOINT=1 AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1)
  WC=$T/wcopy/oq3; mkdir -p "$WC"; cp "$W" "$WC/run-qwen3-v6.sh"
  wr() { local w=$1 b=$2; shift 2; "${base[@]}" "$@" bash "$w" "$b" "$T/out" "$T" "$T" "$T/x" "$T/x" >/dev/null 2>"$T/w.err"; echo $?; }
  chk "refuses a missing OQ3_PART (exit 2)" '[ "$(wr "$W" "$T/b")" = 2 ]'
  chk "refuses a run base other than oq3-v6-runs (exit 2), nothing created" '[ "$(wr "$W" "$T/b" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$T/b" ] && grep -q "RUN_BASE must be $PINNED" "$T/w.err"'
  jq '.pins |= map_values("TO FILL AT FREEZE")' "$FROZEN" >"$WC/frozen-v6.json"
  chk "refuses a real run while the pins are the placeholder (exit 2)" '[ "$(wr "$WC/run-qwen3-v6.sh" "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "pins are not frozen yet" "$T/w.err"'
  jq '.status = "DRAFT"' "$FROZEN" >"$WC/frozen-v6.json"
  chk "refuses a real run while the status is not FROZEN (exit 2)" '[ "$(wr "$WC/run-qwen3-v6.sh" "$PINNED" OQ3_PART=1 "${okenv[@]}")" = 2 ] && [ ! -e "$PINNED" ] && grep -q "status is not FROZEN" "$T/w.err"'
  for v in AIEN_REQUIRE_BLACKWELL=0 AIEN_KV_CONTEXT_TOKENS=2048 AIEN_REQUIRE_CHECKPOINT=0; do
    chk "refuses $v (exit 3)" '[ "$(wr "$W" "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v5.json" "${okenv[@]}" "$v")" = 3 ] && [ ! -e "$T/b" ]'
  done
  for v in AIEN_FORCE_CPU_STUB=1 AIEN_DEV_FALLBACK=1 AIEN_COMPOSE_DOC_BUDGET_MS=600000 AIEN_COMPOSE_DOC_MAX_TOKENS=4096; do
    chk "refuses a preset ${v%%=*} (exit 3)" '[ "$(wr "$W" "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v5.json" "${okenv[@]}" "$v")" = 3 ] && [ ! -e "$T/b" ]'
  done
  chk "refuses a model index file with a wrong digest (exit 3)" '[ "$(wr "$W" "$T/b" OQ3_PART=1 OQ3_DRY_TASKS="$HERE/tasks-oq3-v5.json" "${okenv[@]}")" = 3 ] && grep -q "digest mismatch" "$T/w.err"'
fi
chk "no em dash or en dash in the v6 files" '! LC_ALL=C grep -lP "\xe2\x80[\x93\x94]" QUALIFICATION-v6.md V5-FAILURE-ANALYSIS.md run-qwen3-v6.sh test-v6.sh frozen-v6.json >/dev/null 2>&1'
echo "test-v6: $pass passed, $fail failed"
[ $fail = 0 ]
