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

# CAND-4 qualification harness (ACCEPTANCE-CAND4.md). Shell + jq only. Never retries a
# launch; every launch, failed or not, leaves an attempt record named by its sha256.
#
#   run-cand4.sh check   INPUTS                 recompute every pinned digest (exit 3 on a mismatch)
#   run-cand4.sh hygiene INPUTS OUT             section 3 (hygiene.sh)
#   run-cand4.sh q1      INPUTS RUN_BASE OUT    section 4: 3 rounds x (T1, T2, T3), GPU, one quietlock hold per round
#   run-cand4.sh q2      INPUTS RUN_BASE OUT    section 5: F0 on the GPU (one hold), then the cases on cpu_fault
#   run-cand4.sh q2w     INPUTS RUN_BASE OUT    section 6.1: coverage cases 2 and 5 (run-windows.sh on cpu_fault, after q2)
#
# INPUTS = ACCEPTANCE-CAND4.md (kind QUALIFICATION, refused while UNFROZEN) or a TRIAL inputs
# file (kind TRIAL: OUT must contain TRIAL and stay outside the evidence dir and the repo;
# CAND4_TRIAL_TASKS / CAND4_TRIAL_REPS / CAND4_TRIAL_CASES may narrow a TRIAL run only).
# OUT layout: q1/ (NEXT-PHASE-1 receipts + replies), q2/ (NEXT-PHASE-2 receipt), attempts/,
# scores/, hygiene/ ; every file there is named by the sha256 of its content.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
NP1=$REPO/docs/campaigns/next-phase-1 NP2=$REPO/docs/campaigns/next-phase-2 SCORE=$REPO/docs/campaigns/scoring/score-rows.sh
. "$HERE/inputs.sh"
CMD=${1:?check|hygiene|q1|q2}; INPUTS=${2:?INPUTS}
load_inputs "$INPUTS" || exit 3
KIND=$(inp kind)
SC=$(inp sc_commit) OC=$(inp omega_commit) MD=$(inp model_dir)
A1RULE=$(inp q1_a1_record_mark)
whisper() { crumb whisper laneQ "$REPO" - "$1" >/dev/null 2>&1 || echo "whisper failed: $1" >&2; }
refuse_unless_digests() { verify_digests || { echo "REFUSED: a pinned digest differs (no launch started)" >&2; exit 3; }; }

# content-addressed write: name_by_sha DIR PREFIX < json  -> prints the file name
name_by_sha() {
  local t; t=$(mktemp "$1/.tmp.XXXXXX"); cat >"$t"
  local h; h=$(sha256sum "$t" | cut -d' ' -f1); mv "$t" "$1/$2$h.json"; echo "$2$h.json"
}
# launch env: nothing from the caller's environment but these (ACCEPTANCE-CAND4 section 2)
clean_env() { env -i PATH="$PATH" HOME="$HOME" USER="${USER:-}" LANG="${LANG:-C.UTF-8}" "$@"; }
gpu_env() {
  clean_env AIEN_BIN="$(inp aien_cli_path)" AIEN_MODEL_PATH="$MD/model.safetensors" AIEN_TOKENIZER_PATH="$MD/tokenizer.json" \
    AIEN_COMPOSE_MAX_TOKENS="$(inp max_tokens)" AIEN_REQUIRE_BLACKWELL=1 "$@"
}
# hold MINUTES REASON CMD...: wait for the quiet flag, then run CMD inside our quietlock hold
# with start and release whispers. A hold that never started (the flag was taken first) is
# waited out and taken again; CMD itself is never run twice (it leaves a started marker).
hold() {
  local min=$1 why=$2; shift 2
  local mark; mark=$(mktemp -u "${TMPDIR:-/tmp}/cand4-started.XXXXXX")
  while :; do
    wait_quiet "$why"
    whisper "laneQ GPU start: $why"
    quietlock hold --owner laneQ --minutes "$min" --reason "$why" -- bash -c 'touch "$0"; exec "$@"' "$mark" "$@"
    local rc=$?
    whisper "laneQ GPU release: $why rc=$rc"
    [ -e "$mark" ] && { rm -f "$mark"; return $rc; }
    echo "quietlock: hold not taken (rc $rc), waiting" >&2; sleep 30
  done
}

# a1_read RECEIPT MARK lives in a1-read.sh (shared with selftest.sh).
. "$HERE/a1-read.sh"

case $CMD in
check)
  verify_digests; rc=$?
  jq -c '.[] | {name, ok, got: .got[0:16]}' <<<"$DIGESTS"
  [ $rc = 0 ] && echo "all pinned digests equal the frozen values" || { echo "DIGEST CHECK FAILED"; exit 3; }
  ;;

hygiene)
  OUT=${3:?OUT}; out_ok "$OUT" || exit 3
  bash "$HERE/hygiene.sh" "$INPUTS" "$OUT/hygiene"
  ;;

q1)
  BASE=${3:?RUN_BASE} OUT=${4:?OUT}; out_ok "$OUT" || exit 3
  case $A1RULE in strict|record-mark) ;; *) echo "REFUSED: q1_a1_record_mark must be strict or record-mark (is: $A1RULE)" >&2; exit 3;; esac
  mkdir -p "$BASE" "$OUT/q1" "$OUT/q1-a1" "$OUT/attempts" "$OUT/scores"; BASE=$(cd "$BASE" && pwd); OUT=$(cd "$OUT" && pwd)
  [ -e "$OUT/q1-results.jsonl" ] && { echo "REFUSED: Q1 already ran into $OUT (never repeated)" >&2; exit 3; }
  TASKS="T1 T2 T3" REPS=3
  if [ "$KIND" = TRIAL ]; then TASKS=${CAND4_TRIAL_TASKS:-$TASKS} REPS=${CAND4_TRIAL_REPS:-$REPS}; fi
  for r in $(seq 1 "$REPS"); do
    refuse_unless_digests
    hold 20 "CAND-4 Q1 round $r ($KIND)" bash "$0" _q1_round "$INPUTS" "$BASE" "$OUT" "$r" "$TASKS"
    # a declared launch of this round that left no result line did not run: FAIL, recorded, not retried
    touch "$OUT/q1-results.jsonl"
    for id in $TASKS; do
      jq -e --arg id "$id" --argjson r "$r" 'select(.row == $id and .rep == $r)' "$OUT/q1-results.jsonl" >/dev/null && continue
      a=$(jq -n --arg k "$KIND" --arg sc "$SC" --arg id "$id" --argjson r "$r" \
        '{gate:"CAND4_Q1_ATTEMPT", kind:$k, sc_commit:$sc, row:$id, rep:$r, env:"GPU", verdict:"FAIL",
          outcome:"launch did not run (the round ended before this task; see the hold output)"}' | name_by_sha "$OUT/attempts" q1-)
      jq -nc --arg id "$id" --argjson r "$r" --arg a "$a" '{row:$id, rep:$r, verdict:"FAIL", outcome:("launch did not run; attempt " + $a), env:"GPU"}' >>"$OUT/q1-results.jsonl"
      echo "Q1 $id rep $r: FAIL (launch did not run; attempts/$a)"
    done
  done
  # Section 4.1 negative controls, run as campaign rows on copies of the first real receipt the
  # reading accepted: a damaged mark must be rejected, and a receipt with one more failing row
  # must be rejected. A control row is PASS iff the reading says FAIL. Nothing real is edited.
  if [ "$A1RULE" = record-mark ]; then
    base=$(for f in "$OUT"/q1-a1/a1-*.json; do [ -e "$f" ] || continue; [ "$(jq -r .verdict "$f")" = PASS ] && { echo "$f"; break; }; done)
    cv1=FAIL cv2=FAIL co1="no accepted reading to copy" co2="no accepted reading to copy"
    if [ -n "$base" ]; then
      rcp=$OUT/q1/$(jq -r .v5_receipt "$base"); rid=$(basename "$rcp" .json)
      rroot=$(jq -rs --arg r "q1/$rid.json" '[.[] | select(.receipt == $r) | .run_root][0] // empty' "$OUT"/attempts/q1-*.json)
      CT=$(mktemp -d "${TMPDIR:-/tmp}/cand4-ctl.XXXXXX")
      if [ -n "$rroot" ] && [ -f "$rroot/compose.cortex-mark" ]; then
        cp "$rroot/compose.cortex-mark" "$CT/mark"; printf 'X' | dd of="$CT/mark" bs=1 seek=70 conv=notrunc 2>/dev/null
        d=$(a1_read "$rcp" "$CT/mark"); [ "$(jq -r .verdict <<<"$d")" = FAIL ] && [ "$(jq -r .mark_well_formed <<<"$d")" = false ] && cv1=PASS
        co1="a mark with one byte changed (copy of $rid mark) is read as $(jq -r .verdict <<<"$d"), mark_well_formed $(jq -r .mark_well_formed <<<"$d")"
        jq '(.acceptance_v5.task_quality[] | select(.row == "Q4") | .result) = "FAIL"' "$rcp" >"$CT/rc.json"
        d=$(a1_read "$CT/rc.json" "$rroot/compose.cortex-mark"); [ "$(jq -r .verdict <<<"$d")" = FAIL ] && [ "$(jq -r .only_these_rows <<<"$d")" = false ] && cv2=PASS
        co2="a copy of receipt $rid with Q4 set to FAIL is read as $(jq -r .verdict <<<"$d"), only_these_rows $(jq -r .only_these_rows <<<"$d")"
      fi
      rm -rf "$CT"
    fi
    jq -nc --arg v "$cv1" --arg o "$co1" '{row:"A1-ctl-damaged-mark", rep:1, verdict:$v, outcome:$o, env:"host"}' >>"$OUT/q1-results.jsonl"
    jq -nc --arg v "$cv2" --arg o "$co2" '{row:"A1-ctl-extra-failing-row", rep:1, verdict:$v, outcome:$o, env:"host"}' >>"$OUT/q1-results.jsonl"
    echo "Q1 section 4.1 controls: damaged mark $cv1; extra failing row $cv2"
  fi
  s=$(bash "$SCORE" "$HERE/q1.decl.json" "$OUT/q1-results.jsonl"); src=$?
  f=$(jq -c --arg k "$KIND" '. + {kind:$k, results_sha256:"'"$(sha256sum "$OUT/q1-results.jsonl" | cut -d' ' -f1)"'"}' <<<"$s" | name_by_sha "$OUT/scores" q1-score-)
  echo "Q1 scoring-v5 verdict: $(jq -r .verdict "$OUT/scores/$f") (scores/$f, scorer exit $src)"
  ;;

_q1_round)   # inside the hold: one launch per task, in order, never repeated
  refuse_unless_digests   # recorded in this round's attempt records (issue #240)
  BASE=$3 OUT=$4 r=$5 TASKS=$6
  for id in $TASKS; do
    goal=$(jq -r --arg id "$id" '.tasks[] | select(.id == $id) | .goal' "$NP1/tasks-v5.json")
    R=$BASE/r$r-$id; t0=$(date -u +%FT%TZ)
    gpu_env NP1_GOAL="$goal" bash "$NP1/run-campaign.sh" "$R" >"$R.driver.out" 2>&1; drc=$?
    receipt=null verdict=FAIL outcome="launch failed: no run.json (driver exit $drc)"
    if [ -f "$R/run.json" ]; then
      TASK_ID=$id SPEC="cand4-qualification/ACCEPTANCE-CAND4.md spec_version 1 (rows of next-phase-1/ACCEPTANCE-v5.md)" \
        bash "$NP1/make-receipt.sh" "$R" "$OUT/q1" "$SC" "$OC" "$OC" 0 "CAND-4 Q1 $id rep $r ($KIND)" >"$R.receipt.out" 2>&1
      line=$(head -1 "$R.receipt.out")
      h=$(sed -n 's/^NEXT_PHASE_1 receipt \([0-9a-f]*\)\.json verdict \([A-Z]*\)$/\1/p' <<<"$line")
      v=$(sed -n 's/^NEXT_PHASE_1 receipt [0-9a-f]*\.json verdict \([A-Z]*\)$/\1/p' <<<"$line")
      if [ -n "$h" ]; then receipt="\"q1/$h.json\"" verdict=$v outcome="receipt $h"; else outcome="receipt script failed: $line"; fi
      # ACCEPTANCE-CAND4 section 4.1 (q1_a1_record_mark = record-mark): the v5 receipt stays as it is;
      # a second record reads it under the six conditions of a1_read.
      if [ -n "$h" ] && [ "$v" != PASS ] && [ "$A1RULE" = record-mark ]; then
        rf=$(a1_read "$OUT/q1/$h.json" "$R/compose.cortex-mark" | name_by_sha "$OUT/q1-a1" a1-)
        verdict=$(jq -r .verdict "$OUT/q1-a1/$rf") outcome="receipt $h (v5 $v); section 4.1 reading q1-a1/$rf $verdict"
      fi
    fi
    a=$(jq -n --arg k "$KIND" --arg id "$id" --argjson rep "$r" --arg root "$R" --argjson drc "$drc" \
      --argjson receipt "$receipt" --arg v "$verdict" --arg o "$outcome" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson dg "$DIGESTS" --arg tail "$(tail -5 "$R.driver.out" 2>/dev/null)" --arg sc "$SC" \
      '{gate:"CAND4_Q1_ATTEMPT", kind:$k, sc_commit:$sc, row:$id, rep:$rep, env:"GPU", run_root:$root, driver_exit:$drc,
        receipt:$receipt, verdict:$v, outcome:$o, started:$t0, ended:$t1, digests_checked_before_round:$dg, driver_tail:$tail}' \
      | name_by_sha "$OUT/attempts" q1-)
    jq -nc --arg id "$id" --argjson rep "$r" --arg v "$verdict" --arg o "$outcome; attempt $a" \
      '{row:$id, rep:$rep, verdict:$v, outcome:$o, env:"GPU"}' >>"$OUT/q1-results.jsonl"
    echo "Q1 $id rep $r: $verdict ($outcome; attempts/$a)"
  done
  ;;

q2)
  BASE=${3:?RUN_BASE} OUT=${4:?OUT}; out_ok "$OUT" || exit 3
  mkdir -p "$BASE" "$OUT/q2" "$OUT/attempts" "$OUT/scores"; BASE=$(cd "$BASE" && pwd); OUT=$(cd "$OUT" && pwd)
  if [ -e "$BASE/F0" ] || [ -e "$BASE/cases" ] || [ -e "$OUT/q2-results.jsonl" ]; then
    echo "REFUSED: Q2 already ran in $BASE or $OUT (never repeated)" >&2; exit 3
  fi
  refuse_unless_digests
  hold 20 "CAND-4 Q2 fixture F0 ($KIND)" bash "$0" _q2_fixture "$INPUTS" "$BASE" "$OUT"
  # The cases never touch the GPU (run-faults.sh stops on a non CPU-reference daemon); like
  # builds, they wait while another lane holds the quiet flag.
  refuse_unless_digests
  wait_quiet "Q2 cases"
  CASES= QREPS=3
  if [ "$KIND" = TRIAL ]; then CASES=${CAND4_TRIAL_CASES:-} QREPS=${CAND4_TRIAL_REPS:-3}; fi
  t0=$(date -u +%FT%TZ)
  # shellcheck disable=SC2086
  clean_env AIEN_BIN="$(inp cpu_fault_path)" FIX_OLD="$(inp fix_old_dir)" REPS="$QREPS" \
    bash "$NP2/run-faults.sh" cases "$BASE/F0" "$BASE/cases" $CASES >"$BASE/cases.out" 2>&1; crc=$?
  mkdir -p "$BASE/cases"; touch "$BASE/cases/results.jsonl"
  jq -c --slurpfile d "$HERE/q2.decl.json" '. as $r | {row, rep, verdict, outcome, run_root,
      env: (($d[0].rows | map(select(.row == $r.row)) | .[0].env) // "undeclared")}' "$BASE/cases/results.jsonl" >"$OUT/q2-results.jsonl"
  FJ=$BASE/F0/fixture.json
  if [ -f "$FJ" ]; then
    jq -c '{row:"C3-control", rep:1, env:"GPU",
            verdict:(if .s3_committed == true and ((.backend // "") | test("OmegaGb10Backend")) then "PASS" else "FAIL" end),
            outcome:("F0 daemon with AIEN_REQUIRE_BLACKWELL=1: " + (.backend // "none") + "; s3_committed " + (.s3_committed | tostring))}' "$FJ" >>"$OUT/q2-results.jsonl"
    jq -c '{F0: (.s3_committed == true)}' "$FJ" >"$BASE/external.json"
  else
    echo '{"row":"C3-control","rep":1,"env":"GPU","verdict":"NOT_RUN","outcome":"no fixture.json"}' >>"$OUT/q2-results.jsonl"
    echo '{"F0":false}' >"$BASE/external.json"
  fi
  s=$(bash "$SCORE" "$HERE/q2.decl.json" "$OUT/q2-results.jsonl" "$BASE/external.json"); src=$?
  sf=$(jq -c --arg k "$KIND" --arg h "$(sha256sum "$OUT/q2-results.jsonl" | cut -d' ' -f1)" --slurpfile x "$BASE/external.json" \
       '. + {kind:$k, results_sha256:$h, external:$x[0]}' <<<"$s" | name_by_sha "$OUT/scores" q2-score-)
  # NEXT-PHASE-2 v4-format receipt (evidence: builds, link proof, per-run checks, not-proved list)
  lp() { jq -n --argjson c "$(strings -a "$1" | grep -Eo 'compose\.(candidate\.0|commit|verify)' | sort -u | jq -R . | jq -sc .)" \
      --argjson g "$(strings -a "$1" | grep -Eo 'NVIDIA_DGX_SPARK_GB10_SM121' | sort -u | jq -R . | jq -sc .)" \
      --argjson p "$(strings -a "$1" | grep -c reconcile_panic)" --argjson e "$(strings -a "$1" | grep -c reconcile_error)" \
      '{compose_markers:$c, gpu_markers:$g, reconcile_panic_strings:$p, reconcile_error_strings:$e}'; }
  jq -n --arg ev "$(inp build_evidence)" --slurpfile cf "$(dirname "$(inp cpu_fault_path)")/build-cpu-fault.json" \
    '{gpu:{build_evidence:$ev, recipe:"CAND-4 clean double build (release recipe, scripts/release-build.sh)"}, "cpu-fault":$cf[0]}' >"$BASE/build_env.json"
  jq -n --argjson g "$(lp "$(inp aien_cli_path)")" --argjson c "$(lp "$(inp cpu_fault_path)")" \
    '{method:"strings(1) on each binary (NEXT-PHASE-2 link_proof markers)", gpu:$g, "cpu-fault":$c}' >"$BASE/link_proof.json"
  rr=null
  if [ -f "$FJ" ] && [ -s "$BASE/cases/results.jsonl" ]; then
    FIX_OLD="$(inp fix_old_dir)" bash "$NP2/make-receipt.sh" "$BASE/F0" "$BASE/cases" "$OUT/q2" "$SC" "$OC" \
      "$(inp aien_cli_path)" "$(inp cpu_fault_path)" "$BASE/build_env.json" "$BASE/link_proof.json" \
      "CAND-4 Q2 ($KIND): NEXT-PHASE-2 v4 case set on CAND-4 binaries; this v4-format verdict is evidence only, the scoring-v5 verdict (ACCEPTANCE-CAND4 section 5) decides" \
      >"$BASE/receipt.out" 2>&1
    h=$(sed -n 's/^NEXT_PHASE_2 receipt \([0-9a-f]*\)\.json verdict.*/\1/p' "$BASE/receipt.out" | head -1)
    [ -n "$h" ] && rr="\"q2/$h.json\""
  fi
  a=$(jq -n --arg k "$KIND" --arg sc "$SC" --argjson crc "$crc" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson n "$(wc -l <"$BASE/cases/results.jsonl")" --arg h "$(sha256sum "$BASE/cases/results.jsonl" | cut -d' ' -f1)" \
      --argjson rr "$rr" --arg sf "scores/$sf" --argjson dg "$DIGESTS" --arg tail "$(tail -3 "$BASE/cases.out")" \
      '{gate:"CAND4_Q2_CASES", kind:$k, sc_commit:$sc, binary:"cpu_fault", cases_exit:$crc, started:$t0, ended:$t1,
        result_lines:$n, results_sha256:$h, np2_receipt:$rr, score:$sf, digests_checked_before_cases:$dg, tail:$tail}' \
      | name_by_sha "$OUT/attempts" q2-cases-)
  echo "Q2 cases exit $crc; scoring-v5 verdict: $(jq -r .verdict "$OUT/scores/$sf") (scores/$sf, scorer exit $src); attempts/$a; NEXT-PHASE-2 receipt $rr"
  ;;

_q2_fixture)   # inside the hold: F0 on the production binary, once
  refuse_unless_digests   # recorded in the q2-fixture attempt record (issue #240)
  BASE=$3 OUT=$4; t0=$(date -u +%FT%TZ)
  gpu_env bash "$NP2/run-faults.sh" fixture "$BASE/F0" >"$BASE/F0.out" 2>&1; frc=$?
  fj=null; [ -f "$BASE/F0/fixture.json" ] && fj=$(jq -c . "$BASE/F0/fixture.json")
  a=$(jq -n --arg k "$KIND" --arg sc "$SC" --argjson frc "$frc" --argjson fj "$fj" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson dg "$DIGESTS" --arg tail "$(tail -5 "$BASE/F0.out")" \
      '{gate:"CAND4_Q2_FIXTURE", kind:$k, sc_commit:$sc, binary:"aien_cli", env:"GPU", exit:$frc, fixture:$fj,
        valid:(($fj // {}).s3_committed == true), started:$t0, ended:$t1, digests:$dg, tail:$tail}' | name_by_sha "$OUT/attempts" q2-fixture-)
  echo "Q2 F0 exit $frc valid $(jq -r '.s3_committed // false' "$BASE/F0/fixture.json" 2>/dev/null || echo false) (attempts/$a)"
  ;;
q2w)
  BASE=${3:?RUN_BASE} OUT=${4:?OUT}; out_ok "$OUT" || exit 3
  mkdir -p "$OUT/q2w" "$OUT/attempts" "$OUT/scores"; BASE=$(cd "$BASE" && pwd); OUT=$(cd "$OUT" && pwd)
  [ -f "$BASE/F0/fixture.json" ] || { echo "REFUSED: no F0 under $BASE (run q2 first)" >&2; exit 3; }
  [ -e "$BASE/windows" ] && { echo "REFUSED: the window tests already ran into $BASE/windows (never repeated)" >&2; exit 3; }
  refuse_unless_digests
  clean_env AIEN_BIN="$(inp cpu_fault_path)" bash "$HERE/run-windows.sh" "$BASE/F0" "$BASE/windows" >"$BASE/windows.out" 2>&1; wrc=$?
  refuse_unless_digests
  [ -f "$BASE/windows/results.jsonl" ] || : >"$BASE/windows/results.jsonl"
  cp "$BASE/windows/results.jsonl" "$OUT/q2w-results.jsonl"
  s=$(bash "$SCORE" "$HERE/windows.decl.json" "$OUT/q2w-results.jsonl"); src=$?
  f=$(jq -c --arg k "$KIND" '. + {kind:$k}' <<<"$s" | name_by_sha "$OUT/scores" q2w-score-)
  a=$(jq -n --arg k "$KIND" --arg sc "$SC" --argjson wrc "$wrc" --arg v "$(jq -r .verdict "$OUT/scores/$f")" --argjson dg "$DIGESTS" --arg tail "$(tail -8 "$BASE/windows.out")" \
      '{gate:"CAND4_Q2W_ATTEMPT", kind:$k, sc_commit:$sc, binary:"cpu_fault", env:"CPU fault injection", exit:$wrc, verdict:$v, digests:$dg, tail:$tail}' | name_by_sha "$OUT/attempts" q2w-)
  echo "Q2w (coverage cases 2 and 5) scoring-v5 verdict: $(jq -r .verdict "$OUT/scores/$f") (scores/$f, scorer exit $src; attempts/$a)"
  ;;
*) echo "unknown command $CMD" >&2; exit 2;;
esac
