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

# SMOLLM2-Q qualification harness (the CAND-4 harness, copied; ACCEPTANCE-SMOLLM2-Q.md section 0 lists every change) (ACCEPTANCE-SMOLLM2-Q.md). Shell + jq only. Never retries a
# launch; every launch, failed or not, leaves an attempt record named by its sha256.
#
#   run-smol2q.sh check   INPUTS                 recompute every pinned digest (exit 3 on a mismatch)
#   run-smol2q.sh hygiene INPUTS OUT             section 3 (hygiene.sh)
#   run-smol2q.sh q1      INPUTS RUN_BASE OUT    section 4: 3 rounds x (T1, T2, T3), GPU, one quietlock hold per round
#   run-smol2q.sh q2      INPUTS RUN_BASE OUT    section 5: F0 on the GPU (one hold), then the cases on cpu_fault
#   run-smol2q.sh q2w     INPUTS RUN_BASE OUT    section 6.1: coverage cases 2 and 5 (run-windows.sh on cpu_fault, after q2)
#   run-smol2q.sh mem     INPUTS RUN_BASE OUT    section 8: memory observations (SmolLM2 + Llama control daemon starts, GPU, one hold)
#
# INPUTS = ACCEPTANCE-SMOLLM2-Q.md (kind QUALIFICATION, refused while UNFROZEN) or a TRIAL inputs
# file (kind TRIAL: OUT must contain TRIAL and stay outside the evidence dir and the repo;
# SMOL2Q_TRIAL_TASKS / SMOL2Q_TRIAL_REPS / SMOL2Q_TRIAL_CASES may narrow a TRIAL run only).
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
whisper() { crumb whisper laneSmol2 "$REPO" - "$1" >/dev/null 2>&1 || echo "whisper failed: $1" >&2; }
refuse_unless_digests() { verify_digests || { echo "REFUSED: a pinned digest differs (no launch started)" >&2; exit 3; }; }

# content-addressed write: name_by_sha DIR PREFIX < json  -> prints the file name
name_by_sha() {
  local t; t=$(mktemp "$1/.tmp.XXXXXX"); cat >"$t"
  local h; h=$(sha256sum "$t" | cut -d' ' -f1); mv "$t" "$1/$2$h.json"; echo "$2$h.json"
}
# launch env: nothing from the caller's environment but these (ACCEPTANCE-SMOLLM2-Q section 2)
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
  local mark; mark=$(mktemp -u "${TMPDIR:-/tmp}/smol2q-started.XXXXXX")
  while :; do
    wait_quiet "$why"
    whisper "laneSmol2 GPU start: $why"
    quietlock hold --owner laneSmol2 --minutes "$min" --reason "$why" -- bash -c 'touch "$0"; exec "$@"' "$mark" "$@"
    local rc=$?
    whisper "laneSmol2 GPU release: $why rc=$rc"
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
  if [ "$KIND" = TRIAL ]; then TASKS=${SMOL2Q_TRIAL_TASKS:-$TASKS} REPS=${SMOL2Q_TRIAL_REPS:-$REPS}; fi
  for r in $(seq 1 "$REPS"); do
    refuse_unless_digests
    hold 20 "SMOLLM2-Q Q1 round $r ($KIND)" bash "$0" _q1_round "$INPUTS" "$BASE" "$OUT" "$r" "$TASKS"
    # a declared launch of this round that left no result line did not run: FAIL, recorded, not retried
    touch "$OUT/q1-results.jsonl"
    for id in $TASKS; do
      jq -e --arg id "$id" --argjson r "$r" 'select(.row == $id and .rep == $r)' "$OUT/q1-results.jsonl" >/dev/null && continue
      a=$(jq -n --arg k "$KIND" --arg sc "$SC" --arg id "$id" --argjson r "$r" \
        '{gate:"SMOL2Q_Q1_ATTEMPT", kind:$k, sc_commit:$sc, row:$id, rep:$r, env:"GPU", verdict:"FAIL",
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
      CT=$(mktemp -d "${TMPDIR:-/tmp}/smol2q-ctl.XXXXXX")
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
      TASK_ID=$id SPEC="smollm2-qualification/ACCEPTANCE-SMOLLM2-Q.md spec_version 1 (rows of next-phase-1/ACCEPTANCE-v5.md)" \
        bash "$NP1/make-receipt.sh" "$R" "$OUT/q1" "$SC" "$OC" "$OC" 0 "SMOLLM2-Q Q1 $id rep $r ($KIND)" >"$R.receipt.out" 2>&1
      line=$(head -1 "$R.receipt.out")
      h=$(sed -n 's/^NEXT_PHASE_1 receipt \([0-9a-f]*\)\.json verdict \([A-Z]*\)$/\1/p' <<<"$line")
      v=$(sed -n 's/^NEXT_PHASE_1 receipt [0-9a-f]*\.json verdict \([A-Z]*\)$/\1/p' <<<"$line")
      if [ -n "$h" ]; then receipt="\"q1/$h.json\"" verdict=$v outcome="receipt $h"; else outcome="receipt script failed: $line"; fi
      # ACCEPTANCE-SMOLLM2-Q section 4.1 (q1_a1_record_mark = record-mark): the v5 receipt stays as it is;
      # a second record reads it under the six conditions of a1_read.
      if [ -n "$h" ] && [ "$v" != PASS ] && [ "$A1RULE" = record-mark ]; then
        rf=$(a1_read "$OUT/q1/$h.json" "$R/compose.cortex-mark" | name_by_sha "$OUT/q1-a1" a1-)
        verdict=$(jq -r .verdict "$OUT/q1-a1/$rf") outcome="receipt $h (v5 $v); section 4.1 reading q1-a1/$rf $verdict"
      fi
    fi
    # section 8 observations per launch (not scored): each daemon start's KV pool plan line, warm-up line,
    # VmHWM (from run.json) and any fatal or session-open attempt line
    memobs=$(for dl in "$R"/daemon-1.log "$R"/daemon-2.log; do [ -f "$dl" ] || continue
      sed 's/\x1b\[[0-9;]*m//g' "$dl" | jq -Rsc --arg f "$(basename "$dl")" 'split("\n") as $l | {log:$f,
        kv_pool_plan_line:($l | map(select(test("KV pool: [0-9]+ bytes"))) | .[0] // null),
        warm_up_line:($l | map(select(test("Warm-up:"))) | .[0] // null),
        fatal_lines:($l | map(select(test("Fatal:|STRICT_REAL_MODEL_VIOLATION|0x51")))),
        session_open_attempt_lines:($l | map(select(test("attempt [0-9]+/[0-9]+"))) | length)}'; done | jq -sc .)
    hwms=$(jq -c '[.daemon[]? | {vmhwm_kb, warm_up_ms}]' "$R/run.json" 2>/dev/null || echo null)
    a=$(jq -n --arg k "$KIND" --arg id "$id" --argjson rep "$r" --arg root "$R" --argjson drc "$drc" \
      --argjson receipt "$receipt" --arg v "$verdict" --arg o "$outcome" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson dg "$DIGESTS" --arg tail "$(tail -5 "$R.driver.out" 2>/dev/null)" --arg sc "$SC" --argjson memobs "$memobs" --argjson hwms "${hwms:-null}" \
      '{gate:"SMOL2Q_Q1_ATTEMPT", kind:$k, sc_commit:$sc, row:$id, rep:$rep, env:"GPU", run_root:$root, driver_exit:$drc,
        receipt:$receipt, verdict:$v, outcome:$o, started:$t0, ended:$t1, digests_checked_before_round:$dg, driver_tail:$tail, memory_observations:$memobs, daemon_vmhwm:$hwms}' \
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
  hold 20 "SMOLLM2-Q Q2 fixture F0 ($KIND)" bash "$0" _q2_fixture "$INPUTS" "$BASE" "$OUT"
  # The cases never touch the GPU (run-faults.sh stops on a non CPU-reference daemon); like
  # builds, they wait while another lane holds the quiet flag.
  refuse_unless_digests
  wait_quiet "Q2 cases"
  CASES= QREPS=3
  if [ "$KIND" = TRIAL ]; then CASES=${SMOL2Q_TRIAL_CASES:-} QREPS=${SMOL2Q_TRIAL_REPS:-3}; fi
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
    '{gpu:{build_evidence:$ev, recipe:"SMOLLM2-Q clean double build (the CAND-4 recipe) (release recipe, scripts/release-build.sh)"}, "cpu-fault":$cf[0]}' >"$BASE/build_env.json"
  jq -n --argjson g "$(lp "$(inp aien_cli_path)")" --argjson c "$(lp "$(inp cpu_fault_path)")" \
    '{method:"strings(1) on each binary (NEXT-PHASE-2 link_proof markers)", gpu:$g, "cpu-fault":$c}' >"$BASE/link_proof.json"
  rr=null
  if [ -f "$FJ" ] && [ -s "$BASE/cases/results.jsonl" ]; then
    FIX_OLD="$(inp fix_old_dir)" bash "$NP2/make-receipt.sh" "$BASE/F0" "$BASE/cases" "$OUT/q2" "$SC" "$OC" \
      "$(inp aien_cli_path)" "$(inp cpu_fault_path)" "$BASE/build_env.json" "$BASE/link_proof.json" \
      "SMOLLM2-Q Q2 ($KIND): NEXT-PHASE-2 v4 case set on the SMOLLM2-Q binaries; this v4-format verdict is evidence only, the scoring-v5 verdict (ACCEPTANCE-SMOLLM2-Q section 5) decides" \
      >"$BASE/receipt.out" 2>&1
    h=$(sed -n 's/^NEXT_PHASE_2 receipt \([0-9a-f]*\)\.json verdict.*/\1/p' "$BASE/receipt.out" | head -1)
    [ -n "$h" ] && rr="\"q2/$h.json\""
  fi
  a=$(jq -n --arg k "$KIND" --arg sc "$SC" --argjson crc "$crc" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson n "$(wc -l <"$BASE/cases/results.jsonl")" --arg h "$(sha256sum "$BASE/cases/results.jsonl" | cut -d' ' -f1)" \
      --argjson rr "$rr" --arg sf "scores/$sf" --argjson dg "$DIGESTS" --arg tail "$(tail -3 "$BASE/cases.out")" \
      '{gate:"SMOL2Q_Q2_CASES", kind:$k, sc_commit:$sc, binary:"cpu_fault", cases_exit:$crc, started:$t0, ended:$t1,
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
      '{gate:"SMOL2Q_Q2_FIXTURE", kind:$k, sc_commit:$sc, binary:"aien_cli", env:"GPU", exit:$frc, fixture:$fj,
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
  f=$(jq -c --arg k "$KIND" --arg h "$(sha256sum "$OUT/q2w-results.jsonl" | cut -d' ' -f1)" '. + {kind:$k, results_sha256:$h}' <<<"$s" | name_by_sha "$OUT/scores" q2w-score-)
  a=$(jq -n --arg k "$KIND" --arg sc "$SC" --argjson wrc "$wrc" --arg v "$(jq -r .verdict "$OUT/scores/$f")" --argjson dg "$DIGESTS" --arg tail "$(tail -8 "$BASE/windows.out")" \
      '{gate:"SMOL2Q_Q2W_ATTEMPT", kind:$k, sc_commit:$sc, binary:"cpu_fault", env:"CPU fault injection", exit:$wrc, verdict:$v, digests:$dg, tail:$tail}' | name_by_sha "$OUT/attempts" q2w-)
  echo "Q2w (coverage cases 2 and 5) scoring-v5 verdict: $(jq -r .verdict "$OUT/scores/$f") (scores/$f, scorer exit $src; attempts/$a)"
  ;;
mem)   # section 8 observations (closes #236): one SmolLM2 daemon start and one Llama-3.2-1B control start
  BASE=${3:?RUN_BASE} OUT=${4:?OUT}; out_ok "$OUT" || exit 3
  mkdir -p "$BASE" "$OUT/mem"; BASE=$(cd "$BASE" && pwd); OUT=$(cd "$OUT" && pwd)
  [ -e "$BASE/mem" ] && { echo "REFUSED: the memory observations already ran into $BASE/mem (never repeated)" >&2; exit 3; }
  refuse_unless_digests
  hold 10 "SMOLLM2-Q memory observations ($KIND)" bash "$0" _mem "$INPUTS" "$BASE" "$OUT"
  ;;

_mem)   # inside the hold: SmolLM2 start, then the Llama control start, same binary, once each
  refuse_unless_digests
  BASE=$3 OUT=$4; mkdir -p "$BASE/mem"
  LD=$(inp llama_control_dir) bad=0
  dg llama_control/model.safetensors "$(readlink -f "$LD/model.safetensors")" "$(inp llama_control_safetensors)" || bad=1
  dg llama_control/config.json "$(readlink -f "$LD/config.json")" "$(inp llama_control_config)" || bad=1
  [ $bad = 0 ] || { echo "REFUSED: Llama control files differ from CAND-4 (no start)" >&2; exit 3; }
  # unified memory (DGX Spark known issues): read /proc/meminfo, not nvidia-smi or cudaMemGetInfo
  meminfo() { awk '/^(MemTotal|MemAvailable|SwapFree|HugePages_Total|HugePages_Free|Hugepagesize):/ {printf "%s%s ", $1, $2}' /proc/meminfo; }
  devholders() { for p in /proc/[0-9]*; do ls -l "$p/fd" 2>/dev/null | grep -q /dev/nvidia && printf '%s ' "${p#/proc/}"; done; }
  start_one() {   # NAME MODEL_DIR -> one observation record (not scored)
    local name=$1 d=$2 R=$BASE/mem/$1 dp i t0 st=null hwm= wu plan chk fatal retries ex g0 h0
    mkdir -p "$R/compose" "$R/prov" "$R/state"; t0=$(date -u +%FT%TZ); g0=$(meminfo); h0=$(devholders)
    # exec: $! must be the daemon itself, not a subshell running the clean_env function
    # (first mem run read VmHWM of that subshell: 2716 kB; disclosed in the verdict)
    exec_clean() { exec env -i PATH="$PATH" HOME="$HOME" USER="${USER:-}" LANG="${LANG:-C.UTF-8}" "$@"; }
    # sc#328: the daemon requires the approval desk key.
    AIEN_COMPOSE_DIR="$R/compose" "$(inp aien_cli_path)" compose desk-key --create 1 >/dev/null
    exec_clean AIEN_COMPOSE_DIR="$R/compose" AIEN_PROVENANCE_DIR="$R/prov" AIEN_RUNTIME_SOCK="$R/aien.sock" \
      AIEN_RUNTIME_STATE_DIR="$R/state" AIEN_REQUIRE_CHECKPOINT=1 AIEN_MODEL_PATH="$d/model.safetensors" \
      AIEN_TOKENIZER_PATH="$d/tokenizer.json" AIEN_COMPOSE_MAX_TOKENS="$(inp max_tokens)" AIEN_REQUIRE_BLACKWELL=1 \
      setsid "$(inp aien_cli_path)" daemon >"$R/daemon.log" 2>&1 </dev/null &
    dp=$! i=0
    until grep -q 'Warm-up:\|Fatal:\|STRICT_REAL_MODEL_VIOLATION' "$R/daemon.log" || ! kill -0 "$dp" 2>/dev/null || [ $i -gt 600 ]; do sleep 0.5; i=$((i + 1)); done
    sleep 1
    if kill -0 "$dp" 2>/dev/null; then
      st=$( { echo "measured: $(tr "\\0" " " <"/proc/$dp/cmdline")"; grep -E "^(VmHWM|VmRSS|VmSize)" "/proc/$dp/status"; } | jq -R . | jq -sc .)
      hwm=$(awk '/^VmHWM/ {print $2}' "/proc/$dp/status")
      clean_env AIEN_RUNTIME_SOCK="$R/aien.sock" "$(inp aien_cli_path)" compose shutdown >"$R/shutdown.out" 2>&1
      i=0; while kill -0 "$dp" 2>/dev/null && [ $i -lt 120 ]; do sleep 0.5; i=$((i + 1)); done
    fi
    kill -0 "$dp" 2>/dev/null && ex="still running after the shutdown request (left for the operator, never killed)" || ex="exited"
    plan=$(sed 's/\x1b\[[0-9;]*m//g' "$R/daemon.log" | grep -m1 'KV pool: [0-9]* bytes' | sed 's/^ *//')
    chk=$(sed 's/\x1b\[[0-9;]*m//g' "$R/daemon.log" | grep -m1 'KV pool memory check' | sed 's/^ *//')
    wu=$(sed 's/\x1b\[[0-9;]*m//g' "$R/daemon.log" | grep -m1 'Warm-up:' | sed 's/^ *//')
    fatal=$(sed 's/\x1b\[[0-9;]*m//g' "$R/daemon.log" | grep -m3 -E 'Fatal:|STRICT_REAL_MODEL_VIOLATION|0x51' | jq -R . | jq -sc .)
    retries=$(sed 's/\x1b\[[0-9;]*m//g' "$R/daemon.log" | grep -cE 'attempt [0-9]+/[0-9]+')
    jq -n --arg k "$KIND" --arg sc "$SC" --arg n "$name" --arg d "$d" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --arg g0 "$g0" --arg h0 "$h0" --arg plan "$plan" --arg chk "$chk" --arg wu "$wu" --argjson fatal "$fatal" --argjson st "$st" \
      --argjson hwm "${hwm:-null}" --argjson rt "$retries" --arg ex "$ex" --argjson dg "$DIGESTS" \
      --arg log "$(sha256sum "$R/daemon.log" | cut -d' ' -f1)" \
      '{gate:"SMOL2Q_MEM_OBSERVATION", kind:$k, sc_commit:$sc, model:$n, model_dir:$d, started:$t0, ended:$t1,
        meminfo_before:$g0, gpu_device_holders_before:$h0, kv_pool_plan_line:$plan, kv_pool_memory_check_line:$chk,
        warm_up_line:$wu, fatal_lines:$fatal, session_open_attempt_lines:$rt, proc_status_after_warm_up:$st,
        vmhwm_kb_after_warm_up:$hwm, daemon_end:$ex, daemon_log_sha256:$log, digests:$dg, scored:false}' | name_by_sha "$OUT/mem" mem-
  }
  for m in "smollm2 $MD" "llama-control $LD"; do
    set -- $m; f=$(start_one "$1" "$2")
    echo "memory observation $1: $(jq -r '.kv_pool_plan_line' "$OUT/mem/$f" | cut -c1-60); VmHWM $(jq -r .vmhwm_kb_after_warm_up "$OUT/mem/$f") kB (mem/$f)"
  done
  ;;
*) echo "unknown command $CMD" >&2; exit 2;;
esac
