#!/usr/bin/env bash
# Self-test for the Headstart M0 runner and summarizer (#532). Bash + jq only.
# Runs the summarizer on synthetic data, then the full runner end to end on a
# tiny throwaway Cargo workspace (never on the real repo), plus negative cases
# that must fail: corrupt samples, missing UNIT_DATA, a build that does not
# compile, a missing edit anchor and a build that dirties the source tree.
#
#   bash scripts/bench/headstart-selftest.sh      (exit 0 = all checks passed)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/headstart-aien.sh"
SUM="$HERE/headstart-summarize.sh"
TOOLCHAIN_FILE="$HERE/../../rust-toolchain.toml"
T=$(mktemp -d "${TMPDIR:-/tmp}/headstart-selftest.XXXXXX")
if [ "${HS_SELFTEST_KEEP:-0}" = 1 ]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
PASS=0; FAIL=0

check() { # check <name> <command...>
  local name=$1; shift
  if "$@" >/dev/null 2>&1; then PASS=$((PASS + 1)); echo "ok   $name"
  else FAIL=$((FAIL + 1)); echo "FAIL $name"; fi
}
jqt() { jq -e "$@"; } # jq truthiness test: [jq args] <filter> <file>
jqs() { jq -e -s "$@"; } # same, on a JSONL file slurped into an array

# ------------------------------------------------ 1. summarizer, synthetic
S="$T/samples.jsonl"
mk() { # mk scenario arm wall rc final contaminated aa
  jq -cn --arg s "$1" --arg a "$2" --argjson w "$3" --argjson rc "$4" --argjson f "$5" \
    --argjson c "$6" --arg aa "$7" \
    '{scenario: $s, arm: $a, kind: "cold", command: "cargo x", sample_id: ($s + "-" + $a + "-" + ($w|tostring)),
      wall_s: $w, exit_status: $rc, ok: ($rc == 0), final: $f, contaminated: $c,
      aa_pair: (if $aa == "-" then null else $aa end),
      time_v: {available: true, max_rss_kb: 1000, cpu_percent: 500}}' >> "$S"
}
mk s1 A 10 0 true false -
mk s1 A 12 0 true false -
mk s1 A 11 0 true true -
mk s1 A 5 101 true false -
mk s1 A 99 0 false true -
mk aa A 10 0 true false aa
mk aa A 20 0 true false aa
mk aa B 12 0 true false aa
mk aa B 22 0 true false aa
"$SUM" samples "$S" > "$T/summary.json"
G='.groups[] | select(.scenario == "s1")'
check "summary median excludes failed and superseded" jqt "$G | .wall_s.median == 11 and .wall_s.n == 3" "$T/summary.json"
check "summary min/max/mad/stdev" jqt "$G | .wall_s.min == 10 and .wall_s.max == 12 and .wall_s.mad == 1 and .wall_s.stdev == 1" "$T/summary.json"
check "summary counts failures with exit status" jqt "$G | .n_failed == 1 and .failures[0].exit_status == 101" "$T/summary.json"
check "summary clean subset drops contaminated" jqt "$G | .wall_s_clean.n == 2 and .wall_s_clean.median == 11" "$T/summary.json"
check "summary superseded count" jqt "$G | .n_superseded == 1" "$T/summary.json"
check "A/A delta" jqt '.aa_noise[0] | .median_a_s == 15 and .median_b_s == 17 and .delta_s == 2 and .delta_pct == 13.333' "$T/summary.json"
"$SUM" samples "$S" > "$T/summary2.json"
check "summary is byte-for-byte reproducible" cmp "$T/summary.json" "$T/summary2.json"
cp "$S" "$T/bad.jsonl"; echo '{"scenario": broken' >> "$T/bad.jsonl"
check "NEGATIVE: corrupt samples line is rejected" bash -c "! '$SUM' samples '$T/bad.jsonl'"
: > "$T/empty.jsonl"
check "NEGATIVE: empty samples file is rejected" bash -c "! '$SUM' samples '$T/empty.jsonl'"

# ------------------------------------------------ 2. timings + critical path
cat > "$T/timing.html" <<'EOF'
<html><script>
DURATION = 6;
const UNIT_DATA = [
  {"i": 0, "name": "a", "version": "0.1.0", "mode": "todo", "target": "", "features": [], "start": 0.0, "duration": 4.0,
   "unblocked_units": [], "unblocked_rmeta_units": [1], "sections": [["frontend", {"start": 0.0, "end": 1.0}], ["codegen", {"start": 1.0, "end": 4.0}]]},
  {"i": 1, "name": "b", "version": "0.1.0", "mode": "todo", "target": "", "features": [], "start": 1.1, "duration": 3.0,
   "unblocked_units": [2], "unblocked_rmeta_units": [], "sections": [["frontend", {"start": 0.0, "end": 2.0}], ["codegen", {"start": 2.0, "end": 3.0}]]},
  {"i": 2, "name": "c", "version": "0.1.0", "mode": "todo", "target": " c \"bin\"", "features": [], "start": 4.2, "duration": 1.0,
   "unblocked_units": [], "unblocked_rmeta_units": [], "sections": null}
];
const CONCURRENCY_DATA = [];
</script></html>
EOF
"$SUM" timings "$T/timing.html" > "$T/units.json"
check "timings extracts UNIT_DATA" jqt 'length == 3 and .[2].name == "c"' "$T/units.json"
"$SUM" critical "$T/units.json" 2 > "$T/crit.json"
check "critical chain a -> b -> c" jqt '[.observed.chain[].name] == ["a","b","c"]' "$T/crit.json"
check "critical gates rmeta then full" jqt '[.observed.chain[1:][].gated_by.gate] == ["rmeta","full"] and .observed.chain[1].gate_time_s == 1 and .observed.chain[2].gate_time_s == 4.1' "$T/crit.json"
check "estimate bounds" jqt '.estimate.floor_s == 4 and .estimate.rmeta_wait_on_chain_s == 1 and .estimate.upper_bound_saving_s == 1 and .estimate.scope == "cold cargo build --workspace only"' "$T/crit.json"
# list-schedule replay: a(lib) -> b(lib, rmeta edge) -> c(bin, full edge)
cat > "$T/graph.json" <<'EOF'
[{"name": "a", "version": "0.1.0", "links": null, "build_script": false, "proc_macro": false, "deps": []},
 {"name": "b", "version": "0.1.0", "links": null, "build_script": false, "proc_macro": false,
  "deps": [{"name": "a", "version": "0.1.0", "kinds": ["normal"]}]},
 {"name": "c", "version": "0.1.0", "links": null, "build_script": false, "proc_macro": false,
  "deps": [{"name": "b", "version": "0.1.0", "kinds": ["normal"]}]}]
EOF
"$SUM" replay "$T/units.json" "$T/graph.json" 2 > "$T/replay2.json"
check "replay 2 jobs: baseline 5, no rmeta wait 4, saving 1" jqt '.baseline_model_end_s == 5 and .no_rmeta_wait_model_end_s == 4 and .realistic_saving_s == 1 and .rmeta_edges == 1 and .deadlock == false' "$T/replay2.json"
"$SUM" replay "$T/units.json" "$T/graph.json" 1 > "$T/replay1.json"
check "replay 1 job: job limit removes the saving" jqt '.baseline_model_end_s == 8 and .no_rmeta_wait_model_end_s == 8 and .realistic_saving_s == 0' "$T/replay1.json"
jq '(.[] | select(.name == "a") | .proc_macro) = true' "$T/graph.json" > "$T/graph-pm.json"
"$SUM" replay "$T/units.json" "$T/graph-pm.json" 2 > "$T/replay-pm.json"
check "replay: proc-macro dependency is a full edge (no saving)" jqt '.rmeta_edges == 0 and .realistic_saving_s == 0' "$T/replay-pm.json"
check "NEGATIVE: replay without a graph is rejected" bash -c "! '$SUM' replay '$T/units.json' '$T/missing.json' 2"
printf '<html>no data</html>\n' > "$T/nodata.html"
check "NEGATIVE: report without UNIT_DATA is rejected" bash -c "! '$SUM' timings '$T/nodata.html'"

# ------------------------------------------------ 3. runner end to end on a fixture
mkfixture() { # mkfixture <dir> [broken_runtime] [dirty_buildrs]
  local d=$1 broken=${2:-no} dirty=${3:-no}
  mkdir -p "$d/crates"/{hsbase,spark-harness,spark-supervisor,aien-runtime}/src
  printf '[workspace]\nresolver = "2"\nmembers = ["crates/hsbase", "crates/spark-harness", "crates/spark-supervisor", "crates/aien-runtime"]\n' > "$d/Cargo.toml"
  [ -f "$TOOLCHAIN_FILE" ] && cp "$TOOLCHAIN_FILE" "$d/rust-toolchain.toml"
  pkg() { printf '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n%s\n' "$1" "$2"; }
  pkg hsbase "$([ "$dirty" = yes ] && echo 'build = "build.rs"')" > "$d/crates/hsbase/Cargo.toml"
  printf 'pub fn core_value() -> u64 {\n    7\n}\n' > "$d/crates/hsbase/src/lib.rs"
  if [ "$dirty" = yes ]; then
    printf 'fn main() {\n    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs");\n    let mut s = std::fs::read_to_string(p).unwrap();\n    s.push_str("\\n// dirtied by build script\\n");\n    std::fs::write(p, s).unwrap();\n}\n' > "$d/crates/hsbase/build.rs"
  fi
  pkg spark-harness '[dependencies]
hsbase = { path = "../hsbase" }' > "$d/crates/spark-harness/Cargo.toml"
  printf 'pub fn harness() -> u64 {\n    hsbase::core_value() + 1\n}\n' > "$d/crates/spark-harness/src/lib.rs"
  pkg spark-supervisor '[dependencies]
hsbase = { path = "../hsbase" }' > "$d/crates/spark-supervisor/Cargo.toml"
  printf 'fn main() {\n    println!("{}", hsbase::core_value());\n}\n' > "$d/crates/spark-supervisor/src/main.rs"
  pkg aien-runtime '[dependencies]
spark-harness = { path = "../spark-harness" }' > "$d/crates/aien-runtime/Cargo.toml"
  if [ "$broken" = yes ]; then
    printf 'pub fn runtime() -> u64 {\n    spark_harness::harness() + "not a number"\n}\n' > "$d/crates/aien-runtime/src/lib.rs"
  else
    printf 'pub fn runtime() -> u64 {\n    spark_harness::harness() * 2\n}\n' > "$d/crates/aien-runtime/src/lib.rs"
  fi
  (cd "$d" && git init -q && cargo generate-lockfile -q --offline && git add -A \
     && git -c user.name=selftest -c user.email=selftest@invalid commit -q -m fixture)
}

ANCHOR_DEFAULT='pub fn core_value() -> u64 {'
runner() { # runner <fixture> <tag> [stages...]
  local fx=$1 tag=$2; shift 2
  HS_REPO="$fx" HS_COMMIT=HEAD HS_WORK="$T/work-$tag" HS_OUT="$T/out-$tag" HS_DATE=selftest \
  HS_SAMPLES=1 HS_LOAD_MAX="${T_LOAD_MAX:-100000}" HS_WAIT_CAP_S=0 HS_MAX_RERUNS=0 HS_MON_INTERVAL_S=1 \
  HS_QUIETLOCK_BIN="${T_QL_BIN:-quietlock}" HS_QUIET_WAIT_CAP_S="${T_QL_CAP:-3600}" \
  HS_INCR_STEPS=2 HS_INCR_SERIES=1 HS_BODY_FILE=crates/hsbase/src/lib.rs \
  HS_BODY_ANCHOR="${HS_TEST_ANCHOR:-$ANCHOR_DEFAULT}" HS_IFACE_FILE=crates/hsbase/src/lib.rs \
  HS_EDIT_REASON=selftest bash "$RUNNER" "$@" >> "$T/runner-$tag.log" 2>&1
}

FX="$T/fixture"; mkfixture "$FX"
runner "$FX" good setup timings cold incr cleanup
RC=$?
O="$T/out-good"
check "runner full matrix exits 0" test "$RC" -eq 0
check "runner wrote 18 samples, all ok" jqs 'length == 18 and all(.[]; .ok)' "$O/samples.jsonl"
check "runner manifest has commit, rustc and scenarios" jqt '(.source.commit | length) == 40 and (.toolchain.rustc_vV | test("rustc ")) and (.scenarios | length) >= 11 and .build.registry_cache.state == "warm"' "$O/manifest.json"
check "runner records quiet-flag and load waits per sample" jqs 'all(.[]; (.quiet_flag_wait_s | type) == "number" and (.load_wait_s | type) == "number" and (.loadavg_before | length) == 3)' "$O/samples.jsonl"
check "runner sample ids are unique" jqs 'map(.sample_id) | length == (unique | length)' "$O/samples.jsonl"
check "runner keeps one log tail per sample" bash -c "[ \$(ls '$O/logs' | wc -l) -eq 18 ]"
check "runner summary reproducible from samples" bash -c "'$SUM' samples '$O/samples.jsonl' | cmp - '$O/summary.json'"
check "runner A/A arms present" jqt '.aa_noise[0].n_a == 1 and .aa_noise[0].n_b == 1' "$O/summary.json"
check "runner critical path extracted" jqt '.observed.chain_length >= 2' "$O/timings/critical-path.json"
check "body edits recompiled the edited crate" grep -q 'Checking hsbase' "$O/logs/incr_body_check-A-r1-s1-a1.tail.log"
check "iface edits recorded the edit" jqs 'map(select(.scenario == "incr_iface_check")) | length == 2 and all(.[]; .edit | test("pub fn headstart_m0_probe_"))' "$O/samples.jsonl"
check "caller fixture checkout untouched" git -C "$FX" diff --quiet
check "benchmark worktree removed" bash -c "[ \$(git -C '$FX' worktree list | wc -l) -eq 1 ] && [ ! -e '$T/work-good' ]"

HS_TEST_ANCHOR='no such anchor line' runner "$FX" anchor setup incr
RC=$?
check "NEGATIVE: missing edit anchor aborts" test "$RC" -ne 0
check "NEGATIVE: abort message names the anchor" grep -q 'body anchor must match exactly one line' "$T/runner-anchor.log"
check "NEGATIVE: aborted run left worktree clean" git -C "$T/work-anchor/src" diff --quiet
runner "$FX" anchor cleanup

FXB="$T/fixture-broken"; mkfixture "$FXB" yes
runner "$FXB" broken setup cold cleanup
RC=$?
check "broken build: runner still exits 0 (failure recorded, not hidden)" test "$RC" -eq 0
check "NEGATIVE: failed build recorded with exit status" jqs 'map(select(.scenario == "cold_build_aien_runtime")) | length == 1 and .[0].ok == false and .[0].exit_status != 0' "$T/out-broken/samples.jsonl"
check "NEGATIVE: failed build never counted in median" jqt '.groups[] | select(.scenario == "cold_build_aien_runtime") | .n_failed == 1 and .wall_s.n == 0 and .wall_s.median == null' "$T/out-broken/summary.json"
check "broken build: unaffected crate still measured" jqt '.groups[] | select(.scenario == "cold_build_spark_harness") | .wall_s.n == 1' "$T/out-broken/summary.json"

FXD="$T/fixture-dirty"; mkfixture "$FXD" no yes
runner "$FXD" dirty setup timings
RC=$?
check "NEGATIVE: build that dirties the worktree aborts the run" test "$RC" -ne 0
check "NEGATIVE: abort message names the dirty tree" grep -q 'not clean' "$T/runner-dirty.log"
git -C "$T/work-dirty/src" checkout -q -- . 2>/dev/null
runner "$FXD" dirty cleanup

# Load wait that hits its cap: HS_LOAD_MAX=0 can never be met, cap is 0 s.
T_LOAD_MAX=0 runner "$FX" capped setup timings cleanup
RC=$?
check "capped load wait: run completes" test "$RC" -eq 0
check "NEGATIVE: capped load wait marks the sample contaminated" jqs 'length == 1 and .[0].load_wait_capped == true and .[0].contaminated == true and (.[0].contamination_reasons | index("load_wait_capped"))' "$T/out-capped/samples.jsonl"

# quietlock missing: no check, recorded truthfully.
T_QL_BIN="$T/no-such-quietlock" runner "$FX" noql setup timings cleanup
check "quietlock missing: manifest says not available, not respected" jqt '.protocol.quiet_flag.available == false and .protocol.quiet_flag.respected == false' "$T/out-noql/manifest.json"
check "quietlock missing: sample says not checked" jqs '.[0].quiet_flag_checked == false and .[0].quiet_flag_wait_s == 0' "$T/out-noql/samples.jsonl"

# Stale quiet flag: a fake quietlock that always reports "held" (exit 75).
printf '#!/usr/bin/env bash\necho "fake quietlock: held" >&2\nexit 75\n' > "$T/fake-quietlock"; chmod +x "$T/fake-quietlock"
T_QL_BIN="$T/fake-quietlock" T_QL_CAP=2 timeout 300 bash -c "$(declare -f runner); T='$T' RUNNER='$RUNNER' ANCHOR_DEFAULT='$ANCHOR_DEFAULT' runner '$FX' stale setup timings cleanup"
RC=$?
check "NEGATIVE: stale quiet flag does not hang the run (cap)" test "$RC" -eq 0
check "NEGATIVE: capped quiet-flag wait is logged" grep -q 'quiet flag still held after' "$T/runner-stale.log"
check "NEGATIVE: capped quiet-flag wait marks the sample contaminated" jqs '.[0].quiet_flag_checked == true and .[0].quiet_flag_wait_capped == true and .[0].quiet_flag_wait_s >= 2 and (.[0].contamination_reasons | index("quiet_flag_wait_capped"))' "$T/out-stale/samples.jsonl"
check "quietlock present: manifest says available and respected" jqt '.protocol.quiet_flag.available == true and .protocol.quiet_flag.respected == true and .protocol.quiet_flag.wait_cap_s == 2' "$T/out-stale/manifest.json"

# A foreign "cargo" process (a renamed sleep, outside the runner tree) must be
# seen at start, during (1 s ticks) and in the end-of-sample snapshot.
cp "$(command -v sleep)" "$T/cargo"
"$T/cargo" 60 & FAKE=$!
runner "$FX" foreign setup timings cleanup
kill "$FAKE" 2>/dev/null; wait "$FAKE" 2>/dev/null
check "NEGATIVE: foreign compiler process is detected at start and end" jqs --arg p "$FAKE:cargo:-" '.[0].contaminated == true and (.[0].foreign.at_start | index($p)) and (.[0].foreign.at_end | index($p)) and (.[0].contamination_reasons | index("foreign_at_end"))' "$T/out-foreign/samples.jsonl"
check "timings stage writes the replay estimate" jqt '.deadlock == false and .units_without_package == 0 and .label == "ESTIMATE, not a measurement"' "$T/out-foreign/timings/replay.json"

# ------------------------------------------------ 4. M5 (#536) OFF/ON summary, synthetic
SAB="$T/ab.jsonl"
mkab() { # mkab scenario arm wall rc rss user [contaminated]
  jq -cn --arg s "$1" --arg a "$2" --argjson w "$3" --argjson rc "$4" --argjson rss "$5" --argjson u "$6" --argjson c "${7:-false}" \
    '{scenario: $s, arm: $a, kind: "ab", command: "cargo x", sample_id: ($s + "-" + $a + "-" + ($w|tostring)),
      wall_s: $w, exit_status: $rc, ok: ($rc == 0), final: true, contaminated: $c, aa_pair: null,
      time_v: {available: true, max_rss_kb: $rss, cpu_percent: 100, user_s: $u, sys_s: 1}}' >> "$SAB"
}
mkab w OFF 10 0 1000 9; mkab w OFF 12 0 1000 9; mkab w ON 8 0 1100 10; mkab w ON 9 0 1100 10; mkab w ON 1 101 1 1
mkab w OFF 100 0 9000 90 true; mkab w ON 0.5 0 1 1 true
mkab c OFF 10 0 1000 9 true; mkab c ON 9 0 1000 9 true; mkab f OFF 10 0 1000 9; mkab f ON 1 101 1 1
"$SUM" samples "$SAB" > "$T/ab-summary.json"
AB='.ab[] | select(.scenario == "w")'
check "ab: medians, delta and saving" jqt "$AB | .wall_s_off.median == 11 and .wall_s_on.median == 8.5 and .delta_s == -2.5 and .saving_pct == 22.727" "$T/ab-summary.json"
check "ab: failed ON sample counted as failure, not time" jqt "$AB | .failed_on == 1 and .n_on == 2 and .failures[0].arm == \"ON\" and .failures[0].exit_status == 101" "$T/ab-summary.json"
check "NEGATIVE: contaminated finals excluded from times, RSS and CPU" jqt "$AB | .contaminated_excluded_off == 1 and .contaminated_excluded_on == 1 and .n_off == 2 and .max_rss_kb_median_off == 1000 and .status == \"measured\"" "$T/ab-summary.json"
check "NEGATIVE: only contaminated samples give NOT_MEASURED, no number" jqt '.ab[] | select(.scenario == "c") | .status == "NOT_MEASURED" and .n_off == 0 and .n_on == 0 and .saving_pct == null' "$T/ab-summary.json"
check "NEGATIVE: ON that only fails gives FAILED, no number" jqt '.ab[] | select(.scenario == "f") | .status == "FAILED" and .failed_on == 1 and .saving_pct == null' "$T/ab-summary.json"
check "ab: max RSS and CPU deltas" jqt "$AB | .max_rss_delta_pct == 10 and .cpu_s_median_off == 10 and .cpu_s_median_on == 11 and .cpu_s_delta_pct == 10" "$T/ab-summary.json"
check "ab: section absent when no OFF/ON arms" jqt 'has("ab") | not' "$T/summary.json"

# ------------------------------------------------ 5. M5 A/B stages end to end on the fixture
ABSTAGES=(setup ab abincr abverify cleanup)
HS_ARM_ENV_OFF="HS_FAKE_ARM=off" HS_ARM_ENV_ON="HS_FAKE_ARM=on" HS_AB_JOBS=2 HS_AB_JOBS_SAMPLES=1 \
HS_VERIFY_TEST_PKGS=hsbase HS_VERIFY_ENV_ON="HS_FAKE_VERIFY=1" runner "$FX" ab "${ABSTAGES[@]}"
RC=$?
O="$T/out-ab"
check "ab runner exits 0" test "$RC" -eq 0
check "ab runner wrote 35 samples, all ok" jqs 'length == 35 and all(.[]; .ok)' "$O/samples.jsonl"
check "ab arm environments reach the command" jqs 'all(.[] | select(.arm == "ON"); .env_words | test("HS_FAKE_ARM=on")) and all(.[] | select(.arm != "ON"); .env_words | test("HS_FAKE_ARM=off"))' "$O/samples.jsonl"
check "ab arms use separate target dirs" jqs '[.[] | select(.kind == "ab")] | length == 10 and all(.[]; . as $x | $x.cargo_target_dir | endswith("-" + $x.arm))' "$O/samples.jsonl"
check "ab A/A pair runs with the OFF environment" jqt '.aa_noise[] | select(.scenario == "ab_aa_check_ws") | .n_a == 1 and .n_b == 1' "$O/summary.json"
# Coverage counts contaminated finals too: on a busy host a fixture sample
# can be contaminated (HS_MAX_RERUNS=0 here), which excludes it from times.
check "ab summary covers cold, jobs and incremental scenarios" jqt '[.ab[] | select(.n_off + .contaminated_excluded_off >= 1 and .n_on + .contaminated_excluded_on >= 1) | .scenario] as $s | (["ab_cold_check_ws","ab_cold_build_ws","ab_cold_test_norun_ws","ab_cold_build_ws_j2","abincr_body_check","abincr_iface_check"] - $s) == []' "$O/summary.json"
check "ab incremental: both arms check every edit" jqs '[.[] | select(.scenario == "abincr_body_check")] | (map(select(.arm == "OFF")) | length) == 2 and (map(select(.arm == "ON")) | length) == 2' "$O/samples.jsonl"
check "ab verify: verdict match with test results" jqt '.verdict == "match" and .diagnostics_equal and .test_results_equal and (.arms.ON.test_results | length) >= 1 and .verify_build.exit == 0' "$O/verify.json"
check "ab verify extra env added on top of the ON arm" jqs 'map(select(.scenario == "verify_build_extra")) | length == 1 and (.[0].env_words | test("HS_FAKE_ARM=on") and test("HS_FAKE_VERIFY=1"))' "$O/samples.jsonl"
check "ab manifest records binaries digests and arm envs" jqt '(.toolchain.binaries.sha256 | length) >= 2 and .toolchain.binaries.arm_env.ON == "HS_FAKE_ARM=on"' "$O/manifest.json"
check "ab summary reproducible from samples" bash -c "'$SUM' samples '$O/samples.jsonl' | cmp - '$O/summary.json'"
check "ab caller fixture checkout untouched" git -C "$FX" diff --quiet

HS_ARM_ENV_OFF="HS_FAKE_ARM=off" runner "$FX" noon setup ab
RC=$?
check "NEGATIVE: ab stage without HS_ARM_ENV_ON aborts" test "$RC" -ne 0
check "NEGATIVE: abort message names the arm variables" grep -q 'HS_ARM_ENV_OFF and HS_ARM_ENV_ON' "$T/runner-noon.log"
runner "$FX" noon cleanup
HS_ARM_ENV_OFF="HS_FAKE_ARM=x" HS_ARM_ENV_ON="HS_FAKE_ARM=x" runner "$FX" same setup abverify
RC=$?
check "NEGATIVE: identical arm environments abort" test "$RC" -ne 0
runner "$FX" same cleanup

HS_ARM_ENV_OFF="HS_FAKE_ARM=off" HS_ARM_ENV_ON="RUSTFLAGS=-Wmissing-docs" HS_VERIFY_TEST_PKGS=hsbase \
  runner "$FX" diag setup abverify cleanup
check "NEGATIVE: changed diagnostics give verdict MISMATCH" jqt '.verdict == "MISMATCH" and (.diagnostics_equal | not) and .check_exit_equal and .arms.ON.diagnostics > .arms.OFF.diagnostics' "$T/out-diag/verify.json"

HS_ARM_ENV_OFF="HS_FAKE_ARM=off" HS_ARM_ENV_ON="RUSTFLAGS=-Zheadstart-selftest-bogus" HS_VERIFY_TEST_PKGS=hsbase \
  runner "$FX" brk setup ab abverify cleanup
RC=$?
check "broken ON arm: runner still exits 0 (failure recorded)" test "$RC" -eq 0
check "NEGATIVE: failing ON arm recorded as failures, never as time" jqt '.ab[] | select(.scenario == "ab_cold_build_ws") | .failed_on == 1 and .n_on == 0 and .wall_s_on.median == null and .saving_pct == null' "$T/out-brk/summary.json"
check "NEGATIVE: wrong ON exit code gives verdict MISMATCH" jqt '.verdict == "MISMATCH" and (.check_exit_equal | not) and .arms.ON.check_exit != 0' "$T/out-brk/verify.json"

# ------------------------------------------------ 6. per-sample quiet-flag hold, ICE reports, failure tails
# The real quietlock on a private state dir (QUIETLOCK_DIR), never the host's
# flag. The fake cargo records whether our hold is on the flag while the timed
# command runs, and (with HS_FAKE_ICE=1) fails a check the way the SVH bug
# does: an ICE report in the cwd, the error far above 100 warning lines.
QL="$T/ql"; mkdir -p "$QL"
cat > "$T/fake-cargo" <<'EOF'
#!/usr/bin/env bash
if [ -n "${HS_FAKE_FLAG_LOG:-}" ]; then
  if grep -q hs-selftest "$QUIETLOCK_DIR/.spark-quiet" 2>/dev/null; then echo HELD; else echo FREE; fi >> "$HS_FAKE_FLAG_LOG"
fi
if [ "${HS_FAKE_ICE:-}" = 1 ] && [[ " $* " == *" check "* ]]; then
  printf "thread 'rustc' panicked at compiler/selftest.rs:1:1:\nassertion failed: selftest svh\nstack backtrace:\n   0: selftest-backtrace-frame\n" > "rustc-ice-selftest-$$.txt"
  echo 'error: could not compile `hsbase` (lib)'
  for i in $(seq 100); do echo "warning: selftest noise $i"; done
  exit 101
fi
exec cargo "$@"
EOF
chmod +x "$T/fake-cargo"
QL_BIN=$(command -v quietlock || true)
if [ -n "$QL_BIN" ]; then
  export QUIETLOCK_DIR="$QL"
  hrunner() {
    HS_HOLD=1 HS_HOLD_OWNER=hs-selftest HS_HOLD_MINUTES=1 HS_HOLD_REASON=selftest \
    HS_CARGO="$T/fake-cargo" HS_FAKE_FLAG_LOG="$T/flag.log" runner "$@"
  }
  hrunner "$FX" hold setup
  # A foreign hold taken between stages: the first sample must queue behind it.
  "$QL_BIN" hold --owner selftest-other --minutes 1 --reason selftest -- sleep 5 >/dev/null 2>&1 &
  OTHER=$!; sleep 1
  hrunner "$FX" hold cold
  RC=$?; wait "$OTHER" 2>/dev/null
  O="$T/out-hold"; N=$(jq -s length "$O/samples.jsonl")
  check "hold: run exits 0 with samples" test "$RC" -eq 0 -a "$N" -ge 2
  check "hold: every sample held, one distinct hold id each" jqs 'all(.[]; .quiet_hold.held and .quiet_hold.requested) and ([.[].quiet_hold.id] | unique | length) == length and all(.[]; .contamination_reasons | index("quiet_hold_not_taken") | not)' "$O/samples.jsonl"
  check "hold: our hold is on the flag during every timed command" test "$(grep -c HELD "$T/flag.log")" -eq "$N"
  check "hold: setup commands outside samples run unheld" grep -q FREE "$T/flag.log"
  check "hold: every hold acquired and released (history)" test "$(grep -c 'hold acquired: hs-selftest' "$QL/.spark-quiet.history")" -eq "$N" -a "$(grep -c 'released' "$QL/.spark-quiet.history")" -ge "$((N + 1))"
  check "hold: flag clear after the run" "$QL_BIN" check
  check "NEGATIVE: foreign hold is waited out before our hold" jqs '.[0].quiet_flag_wait_s >= 2 and .[0].quiet_hold.held' "$O/samples.jsonl"
  check "hold: manifest lists the held samples" jqt --argjson n "$N" '.protocol.quiet_hold.enabled and .protocol.quiet_hold.owner == "hs-selftest" and (.protocol.quiet_hold.samples_held | length) == $n and .protocol.quiet_hold.samples_unheld == []' "$O/manifest.json"
  runner "$FX" hold cleanup

  # A quietlock whose check says clear but whose hold is always refused.
  printf '#!/usr/bin/env bash\n[ "$1" = check ] && exit 0\necho "quietlock: QUIETLOCK_REFUSED fake" >&2\nexit 75\n' > "$T/refusing-quietlock"
  chmod +x "$T/refusing-quietlock"
  T_QL_BIN="$T/refusing-quietlock" T_QL_CAP=2 HS_HOLD=1 runner "$FX" refused setup cold cleanup
  RC=$?
  check "NEGATIVE: refused hold does not hang the run (cap)" test "$RC" -eq 0
  check "NEGATIVE: refused hold marks the sample contaminated, unheld" jqs '.[0].quiet_hold.held == false and .[0].quiet_hold.refused >= 1 and .[0].contaminated and (.[0].contamination_reasons | index("quiet_hold_not_taken"))' "$T/out-refused/samples.jsonl"
  unset QUIETLOCK_DIR
else
  check "quietlock installed for the hold tests" false
fi

HS_CARGO="$T/fake-cargo" HS_ARM_ENV_OFF="HS_FAKE_ARM=off" HS_ARM_ENV_ON="HS_FAKE_ICE=1" runner "$FX" ice setup ab
RC=$?
ICE='.[] | select(.scenario == "ab_cold_check_ws" and .arm == "ON")'
check "ICE ON arm: runner exits 0, failure recorded" test "$RC" -eq 0
check "NEGATIVE: ICE sample has exit 101 and one ICE report" jqs "[$ICE] | length == 1 and .[0].exit_status == 101 and .[0].ice_reports == 1" "$T/out-ice/samples.jsonl"
check "OFF samples report no ICE" jqs 'all(.[] | select(.arm == "OFF"); .ice_reports == 0)' "$T/out-ice/samples.jsonl"
TAILF="$T/out-ice/$(jq -rs "[$ICE][0].log_tail" "$T/out-ice/samples.jsonl")"
check "failure tail starts with the error lines" bash -c "head -1 '$TAILF' | grep -q '^== error lines' && sed -n 2p '$TAILF' | grep -q 'could not compile'"
check "failure tail carries the ICE panic head" grep -q 'assertion failed: selftest svh' "$TAILF"
check "NEGATIVE: ICE backtrace is not copied" bash -c "! grep -q selftest-backtrace-frame '$TAILF'"
check "ICE report removed from the worktree root" bash -c "! ls '$T/work-ice/src'/rustc-ice-* 2>/dev/null"
runner "$FX" ice cleanup

echo "selftest: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
