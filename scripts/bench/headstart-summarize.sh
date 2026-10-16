#!/usr/bin/env bash
# Headstart M0 (#532) summarizer. Pure bash + jq + sed, no Python.
#
#   headstart-summarize.sh samples <samples.jsonl>            -> summary JSON on stdout
#   headstart-summarize.sh timings <cargo-timing.html>        -> UNIT_DATA as a JSON array
#   headstart-summarize.sh critical <units.json> [jobs]       -> critical-path JSON on stdout
#   headstart-summarize.sh graph <cargo-metadata.json>        -> package graph (normal + build edges)
#   headstart-summarize.sh replay <units.json> <graph.json> <jobs> -> list-schedule estimate JSON
#
# `samples` is deterministic: running it again on the same samples.jsonl
# reproduces summary.json byte for byte (see benchmarks/headstart/README.md).
set -euo pipefail

usage() {
  sed -n '2,11p' "$0" >&2
  exit 64
}

# Shared jq helpers: median, MAD (median absolute deviation), sample stdev.
JQ_STATS='
def r3: (. * 1000 | round) / 1000;
def median: sort | length as $n
  | if $n == 0 then null
    elif $n % 2 == 1 then .[($n - 1) / 2]
    else (.[$n / 2 - 1] + .[$n / 2]) / 2 end;
def mad: median as $m | if $m == null then null else map(. - $m | fabs) | median end;
def stdev: length as $n
  | if $n < 2 then null
    else (add / $n) as $mu | (map((. - $mu) * (. - $mu)) | add / ($n - 1)) | sqrt end;
def stats: if length == 0 then {n: 0, median: null, min: null, max: null, mad: null, stdev: null}
  else {n: length, median: (median | r3), min: (min | r3), max: (max | r3),
        mad: (mad | r3), stdev: (stdev | if . == null then null else r3 end)} end;
def nn_median: map(select(. != null)) | if length == 0 then null else median | r3 end;
'

cmd_samples() {
  local f=$1
  [ -s "$f" ] || { echo "summarize: $f is missing or empty" >&2; exit 65; }
  # Every line must parse; a corrupt line is an error, never silently skipped.
  jq -e -s 'length > 0' "$f" >/dev/null
  jq -S -s "$JQ_STATS"'
    . as $all
    | (map(select(.final == true and .ok == true))) as $okf
    | {
        schema: "headstart-summary/v1",
        source: "samples.jsonl",
        n_lines: ($all | length),
        rules: {
          counted: "final attempt, exit status 0 (failed builds are never counted)",
          clean: "counted and contaminated == false",
          dispersion: "mad = median absolute deviation; stdev = sample standard deviation (n-1)",
          units: "seconds for wall_s, kilobytes for max_rss_kb, percent for cpu_percent"
        },
        groups: (
          group_by([.scenario, .arm])
          | map(
              . as $g
              | (map(select(.final == true and .ok == true))) as $ok
              | (map(select(.final == true and .ok == true and .contaminated == false))) as $clean
              | {
                  scenario: $g[0].scenario,
                  arm: $g[0].arm,
                  kind: $g[0].kind,
                  command: $g[0].command,
                  n_lines: ($g | length),
                  n_final: (map(select(.final == true)) | length),
                  n_failed: (map(select(.final == true and .ok == false)) | length),
                  n_superseded: (map(select(.final == false)) | length),
                  n_contaminated_counted: ($ok | map(select(.contaminated == true)) | length),
                  failures: (map(select(.ok == false)) | map({sample_id, exit_status})),
                  wall_s: ($ok | map(.wall_s) | stats),
                  wall_s_clean: ($clean | map(.wall_s) | stats),
                  max_rss_kb_median: ($ok | map(.time_v.max_rss_kb) | nn_median),
                  cpu_percent_median: ($ok | map(.time_v.cpu_percent) | nn_median)
                }
            )
        ),
        aa_noise: (
          [ $okf[] | select(.aa_pair != null) ] | group_by(.aa_pair)
          | map(
              (map(select(.arm == "A") | .wall_s) | median) as $a
              | (map(select(.arm == "B") | .wall_s) | median) as $b
              | {
                  scenario: .[0].aa_pair,
                  n_a: (map(select(.arm == "A")) | length),
                  n_b: (map(select(.arm == "B")) | length),
                  median_a_s: ($a | if . == null then null else r3 end),
                  median_b_s: ($b | if . == null then null else r3 end),
                  delta_s: (if $a == null or $b == null then null else ($b - $a) | r3 end),
                  delta_pct: (if $a == null or $b == null or $a == 0 then null
                              else (($b - $a) / $a * 100) | r3 end)
                }
            )
        )
      }
    # M5 (#536): OFF vs ON per scenario, present only when such arms exist,
    # so summaries of runs without them stay byte-identical.
    | ([ $all[] | select(.final == true and (.arm == "OFF" or .arm == "ON")) ]) as $abf
    # Times, RSS and CPU come from clean samples only (final, exit 0, not
    # contaminated); an arm without one is NOT_MEASURED, never a number.
    | if ($abf | length) == 0 then . else . + {ab_rules: {
          counted: "final attempt, exit status 0 and contaminated == false; contaminated finals are excluded and counted, failures are listed and never timed",
          status: "measured when both arms have a clean sample; else FAILED when a final sample failed, else NOT_MEASURED (no clean final sample: contaminated host)"},
        aa_noise_clean: (
          [ $okf[] | select(.aa_pair != null and .contaminated == false) ] | group_by(.aa_pair)
          | map(
              (map(select(.arm == "A") | .wall_s) | median) as $a
              | (map(select(.arm == "B") | .wall_s) | median) as $b
              | {
                  scenario: .[0].aa_pair,
                  n_a: (map(select(.arm == "A")) | length),
                  n_b: (map(select(.arm == "B")) | length),
                  median_a_s: ($a | if . == null then null else r3 end),
                  median_b_s: ($b | if . == null then null else r3 end),
                  delta_pct: (if $a == null or $b == null or $a == 0 then null
                              else (($b - $a) / $a * 100) | r3 end)
                }
            )
        ),
        ab: (
        $abf | group_by(.scenario) | map(
          . as $g
          | (map(select(.arm == "OFF" and .ok == true and .contaminated == false))) as $off
          | (map(select(.arm == "ON" and .ok == true and .contaminated == false))) as $on
          | ($off | map(.wall_s) | median) as $mo
          | ($on | map(.wall_s) | median) as $mn
          | ($off | map(.time_v.max_rss_kb) | nn_median) as $ro
          | ($on | map(.time_v.max_rss_kb) | nn_median) as $rn
          | ($off | map(if .time_v.user_s == null then null else .time_v.user_s + .time_v.sys_s end) | nn_median) as $co
          | ($on | map(if .time_v.user_s == null then null else .time_v.user_s + .time_v.sys_s end) | nn_median) as $cn
          | {
              scenario: $g[0].scenario,
              n_off: ($off | length), n_on: ($on | length),
              failed_off: (map(select(.arm == "OFF" and .ok == false)) | length),
              failed_on: (map(select(.arm == "ON" and .ok == false)) | length),
              failures: (map(select(.ok == false)) | map({sample_id, arm, exit_status})),
              contaminated_excluded_off: (map(select(.arm == "OFF" and .ok == true and .contaminated == true)) | length),
              contaminated_excluded_on: (map(select(.arm == "ON" and .ok == true and .contaminated == true)) | length),
              status: (if ($off | length) > 0 and ($on | length) > 0 then "measured"
                       elif any(.[]; .ok == false) then "FAILED"
                       else "NOT_MEASURED" end),
              ice_reports_on: (map(select(.arm == "ON" and has("ice_reports")) | .ice_reports)
                               | if length == 0 then null else add end),
              wall_s_off: ($off | map(.wall_s) | stats),
              wall_s_on: ($on | map(.wall_s) | stats),
              delta_s: (if $mo == null or $mn == null then null else ($mn - $mo) | r3 end),
              saving_pct: (if $mo == null or $mn == null or $mo == 0 then null
                           else (($mo - $mn) / $mo * 100) | r3 end),
              max_rss_kb_median_off: $ro, max_rss_kb_median_on: $rn,
              max_rss_delta_pct: (if $ro == null or $rn == null or $ro == 0 then null
                                  else (($rn - $ro) / $ro * 100) | r3 end),
              cpu_s_median_off: $co, cpu_s_median_on: $cn,
              cpu_s_delta_pct: (if $co == null or $cn == null or $co == 0 then null
                                else (($cn - $co) / $co * 100) | r3 end)
            }
        )
      )} end' "$f"
}

cmd_timings() {
  local html=$1
  [ -s "$html" ] || { echo "summarize: $html is missing or empty" >&2; exit 65; }
  # Cargo embeds the per-unit data as `const UNIT_DATA = [ ... ];` (JSON) in
  # the timing report. Cut that block out and validate it as JSON.
  local block
  block=$(sed -n '/^const UNIT_DATA = \[/,/^\];/p' "$html" | sed -e '1s/^const UNIT_DATA = //' -e '$s/;$//')
  [ -n "$block" ] || { echo "summarize: no UNIT_DATA block in $html" >&2; exit 65; }
  printf '%s\n' "$block" | jq -e 'type == "array" and length > 0' >/dev/null
  printf '%s\n' "$block" | jq -c .
}

cmd_critical() {
  local units=$1 jobs=${2:-null}
  jq -S --argjson jobs "$jobs" "$JQ_STATS"'
    # frontend end (seconds after unit start) = when its .rmeta is ready.
    def fe_end: if (.sections // null) == null then null
      else ([.sections[] | select(.[0] == "frontend") | .[1].end] | first) end;
    . as $u
    | ($u | map({key: (.i | tostring), value: .}) | from_entries) as $by
    # Each unit records which units became runnable when it finished
    # (unblocked_units) or when its metadata was ready (unblocked_rmeta_units).
    # That edge is the dependency that actually gated the unblocked unit.
    | ([ $u[] | .i as $d
         | ((.unblocked_units[]? | {key: tostring, value: {pred: $d, gate: "full"}}),
            (.unblocked_rmeta_units[]? | {key: tostring, value: {pred: $d, gate: "rmeta"}})) ]
       | from_entries) as $pred
    | ($u | max_by(.start + .duration)) as $last
    | [ $last.i | recurse(($pred[tostring] // empty).pred) ] | reverse
    | map($by[tostring]) as $chain
    | ($chain | to_entries | map(
        .value as $c
        | ($pred[$c.i | tostring]) as $p
        | ($p | if . == null then null else $by[.pred | tostring] end) as $pu
        | {
            i: $c.i, name: $c.name, version: $c.version, target: $c.target, mode: $c.mode,
            start_s: ($c.start | r3), duration_s: ($c.duration | r3),
            frontend_s: ($c | fe_end | if . == null then null else r3 end),
            gated_by: (if $p == null then null else {i: $p.pred, name: $pu.name, gate: $p.gate} end),
            gate_time_s: (if $p == null then null
                          elif $p.gate == "full" then ($pu.start + $pu.duration) | r3
                          else (($pu | fe_end) as $fe | if $fe == null then null else ($pu.start + $fe) | r3 end) end),
            pred_frontend_s: (if $p == null or $p.gate != "rmeta" then null
                              else ($pu | fe_end | if . == null then null else r3 end) end)
          })) as $links
    | ($u | map(.duration) | add) as $cpu
    | ($u | max_by(.duration)) as $longest
    | ($last.start + $last.duration) as $wall
    | ([ $links[] | select(.gated_by.gate == "rmeta") | .pred_frontend_s // 0 ] | add // 0) as $rmeta_wait
    | ([ $links[] | select(.gated_by.gate == "full") ] | length) as $n_full
    | (if $jobs == null then null else $cpu / $jobs end) as $cpu_floor
    | ([$longest.duration, ($cpu_floor // 0)] | max) as $floor
    | {
        schema: "headstart-critical-path/v1",
        units: ($u | length),
        observed: {
          build_end_s: ($wall | r3),
          sum_unit_duration_s: ($cpu | r3),
          longest_unit: {name: $longest.name, target: $longest.target, mode: $longest.mode, duration_s: ($longest.duration | r3)},
          chain_length: ($links | length),
          chain_full_gated_links: $n_full,
          chain_rmeta_gated_links: ([ $links[] | select(.gated_by.gate == "rmeta") ] | length),
          chain: $links
        },
        estimate: {
          label: "ESTIMATE, not a measurement",
          method: ("Walk back from the last unit to finish along the edge that unblocked each unit "
            + "(cargo unblocked_units / unblocked_rmeta_units). On rmeta-gated links the dependent "
            + "waited for the predecessor frontend (sections.frontend.end). A metadata-pipelining "
            + "compiler change can at most remove those waits on this chain; full-gated links "
            + "(build scripts, proc macros, linked binaries) need finished artifacts and are not "
            + "shortened. Upper bound = min(sum of predecessor frontend time on rmeta-gated chain links, "
            + "build_end - floor) where floor = max(longest unit, sum of unit durations / jobs). "
            + "This single-chain bound ignores that another chain may become critical; the realistic "
            + "estimate that accounts for the whole graph and the job limit is the list-schedule "
            + "replay (headstart-summarize.sh replay, timings/replay.json). Covers cold cargo build only."),
          scope: "cold cargo build --workspace only",
          jobs: $jobs,
          floor_s: ($floor | r3),
          rmeta_wait_on_chain_s: ($rmeta_wait | r3),
          upper_bound_saving_s: ([$rmeta_wait, ($wall - $floor)] | min | if . < 0 then 0 else . end | r3),
          upper_bound_saving_pct: (([$rmeta_wait, ($wall - $floor)] | min | if . < 0 then 0 else . end) / $wall * 100 | r3),
          realistic_estimate: "see timings/replay.json"
        }
      }' "$units"
}

# Package dependency graph from `cargo metadata --format-version 1
# --filter-platform <host>`: normal and build edges only (dev edges dropped,
# they are not part of `cargo build --workspace`).
cmd_graph() {
  local meta=$1
  [ -s "$meta" ] || { echo "summarize: $meta is missing or empty" >&2; exit 65; }
  jq -S '
    (.packages | map({key: .id, value: {
        name, version, links,
        build_script: any(.targets[]; .kind | index("custom-build")),
        proc_macro: any(.targets[]; .kind | index("proc-macro"))}}) | from_entries) as $p
    | [ .resolve.nodes[] | . as $n | $p[$n.id] + {
          deps: ([ $n.deps[]
                   | {pkg: $p[.pkg], kinds: ([.dep_kinds[].kind | . // "normal"] | unique | map(select(. != "dev")))}
                   | select(.kinds | length > 0)
                   | {name: .pkg.name, version: .pkg.version, kinds} ]
                 | sort_by(.name, .version)) } ]
    | sort_by(.name, .version)' "$meta"
}

# List-schedule replay of one cold `cargo build` (#532 estimate).
# Units and durations come from cargo-timing UNIT_DATA; edges come from the
# package graph with cargo unit rules: build-script compile needs its build
# dependencies (full); build-script run needs its compile and the runs of
# normal deps that declare `links`; a lib needs its own build-script run
# (full) and its normal deps (rmeta, or full for proc macros); a bin needs its
# own lib and its normal deps (full). Units with the same name, version and
# kind (feature variants) are treated as one group: a dependent waits for all
# of them (conservative). The replay runs twice at the given job count, with
# observed start order as priority: "baseline" (rmeta ready at frontend end)
# and "no_rmeta_wait" (rmeta ready when the dependency starts). The saving is
# baseline minus no_rmeta_wait; comparing baseline with the observed build
# end shows how well the model fits.
cmd_replay() {
  local units=$1 graph=$2 jobs=$3
  [ -s "$units" ] && [ -s "$graph" ] || { echo "summarize: replay needs units.json and graph.json" >&2; exit 65; }
  jq -S -n --slurpfile u "$units" --slurpfile g "$graph" --argjson jobs "$jobs" "$JQ_STATS"'
    def kind: if .mode == "run-custom-build" then "bsrun"
      elif (.target | test("build-script")) then "bscomp"
      elif (.target | test("\"bin\"")) then "bin" else "lib" end;
    def fe_end: if (.sections // null) == null then null
      else ([.sections[] | select(.[0] == "frontend") | .[1].end] | first) end;
    $u[0] as $U | $g[0] as $G
    | ($G | map({key: (.name + "@" + .version), value: .}) | from_entries) as $pkg
    | (reduce ($U | to_entries[]) as $e ({};
        .[$e.value.name + "@" + $e.value.version + "#" + ($e.value | kind)] += [$e.key])) as $lk
    | def look($nv; $k): $lk[$nv + "#" + $k] // [];
      [ $U | to_entries[] | .key as $i | .value as $x | ($x.name + "@" + $x.version) as $nv
        | ($pkg[$nv]) as $P
        | ($x | kind) as $k
        | (if $P == null then []
           elif $k == "bscomp" then
             [ $P.deps[] | select(.kinds | index("build")) | look(.name + "@" + .version; "lib")[] | {d: ., gate: "full"} ]
           elif $k == "bsrun" then
             [ look($nv; "bscomp")[] | {d: ., gate: "full"} ]
             + [ $P.deps[] | select(.kinds | index("normal"))
                 | select(($pkg[.name + "@" + .version].links // null) != null)
                 | look(.name + "@" + .version; "bsrun")[] | {d: ., gate: "full"} ]
           elif $k == "lib" then
             [ look($nv; "bsrun")[] | {d: ., gate: "full"} ]
             + [ $P.deps[] | select(.kinds | index("normal")) | (.name + "@" + .version) as $dn
                 | look($dn; "lib")[] | {d: ., gate: (if $pkg[$dn].proc_macro then "full" else "rmeta" end)} ]
           else
             [ look($nv; "lib")[] | {d: ., gate: "full"} ]
             + [ $P.deps[] | select(.kinds | index("normal")) | look(.name + "@" + .version; "lib")[] | {d: ., gate: "full"} ]
           end) as $deps
        | {i: $i, dur: $x.duration, fe: ($x | fe_end // $x.duration), prio: $x.start, matched: ($P != null),
           deps: ($deps | map(select(.d != $i)) | unique)} ] as $N
    | ($N | length) as $n
    | def sim($mode):
        def avail($st; $e): $st[$e.d] as $s
          | if $s == null then null
            elif $e.gate == "full" then $s + $N[$e.d].dur
            elif $mode == "nowait" then $s
            else $s + $N[$e.d].fe end;
        def rt($st; $i): [ $N[$i].deps[] | avail($st; .) ] as $a
          | if any($a[]; . == null) then null else ($a | max // 0) end;
        {t: 0, st: [range($n) | null], stuck: false}
        | until((.stuck) or all(.st[]; . != null);
            . as $s
            | ([ range($n) | select($s.st[.] != null and $s.st[.] + $N[.].dur > $s.t) ] | length) as $busy
            | ([ range($n) | select($s.st[.] == null) | {i: ., r: rt($s.st; .)} | select(.r != null) ]) as $rd
            | ([ $rd[] | select(.r <= $s.t) ] | sort_by($N[.i].prio, .i) | .[0:([$jobs - $busy, 0] | max)]) as $go
            | if ($go | length) > 0 then
                reduce $go[] as $x ($s; .st[$x.i] = $s.t)
              else
                ([ (range($n) | select($s.st[.] != null) | $s.st[.] + $N[.].dur | select(. > $s.t)),
                   ($rd[] | .r | select(. > $s.t)) ] | min) as $nt
                | if $nt == null then .stuck = true else .t = $nt end
              end)
        | . as $s
        | {end_s: ([ range($n) | select($s.st[.] != null) | $s.st[.] + $N[.].dur ] | max // 0 | r3),
           deadlock: .stuck};
    (sim("baseline")) as $b | (sim("nowait")) as $w
    | ($U | map(.start + .duration) | max) as $obs
    | {
        schema: "headstart-replay/v1",
        label: "ESTIMATE, not a measurement",
        scope: "cold cargo build --workspace only",
        jobs: $jobs,
        units: $n,
        units_without_package: ($N | map(select(.matched | not)) | length),
        edges: ($N | map(.deps | length) | add),
        rmeta_edges: ($N | map(.deps[] | select(.gate == "rmeta")) | length),
        observed_build_end_s: ($obs | r3),
        baseline_model_end_s: $b.end_s,
        model_error_pct: (if $obs == 0 then null else (($b.end_s - $obs) / $obs * 100 | r3) end),
        no_rmeta_wait_model_end_s: $w.end_s,
        realistic_saving_s: (($b.end_s - $w.end_s) | r3),
        realistic_saving_pct: (if $b.end_s == 0 then null else (($b.end_s - $w.end_s) / $b.end_s * 100 | r3) end),
        deadlock: ($b.deadlock or $w.deadlock),
        method: ("List-schedule replay of UNIT_DATA durations over the cargo metadata package graph "
          + "(see headstart-summarize.sh cmd_replay for the unit rules) at the given job count, priority "
          + "= observed start order. baseline: an rmeta edge is ready at the dependency frontend end; "
          + "no_rmeta_wait: an rmeta edge is ready when the dependency starts, the most a metadata-pipelining "
          + "change could give. realistic_saving = baseline - no_rmeta_wait, both from the same model, so "
          + "model error largely cancels. Feature-variant units are grouped conservatively.")
      }'
}

[ $# -ge 2 ] || usage
case "$1" in
  samples) cmd_samples "$2" ;;
  timings) cmd_timings "$2" ;;
  critical) cmd_critical "$2" "${3:-null}" ;;
  graph) cmd_graph "$2" ;;
  replay) [ $# -ge 4 ] || usage; cmd_replay "$2" "$3" "$4" ;;
  *) usage ;;
esac
