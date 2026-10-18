//! Verdict set and derivation (ADR 0028 Decision 5). A test program only
//! reports observations; the verdict is derived here from the manifest's
//! expectations and facts the runner measured. Only the last row can yield
//! PASS.
//!
//! Scope (Slice B): rows 2 to 15 are implemented as below, including the
//! hardware verdict (row 3) and the dependency verdicts (rows 4 and 5). Where
//! the runner still cannot do what a row presumes (build, symbol checks,
//! operator attestations) it refuses with NOT_RUN or BLOCKED_OPERATOR and a
//! reason code; it never guesses a PASS.
//!
//! Deviation from the ADR table order (recorded): timeout (row 14) is tested
//! before the exit status (row 10), because a killed process has no exit code
//! and would otherwise always be reported as rc_nonzero.

use crate::manifest::{parse_json_scalar, valid_obs_name, Manifest, Op, Quant, Rule};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    NotRun,
    BlockedHardware,
    BlockedOperator,
    MissingImplementation,
}

impl Verdict {
    pub const ALL: [Verdict; 6] = [
        Verdict::Pass,
        Verdict::Fail,
        Verdict::NotRun,
        Verdict::BlockedHardware,
        Verdict::BlockedOperator,
        Verdict::MissingImplementation,
    ];

    /// The verdict with this exact name, as written by `as_str`.
    pub fn parse(s: &str) -> Option<Verdict> {
        Verdict::ALL.into_iter().find(|v| v.as_str() == s)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::NotRun => "NOT_RUN",
            Verdict::BlockedHardware => "BLOCKED_HARDWARE",
            Verdict::BlockedOperator => "BLOCKED_OPERATOR",
            Verdict::MissingImplementation => "MISSING_IMPLEMENTATION",
        }
    }

    /// CLI exit code: 0 PASS, 1 FAIL, 2 NOT_RUN, 3 BLOCKED_HARDWARE,
    /// 4 BLOCKED_OPERATOR, 5 MISSING_IMPLEMENTATION.
    pub fn exit_code(self) -> i32 {
        match self {
            Verdict::Pass => 0,
            Verdict::Fail => 1,
            Verdict::NotRun => 2,
            Verdict::BlockedHardware => 3,
            Verdict::BlockedOperator => 4,
            Verdict::MissingImplementation => 5,
        }
    }
}

/// The verdict of a campaign of gates: any FAIL makes it FAIL, otherwise the
/// highest-numbered of NOT_RUN, BLOCKED_HARDWARE, BLOCKED_OPERATOR and
/// MISSING_IMPLEMENTATION (ADR 0028 Decision 5), otherwise PASS. An empty list
/// is PASS here; a caller that ran nothing must say so itself.
pub fn campaign_verdict(vs: &[Verdict]) -> Verdict {
    if vs.contains(&Verdict::Fail) {
        return Verdict::Fail;
    }
    vs.iter()
        .copied()
        .max_by_key(|v| v.exit_code())
        .unwrap_or(Verdict::Pass)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub name: String,
    pub value: Value,
}

/// Parse `AIEN-OBS <name> <json-value>` lines. Anything else, including a
/// malformed AIEN-OBS line (bad name, bad JSON, float), is ordinary log text.
pub fn parse_observations(stdout: &[u8]) -> Vec<Observation> {
    let text = String::from_utf8_lossy(stdout);
    let mut out = Vec::new();
    for line in text.lines() {
        let rest = match line.strip_prefix("AIEN-OBS ") {
            Some(r) => r,
            None => continue,
        };
        let (name, json) = match rest.split_once(' ') {
            Some(p) => p,
            None => continue,
        };
        if !valid_obs_name(name) {
            continue;
        }
        if let Some(value) = parse_json_scalar(json.trim()) {
            out.push(Observation {
                name: name.to_string(),
                value,
            });
        }
    }
    out
}

fn compare(op: Op, a: &Value, b: &Value) -> bool {
    match op {
        Op::Eq => a == b,
        Op::Ne => a != b,
        _ => match (a.as_i64(), b.as_i64()) {
            (Some(x), Some(y)) => match op {
                Op::Lt => x < y,
                Op::Le => x <= y,
                Op::Gt => x > y,
                _ => x >= y,
            },
            _ => false,
        },
    }
}

pub fn eval_rule(rule: &Rule, obs: &[Observation]) -> bool {
    let vals: Vec<&Value> = obs
        .iter()
        .filter(|o| o.name == rule.name)
        .map(|o| &o.value)
        .collect();
    if vals.is_empty() {
        return false;
    }
    match rule.quant {
        Quant::Last => compare(rule.op, vals[vals.len() - 1], &rule.value),
        Quant::All => vals.iter().all(|v| compare(rule.op, v, &rule.value)),
        Quant::Any => vals.iter().any(|v| compare(rule.op, v, &rule.value)),
    }
}

/// What the runner knows about one dependency of a gate: its current verdict
/// (None when there is no usable receipt: absent, or not for this commit) and
/// the receipt that verdict came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepFact {
    pub gate: String,
    pub verdict: Option<Verdict>,
    pub receipt: Option<String>,
}

fn dep_verdict(deps: &[DepFact], gate: &str) -> Option<Verdict> {
    deps.iter().find(|d| d.gate == gate).and_then(|d| d.verdict)
}

/// Facts the runner measured. For a gate that did not run, `exit_status` is
/// None and `observations` is empty.
#[derive(Debug, Clone, Default)]
pub struct Facts {
    /// Refusal id decided before running (dirty_tree, evidence_inside, ...).
    pub refusal: Option<String>,
    pub exec_exists: bool,
    /// Current verdicts of the gate's `depends_on` gates. A dependency with no
    /// entry here counts as having no verdict.
    pub deps: Vec<DepFact>,
    /// `machine.gb10_present` and `machine.qemu_available` (row 3).
    pub gb10_present: bool,
    pub qemu_available: bool,
    pub exit_status: Option<i64>,
    pub observations: Vec<Observation>,
    pub timed_out: bool,
    /// `git status --porcelain` differed after the run.
    pub tree_changed_after: bool,
    pub head_moved: bool,
    /// The operator started this run by hand (`--operator`). It is the only
    /// thing that lets a `requires: operator` gate run (ADR 0033 Decision 2).
    pub operator: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Derived {
    pub verdict: Verdict,
    pub reason: String,
    /// Each `expects.observe` rule and whether it held.
    pub rules: Vec<(String, bool)>,
}

/// Rows 2 to 7: decided before anything runs. Some(..) means do not run.
pub fn pre_run(m: &Manifest, f: &Facts) -> Option<(Verdict, String)> {
    if !f.operator && m.requires.iter().any(|r| r == "operator") {
        // Row 2: an operator gate is never scheduled automatically (ADR 0033
        // Decision 2). The operator starting the run by hand (`--operator`)
        // is the attestation; nothing else lifts this (sc #614).
        return Some((
            Verdict::BlockedOperator,
            "OPERATOR_STEP_PENDING".to_string(),
        ));
    }
    // Row 3: the gate's pool needs hardware this machine does not have.
    match m.pool() {
        "gb10" if !f.gb10_present => {
            return Some((Verdict::BlockedHardware, "NO_GB10".to_string()));
        }
        "qemu" if !f.qemu_available => {
            return Some((Verdict::BlockedHardware, "NO_QEMU".to_string()));
        }
        _ => {}
    }
    // Row 4 before row 5: a blocked dependency passes its blocked verdict on.
    for d in &m.depends_on {
        if let Some(v @ (Verdict::BlockedHardware | Verdict::BlockedOperator)) =
            dep_verdict(&f.deps, d)
        {
            return Some((v, format!("DEP_BLOCKED:{d}")));
        }
    }
    // Row 5: every other dependency must currently be PASS.
    if let Some(d) = m
        .depends_on
        .iter()
        .find(|d| dep_verdict(&f.deps, d) != Some(Verdict::Pass))
    {
        return Some((Verdict::NotRun, format!("DEP_NOT_PASS:{d}")));
    }
    if let Some(b) = &m.build {
        if b.tool != "none" {
            return Some((Verdict::NotRun, "build_unsupported".to_string()));
        }
    }
    if let Some(c) = m.checks.iter().find(|c| c.as_str() != "clean_tree_after") {
        return Some((Verdict::NotRun, format!("check_unsupported:{c}")));
    }
    if let Some(r) = &f.refusal {
        return Some((Verdict::NotRun, r.clone()));
    }
    if !f.exec_exists {
        return Some((Verdict::MissingImplementation, "NO_RUN_EXEC".to_string()));
    }
    None
}

/// Exit status with which a program says it could not run because an input it
/// does not own is absent (a model checkpoint, say), together with a
/// `not_run_reason` observation holding a plain lowercase name. The verdict is
/// then NOT_RUN with that reason: a refusal, never PASS. Without a valid
/// reason, or when the manifest itself expects 77, it is an ordinary exit.
pub const SELF_NOT_RUN_EXIT: i64 = 77;

fn self_not_run(m: &Manifest, f: &Facts) -> Option<String> {
    if f.timed_out || f.exit_status != Some(SELF_NOT_RUN_EXIT) || m.expect_exit == SELF_NOT_RUN_EXIT
    {
        return None;
    }
    let o = f
        .observations
        .iter()
        .rev()
        .find(|o| o.name == "not_run_reason")?;
    let s = o.value.as_str()?;
    let plain = !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    plain.then(|| s.to_string())
}

pub fn derive(m: &Manifest, f: &Facts) -> Derived {
    if let Some((verdict, reason)) = pre_run(m, f) {
        return Derived {
            verdict,
            reason,
            rules: m.rules.iter().map(|r| (r.text.clone(), false)).collect(),
        };
    }
    let rules: Vec<(String, bool)> = m
        .rules
        .iter()
        .map(|r| (r.text.clone(), eval_rule(r, &f.observations)))
        .collect();
    let (verdict, reason) = if f.timed_out {
        (Verdict::Fail, "timeout".to_string())
    } else if let Some(r) = self_not_run(m, f) {
        (Verdict::NotRun, r)
    } else if f.exit_status != Some(m.expect_exit) {
        (Verdict::Fail, "rc_nonzero".to_string())
    } else if let Some(idx) = rules.iter().position(|r| !r.1) {
        (Verdict::Fail, format!("rule:{idx}"))
    } else if f.tree_changed_after {
        (Verdict::Fail, "dirty_after".to_string())
    } else if f.head_moved {
        (Verdict::Fail, "head_moved".to_string())
    } else {
        (Verdict::Pass, String::new())
    };
    Derived {
        verdict,
        reason,
        rules,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;
    use serde_json::json;

    fn m(extra_replace: Option<(&str, &str)>) -> Manifest {
        let mut t = String::from(
            "gate: G\nmanifest_version: 1\nowner: o\nrequires:\n  - host\nrun:\n  exec: x.sh\nexpects:\n  exit: 0\n  observe: [\"n == 3\", \"all k <= 2\"]\n  verdict: PASS\ntimeout: 5s\n",
        );
        if let Some((a, b)) = extra_replace {
            assert!(t.contains(a), "replace target missing: {a}");
            t = t.replace(a, b);
        }
        manifest::parse(t.as_bytes()).unwrap()
    }

    fn obs(name: &str, v: Value) -> Observation {
        Observation {
            name: name.to_string(),
            value: v,
        }
    }

    fn good_facts() -> Facts {
        Facts {
            refusal: None,
            exec_exists: true,
            deps: Vec::new(),
            gb10_present: true,
            qemu_available: true,
            exit_status: Some(0),
            observations: vec![obs("n", json!(3)), obs("k", json!(1)), obs("k", json!(2))],
            timed_out: false,
            tree_changed_after: false,
            head_moved: false,
            operator: false,
        }
    }

    #[test]
    fn exit_codes_and_names_are_fixed() {
        let all = [
            (Verdict::Pass, "PASS", 0),
            (Verdict::Fail, "FAIL", 1),
            (Verdict::NotRun, "NOT_RUN", 2),
            (Verdict::BlockedHardware, "BLOCKED_HARDWARE", 3),
            (Verdict::BlockedOperator, "BLOCKED_OPERATOR", 4),
            (Verdict::MissingImplementation, "MISSING_IMPLEMENTATION", 5),
        ];
        for (v, name, code) in all {
            assert_eq!(v.as_str(), name);
            assert_eq!(v.exit_code(), code);
        }
    }

    #[test]
    fn all_expectations_met_is_pass() {
        let d = derive(&m(None), &good_facts());
        assert_eq!(d.verdict, Verdict::Pass);
        assert_eq!(d.reason, "");
        assert!(d.rules.iter().all(|r| r.1));
    }

    #[test]
    fn rule_mismatch_is_fail_with_index() {
        let mut f = good_facts();
        f.observations[0] = obs("n", json!(4));
        let d = derive(&m(None), &f);
        assert_eq!(d.verdict, Verdict::Fail);
        assert_eq!(d.reason, "rule:0");
        assert!(!d.rules[0].1);
        assert!(d.rules[1].1);
    }

    #[test]
    fn printing_pass_cannot_make_a_pass() {
        let mut f = good_facts();
        f.observations = parse_observations(b"PASS\nAIEN-OBS verdict \"PASS\"\n");
        let d = derive(&m(None), &f);
        assert_eq!(d.verdict, Verdict::Fail);
    }

    #[test]
    fn missing_observation_does_not_hold() {
        let mut f = good_facts();
        f.observations.retain(|o| o.name != "n");
        assert_eq!(derive(&m(None), &f).reason, "rule:0");
    }

    #[test]
    fn exit_77_with_a_reason_is_not_run_and_nothing_else_is() {
        let reason = |v: Value| Observation {
            name: "not_run_reason".into(),
            value: v,
        };
        let mut f = good_facts();
        f.exit_status = Some(77);
        f.observations.push(reason(json!("model_missing")));
        let d = derive(&m(None), &f);
        assert_eq!(
            (d.verdict, d.reason.as_str()),
            (Verdict::NotRun, "model_missing")
        );
        // No reason, or a reason that is not a plain name: an ordinary failure.
        f.observations.retain(|o| o.name != "not_run_reason");
        assert_eq!(derive(&m(None), &f).reason, "rc_nonzero");
        for bad in [json!("Model Missing"), json!(""), json!(3), json!("a;b")] {
            let mut g = f.clone();
            g.observations.push(reason(bad));
            assert_eq!(derive(&m(None), &g).reason, "rc_nonzero");
        }
        // The reason never turns another exit into NOT_RUN, and a timeout stays a timeout.
        let mut g = good_facts();
        g.exit_status = Some(1);
        g.observations.push(reason(json!("model_missing")));
        assert_eq!(derive(&m(None), &g).reason, "rc_nonzero");
        let mut g = f.clone();
        g.observations.push(reason(json!("model_missing")));
        g.timed_out = true;
        assert_eq!(derive(&m(None), &g).reason, "timeout");
        // A manifest that expects 77 keeps its own meaning.
        let m77 = m(Some(("exit: 0", "exit: 77")));
        assert_eq!(derive(&m77, &g.clone()).reason, "timeout");
        g.timed_out = false;
        assert_eq!(derive(&m77, &g).verdict, Verdict::Pass);
    }
    #[test]
    fn exit_status_must_match_expectation() {
        let mut f = good_facts();
        f.exit_status = Some(1);
        let d = derive(&m(None), &f);
        assert_eq!(
            (d.verdict, d.reason.as_str()),
            (Verdict::Fail, "rc_nonzero")
        );
        f.exit_status = None;
        assert_eq!(derive(&m(None), &f).reason, "rc_nonzero");
        // A manifest that expects exit 7 passes on 7 and fails on 0.
        let m7 = m(Some(("exit: 0", "exit: 7")));
        f.exit_status = Some(7);
        assert_eq!(derive(&m7, &f).verdict, Verdict::Pass);
        f.exit_status = Some(0);
        assert_eq!(derive(&m7, &f).verdict, Verdict::Fail);
    }

    #[test]
    fn timeout_beats_everything_after_run() {
        let mut f = good_facts();
        f.timed_out = true;
        f.exit_status = None;
        let d = derive(&m(None), &f);
        assert_eq!((d.verdict, d.reason.as_str()), (Verdict::Fail, "timeout"));
    }

    #[test]
    fn tree_and_head_changes_fail() {
        let mut f = good_facts();
        f.tree_changed_after = true;
        assert_eq!(derive(&m(None), &f).reason, "dirty_after");
        f.tree_changed_after = false;
        f.head_moved = true;
        assert_eq!(derive(&m(None), &f).reason, "head_moved");
    }

    #[test]
    fn first_reason_wins() {
        let mut f = good_facts();
        f.exit_status = Some(9);
        f.observations.clear();
        f.tree_changed_after = true;
        assert_eq!(derive(&m(None), &f).reason, "rc_nonzero");
    }

    #[test]
    fn refusal_is_not_run() {
        let mut f = good_facts();
        f.refusal = Some("dirty_tree".to_string());
        let d = derive(&m(None), &f);
        assert_eq!(
            (d.verdict, d.reason.as_str()),
            (Verdict::NotRun, "dirty_tree")
        );
    }

    #[test]
    fn missing_exec_is_missing_implementation() {
        let mut f = good_facts();
        f.exec_exists = false;
        let d = derive(&m(None), &f);
        assert_eq!(d.verdict, Verdict::MissingImplementation);
        assert_eq!(d.reason, "NO_RUN_EXEC");
    }

    #[test]
    fn operator_gate_is_blocked_operator() {
        let mo = m(Some(("  - host\n", "  - host\n  - operator\n")));
        let d = derive(&mo, &good_facts());
        assert_eq!(d.verdict, Verdict::BlockedOperator);
        assert_eq!(d.reason, "OPERATOR_STEP_PENDING");
    }

    #[test]
    fn operator_gate_runs_when_the_operator_started_it() {
        // sc #614: `--operator` is the operator starting the gate by hand
        // (ADR 0033 Decision 2); the facts then decide like any other gate.
        let mo = m(Some(("  - host\n", "  - host\n  - operator\n")));
        let f = Facts {
            operator: true,
            ..good_facts()
        };
        assert_eq!(pre_run(&mo, &f), None);
        let d = derive(&mo, &f);
        assert_eq!((d.verdict, d.reason.as_str()), (Verdict::Pass, ""));
        // The other rows still apply to an operator-started gate.
        let gb = gate_with("  - gb10\n  - operator\n", "  - OTHER\n");
        let f = Facts {
            operator: true,
            gb10_present: false,
            ..good_facts()
        };
        assert_eq!(derive(&gb, &f).reason, "NO_GB10");
    }

    #[test]
    fn missing_hardware_blocks_the_gate() {
        let mut f = good_facts();
        let g = m(Some(("  - host\n", "  - gb10\n")));
        assert_eq!(derive(&g, &f).verdict, Verdict::Pass);
        f.gb10_present = false;
        let d = derive(&g, &f);
        assert_eq!(
            (d.verdict, d.reason.as_str()),
            (Verdict::BlockedHardware, "NO_GB10")
        );
        let q = m(Some(("  - host\n", "  - qemu\n")));
        assert_eq!(
            derive(&q, &f).verdict,
            Verdict::Pass,
            "gb10 is irrelevant to qemu"
        );
        f.qemu_available = false;
        let d = derive(&q, &f);
        assert_eq!(
            (d.verdict, d.reason.as_str()),
            (Verdict::BlockedHardware, "NO_QEMU")
        );
        // A host gate never needs the special hardware.
        assert_eq!(derive(&m(None), &f).verdict, Verdict::Pass);
        // gb10 beats qemu when both are listed (the scarcest pool decides).
        let both = m(Some(("  - host\n", "  - qemu\n  - gb10\n")));
        f.qemu_available = true;
        assert_eq!(derive(&both, &f).reason, "NO_GB10");
    }

    fn dep(gate: &str, verdict: Option<Verdict>) -> DepFact {
        DepFact {
            gate: gate.to_string(),
            verdict,
            receipt: verdict.map(|_| format!("digest-of-{gate}")),
        }
    }

    fn gate_with(requires: &str, deps: &str) -> Manifest {
        let dep_block = if deps.is_empty() {
            String::new()
        } else {
            format!("depends_on:\n{deps}")
        };
        let text = format!(
            "gate: G\nmanifest_version: 1\nowner: o\nrequires:\n{requires}run:\n  exec: x.sh\nexpects:\n  exit: 0\n  observe: [\"n == 3\", \"all k <= 2\"]\n  verdict: PASS\ntimeout: 5s\n{dep_block}"
        );
        manifest::parse(text.as_bytes()).unwrap()
    }

    fn with_deps(list: &str) -> Manifest {
        gate_with("  - host\n", list)
    }

    #[test]
    fn a_passing_dependency_lets_the_gate_run() {
        let mut f = good_facts();
        f.deps = vec![dep("OTHER", Some(Verdict::Pass))];
        let d = derive(&with_deps("  - OTHER\n"), &f);
        assert_eq!(d.verdict, Verdict::Pass);
    }

    #[test]
    fn a_dependency_that_is_not_pass_makes_not_run() {
        let one = with_deps("  - OTHER\n");
        let mut f = good_facts();
        // No fact at all (absent or stale).
        let r = derive(&one, &f);
        assert_eq!(
            (r.verdict, r.reason.as_str()),
            (Verdict::NotRun, "DEP_NOT_PASS:OTHER")
        );
        for v in [
            None,
            Some(Verdict::Fail),
            Some(Verdict::NotRun),
            Some(Verdict::MissingImplementation),
        ] {
            f.deps = vec![dep("OTHER", v)];
            let r = derive(&one, &f);
            assert_eq!(
                (r.verdict, r.reason.as_str()),
                (Verdict::NotRun, "DEP_NOT_PASS:OTHER"),
                "dependency verdict {v:?}"
            );
            assert!(r.rules.iter().all(|x| !x.1), "nothing was checked");
        }
    }

    #[test]
    fn the_first_dependency_that_is_not_pass_is_named() {
        let two = with_deps("  - FIRST\n  - SECOND\n");
        let mut f = good_facts();
        f.deps = vec![dep("FIRST", Some(Verdict::Pass)), dep("SECOND", None)];
        assert_eq!(derive(&two, &f).reason, "DEP_NOT_PASS:SECOND");
        f.deps = vec![dep("FIRST", Some(Verdict::Fail)), dep("SECOND", None)];
        assert_eq!(derive(&two, &f).reason, "DEP_NOT_PASS:FIRST");
    }

    #[test]
    fn a_blocked_dependency_passes_its_verdict_on() {
        let one = with_deps("  - OTHER\n");
        let mut f = good_facts();
        f.deps = vec![dep("OTHER", Some(Verdict::BlockedOperator))];
        let r = derive(&one, &f);
        assert_eq!(
            (r.verdict, r.reason.as_str()),
            (Verdict::BlockedOperator, "DEP_BLOCKED:OTHER")
        );
        f.deps = vec![dep("OTHER", Some(Verdict::BlockedHardware))];
        let r = derive(&one, &f);
        assert_eq!(
            (r.verdict, r.reason.as_str()),
            (Verdict::BlockedHardware, "DEP_BLOCKED:OTHER")
        );
    }

    #[test]
    fn blocked_beats_failed_across_dependencies() {
        // Row 4 is checked before row 5, whatever the order in the manifest.
        let two = with_deps("  - FAILED\n  - BLOCKED\n");
        let mut f = good_facts();
        f.deps = vec![
            dep("FAILED", Some(Verdict::Fail)),
            dep("BLOCKED", Some(Verdict::BlockedHardware)),
        ];
        let r = derive(&two, &f);
        assert_eq!(
            (r.verdict, r.reason.as_str()),
            (Verdict::BlockedHardware, "DEP_BLOCKED:BLOCKED")
        );
    }

    #[test]
    fn rows_are_checked_in_order() {
        let mut f = good_facts();
        f.gb10_present = false;
        f.refusal = Some("dirty_tree".to_string());
        // Row 2 (operator) beats row 3 (hardware).
        let op = gate_with("  - gb10\n  - operator\n", "  - OTHER\n");
        assert_eq!(derive(&op, &f).reason, "OPERATOR_STEP_PENDING");
        // Row 3 (hardware) beats rows 4 and 5 (dependencies).
        let gb = gate_with("  - gb10\n", "  - OTHER\n");
        assert_eq!(derive(&gb, &f).reason, "NO_GB10");
        // Rows 4 and 5 beat the refusal (row 6).
        let host = gate_with("  - host\n", "  - OTHER\n");
        assert_eq!(derive(&host, &f).reason, "DEP_NOT_PASS:OTHER");
        f.deps = vec![dep("OTHER", Some(Verdict::Pass))];
        assert_eq!(derive(&host, &f).reason, "dirty_tree");
        // Row 6 beats row 7.
        f.exec_exists = false;
        assert_eq!(derive(&host, &f).reason, "dirty_tree");
        f.refusal = None;
        assert_eq!(derive(&host, &f).reason, "NO_RUN_EXEC");
    }

    #[test]
    fn campaign_verdict_follows_the_exit_code_rule() {
        use Verdict::*;
        assert_eq!(campaign_verdict(&[]), Pass);
        assert_eq!(campaign_verdict(&[Pass, Pass]), Pass);
        assert_eq!(campaign_verdict(&[Pass, NotRun]), NotRun);
        assert_eq!(
            campaign_verdict(&[NotRun, BlockedHardware]),
            BlockedHardware
        );
        assert_eq!(
            campaign_verdict(&[BlockedOperator, MissingImplementation, NotRun]),
            MissingImplementation
        );
        // Any FAIL wins, even over the higher-numbered codes.
        assert_eq!(campaign_verdict(&[MissingImplementation, Fail, Pass]), Fail);
        assert_eq!(campaign_verdict(&[Fail]).exit_code(), 1);
        assert_eq!(campaign_verdict(&[BlockedOperator, Pass]).exit_code(), 4);
    }

    #[test]
    fn verdict_names_parse_back() {
        for v in Verdict::ALL {
            assert_eq!(Verdict::parse(v.as_str()), Some(v));
        }
        assert_eq!(Verdict::parse("pass"), None);
        assert_eq!(Verdict::parse(""), None);
        assert_eq!(Verdict::ALL.len(), 6);
    }

    #[test]
    fn build_and_checks_are_not_run() {
        let b = m(Some((
            "timeout: 5s\n",
            "timeout: 5s\nbuild:\n  target: t\n  tool: make\n",
        )));
        assert_eq!(derive(&b, &good_facts()).reason, "build_unsupported");
        let c = m(Some((
            "timeout: 5s\n",
            "timeout: 5s\nchecks:\n  - no_libm_symbols\n",
        )));
        assert_eq!(
            derive(&c, &good_facts()).reason,
            "check_unsupported:no_libm_symbols"
        );
    }

    #[test]
    fn build_none_and_clean_tree_check_are_fine() {
        let b = m(Some((
            "timeout: 5s\n",
            "timeout: 5s\nbuild:\n  target: t\n  tool: none\nchecks:\n  - clean_tree_after\n",
        )));
        assert_eq!(derive(&b, &good_facts()).verdict, Verdict::Pass);
    }

    #[test]
    fn observation_parsing() {
        let o = parse_observations(
            b"noise\nAIEN-OBS a 1\nAIEN-OBS b \"x y\"\nAIEN-OBS c 1.5\nAIEN-OBS BAD 1\nAIEN-OBS d\nAIEN-OBS a 2\nAIEN-OBS e true\nAIEN-OBS f nope\n",
        );
        let names: Vec<&str> = o.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "a", "e"]);
        assert_eq!(o[1].value, json!("x y"));
    }

    #[test]
    fn quantifiers_and_ordering() {
        let r = manifest::parse_rule("any k > 5").unwrap();
        let o = vec![obs("k", json!(1)), obs("k", json!(9))];
        assert!(eval_rule(&r, &o));
        let r = manifest::parse_rule("k > 5").unwrap();
        assert!(eval_rule(&r, &o), "last value 9 > 5");
        let r = manifest::parse_rule("all k > 5").unwrap();
        assert!(!eval_rule(&r, &o));
        let r = manifest::parse_rule("k < 5").unwrap();
        assert!(!eval_rule(&r, &o));
        // Ordering over strings never holds.
        let r = manifest::parse_rule("s < 5").unwrap();
        assert!(!eval_rule(&r, &[obs("s", json!("a"))]));
        let r = manifest::parse_rule("s != \"a\"").unwrap();
        assert!(eval_rule(&r, &[obs("s", json!("b"))]));
    }
}
