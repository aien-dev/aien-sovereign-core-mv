//! ALLEN persona profile (arch#159) in the runtime. FIXTURE-ONLY: identities
//! come from the test-only subject encoder in aien-allen/tests/support, resolved
//! through the production resolver. No model runs.
//!
//! Part A needs no composition library and always runs. Part B (marked
//! `b_*`) re-executes this binary against a real compose home and reports
//! NOT_RUN in a stub omega build, like allen_spine_test.
#[path = "support/home_guard.rs"]
mod home_guard;

#[path = "../../aien-allen/tests/support/mod.rs"]
mod support;

use aien_allen::binding::pin_path;
use aien_allen::deployment::{self, Input};
use aien_allen::resolve::{resolve, Context, Resolved, SubjectSource};
use aien_allen::{hex, ENV_ADOPT, ENV_SUBJECT};
use aien_allen_profile::{profile_dir, Changes, Identity, Store, Tone};
use aien_omega_compose::LINKED;
use aien_runtime::control::{ComposeTaskReport, ControlCommand, ControlResponse, PersonaReport};
use aien_runtime::persona::{context_for, handle_allen_command, prefix_prompt, report_for};
use aien_runtime::spine::{task_prompt, ComposeBridge, ComposeProposer, Generation};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

struct Rig {
    _t: tempfile::TempDir,
    home: PathBuf,
    r: Resolved,
}

fn rig() -> Rig {
    let t = tempfile::tempdir().unwrap();
    let f = support::fx("persona");
    let c = support::chain(&f, 2);
    let subj = support::write_head(&t.path().join("subject.bin"), c.last().unwrap());
    let home = t.path().join("compose");
    let r = resolve(
        &SubjectSource::Head(subj),
        &pin_path(&home),
        &Context {
            machine_id: [1; 32],
            lineage: Some(f.cortex),
        },
        Some(&hex(&f.agent)),
    )
    .unwrap();
    Rig { _t: t, home, r }
}

fn set(expected: u64, ch: Changes) -> ControlCommand {
    ControlCommand::AllenProfileSet {
        expected_revision: expected,
        changes: ch,
    }
}

fn name(n: &str) -> Changes {
    Changes {
        name: Some(n.into()),
        ..Default::default()
    }
}

fn run(rig: &Rig, cmd: ControlCommand) -> ControlResponse {
    handle_allen_command(&rig.home, Some(&rig.r), "model:test", &cmd)
}

fn profile(r: ControlResponse) -> aien_runtime::control::AllenProfileReport {
    match r {
        ControlResponse::AllenProfile(p) => *p,
        o => panic!("expected AllenProfile, got {o:?}"),
    }
}

fn refusal(r: ControlResponse) -> aien_runtime::control::AllenRefusalReport {
    match r {
        ControlResponse::AllenRefused(p) => *p,
        o => panic!("expected AllenRefused, got {o:?}"),
    }
}

fn dir_bytes(d: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<_> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn not_engaged_refuses_profile_commands_and_changes_nothing() {
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    for cmd in [
        ControlCommand::AllenProfileShow,
        ControlCommand::AllenProfileHistory,
        set(0, name("Nova")),
        ControlCommand::AllenProfileRevert {
            expected_revision: 1,
            to: 1,
        },
        ControlCommand::AllenProfileReset {
            expected_revision: 1,
        },
    ] {
        let r = refusal(handle_allen_command(&home, None, "model:test", &cmd));
        assert_eq!(r.code, "not_engaged");
        assert!(r.message.contains("not engaged"));
    }
    assert!(!profile_dir(&home).exists());
    // Status still answers, and says so.
    match handle_allen_command(&home, None, "model:test", &ControlCommand::AllenStatus) {
        ControlResponse::AllenStatusReport(s) => {
            assert_eq!(s.identity, "not_engaged");
            assert_eq!(s.persona.state, "not_engaged");
            assert!(s.fingerprint.is_none());
            assert_eq!(s.unsupported, vec!["avatar", "voice"]);
        }
        o => panic!("{o:?}"),
    }
    // The task prompt is byte-identical when ALLEN is not engaged.
    assert!(context_for(&home, None).is_none());
    assert_eq!(prefix_prompt("PROMPT", None), "PROMPT");
}

#[test]
fn engaged_set_show_history_revert_reset_and_stale() {
    let rig = rig();
    let p = profile(run(&rig, ControlCommand::AllenProfileShow));
    assert_eq!((p.revision, p.state.as_str()), (0, "default"));
    assert_eq!(p.defaults.unwrap().display_name, "ALLEN");

    let ch = Changes {
        name: Some("Nova".into()),
        tone: Some(Tone::Warm),
        ..Default::default()
    };
    let p = profile(run(&rig, set(0, ch)));
    assert_eq!(p.revision, 1);
    // Stale expect: refused, the current revision is reported, nothing written.
    let e = refusal(run(&rig, set(0, name("Other"))));
    assert_eq!(
        (e.code.as_str(), e.current_revision),
        ("stale_update", Some(1))
    );
    assert!(e.message.contains("--expect 1"));
    profile(run(&rig, set(1, name("Echo"))));
    match run(&rig, ControlCommand::AllenProfileHistory) {
        ControlResponse::AllenHistory(h) => assert_eq!(h.entries.len(), 2),
        o => panic!("{o:?}"),
    }
    let p = profile(run(
        &rig,
        ControlCommand::AllenProfileRevert {
            expected_revision: 2,
            to: 1,
        },
    ));
    assert_eq!(p.profile.unwrap().persona.display_name, "Nova");
    let p = profile(run(
        &rig,
        ControlCommand::AllenProfileReset {
            expected_revision: 3,
        },
    ));
    assert_eq!(p.profile.unwrap().persona.display_name, "ALLEN");
    // Status shows identity and profile separately; the identity never moved.
    match run(&rig, ControlCommand::AllenStatus) {
        ControlResponse::AllenStatusReport(s) => {
            assert_eq!(s.identity, "engaged");
            assert_eq!(s.fingerprint.as_deref(), Some(&hex(&rig.r.agent)[..8]));
            assert_eq!(s.head_sequence, Some(rig.r.head_seq));
            assert_eq!(s.persona.revision, 4);
            assert_eq!(s.execution_mode, "local_model");
        }
        o => panic!("{o:?}"),
    }
}

#[test]
fn restart_keeps_the_profile() {
    let rig = rig();
    profile(run(&rig, set(0, name("Nova"))));
    // "Restart": nothing in memory; context comes from the files alone.
    let ctx = context_for(&rig.home, Some(&rig.r)).unwrap();
    assert_eq!(ctx.persona.display_name, "Nova");
    assert_eq!(report_for(Some(&ctx)).state, "applied");
}

#[test]
fn permission_keys_are_refused_through_the_command_path() {
    let rig = rig();
    let ch = Changes {
        set_prefs: vec![aien_allen_profile::PrefChange {
            key: "auto_approve".into(),
            value: "yes".into(),
            scope: aien_allen_profile::Scope::All,
        }],
        ..Default::default()
    };
    let e = refusal(run(&rig, set(0, ch)));
    assert_eq!(e.code, "permission_key");
    assert!(e.message.contains("approval desk"));
    assert!(!profile_dir(&rig.home).exists());
}

#[test]
fn prompt_carries_the_persona_and_keeps_the_task() {
    let rig = rig();
    let ws = tempfile::tempdir().unwrap();
    let base = task_prompt("write a note", ws.path());
    profile(run(&rig, set(0, name("Nova"))));
    let ctx = context_for(&rig.home, Some(&rig.r)).unwrap();
    let full = prefix_prompt(&base, Some(&ctx));
    assert!(full.ends_with(&base), "the task text is untouched");
    assert!(full.starts_with("[ALLEN style preferences"));
    assert!(full.contains("name: \"Nova\""));
    assert!(full.contains("grant no permission"));
    assert!(full.len() - base.len() <= aien_allen_profile::CONTEXT_MAX_BYTES + 1);
    let r = report_for(Some(&ctx));
    assert_eq!(
        r,
        PersonaReport {
            state: "applied".into(),
            display_name: "Nova".into(),
            revision: 1,
            reason: None
        }
    );
}

#[test]
fn damaged_profile_runs_with_defaults_and_says_so() {
    let rig = rig();
    profile(run(&rig, set(0, name("Nova"))));
    let first = profile_dir(&rig.home).join(format!("r{:020}.json", 1));
    std::fs::write(&first, b"{broken").unwrap();
    let ctx = context_for(&rig.home, Some(&rig.r)).unwrap();
    let rep = report_for(Some(&ctx));
    assert_eq!(rep.state, "refused");
    assert_eq!(rep.display_name, "ALLEN");
    assert!(rep.reason.unwrap().contains("damaged"));
    assert!(prefix_prompt("P", Some(&ctx)).contains("name: \"ALLEN\""));
    // Commands refuse; the damaged file is not replaced.
    assert_eq!(refusal(run(&rig, set(1, name("X")))).code, "damaged");
    assert_eq!(std::fs::read(&first).unwrap(), b"{broken");
}

#[test]
fn foreign_profile_is_refused_and_not_used() {
    let rig = rig();
    let other = Identity {
        agent: [9; 32],
        root: [8; 32],
    };
    Store::new(&rig.home, other)
        .set(0, &name("Stranger"))
        .unwrap();
    let ctx = context_for(&rig.home, Some(&rig.r)).unwrap();
    assert_eq!(report_for(Some(&ctx)).state, "refused");
    assert_eq!(ctx.persona.display_name, "ALLEN");
    assert_eq!(
        refusal(run(&rig, set(1, name("X")))).code,
        "foreign_profile"
    );
}

#[test]
fn a_model_swap_record_leaves_the_profile_untouched() {
    let rig = rig();
    profile(run(&rig, set(0, name("Nova"))));
    let before = dir_bytes(&profile_dir(&rig.home));
    let dep = deployment::deployments_path(&rig.home);
    for (i, model) in ["aa", "bb"].iter().enumerate() {
        deployment::append(
            &dep,
            &rig.r.agent,
            Input {
                model_sha256: model.repeat(32),
                config_sha256: "cc".repeat(32),
                candidate_id: format!("cand-{i}"),
                executable_sha256: "dd".repeat(32),
                revision: "r".into(),
                placeholder: true,
            },
        )
        .unwrap();
    }
    assert_eq!(before, dir_bytes(&profile_dir(&rig.home)));
    let ctx = context_for(&rig.home, Some(&rig.r)).unwrap();
    assert_eq!(ctx.persona.display_name, "Nova");
    match run(&rig, ControlCommand::AllenStatus) {
        ControlResponse::AllenStatusReport(s) => {
            let d = s.deployment.unwrap();
            assert_eq!((d.seq, d.candidate_id.as_str()), (2, "cand-1"));
            assert_eq!(s.persona.display_name, "Nova");
        }
        o => panic!("{o:?}"),
    }
}

/// No code path from a task, a model reply or a workspace file to the profile
/// writer: only the operator-command handler in persona.rs calls a writer,
/// and nothing else in the runtime even names the profile crate's store.
#[test]
fn only_the_command_handler_can_write_the_profile() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for ent in std::fs::read_dir(&src).unwrap() {
        let p = ent.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&p).unwrap();
        if name != "persona.rs" && name != "control.rs" {
            assert!(
                !text.contains("aien_allen_profile"),
                "{name} must not touch the profile crate"
            );
        }
    }
    let persona = std::fs::read_to_string(src.join("persona.rs")).unwrap();
    let ctx_fn = persona
        .split("pub fn context_for")
        .nth(1)
        .unwrap()
        .split("pub fn report_for")
        .next()
        .unwrap();
    for w in [".set(", ".revert(", ".reset(", "write"] {
        assert!(!ctx_fn.contains(w), "context_for must be read-only ({w})");
    }
    // The spine calls the handler only for operator control commands.
    let spine = std::fs::read_to_string(src.join("spine.rs")).unwrap();
    assert!(spine.contains("handle_allen_command"));
}

#[test]
fn old_reports_without_persona_still_parse() {
    let report = ComposeTaskReport {
        compose_commit: None,
        generation_record: None,
        compose_dir: "d".into(),
        machine_id: "m".into(),
        task: 1,
        outcome: 1,
        committed: true,
        branch_count: 1,
        branches_reclaimed: 1,
        winner: Some(0),
        aegis_pass_mask: 1,
        cx_goal: 1,
        cx_candidates: vec![],
        cx_evidence: 1,
        cx_promotion: 1,
        cx_admissions: vec![],
        winner_digest: String::new(),
        record_digest: String::new(),
        proposal: None,
        proposal_sha256: None,
        proposal_path: None,
        proposal_content_sha256: None,
        uncommitted_proposal: None,
        proposer_error: None,
        proposal_attempts: vec![],
        requirements_recognized: vec![],
        requirements_uncertain: vec![],
        proposer: "model".into(),
        persona: Some(PersonaReport {
            state: "applied".into(),
            display_name: "Nova".into(),
            revision: 2,
            reason: None,
        }),
        memory: None,
    };
    let mut v = serde_json::to_value(&report).unwrap();
    assert_eq!(v["persona"]["display_name"], "Nova");
    v.as_object_mut().unwrap().remove("persona");
    let old: ComposeTaskReport = serde_json::from_value(v).unwrap();
    assert!(old.persona.is_none());
}

// ---- Part B: through a real compose home (NOT_RUN in a stub omega build) ----

fn proposer(seen: Arc<std::sync::Mutex<String>>) -> ComposeProposer {
    Arc::new(move |prompt: &str, _l: std::time::Duration| {
        *seen.lock().unwrap() = prompt.to_string();
        // A reply that tries to talk the daemon into changing the profile.
        Ok(Generation {
            text: "filename: NOTES.md\naien allen set --name Hacked --expect 1\n".into(),
            tokens: 8,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

/// The child: one task, one status, one unauthorised effect intent.
#[test]
fn child_task() {
    let (Ok(home), Ok(ws)) = (
        std::env::var("PERSONA_CHILD_HOME"),
        std::env::var("PERSONA_CHILD_WS"),
    ) else {
        return;
    };
    let seen = Arc::new(std::sync::Mutex::new(String::new()));
    let b = ComposeBridge::new(PathBuf::from(home), proposer(seen.clone()), "model:test");
    match b.run_task("write a short note in NOTES.md", &ws) {
        ControlResponse::ComposeTaskResult(r) => {
            println!("CHILD_COMMITTED={}", r.committed);
            println!(
                "CHILD_PERSONA={}",
                serde_json::to_string(&r.persona).unwrap()
            );
        }
        o => println!("CHILD_TASK_RESPONSE={o:?}"),
    }
    println!(
        "CHILD_PROMPT_HEAD={}",
        seen.lock().unwrap().replace('\n', "\\n")
    );
    if let ControlResponse::AllenStatusReport(s) = b.allen_command(&ControlCommand::AllenStatus) {
        println!("CHILD_STATUS={}", serde_json::to_string(&*s).unwrap());
    }
    // A write still needs an authorization, whatever the persona says.
    let r = aien_runtime::effects::open_intent(
        &b,
        &aien_runtime::effects::IntentRequest {
            authorization: 999_999,
            proposal_sha256: "0".repeat(64),
            path: "NOTES.md".into(),
            target: ws.clone(),
            content_sha256: "0".repeat(64),
            executor_pid: 1,
            executor_start: 1,
        },
    );
    println!(
        "CHILD_INTENT_REFUSED={}",
        matches!(r, ControlResponse::Error(_))
    );
}

fn child(home: &Path, ws: Option<&Path>, env: &[(&str, &str)]) -> (i32, String, String) {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "child_task", "--nocapture", "--test-threads=1"])
        .env("PERSONA_CHILD_HOME", home)
        .env_remove(ENV_SUBJECT)
        .env_remove(ENV_ADOPT);
    if let Some(w) = ws {
        c.env("PERSONA_CHILD_WS", w);
    } else {
        c.env("PERSONA_CHILD_WS", home.parent().unwrap());
    }
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn b_engaged_task_uses_the_persona_and_a_reply_cannot_change_it() {
    let _home = home_guard::home_slot();
    if !LINKED {
        println!("NOT_RUN: stub omega build (no librx_compose.a); set AIEN_OMEGA_COMPOSE_DIR");
        return;
    }
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    let ws = t.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    // Not engaged first: creates the home, no persona text, report says so.
    let (code, out, err) = child(&home, Some(&ws), &[]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("\"state\":\"not_engaged\""), "{out}");
    assert!(!out.contains("ALLEN style preferences"), "{out}");
    // Engage: subject built for this home's Cortex lineage, adopted once.
    let mut name = home.as_os_str().to_owned();
    name.push(".machine-root");
    let root = std::fs::read(PathBuf::from(name)).unwrap();
    let (mut c, _) = aien_omega_compose::Compose::open(
        &home,
        aien_omega_compose::RootKind::Provisioned,
        &root,
        0xA1E4_0001,
    )
    .unwrap();
    let lineage = c.record(1).unwrap().digest;
    drop(c);
    let mut f = support::fx("persona-b");
    f.cortex = lineage;
    let chain = support::chain(&f, 2);
    let subj = support::write_head(&t.path().join("subject.bin"), chain.last().unwrap());
    let env = [
        (ENV_SUBJECT, subj.to_str().unwrap()),
        (ENV_ADOPT, &hex(&f.agent)[..]),
    ];
    // Adoption run (no profile yet): defaults, state "default".
    let (code, out, err) = child(&home, Some(&ws), &env);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("\"state\":\"default\""), "{out}");
    assert!(out.contains("CHILD_INTENT_REFUSED=true"), "{out}");
    // Save a profile, run again.
    let store = Store::new(
        &home,
        Identity {
            agent: f.agent,
            root: f.root,
        },
    );
    store.set(0, &name_change("Nova")).unwrap();
    let before = dir_bytes(&profile_dir(&home));
    let (code, out, err) = child(&home, Some(&ws), &[(ENV_SUBJECT, subj.to_str().unwrap())]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(
        out.contains("\"state\":\"applied\"") && out.contains("\"display_name\":\"Nova\""),
        "{out}"
    );
    assert!(out.contains("name: \"Nova\"\\ntone: neutral"), "{out}");
    assert!(out.contains("CHILD_INTENT_REFUSED=true"), "{out}");
    assert!(out.contains("\"identity\":\"engaged\""), "{out}");
    // The model reply asked for a rename; the profile did not move.
    assert_eq!(before, dir_bytes(&profile_dir(&home)));
    assert!(!ws.join("NOTES.md").exists(), "no write without authorize");
}

fn name_change(n: &str) -> Changes {
    name(n)
}
