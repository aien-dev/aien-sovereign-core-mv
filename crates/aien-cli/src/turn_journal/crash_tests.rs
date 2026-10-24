//! Process-death checks for #665. A real durable append is the disposable tool.
//! These exercise the production journal and settlement functions without a model.

use super::*;
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CHILD_TEST: &str = "turn_journal::crash_tests::crash_child";

// Reap the subprocess even if a parent assertion or readiness timeout fails.
struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn append_effect(dir: &Path) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("effects"))
        .unwrap();
    file.write_all(b"committed\n").unwrap();
    file.sync_all().unwrap();
}

fn hold(dir: &Path) -> ! {
    std::fs::write(dir.join("ready"), b"ready").unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
#[ignore = "subprocess entry point, invoked by crash boundary tests"]
fn crash_child() {
    let dir = PathBuf::from(std::env::var_os("AIEN_CRASH_TEST_DIR").unwrap());
    let phase = std::env::var("AIEN_CRASH_TEST_PHASE").unwrap();
    assert!(matches!(
        phase.as_str(),
        "before_effect" | "after_effect" | "after_outcome"
    ));
    // Pre-create and sync the effect ledger so the test measures process death,
    // not persistence of a newly created directory entry across power failure.
    File::create(dir.join("effects"))
        .unwrap()
        .sync_all()
        .unwrap();
    let mut journal = Journal::create(&dir.join("session.jsonl")).unwrap();
    let mut messages = Vec::new();
    run_tool_calls(
        &mut journal,
        &mut messages,
        vec![("write_to_file".into(), json!({"path": "effects"}))],
        |_, _| {
            if phase == "before_effect" {
                hold(&dir);
            }
            append_effect(&dir);
            if phase == "after_effect" {
                // The effect is durable, but dispatch has not returned, so the
                // journal cannot yet contain its successful outcome.
                hold(&dir);
            }
            json!({"status": "committed"})
        },
    )
    .unwrap();
    hold(&dir);
}

fn crash_and_resume(phase: &str, expected_effect: &str, expected_pending: usize) {
    let dir = tempfile::tempdir().unwrap();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CHILD_TEST, "--ignored", "--test-threads=1"])
            .env("AIEN_CRASH_TEST_DIR", dir.path())
            .env("AIEN_CRASH_TEST_PHASE", phase)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(20);
    while !dir.path().join("ready").exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "child exited before {phase}"
        );
        assert!(Instant::now() < deadline, "child never reached {phase}");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        std::fs::read_to_string(dir.path().join("effects")).unwrap(),
        expected_effect
    );
    child.0.kill().unwrap();
    assert_eq!(
        child.0.wait().unwrap().signal(),
        Some(9),
        "expected SIGKILL"
    );

    let path = dir.path().join("session.jsonl");
    let (mut journal, replay) = Journal::resume(&path).unwrap();
    assert_eq!(replay.in_doubt.len(), expected_pending);
    assert!(!replay.torn_tail);
    let mut messages = replay.messages;
    let steps = settle_in_doubt(&mut journal, &mut messages, &replay.in_doubt, |_, _| {
        // A regression that replays a write produces an observable extra effect.
        append_effect(dir.path());
        json!({"status": "committed"})
    })
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("effects")).unwrap(),
        expected_effect,
        "recovery must not execute an effectful tool"
    );
    assert_eq!(
        messages.len(),
        1,
        "exactly one outcome reaches the conversation"
    );
    let content = messages[0]["content"].as_str().unwrap();
    if expected_pending == 1 {
        assert!(matches!(
            steps.as_slice(),
            [ResumeStep::ReportUnknown { .. }]
        ));
        assert!(content.contains("\"status\":\"outcome_unknown\""));
        assert!(
            !content.contains("\"status\":\"committed\""),
            "lost response is not success"
        );
    } else {
        assert!(steps.is_empty());
        assert!(content.contains("\"status\":\"committed\""));
    }
    drop(journal);

    // A second journal reopen must neither replay the effect nor add a second result.
    let (mut journal, replay) = Journal::resume(&path).unwrap();
    assert!(replay.in_doubt.is_empty());
    assert_eq!(replay.messages, messages);
    let mut again = replay.messages;
    settle_in_doubt(&mut journal, &mut again, &replay.in_doubt, |_, _| {
        append_effect(dir.path());
        json!({"status": "committed"})
    })
    .unwrap();
    assert_eq!(again, messages);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("effects")).unwrap(),
        expected_effect
    );
}

#[test]
fn killed_before_effect_does_not_invent_success_or_execute_on_resume() {
    crash_and_resume("before_effect", "", 1);
}

#[test]
fn killed_after_effect_before_outcome_does_not_repeat_the_committed_effect() {
    crash_and_resume("after_effect", "committed\n", 1);
}

#[test]
fn killed_after_outcome_preserves_success_without_repeating_the_effect() {
    crash_and_resume("after_outcome", "committed\n", 0);
}
