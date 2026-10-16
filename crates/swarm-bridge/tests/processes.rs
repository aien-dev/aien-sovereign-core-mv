//! The real binary against the real synapse#83 board, one OS process per
//! call, the way Claude Code runs hooks and agents run Bash commands.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_swarm-bridge");
static N: AtomicU32 = AtomicU32::new(0);

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "swarm-bridge-proc-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn init(dir: &Path) -> PathBuf {
    let root = dir.join("board");
    let o = Command::new(BIN)
        .args(["init-board"])
        .arg(&root)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    root
}

/// Start one hook process (not yet waited on).
fn spawn_hook(
    dir: &Path,
    root: &Path,
    session: &str,
    event: &str,
    env: &[(&str, &str)],
) -> std::process::Child {
    let transcript = dir.join(format!("{session}.jsonl"));
    if !transcript.exists() {
        fs::write(&transcript, b"{\"type\":\"user\"}\n").unwrap();
    }
    let input = serde_json::json!({
        "session_id": session,
        "transcript_path": transcript,
        "cwd": dir,
        "hook_event_name": event,
    });
    let mut c = Command::new(BIN);
    c.arg("hook")
        .env("SWARM_BRIDGE_BOARD_ROOT", root)
        .env("SWARM_BRIDGE_STATE_DIR", dir.join("state"))
        .env_remove("SWARM_TAG")
        .env_remove("SWARM_BRIDGE_SESSION")
        .env_remove("SWARM_BRIDGE_SUBSCRIBE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        c.env(k, v);
    }
    let mut child = c.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    child
}

/// Wait for a hook; record its context into the transcript like Claude Code
/// does, and return the context.
fn finish_hook(dir: &Path, session: &str, child: std::process::Child) -> Option<String> {
    let o = child.wait_with_output().unwrap();
    assert!(o.status.success(), "hook exit {:?}", o.status);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.is_empty(), "hook stderr: {err}");
    if o.stdout.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_slice(&o.stdout).unwrap();
    let ctx = v["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap()
        .to_string();
    let line = serde_json::json!({
        "type": "attachment",
        "attachment": {"type": "hook_additional_context", "content": [ctx]}
    });
    let mut f = fs::OpenOptions::new()
        .append(true)
        .open(dir.join(format!("{session}.jsonl")))
        .unwrap();
    writeln!(f, "{line}").unwrap();
    Some(ctx)
}

fn hook(
    dir: &Path,
    root: &Path,
    session: &str,
    event: &str,
    env: &[(&str, &str)],
) -> Option<String> {
    let c = spawn_hook(dir, root, session, event, env);
    finish_hook(dir, session, c)
}

fn cli(dir: &Path, root: &Path, session: &str, args: &[&str]) -> String {
    let o = Command::new(BIN)
        .args(args)
        .env("SWARM_BRIDGE_BOARD_ROOT", root)
        .env("SWARM_BRIDGE_STATE_DIR", dir.join("state"))
        .env("SWARM_BRIDGE_SESSION", session)
        .env_remove("SWARM_TAG")
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

fn posted_id(out: &str) -> u64 {
    out.strip_prefix("posted #")
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no id in {out:?}"))
}

#[test]
fn two_processes_claim_one_tag_loser_gets_suffix_and_banner_says_so() {
    for _ in 0..5 {
        let dir = tmp();
        let root = init(&dir);
        let env = [("SWARM_TAG", "alpha")];
        let c1 = spawn_hook(&dir, &root, "sess-one", "SessionStart", &env);
        let c2 = spawn_hook(&dir, &root, "sess-two", "SessionStart", &env);
        let a = finish_hook(&dir, "sess-one", c1).unwrap();
        let b = finish_hook(&dir, "sess-two", c2).unwrap();
        let (win, lose) = if a.starts_with("You are alpha (") {
            (a, b)
        } else {
            (b, a)
        };
        assert!(win.starts_with("You are alpha (board "), "{win}");
        assert!(lose.starts_with("You are alpha-2 (board "), "{lose}");
        assert!(
            lose.contains("Tag alpha is taken by another session; you are alpha-2."),
            "{lose}"
        );
    }
}

#[test]
fn question_answer_follower_same_ids_follower_gets_both_as_subscribed() {
    let dir = tmp();
    let root = init(&dir);
    let thread = "contract/ledger-verify";
    let sub = format!("thread:{thread}");
    // All three sessions start (banner first).
    for (s, t) in [
        ("s-alpha", "alpha"),
        ("s-beta", "beta"),
        ("s-gamma", "gamma"),
    ] {
        let mut env = vec![("SWARM_TAG", t)];
        if t == "gamma" {
            env.push(("SWARM_BRIDGE_SUBSCRIBE", sub.as_str()));
        }
        let ctx = hook(&dir, &root, s, "SessionStart", &env).unwrap();
        assert!(ctx.starts_with(&format!("You are {t} (board ")), "{ctx}");
    }
    let q = posted_id(&cli(
        &dir,
        &root,
        "s-alpha",
        &[
            "question",
            "--to",
            "beta",
            "--thread",
            thread,
            "[Q1] empty slice?",
        ],
    ));
    // beta gets it addressed; gamma gets it subscribed, same id.
    let bctx = hook(
        &dir,
        &root,
        "s-beta",
        "PostToolUse",
        &[("SWARM_TAG", "beta")],
    )
    .unwrap();
    let (baddr, _) = bctx
        .split_once("From your subscriptions:")
        .unwrap_or((&bctx, ""));
    assert!(
        baddr.contains(&format!("#{q} question from=alpha")),
        "{bctx}"
    );
    let a = posted_id(&cli(
        &dir,
        &root,
        "s-beta",
        &["answer", "--parent", &q.to_string(), "[A1] Ok(genesis)."],
    ));
    assert_ne!(a, q);
    let actx = hook(
        &dir,
        &root,
        "s-alpha",
        "PostToolUse",
        &[("SWARM_TAG", "alpha")],
    )
    .unwrap();
    let (aaddr, _) = actx
        .split_once("From your subscriptions:")
        .unwrap_or((&actx, ""));
    assert!(
        aaddr.contains(&format!("#{a} answer from=beta")) && aaddr.contains("to=alpha"),
        "{actx}"
    );
    let gctx = hook(
        &dir,
        &root,
        "s-gamma",
        "PostToolUse",
        &[("SWARM_TAG", "gamma")],
    )
    .unwrap();
    let (gaddr, gsub) = gctx.split_once("From your subscriptions:").unwrap();
    assert!(!gaddr.contains(&format!("#{q} ")) && !gaddr.contains(&format!("#{a} ")));
    assert!(
        gsub.contains(&format!("#{q} question from=alpha")),
        "{gctx}"
    );
    assert!(gsub.contains(&format!("#{a} answer from=beta")), "{gctx}");
    // Next boundary: nothing repeats.
    assert!(hook(
        &dir,
        &root,
        "s-gamma",
        "PostToolUse",
        &[("SWARM_TAG", "gamma")]
    )
    .is_none());
    // Exactly two board writes.
    let all = cli(&dir, &root, "s-gamma", &["board-events"]);
    assert_eq!(all.lines().count(), 2, "{all}");
}

#[test]
fn watch_wakes_on_a_post_from_another_process() {
    let dir = tmp();
    let root = init(&dir);
    hook(&dir, &root, "s-b", "SessionStart", &[("SWARM_TAG", "beta")]).unwrap();
    hook(
        &dir,
        &root,
        "s-a",
        "SessionStart",
        &[("SWARM_TAG", "alpha")],
    )
    .unwrap();
    let w = Command::new(BIN)
        .args(["watch", "--timeout", "60"])
        .env("SWARM_BRIDGE_BOARD_ROOT", &root)
        .env("SWARM_BRIDGE_STATE_DIR", dir.join("state"))
        .env("SWARM_BRIDGE_SESSION", "s-b")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let started = std::time::Instant::now();
    let q = posted_id(&cli(
        &dir,
        &root,
        "s-a",
        &["question", "--to", "beta", "--thread", "t", "ping?"],
    ));
    let o = w.wait_with_output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(started.elapsed().as_secs() < 30, "woken, not timed out");
    assert!(
        out.contains("woken by a board notification") || out.contains("already waiting"),
        "{out}"
    );
    assert!(out.contains(&format!("#{q} question from=alpha")), "{out}");
}

/// bench pair 1 and 2 (2026-10-16): a board root deep enough that the wake
/// socket path passes the 107-byte `sun_path` limit made `watch` fail
/// outright, so the answering agent never saw the question. `watch` now
/// falls back to polling the cursor and says why.
#[test]
fn watch_polls_when_the_wake_socket_path_is_too_long() {
    let dir = tmp();
    let deep = dir.join("d".repeat(70));
    fs::create_dir_all(&deep).unwrap();
    let root = init(&deep);
    let sock_len = root.join("wake/0123456789abcdef.sock").as_os_str().len();
    assert!(sock_len > 107, "test must exceed sun_path: {sock_len}");
    hook(&dir, &root, "s-b", "SessionStart", &[("SWARM_TAG", "beta")]).unwrap();
    hook(
        &dir,
        &root,
        "s-a",
        "SessionStart",
        &[("SWARM_TAG", "alpha")],
    )
    .unwrap();
    let watch = |secs: &str| {
        Command::new(BIN)
            .args(["watch", "--timeout", secs])
            .env("SWARM_BRIDGE_BOARD_ROOT", &root)
            .env("SWARM_BRIDGE_STATE_DIR", dir.join("state"))
            .env("SWARM_BRIDGE_SESSION", "s-b")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    // Negative control: nothing posted, the watch times out cleanly.
    let o = watch("1").wait_with_output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("timed out"));
    let w = watch("60");
    std::thread::sleep(std::time::Duration::from_millis(300));
    let started = std::time::Instant::now();
    let q = posted_id(&cli(
        &dir,
        &root,
        "s-a",
        &["question", "--to", "beta", "--thread", "t", "ping?"],
    ));
    let o = w.wait_with_output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(
        started.elapsed().as_secs() < 30,
        "found by polling, not timed out"
    );
    assert!(out.contains("SocketPathTooLong"), "{out}");
    assert!(out.contains("polling the cursor found events"), "{out}");
    assert!(out.contains(&format!("#{q} question from=alpha")), "{out}");
}
