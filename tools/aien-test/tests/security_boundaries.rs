//! Offline security regression shared with the origin/main baseline.
use aien_test::{plan, verify, worker::StateDir};
use std::{fs, path::Path, process::Command};

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .current_dir(root)
        .args([
            "-c",
            "user.name=Security Test",
            "-c",
            "user.email=security@example.invalid",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn candidate_checker_cannot_execute_without_valid_evidence() {
    let temp = std::env::temp_dir().join(format!("sc-sec-1-{}", std::process::id()));
    fs::create_dir(&temp).unwrap();
    let root = temp.join("aien-architecture");
    fs::create_dir_all(root.join("tools/crumb")).unwrap();
    let built = temp.join("build-executed");
    let ran = temp.join("checker-executed");
    fs::write(
        root.join("tools/crumb/crumb.c"),
        "int main(void){return 0;}\n",
    )
    .unwrap();
    fs::write(
        root.join("tools/crumb/Makefile"),
        format!(
            "crumb:\n\ttouch {}\n\tprintf '#!/bin/sh\\ntouch {}\\n' > crumb\n\tchmod +x crumb\n",
            built.display(),
            ran.display()
        ),
    )
    .unwrap();
    git(&root, &["init", "-q"]);
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "candidate"]);
    let head = plan::resolve(&root, "HEAD").unwrap();
    let report = verify::run(
        &root,
        &verify::Ask {
            state: StateDir::new(&temp.join("state")),
            commit: head.clone(),
            base: head.clone(),
            repository: Some("github.com/test/aien-architecture".into()),
            release: false,
            operator: false,
            hardware_identity: String::new(),
            crumb: Some("crumb".into()),
            keys_override: None,
        },
    );
    let executed = built.exists() || ran.exists();
    // Negative control: the candidate payload really does execute if promoted explicitly.
    let tool = verify::crumb_tool_for(&root, &head, "crumb", &temp.join("control"));
    let verify::CrumbTool::Pinned { path, .. } = tool else {
        panic!("control unavailable")
    };
    assert_eq!(
        verify::crumb_status(&root, &head, path.to_str()).0,
        "CURRENT"
    );
    assert!(
        built.exists() && ran.exists(),
        "inert negative-control payload"
    );
    fs::remove_dir_all(temp).unwrap();
    assert!(!report.ready);
    assert!(
        !executed,
        "candidate Makefile/checker executed as trusted worker despite invalid evidence"
    );
}
