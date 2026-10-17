//! `switchboard doctor`: reports presence and permissions, never contents.

mod common;

use aien_switchboard::*;
use common::*;
use std::fs;
use std::os::unix::fs::PermissionsExt;

fn probe() -> FakeProbe {
    FakeProbe {
        answer: Ok("2.1.301 (Claude Code)".into()),
        calls: Vec::new(),
    }
}

fn text(r: &Report) -> String {
    r.to_string()
}

#[test]
fn signed_in_profile_reports_presence_without_opening_the_credentials_file() {
    let f = Fixture::new();
    f.install_fake_claude();
    let dir = f.profile_dir("anthropic-1");
    f.make_private_dir(&dir);
    let cred = dir.join(".credentials.json");
    fs::write(&cred, "{\"token\":\"CRED-SENTINEL\"}").unwrap();
    // Unreadable: if doctor tried to open it, it could not, and the report
    // would show an error instead of "present".
    fs::set_permissions(&cred, fs::Permissions::from_mode(0o000)).unwrap();

    let cwd = f.tmp.path().join("work");
    fs::create_dir_all(&cwd).unwrap();
    let r = doctor(&f.rules(""), &f.base_env(), &cwd, &mut probe());
    let t = text(&r);
    fs::set_permissions(&cred, fs::Permissions::from_mode(0o600)).unwrap();

    assert!(!t.contains("CRED-SENTINEL"), "{t}");
    let found = r
        .findings
        .iter()
        .find(|x| x.subject == "anthropic-1" && x.message.contains("credentials file present"))
        .unwrap_or_else(|| panic!("no presence finding: {t}"));
    assert_eq!(found.level, Level::Warn, "mode 000 is not 0600: {t}");
    assert!(found.message.contains("000"), "{t}");
}

#[test]
fn credentials_with_0600_are_ok_and_missing_ones_say_not_signed_in() {
    let f = Fixture::new();
    f.install_fake_claude();
    let d1 = f.profile_dir("anthropic-1");
    f.make_private_dir(&d1);
    let cred = d1.join(".credentials.json");
    fs::write(&cred, "x").unwrap();
    fs::set_permissions(&cred, fs::Permissions::from_mode(0o600)).unwrap();
    f.make_private_dir(&f.profile_dir("anthropic-2"));

    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut probe());
    let t = text(&r);
    assert!(
        r.findings.iter().any(|x| x.subject == "anthropic-1"
            && x.level == Level::Ok
            && x.message.contains("credentials file present")),
        "{t}"
    );
    assert!(
        r.findings.iter().any(|x| x.subject == "anthropic-2"
            && x.level == Level::Warn
            && x.message.contains("not signed in")),
        "{t}"
    );
    assert!(!r.failed(), "{t}");
}

#[test]
fn executable_and_version_are_reported() {
    let f = Fixture::new();
    let exe = f.install_fake_claude();
    let mut p = probe();
    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut p);
    let t = text(&r);
    assert!(t.contains("2.1.301 (Claude Code)"), "{t}");
    assert!(t.contains(exe.to_str().unwrap()), "{t}");
    assert_eq!(p.calls, vec![exe]);
}

#[test]
fn missing_executable_fails_the_doctor() {
    let f = Fixture::new();
    fs::create_dir_all(f.bin()).unwrap();
    let mut p = probe();
    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut p);
    assert!(r.failed(), "{}", text(&r));
    assert!(r
        .findings
        .iter()
        .any(|x| x.level == Level::Fail && x.message.contains("not found")));
    assert!(p.calls.is_empty());
}

#[test]
fn missing_profile_folder_is_a_warning_and_open_folder_is_a_failure() {
    let f = Fixture::new();
    f.install_fake_claude();
    let d2 = f.profile_dir("anthropic-2");
    fs::create_dir_all(&d2).unwrap();
    fs::set_permissions(&d2, fs::Permissions::from_mode(0o750)).unwrap();
    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut probe());
    let t = text(&r);
    assert!(
        r.findings.iter().any(|x| x.subject == "anthropic-1"
            && x.level == Level::Warn
            && x.message.contains("folder not created yet")),
        "{t}"
    );
    assert!(
        r.findings.iter().any(|x| x.subject == "anthropic-2"
            && x.level == Level::Fail
            && x.message.contains("750")),
        "{t}"
    );
    assert!(r.failed());
}

#[test]
fn api_key_helper_and_env_block_in_profile_settings_warn_without_values() {
    let f = Fixture::new();
    f.install_fake_claude();
    let d1 = f.profile_dir("anthropic-1");
    f.make_private_dir(&d1);
    fs::write(
        d1.join("settings.json"),
        r#"{"apiKeyHelper":"/bin/echo HELPER-SENTINEL","env":{"ANTHROPIC_API_KEY":"sk-ant-SENTINEL","FOO":"bar"}}"#,
    )
    .unwrap();
    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut probe());
    let t = text(&r);
    assert!(!t.contains("SENTINEL"), "{t}");
    assert!(
        r.findings
            .iter()
            .any(|x| x.level == Level::Warn && x.message.contains("apiKeyHelper")),
        "{t}"
    );
    assert!(
        r.findings.iter().any(|x| x.level == Level::Warn
            && x.message.contains("env block")
            && x.message.contains("ANTHROPIC_API_KEY")),
        "{t}"
    );
}

#[test]
fn repo_settings_files_are_checked() {
    let f = Fixture::new();
    f.install_fake_claude();
    let repo = f.tmp.path().join("repo");
    let sub = repo.join("crates").join("x");
    fs::create_dir_all(&sub).unwrap();
    fs::create_dir_all(repo.join(".git")).unwrap();
    fs::create_dir_all(repo.join(".claude")).unwrap();
    fs::write(
        repo.join(".claude").join("settings.local.json"),
        r#"{"env":{"CLAUDE_CODE_OAUTH_TOKEN":"OAUTH-SENTINEL"}}"#,
    )
    .unwrap();
    fs::write(
        repo.join(".claude").join("settings.json"),
        "{not json OTHER-SENTINEL",
    )
    .unwrap();
    let r = doctor(&f.rules(""), &f.base_env(), &sub, &mut probe());
    let t = text(&r);
    assert!(!t.contains("SENTINEL"), "{t}");
    assert!(
        r.findings
            .iter()
            .any(|x| x.subject.ends_with("settings.local.json")
                && x.message.contains("CLAUDE_CODE_OAUTH_TOKEN")),
        "{t}"
    );
    assert!(
        r.findings
            .iter()
            .any(|x| x.subject.ends_with("settings.json") && x.message.contains("cannot parse")),
        "{t}"
    );
}

#[test]
fn credential_variables_in_the_shell_are_named_not_shown() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut e = f.base_env();
    e.insert("ANTHROPIC_API_KEY".into(), "sk-ant-SENTINEL".into());
    e.insert("AWS_SECRET_ACCESS_KEY".into(), "AWS-SENTINEL".into());
    let r = doctor(&f.rules(""), &e, f.tmp.path(), &mut probe());
    let t = text(&r);
    assert!(!t.contains("SENTINEL"), "{t}");
    assert!(
        t.contains("ANTHROPIC_API_KEY") && t.contains("AWS_SECRET_ACCESS_KEY"),
        "{t}"
    );
}

#[test]
fn anthropic_profile_files_in_home_are_flagged_by_presence() {
    let f = Fixture::new();
    f.install_fake_claude();
    let cfg = f.home().join(".config").join("anthropic");
    fs::create_dir_all(&cfg).unwrap();
    fs::write(cfg.join("active_config"), "PROFILE-SENTINEL").unwrap();
    let r = doctor(&f.rules(""), &f.base_env(), f.tmp.path(), &mut probe());
    let t = text(&r);
    assert!(!t.contains("PROFILE-SENTINEL"), "{t}");
    assert!(
        r.findings
            .iter()
            .any(|x| x.level == Level::Warn && x.message.contains("Anthropic profile")),
        "{t}"
    );
}
