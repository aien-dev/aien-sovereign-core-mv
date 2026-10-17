//! `switchboard launch`: isolated profile folder, explicit environment
//! allowlist, typed errors. Uses a temp HOME and the fake spawner only.

mod common;

use aien_switchboard::launch::{
    check_profile_dir, launch_env, DirProblem, DirState, PROXY_ENV, STRIPPED_ENV,
};
use aien_switchboard::*;
use common::*;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;

/// Every variable ADR 0030 amendment A and issue #592 name, plus the cloud
/// credential families.
const MUST_STRIP: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_BASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "CLOUD_ML_REGION",
    "GH_TOKEN",
    "OPENAI_API_KEY",
];

#[test]
fn stripped_list_names_every_variable_from_the_issue() {
    for name in &MUST_STRIP[..10] {
        assert!(
            STRIPPED_ENV.contains(name),
            "{name} missing from STRIPPED_ENV"
        );
    }
}

#[test]
fn allowlist_strips_every_listed_variable() {
    let f = Fixture::new();
    let mut inherited = f.base_env();
    for name in MUST_STRIP {
        inherited.insert(OsString::from(name), OsString::from("SENTINEL-VALUE"));
    }
    inherited.insert("LANG".into(), "C.UTF-8".into());
    inherited.insert("LC_ALL".into(), "C.UTF-8".into());
    inherited.insert("TERM".into(), "xterm-256color".into());

    let dir = f.profile_dir("anthropic-1");
    let out = launch_env(&inherited, &dir, &[]);
    for name in MUST_STRIP {
        assert!(
            !has(&out, name),
            "{name} leaked into the launch environment"
        );
    }
    for keep in ["PATH", "HOME", "LANG", "LC_ALL", "TERM"] {
        assert!(has(&out, keep), "{keep} should pass");
    }
    assert_eq!(out[&OsString::from("CLAUDE_CONFIG_DIR")], dir.as_os_str());
    assert!(!out.values().any(|v| v == "SENTINEL-VALUE"));
}

#[test]
fn inherited_claude_config_dir_is_overridden() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut inherited = f.base_env();
    inherited.insert("CLAUDE_CONFIG_DIR".into(), "/somewhere/else".into());
    let plan = plan_launch(&f.rules(""), "anthropic-2", &inherited).unwrap();
    assert_eq!(
        plan.env[&OsString::from("CLAUDE_CONFIG_DIR")],
        f.profile_dir("anthropic-2").as_os_str()
    );
    assert_eq!(plan.config_dir, f.profile_dir("anthropic-2"));
}

#[test]
fn proxy_variables_pass_only_when_listed_in_local_config() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut inherited = f.base_env();
    inherited.insert("HTTPS_PROXY".into(), "http://proxy:3128".into());
    inherited.insert("NO_PROXY".into(), "localhost".into());

    let plain = plan_launch(&f.rules(""), "anthropic-1", &inherited).unwrap();
    assert!(!has(&plain.env, "HTTPS_PROXY"));
    assert!(!has(&plain.env, "NO_PROXY"));

    let listed = f.rules("[launch]\npass_env = [\"HTTPS_PROXY\"]\n");
    assert_eq!(listed.pass_env(), ["HTTPS_PROXY".to_string()]);
    let plan = plan_launch(&listed, "anthropic-1", &inherited).unwrap();
    assert!(has(&plan.env, "HTTPS_PROXY"));
    assert!(!has(&plan.env, "NO_PROXY"));
}

#[test]
fn local_config_may_only_pass_proxy_variables() {
    let f = Fixture::new();
    assert!(PROXY_ENV.contains(&"HTTPS_PROXY"));
    for bad in [
        "ANTHROPIC_API_KEY",
        "CLAUDE_CONFIG_DIR",
        "AWS_PROFILE",
        "EDITOR",
    ] {
        let toml = f.rules_toml(&format!("[launch]\npass_env = [\"{bad}\"]\n"));
        match RuleTable::from_toml_str(&toml) {
            Err(RulesError::Invalid(m)) => assert!(m.contains(bad), "{m}"),
            other => panic!("{bad} accepted: {other:?}"),
        }
    }
}

#[test]
fn unknown_alias_is_a_typed_error() {
    let f = Fixture::new();
    f.install_fake_claude();
    let err = plan_launch(&f.rules(""), "anthropic-9", &f.base_env()).unwrap_err();
    match err {
        LaunchError::UnknownAlias { alias, known } => {
            assert_eq!(alias, "anthropic-9");
            assert_eq!(known, vec!["anthropic-1", "anthropic-2"]);
        }
        other => panic!("wrong error {other:?}"),
    }
}

#[test]
fn non_claude_account_is_a_typed_error() {
    let f = Fixture::new();
    f.install_fake_claude();
    let err = plan_launch(&f.rules(""), "openrouter-1", &f.base_env()).unwrap_err();
    assert_eq!(
        err,
        LaunchError::NotAClaudeProfile {
            alias: "openrouter-1".into()
        }
    );
}

#[test]
fn missing_executable_is_a_typed_error() {
    let f = Fixture::new();
    fs::create_dir_all(f.bin()).unwrap(); // empty: no claude on PATH
    let err = plan_launch(&f.rules(""), "anthropic-1", &f.base_env()).unwrap_err();
    assert_eq!(
        err,
        LaunchError::ExecutableNotFound {
            program: "claude".into()
        }
    );
}

#[test]
fn non_executable_file_on_path_is_not_found() {
    let f = Fixture::new();
    let exe = f.install_fake_claude();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o644)).unwrap();
    let err = plan_launch(&f.rules(""), "anthropic-1", &f.base_env()).unwrap_err();
    assert!(matches!(err, LaunchError::ExecutableNotFound { .. }));
}

#[test]
fn launch_creates_private_folder_and_starts_resolved_program_once() {
    let f = Fixture::new();
    let exe = f.install_fake_claude();
    let mut sp = FakeSpawner {
        exit_code: 7,
        ..FakeSpawner::default()
    };
    let dir = f.profile_dir("anthropic-1");
    assert_eq!(check_profile_dir(&dir), Ok(DirState::Missing));

    let code = launch(&f.rules(""), "anthropic-1", &f.base_env(), &mut sp).unwrap();
    assert_eq!(code, 7);
    assert_eq!(sp.plans.len(), 1);
    assert_eq!(sp.plans[0].program, exe);
    assert_eq!(sp.plans[0].alias, "anthropic-1");
    let mode = fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700);
    assert_eq!(check_profile_dir(&dir), Ok(DirState::Private));
}

#[test]
fn launch_refuses_a_folder_others_can_read() {
    let f = Fixture::new();
    f.install_fake_claude();
    let dir = f.profile_dir("anthropic-1");
    fs::create_dir_all(&dir).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    let mut sp = FakeSpawner::default();
    let err = launch(&f.rules(""), "anthropic-1", &f.base_env(), &mut sp).unwrap_err();
    assert_eq!(
        err,
        LaunchError::ProfileDir {
            path: dir.clone(),
            problem: DirProblem::TooOpen { mode: 0o755 }
        }
    );
    assert!(sp.plans.is_empty(), "nothing may start");
}

#[test]
fn launch_refuses_a_symlinked_folder() {
    let f = Fixture::new();
    f.install_fake_claude();
    let real = f.tmp.path().join("shared");
    f.make_private_dir(&real);
    let dir = f.profile_dir("anthropic-1");
    fs::create_dir_all(dir.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&real, &dir).unwrap();
    let mut sp = FakeSpawner::default();
    let err = launch(&f.rules(""), "anthropic-1", &f.base_env(), &mut sp).unwrap_err();
    assert!(matches!(
        err,
        LaunchError::ProfileDir {
            problem: DirProblem::Symlink,
            ..
        }
    ));
    assert!(sp.plans.is_empty());
}

#[test]
fn plan_debug_and_error_text_never_show_environment_values() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut inherited = f.base_env();
    inherited.insert("ANTHROPIC_API_KEY".into(), "sk-ant-SENTINEL".into());
    inherited.insert("LANG".into(), "LANG-SENTINEL".into());
    let plan = plan_launch(&f.rules(""), "anthropic-1", &inherited).unwrap();
    let dbg = format!("{plan:?}");
    // Assertion messages never echo the text under test: it was built from
    // an environment holding a (fake) key.
    assert!(dbg.contains("LANG"), "variable names should be shown");
    assert!(
        !dbg.contains("SENTINEL"),
        "variable values must not be shown"
    );
    let err = plan_launch(&f.rules(""), "nope", &inherited).unwrap_err();
    assert!(!err.to_string().contains("SENTINEL"));
}
