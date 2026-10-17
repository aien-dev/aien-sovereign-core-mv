//! The `switchboard` command surface, driven in-process with fakes.

mod common;

use aien_switchboard::cli::run;
use aien_switchboard::*;
use common::*;
use std::fs;

struct Out {
    code: i32,
    out: String,
    err: String,
}

fn call(f: &Fixture, args: &[&str], sp: &mut FakeSpawner) -> Out {
    let rules = f.home().join(".switchboard").join("rules.toml");
    fs::create_dir_all(rules.parent().unwrap()).unwrap();
    if !rules.exists() {
        fs::write(&rules, f.rules_toml("")).unwrap();
    }
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut probe = FakeProbe {
        answer: Ok("2.1.301 (Claude Code)".into()),
        calls: Vec::new(),
    };
    let (mut o, mut e) = (Vec::new(), Vec::new());
    let code = run(
        &args,
        &f.base_env(),
        f.tmp.path(),
        sp,
        &mut probe,
        &mut o,
        &mut e,
    );
    Out {
        code,
        out: String::from_utf8(o).unwrap(),
        err: String::from_utf8(e).unwrap(),
    }
}

#[test]
fn accounts_lists_profiles_from_the_default_rule_file_in_home() {
    let f = Fixture::new();
    let o = call(&f, &["accounts"], &mut FakeSpawner::default());
    assert_eq!(o.code, 0, "{}", o.err);
    for a in ["anthropic-1", "anthropic-2", "openrouter-1", "local"] {
        assert!(o.out.contains(a), "{}", o.out);
    }
    assert!(o
        .out
        .contains(f.profile_dir("anthropic-1").to_str().unwrap()));
}

#[test]
fn launch_runs_through_the_spawner_and_returns_its_exit_code() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut sp = FakeSpawner {
        exit_code: 3,
        ..FakeSpawner::default()
    };
    let o = call(&f, &["launch", "anthropic-2"], &mut sp);
    assert_eq!(o.code, 3, "{}", o.err);
    assert_eq!(sp.plans.len(), 1);
    assert_eq!(sp.plans[0].config_dir, f.profile_dir("anthropic-2"));
}

#[test]
fn unknown_alias_exits_nonzero_with_known_names() {
    let f = Fixture::new();
    f.install_fake_claude();
    let mut sp = FakeSpawner::default();
    let o = call(&f, &["launch", "nobody"], &mut sp);
    assert_eq!(o.code, 1);
    assert!(
        o.err.contains("nobody") && o.err.contains("anthropic-1"),
        "{}",
        o.err
    );
    assert!(sp.plans.is_empty());
}

#[test]
fn doctor_exit_code_follows_failures() {
    let f = Fixture::new();
    f.install_fake_claude();
    let ok = call(&f, &["doctor"], &mut FakeSpawner::default());
    assert_eq!(ok.code, 0, "{}{}", ok.out, ok.err);
    fs::remove_file(f.bin().join("claude")).unwrap();
    let bad = call(&f, &["doctor"], &mut FakeSpawner::default());
    assert_eq!(bad.code, 1, "{}", bad.out);
}

#[test]
fn usage_errors_exit_2() {
    let f = Fixture::new();
    assert_eq!(call(&f, &[], &mut FakeSpawner::default()).code, 2);
    assert_eq!(call(&f, &["launch"], &mut FakeSpawner::default()).code, 2);
    assert_eq!(
        call(&f, &["frobnicate"], &mut FakeSpawner::default()).code,
        2
    );
    assert_eq!(
        call(&f, &["launch", "a", "b"], &mut FakeSpawner::default()).code,
        2
    );
}

#[test]
fn rules_flag_points_at_another_file() {
    let f = Fixture::new();
    let other = f.tmp.path().join("other.toml");
    fs::write(&other, f.rules_toml("")).unwrap();
    let o = call(
        &f,
        &["--rules", other.to_str().unwrap(), "accounts"],
        &mut FakeSpawner::default(),
    );
    assert_eq!(o.code, 0, "{}", o.err);
}
