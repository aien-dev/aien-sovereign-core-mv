use aien_switchboard::*;

const RULES: &str = r#"
[settings]
ceiling_percent = 90
default_backoff_secs = 600

[[accounts]]
id = "anthropic-1"
provider = "anthropic"
program = "claude"
settings_dir = "/tmp/sb/anthropic-1"

[[accounts]]
id = "google-1"
provider = "google"
program = "agy"
settings_dir = "/tmp/sb/google-1"

[[accounts]]
id = "openrouter-1"
provider = "openrouter"
program = "http"
key_ref = "openrouter-key"

[[accounts]]
id = "local"
provider = "local"
program = "aien"

[routes]
code = ["anthropic-1", "google-1", "openrouter-1", "local"]
review = ["google-1", "anthropic-1"]
default = ["openrouter-1"]
"#;

fn rules() -> RuleTable {
    RuleTable::from_toml_str(RULES).unwrap()
}
fn id(s: &str) -> AccountId {
    AccountId::new(s)
}
fn code() -> Job {
    Job::new(JobKind::Code, JobSize::Small)
}
fn win(pct: f64, resets_at: Option<u64>) -> Vec<Window> {
    vec![Window {
        name: "five_hour".into(),
        used_percent: pct,
        resets_at,
    }]
}

#[test]
fn rule_order_is_respected() {
    let r = rules();
    let d = route(&code(), &r, &Gauges::new(), 1000);
    assert_eq!(d.account, id("anthropic-1"));
    assert_eq!(
        d.alternates,
        vec![id("google-1"), id("openrouter-1"), id("local")]
    );
    let d = route(
        &Job::new(JobKind::Review, JobSize::Small),
        &r,
        &Gauges::new(),
        1000,
    );
    assert_eq!(d.account, id("google-1"));
}

#[test]
fn unlisted_kind_uses_default_and_local_is_appended() {
    let r = rules();
    assert_eq!(
        r.route_for(JobKind::Research),
        &[id("openrouter-1"), id("local")]
    );
    assert_eq!(
        r.route_for(JobKind::Review),
        &[id("google-1"), id("anthropic-1"), id("local")]
    );
}

#[test]
fn cooled_down_account_is_skipped_with_a_plain_reason() {
    let r = rules();
    let mut g = Gauges::new();
    g.on_limit_hit(&id("anthropic-1"), Some(5000), 1000, 600);
    let d = route(&code(), &r, &g, 1000);
    assert_eq!(d.account, id("google-1"));
    assert!(
        d.reason.contains("anthropic-1 is cooling down"),
        "{}",
        d.reason
    );
}

#[test]
fn account_over_the_ceiling_is_skipped_and_stale_window_is_not() {
    let r = rules();
    let mut g = Gauges::new();
    g.set_windows(&id("anthropic-1"), win(95.0, Some(5000)));
    let d = route(&code(), &r, &g, 1000);
    assert_eq!(d.account, id("google-1"));
    assert!(d.reason.contains("95% used"), "{}", d.reason);
    // Once the window's reset time has passed, the account has room again.
    let d = route(&code(), &r, &g, 5000);
    assert_eq!(d.account, id("anthropic-1"));
    // Exactly under the ceiling is fine.
    g.set_windows(&id("anthropic-1"), win(89.9, None));
    assert_eq!(route(&code(), &r, &g, 1000).account, id("anthropic-1"));
}

#[test]
fn everything_exhausted_goes_to_local() {
    let r = rules();
    let mut g = Gauges::new();
    for a in ["anthropic-1", "google-1", "openrouter-1"] {
        g.on_limit_hit(&id(a), None, 1000, 600);
    }
    let d = route(&code(), &r, &g, 1000);
    assert_eq!(d.account, id("local"));
    assert!(d.alternates.is_empty());
    assert!(
        d.reason.contains("stays on the local model"),
        "{}",
        d.reason
    );
}

#[test]
fn cooldown_expires() {
    let r = rules();
    let mut g = Gauges::new();
    let until = g.on_limit_hit(&id("anthropic-1"), None, 1000, 600);
    assert_eq!(until, 1600);
    assert_eq!(route(&code(), &r, &g, 1599).account, id("google-1"));
    assert_eq!(route(&code(), &r, &g, 1600).account, id("anthropic-1"));
}

#[test]
fn reset_time_in_the_past_falls_back_to_default_backoff_and_longer_cooldown_is_kept() {
    let mut g = Gauges::new();
    assert_eq!(g.on_limit_hit(&id("a"), Some(500), 1000, 600), 1600);
    assert_eq!(g.on_limit_hit(&id("a"), Some(9000), 1000, 600), 9000);
    assert_eq!(g.on_limit_hit(&id("a"), None, 1000, 600), 9000);
}

#[test]
fn failover_on_limit_hit_then_cooldown_sticks() {
    let r = rules();
    let mut g = Gauges::new();
    let mut run = FakeRunner::new();
    run.push(
        &id("anthropic-1"),
        RunResult::LimitHit {
            resets_at: Some(4000),
        },
    );
    let done = run_job(&code(), "hi", &r, &mut g, &mut run, 1000).unwrap();
    assert_eq!(done.account, id("google-1"));
    assert_eq!(done.tried, vec![id("anthropic-1"), id("google-1")]);
    assert!(done.output.contains("google-1"));
    // The next job skips the cooled account without trying it.
    let done = run_job(&code(), "again", &r, &mut g, &mut run, 1001).unwrap();
    assert_eq!(done.tried, vec![id("google-1")]);
}

#[test]
fn all_outside_accounts_hit_limits_then_local_answers() {
    let r = rules();
    let mut g = Gauges::new();
    let mut run = FakeRunner::new();
    for a in ["anthropic-1", "google-1", "openrouter-1"] {
        run.push(&id(a), RunResult::LimitHit { resets_at: None });
    }
    let done = run_job(&code(), "x", &r, &mut g, &mut run, 1000).unwrap();
    assert_eq!(done.account, id("local"));
    assert_eq!(done.tried.len(), 4);
}

#[test]
fn local_limit_hit_is_a_clear_error_not_a_loop() {
    let r = rules();
    let mut g = Gauges::new();
    let mut run = FakeRunner::new();
    run.push(&id("local"), RunResult::LimitHit { resets_at: None });
    let only_local = RuleTable::from_toml_str(
        "[[accounts]]\nid=\"local\"\nprovider=\"local\"\nprogram=\"aien\"\n[routes]\ndefault=[\"local\"]\n",
    )
    .unwrap();
    let e = run_job(&code(), "x", &only_local, &mut g, &mut run, 1000).unwrap_err();
    assert!(e.to_string().contains("out of room"), "{e}");
    let _ = r;
}

#[test]
fn local_listed_early_is_moved_last() {
    let r = RuleTable::from_toml_str(
        "[[accounts]]\nid=\"local\"\nprovider=\"local\"\nprogram=\"aien\"\n\
         [[accounts]]\nid=\"g1\"\nprovider=\"google\"\nprogram=\"agy\"\nsettings_dir=\"/tmp/sb/g1\"\n\
         [routes]\ndefault=[\"local\",\"g1\"]\n",
    )
    .unwrap();
    assert_eq!(r.route_for(JobKind::Code), &[id("g1"), id("local")]);
}

fn parse_err(text: &str) -> String {
    RuleTable::from_toml_str(text).unwrap_err().to_string()
}

#[test]
fn parse_errors_are_clear() {
    // Not TOML at all: parser reports line and column.
    let e = parse_err("[routes\ndefault = [");
    assert!(e.starts_with("rule file is not valid:"), "{e}");
    assert!(e.contains("line 1"), "{e}");
    // Missing the required default route.
    assert!(parse_err("[routes]\n").contains("default"));
    // Misspelled key.
    assert!(parse_err("[routes]\ndefalt = []\n").contains("defalt"));
    // Unknown provider.
    let e = parse_err(
        "[[accounts]]\nid=\"a\"\nprovider=\"bing\"\nprogram=\"x\"\n[routes]\ndefault=[\"a\"]\n",
    );
    assert!(e.contains("bing"), "{e}");
}

#[test]
fn rule_mistakes_name_the_problem() {
    let local = "[[accounts]]\nid=\"local\"\nprovider=\"local\"\nprogram=\"aien\"\n";
    let g1 = "[[accounts]]\nid=\"g1\"\nprovider=\"google\"\nprogram=\"agy\"\nsettings_dir=\"/tmp/sb/g1\"\n";
    assert!(parse_err(&format!("{g1}[routes]\ndefault=[\"g1\"]\n"))
        .contains("no account with provider = \"local\""));
    assert!(parse_err(&format!("{local}[routes]\ndefault=[\"nope\"]\n"))
        .contains("\"nope\", which is not in [[accounts]]"));
    assert!(parse_err(&format!("{local}[routes]\ndefault=[]\n")).contains("empty list"));
    assert!(
        parse_err(&format!("{local}{local}[routes]\ndefault=[\"local\"]\n"))
            .contains("listed twice")
    );
    assert!(
        parse_err(&format!("{local}{g1}[routes]\ndefault=[\"g1\",\"g1\"]\n"))
            .contains("lists g1 twice")
    );
    assert!(parse_err(&format!(
        "[settings]\nceiling_percent=150\n{local}[routes]\ndefault=[\"local\"]\n"
    ))
    .contains("ceiling_percent"));
    let both = "[[accounts]]\nid=\"x\"\nprovider=\"xai\"\nprogram=\"grok\"\nsettings_dir=\"/a\"\nkey_ref=\"k\"\n";
    assert!(parse_err(&format!("{local}{both}[routes]\ndefault=[\"x\"]\n")).contains("pick one"));
    let secret = "[[accounts]]\nid=\"o\"\nprovider=\"openrouter\"\nprogram=\"http\"\nkey_ref=\"sk-or v1 abc def\"\n";
    assert!(
        parse_err(&format!("{local}{secret}[routes]\ndefault=[\"o\"]\n"))
            .contains("never the key itself")
    );
}

#[test]
fn missing_file_is_a_read_error() {
    let e =
        RuleTable::from_path(std::path::Path::new("/nonexistent/switchboard.toml")).unwrap_err();
    assert!(e.to_string().starts_with("cannot read rule file:"), "{e}");
}

#[test]
fn routing_is_deterministic() {
    let r = rules();
    let mut g = Gauges::new();
    g.set_windows(&id("anthropic-1"), win(91.0, None));
    let a = route(&code(), &r, &g, 1000);
    let b = route(&code(), &r, &g, 1000);
    assert_eq!(a, b);
}

const LOCAL: &str = "[[accounts]]\nid=\"local\"\nprovider=\"local\"\nprogram=\"aien\"\n";

fn sub(id: &str, provider: &str, program: &str) -> String {
    format!("[[accounts]]\nid=\"{id}\"\nprovider=\"{provider}\"\nprogram=\"{program}\"\nsettings_dir=\"/tmp/sb/{id}\"\n")
}

#[test]
fn route_with_two_subscriptions_of_one_provider_is_rejected() {
    // Operator decision 2026-10-16 (aien-architecture#236, issue #592): no automatic
    // routing between two subscription accounts of the same provider.
    let accts = format!(
        "{LOCAL}{}{}",
        sub("a1", "anthropic", "claude"),
        sub("a2", "anthropic", "claude")
    );
    let e = RuleTable::from_toml_str(&format!(
        "{accts}[routes]\ndefault=[\"local\"]\ncode=[\"a1\",\"a2\"]\n"
    ))
    .unwrap_err();
    let m = e.to_string();
    assert!(matches!(e, RulesError::Invalid(_)), "{m}");
    assert!(
        m.contains("route \"code\"") && m.contains("a1") && m.contains("a2"),
        "{m}"
    );
    assert!(m.contains("anthropic"), "{m}");
    // The default route is checked too.
    assert!(
        RuleTable::from_toml_str(&format!("{accts}[routes]\ndefault=[\"a2\",\"a1\"]\n")).is_err()
    );
    // Same accounts on separate routes are fine: manual launch may open any.
    assert!(RuleTable::from_toml_str(&format!(
        "{accts}[routes]\ndefault=[\"local\"]\ncode=[\"a1\"]\nreview=[\"a2\"]\n"
    ))
    .is_ok());
}

#[test]
fn subscription_plus_api_key_or_other_provider_is_allowed() {
    let key = "[[accounts]]\nid=\"a-key\"\nprovider=\"anthropic\"\nprogram=\"http\"\nkey_ref=\"anthropic-console\"\n";
    let accts = format!(
        "{LOCAL}{}{key}{}",
        sub("a1", "anthropic", "claude"),
        sub("g1", "google", "agy")
    );
    let r = RuleTable::from_toml_str(&format!(
        "{accts}[routes]\ndefault=[\"a1\",\"a-key\",\"g1\"]\n"
    ));
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn settings_folders_must_be_absolute_and_not_shared() {
    let rel = "[[accounts]]\nid=\"a1\"\nprovider=\"anthropic\"\nprogram=\"claude\"\nsettings_dir=\"rel/a1\"\n";
    let e = RuleTable::from_toml_str(&format!("{LOCAL}{rel}[routes]\ndefault=[\"local\"]\n"))
        .unwrap_err();
    assert!(e.to_string().contains("absolute"), "{e}");
    let shared = "[[accounts]]\nid=\"a2\"\nprovider=\"anthropic\"\nprogram=\"claude\"\nsettings_dir=\"/tmp/sb/a1\"\n";
    let e = RuleTable::from_toml_str(&format!(
        "{LOCAL}{}{shared}[routes]\ndefault=[\"local\"]\n",
        sub("a1", "anthropic", "claude")
    ))
    .unwrap_err();
    assert!(e.to_string().contains("same settings_dir"), "{e}");
}

#[test]
fn route_error_cites_the_operator_decision_only() {
    let accts = format!(
        "{LOCAL}{}{}",
        sub("a1", "anthropic", "claude"),
        sub("a2", "anthropic", "claude")
    );
    let m = RuleTable::from_toml_str(&format!("{accts}[routes]\ndefault=[\"a1\",\"a2\"]\n"))
        .unwrap_err()
        .to_string();
    assert!(m.contains("aien-architecture#236"), "{m}");
    assert!(!m.contains("amendment"), "{m}");
}

#[test]
fn non_local_account_without_a_login_setting_is_refused() {
    // A claude account with no settings_dir would use the default ~/.claude
    // login, a subscription the route rule could not see. Review of #619.
    let bare = "[[accounts]]\nid=\"a2\"\nprovider=\"anthropic\"\nprogram=\"claude\"\n";
    let accts = format!("{LOCAL}{}{bare}", sub("a1", "anthropic", "claude"));
    let e = RuleTable::from_toml_str(&format!(
        "{accts}[routes]\ndefault=[\"local\"]\ncode=[\"a1\",\"a2\"]\n"
    ))
    .unwrap_err();
    assert!(matches!(e, RulesError::Invalid(_)), "{e}");
    assert!(e.to_string().contains("\"a2\""), "{e}");
    // Not routed anywhere: still refused, for any non-local provider.
    let g = "[[accounts]]\nid=\"g\"\nprovider=\"google\"\nprogram=\"agy\"\n";
    let e = RuleTable::from_toml_str(&format!("{LOCAL}{g}[routes]\ndefault=[\"local\"]\n"))
        .unwrap_err();
    assert!(e.to_string().contains("settings_dir or key_ref"), "{e}");
}

#[test]
fn program_must_be_a_bare_name_or_an_absolute_path() {
    for bad in ["bin/claude", "./claude", "../x/claude"] {
        let a = format!("[[accounts]]\nid=\"a1\"\nprovider=\"anthropic\"\nprogram=\"{bad}\"\nsettings_dir=\"/tmp/sb/a1\"\n");
        let e = RuleTable::from_toml_str(&format!("{LOCAL}{a}[routes]\ndefault=[\"local\"]\n"))
            .unwrap_err();
        assert!(e.to_string().contains("absolute"), "{bad}: {e}");
    }
    let ok = "[[accounts]]\nid=\"a1\"\nprovider=\"anthropic\"\nprogram=\"/opt/claude/bin/claude\"\nsettings_dir=\"/tmp/sb/a1\"\n";
    assert!(
        RuleTable::from_toml_str(&format!("{LOCAL}{ok}[routes]\ndefault=[\"local\"]\n")).is_ok()
    );
}

#[test]
fn settings_dir_with_parent_steps_is_refused() {
    let a = "[[accounts]]\nid=\"a2\"\nprovider=\"anthropic\"\nprogram=\"claude\"\nsettings_dir=\"/tmp/sb/x/../a1\"\n";
    let e = RuleTable::from_toml_str(&format!("{LOCAL}{a}[routes]\ndefault=[\"local\"]\n"))
        .unwrap_err();
    assert!(e.to_string().contains(".."), "{e}");
}
