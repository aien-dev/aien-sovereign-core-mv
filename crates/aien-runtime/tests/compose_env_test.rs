// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Startup refusal of compose limit settings, read from the environment the
//! way the daemon reads them at start (`compose_budgets_from_env`,
//! `compose_doc_max_tokens_from_env`). Its own test binary, because it sets
//! process environment variables; every test holds `ENV` while it does.
use aien_runtime::compose::{compose_budgets_from_env, compose_doc_max_tokens_from_env};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

const EDIT: &str = "AIEN_COMPOSE_EDIT_BUDGET_MS";
const DOC: &str = "AIEN_COMPOSE_DOC_BUDGET_MS";
const TOKENS: &str = "AIEN_COMPOSE_DOC_MAX_TOKENS";
const RETIRED: &str = "AIEN_COMPOSE_BUDGET_MS";
const ALL: [&str; 4] = [EDIT, DOC, TOKENS, RETIRED];

/// The environment is process-global: one test at a time touches it.
static ENV: Mutex<()> = Mutex::new(());

/// Holds `ENV` with every compose limit variable unset; unsets them again when dropped.
struct CleanEnv {
    _guard: MutexGuard<'static, ()>,
}

fn clean_env() -> CleanEnv {
    let g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    for k in ALL {
        std::env::remove_var(k);
    }
    CleanEnv { _guard: g }
}

impl Drop for CleanEnv {
    fn drop(&mut self) {
        for k in ALL {
            std::env::remove_var(k);
        }
    }
}

/// `v` = None leaves `name` unset.
fn set(name: &str, v: Option<&str>) {
    match v {
        Some(v) => std::env::set_var(name, v),
        None => std::env::remove_var(name),
    }
}

#[test]
fn environment_limits_are_read_and_bad_ones_refused() {
    let _env = clean_env();
    let b = compose_budgets_from_env().expect("defaults");
    assert_eq!(b.edit, Duration::from_secs(29));
    assert_eq!(b.doc, Duration::from_secs(120));
    assert_eq!(compose_doc_max_tokens_from_env(), Ok(1024));

    std::env::set_var(DOC, "90000");
    std::env::set_var(TOKENS, "512");
    assert_eq!(
        compose_budgets_from_env().expect("set").doc,
        Duration::from_secs(90)
    );
    assert_eq!(compose_doc_max_tokens_from_env(), Ok(512));

    std::env::set_var(DOC, "5");
    assert!(compose_budgets_from_env()
        .expect_err("out of range")
        .contains(DOC));
    std::env::set_var(DOC, "90000");
    std::env::set_var(TOKENS, "0");
    assert!(compose_doc_max_tokens_from_env().is_err());

    // the retired single-budget name is refused, not ignored
    std::env::set_var(RETIRED, "29000");
    assert!(compose_budgets_from_env()
        .expect_err("retired")
        .contains("retired"));
}

/// Both budget settings: unset gives the default; 1 000..=599 000 ms is
/// accepted; anything else is refused with a message naming the setting.
#[test]
fn compose_budget_settings_parse_and_refuse() {
    let _env = clean_env();
    // One setting at a time; the other stays unset (its default).
    let edit = |v: Option<&str>| {
        set(DOC, None);
        set(EDIT, v);
        compose_budgets_from_env().map(|b| b.edit)
    };
    let doc = |v: Option<&str>| {
        set(EDIT, None);
        set(DOC, v);
        compose_budgets_from_env().map(|b| b.doc)
    };
    assert_eq!(edit(None), Ok(Duration::from_secs(29)));
    assert_eq!(doc(None), Ok(Duration::from_secs(120)));
    assert_eq!(edit(Some("29000")), Ok(Duration::from_secs(29)));
    assert_eq!(doc(Some("90000")), Ok(Duration::from_secs(90)));
    assert_eq!(doc(Some(" 1000 ")), Ok(Duration::from_secs(1)));
    assert_eq!(doc(Some("599000")), Ok(Duration::from_millis(599_000)));
    for bad in [
        "",
        "abc",
        "-5",
        "1.5",
        "999",
        "0",
        "599001",
        "600000",
        "99999999999999999999",
    ] {
        let e = edit(Some(bad)).expect_err(bad);
        assert!(e.contains(EDIT), "{bad}: {e}");
        let e = doc(Some(bad)).expect_err(bad);
        assert!(e.contains(DOC), "{bad}: {e}");
    }
}

/// AIEN_COMPOSE_DOC_MAX_TOKENS: default 1024, 16..=4096, bad values refused.
#[test]
fn compose_doc_token_cap_parses_and_refuses() {
    let _env = clean_env();
    let cap = |v: Option<&str>| {
        set(TOKENS, v);
        compose_doc_max_tokens_from_env()
    };
    assert_eq!(cap(None), Ok(1024));
    assert_eq!(cap(Some("16")), Ok(16));
    assert_eq!(cap(Some(" 4096 ")), Ok(4096));
    for bad in ["", "x", "-1", "15", "0", "4097", "1e3"] {
        let e = cap(Some(bad)).expect_err(bad);
        assert!(e.contains(TOKENS), "{bad}: {e}");
    }
}
