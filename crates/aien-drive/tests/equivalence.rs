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

//! Equivalence with the removed Python `scripts/aien_drive.py` (issue #179).
//!
//! `golden_from_python_original.json` was produced by running the original
//! Python `extract_tool_calls` and `dispatch_tool` (git history of
//! aien-sovereign-core before #182) on exactly these inputs. Results are
//! compared as parsed JSON (key order is not significant to the model).
//! The one allowed difference: error TEXT from the OS (Python prints
//! "[Errno 2] ...", Rust prints "No such file or directory (os error 2)"),
//! so for errors that come from the OS only the `error` key is compared.

use aien_drive::{dispatch_tool, extract_tool_calls, Config};
use serde_json::Value;
use std::time::Duration;

fn cfg() -> Config {
    Config {
        endpoint: String::new(),
        model: String::new(),
        cortex_endpoint: "http://127.0.0.1:1".into(),
        cortex_space: "aien-sources".into(),
        request_timeout: Duration::from_secs(5),
    }
}

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_from_python_original.json")).unwrap()
}

#[test]
fn extract_matches_python() {
    let g = golden();
    let cases = g["extract"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for c in cases {
        let got = extract_tool_calls(c["text"].as_str().unwrap());
        assert_eq!(Value::Array(got), c["calls"], "text: {}", c["text"]);
    }
}

fn prepare(tmp: &std::path::Path) {
    std::fs::create_dir_all(tmp.join("ls")).unwrap();
    std::fs::write(tmp.join("f.txt"), "one\ntwo\nthree\nfour\n").unwrap();
    std::fs::write(tmp.join("e.txt"), "aa bb aa\n").unwrap();
    for n in ["b", "a", "c"] {
        std::fs::write(tmp.join("ls").join(n), "x").unwrap();
    }
    std::fs::write(tmp.join("g.txt"), "needle here\nother\n").unwrap();
}

fn run_dispatch_cases(mutate_expected: impl Fn(&mut Value)) -> Vec<String> {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let tmp = std::env::temp_dir().join(format!("aien-drive-eq-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    prepare(&tmp);
    let t = tmp.to_str().unwrap();
    let g = golden();
    let mut failures = vec![];
    for c in g["dispatch"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let args: Value = serde_json::from_str(&c["args"].to_string().replace("{TMP}", t)).unwrap();
        let got = dispatch_tool(&cfg(), name, &args);
        let got: Value = serde_json::from_str(&got.to_string().replace(t, "{TMP}")).unwrap();
        let mut want = c["result"].clone();
        mutate_expected(&mut want);
        let os_error = want["error"]
            .as_str()
            .is_some_and(|e| e.starts_with("[Errno"));
        let ok = if os_error {
            got.get("error").is_some()
        } else {
            got == want
        };
        if !ok {
            failures.push(format!("{name}: got {got} want {want}"));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    failures
}

#[test]
fn dispatch_matches_python() {
    let f = run_dispatch_cases(|_| {});
    assert!(f.is_empty(), "{f:#?}");
}

/// Negative control: corrupt one expected value and the comparison must fail.
#[test]
fn comparison_detects_a_difference() {
    let f = run_dispatch_cases(|w| {
        if w.get("milestone_done").is_some() {
            w["milestone_done"] = Value::Bool(false);
        }
    });
    assert_eq!(f.len(), 1, "{f:#?}");
}
