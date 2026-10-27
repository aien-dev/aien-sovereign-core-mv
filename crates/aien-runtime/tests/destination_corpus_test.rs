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

//! The destination parser over every campaign goal corpus in this repository
//! (v5 G6 finding: two independently authored v5 goals were refused before
//! the model). For each goal the probe records what `named_destination` reads
//! and what the compose bridge then decides, and compares the whole table
//! with the committed record
//! `docs/campaigns/open-model-qwen3/evidence-v5/destination-probe.tsv`.
//! Any change to how a real goal is read shows up here as a diff.
//!
//! The verdict is observed through `ComposeBridge::run_task`
//! (tests/support/compose_harness.rs) with a silent model:
//! - `Refused(why)`: the bridge refused the goal before any model call; its
//!   refusal text is `RunComposeTask: ` followed by `why`.
//! - `Edit(path)`: the edit proposer was asked, and the prompt carries the
//!   edit block of `path`.
//! - `New`: the document proposer was asked; or the run was refused before the
//!   model for a goal requirement it could not read (no call to look at), and
//!   the workspace is empty, so no file exists that could have been an edit.
//!
//! The workspace is empty, so an edit goal whose file a campaign seeds reads
//! as `New` here; the parse column is what matters. To rewrite the record:
//! `DEST_PROBE_WRITE=<path> cargo test -p aien-runtime --test
//! destination_corpus_test` (linked build), then review the diff.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::destination::named_destination;
use compose_harness::{silent, Asked, Harness};
use std::path::{Path, PathBuf};

const CORPUS: &[&str] = &[
    "docs/campaigns/open-model-qwen3/tasks-oq3-v5.json",
    "docs/campaigns/open-model-qwen3/tasks-oq3-v4.json",
    "docs/campaigns/open-model-qwen3/tasks-oq3-v3.json",
    "docs/campaigns/open-model-qwen3/dryrun-tasks-v2.json",
    "docs/campaigns/open-model-qwen3/diagnostic-long-1-tasks.json",
    "docs/campaigns/next-phase-1/tasks-v5.json",
    "docs/campaigns/next-phase-1/tasks-v6.json",
    "docs/campaigns/next-phase-1/tasks-v8.json",
];
const RECORD: &str = "docs/campaigns/open-model-qwen3/evidence-v5/destination-probe.tsv";

/// The prefix of a task refusal made before any model call.
const REFUSAL_PREFIX: &str = "RunComposeTask: ";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// (corpus, id, goal) for every task of every corpus file.
fn goals() -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for f in CORPUS {
        let text = std::fs::read_to_string(repo().join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        for t in v["tasks"]
            .as_array()
            .unwrap_or_else(|| panic!("{f}: no tasks"))
        {
            out.push((
                f.to_string(),
                t["id"].as_str().unwrap().to_string(),
                t["goal"].as_str().unwrap().to_string(),
            ));
        }
    }
    out
}

/// The path of the edit block in `prompt`, if it carries one.
fn edit_path(prompt: &str) -> Option<String> {
    const HEAD: &str = "\nThe file ";
    const TAIL: &str = " already exists. Its current content is:\n";
    let end = prompt.rfind(TAIL)?;
    let start = prompt[..end].rfind(HEAD)? + HEAD.len();
    Some(prompt[start..end].to_string())
}

/// What the bridge decides for `goal` in the harness workspace.
fn verdict(h: &Harness, goal: &str) -> String {
    h.clear();
    let r = match h.run(goal) {
        Err(e) => {
            assert!(
                h.calls().is_empty(),
                "{goal}: a refused goal reached the model"
            );
            let why = e
                .strip_prefix(REFUSAL_PREFIX)
                .unwrap_or_else(|| panic!("{goal}: not a task refusal: {e}"));
            return format!("Refused({why})");
        }
        Ok(r) => r,
    };
    match h.calls().first() {
        Some(c) if c.asked == Asked::Document => {
            assert_eq!(edit_path(&c.prompt), None, "{goal}: {}", c.prompt);
            "New".into()
        }
        Some(c) => format!(
            "Edit({})",
            edit_path(&c.prompt).unwrap_or_else(|| panic!("{goal}: no edit block: {}", c.prompt))
        ),
        None => {
            // Refused before the model for a requirement it could not read:
            // one refused attempt, and nothing in the workspace to edit.
            assert_eq!(r.proposal_attempts.len(), 1, "{goal}: {r:?}");
            assert_eq!(r.proposal_attempts[0].outcome, "refused", "{goal}: {r:?}");
            assert_eq!(std::fs::read_dir(h.ws()).unwrap().count(), 0, "{goal}");
            "New".into()
        }
    }
}

fn table(h: &Harness) -> String {
    let mut s = String::from("corpus\tid\tparse\tverdict\n");
    for (f, id, goal) in goals() {
        let parse = match named_destination(&goal, &|_| false) {
            Ok(Some(p)) => format!("dest {p}"),
            Ok(None) => "none".into(),
            Err(e) => format!("error {e}"),
        };
        s.push_str(&format!("{f}\t{id}\t{parse}\t{}\n", verdict(h, &goal)));
    }
    s
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn every_campaign_goal_reads_as_recorded() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    let now = table(&h);
    assert!(goals().len() >= 60, "a corpus file went missing");
    if let Ok(p) = std::env::var("DEST_PROBE_WRITE") {
        std::fs::write(&p, &now).unwrap();
    }
    let record = std::fs::read_to_string(repo().join(RECORD)).unwrap_or_default();
    assert!(
        now == record,
        "the destination parser reads a campaign goal differently from {RECORD}:\n{}",
        now.lines()
            .zip(record.lines().chain(std::iter::repeat("")))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| format!("  now:    {a}\n  record: {b}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The three v5 goals the G6 finding is about, by their declared destination.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn v5_goals_resolve_to_their_declared_destinations() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    let v5: Vec<_> = goals()
        .into_iter()
        .filter(|(f, _, _)| f.ends_with("tasks-oq3-v5.json"))
        .collect();
    let goal = |id: &str| v5.iter().find(|(_, i, _)| i == id).unwrap().2.clone();
    for (id, want) in [
        ("D2", "docs/rename-photos.md"),
        ("D4", "docs/choir-summary.md"),
    ] {
        assert_eq!(
            named_destination(&goal(id), &|_| false),
            Ok(Some(want.to_string())),
            "{id}"
        );
        assert_eq!(verdict(&h, &goal(id)), "New", "{id}");
    }
    // N1 is read, then refused as unsafe (outside the workspace).
    assert_eq!(
        named_destination(&goal("N1"), &|_| false),
        Ok(Some("../shared-stuff/reminder.txt".to_string()))
    );
    let v = verdict(&h, &goal("N1"));
    assert!(
        v.starts_with("Refused(") && v.contains("cannot be written safely"),
        "{v}"
    );
}
