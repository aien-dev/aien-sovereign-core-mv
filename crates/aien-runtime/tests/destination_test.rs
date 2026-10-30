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

//! Issue #288: the task kind, prompt and budget follow the DESTINATION the
//! goal names, not the first path that happens to exist.
//!
//! Every test drives `ComposeBridge::run_task` (tests/support/compose_harness.rs).
//! The task kind is which proposer the bridge asked (`Asked::Edit` = the goal
//! edits an existing file, `Asked::Document` = a new document), the edit target
//! and the destination are the blocks of the prompt the model was handed, and a
//! refused goal is the bridge's refusal text (no model call).
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use compose_harness::{silent, Asked, Call, Harness};

/// A harness whose workspace holds `files` and an empty `docs` folder.
fn ws_with(files: &[(&str, &str)]) -> Harness {
    let h = Harness::new(silent());
    std::fs::create_dir_all(h.ws().join("docs")).unwrap();
    for (n, c) in files {
        h.write(n, c);
    }
    h
}

/// The first model call of a run of `goal` (the run must not be refused).
fn first(h: &Harness, goal: &str) -> Call {
    h.clear();
    let r = h
        .run(goal)
        .unwrap_or_else(|e| panic!("{goal}: refused: {e}"));
    h.calls()
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("{goal}: no model call: {r:?}"))
}

/// The bridge's refusal of `goal`, without the `RunComposeTask: ` prefix; no
/// model call was made.
fn refused(h: &Harness, goal: &str) -> String {
    h.clear();
    let e = h.run(goal).expect_err(goal);
    assert!(
        h.calls().is_empty(),
        "{goal}: a refused goal reached the model"
    );
    e.strip_prefix("RunComposeTask: ")
        .unwrap_or_else(|| panic!("{goal}: not a task refusal: {e}"))
        .to_string()
}

/// The new-document block the prompt ends with when the goal names a
/// destination that does not exist yet (the product's text, verbatim).
fn new_block(path: &str) -> String {
    format!("\nWrite the new file {path}. The filename line of your answer must be exactly: filename: {path}")
}

/// The edit block the prompt ends with when the goal's destination is an
/// existing file (the product's text, verbatim).
fn edit_block(path: &str, content: &str) -> String {
    let nl = if content.ends_with('\n') { "" } else { "\n" };
    format!(
        "\nThe file {path} already exists. Its current content is:\n{content}{nl}\
         Write the complete new content of {path}: keep every existing line and make the requested change."
    )
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn create_new_file_about_an_existing_one_is_a_new_document() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("README.md", "# readme\n")]);
    let goal = "Create docs/SUMMARY.md about README.md";
    let c = first(&h, goal);
    // A new document (no edit target), destination docs/SUMMARY.md.
    assert_eq!(c.asked, Asked::Document);
    assert!(
        c.prompt.ends_with(&new_block("docs/SUMMARY.md")),
        "{}",
        c.prompt
    );
    assert!(!c.prompt.contains("already exists"));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn update_destination_using_a_source_is_an_edit_of_the_destination() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("README.md", "# readme\n"), ("docs/NOTES.md", "n\n")]);
    let c = first(&h, "Update README.md using docs/NOTES.md");
    assert_eq!(c.asked, Asked::Edit);
    // The edit target: README.md with its exact content.
    assert!(
        c.prompt.ends_with(&edit_block("README.md", "# readme\n")),
        "{}",
        c.prompt
    );
    assert!(c.prompt.contains("The file README.md already exists"));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn append_to_existing_guide_is_an_edit() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("docs/GUIDE.md", "# guide\n")]);
    let c = first(&h, "Append a section to docs/GUIDE.md");
    assert_eq!(c.asked, Asked::Edit);
    assert!(
        c.prompt
            .ends_with(&edit_block("docs/GUIDE.md", "# guide\n")),
        "{}",
        c.prompt
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn missing_destination_with_existing_source_is_new() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("src.txt", "s\n")]);
    let c = first(&h, "Write out.md based on src.txt");
    assert_eq!(c.asked, Asked::Document);
    assert!(!c.prompt.contains("already exists"), "{}", c.prompt);
}

// Refused before any model call: runs in the stub build too.
#[test]
fn unsafe_destinations_are_refused_with_a_reason() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    let outside = h.tmp.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("s.txt"), "s\n").unwrap();
    std::fs::create_dir_all(h.ws().join("docs")).unwrap();
    std::os::unix::fs::symlink(outside.join("s.txt"), h.ws().join("link.md")).unwrap();
    std::os::unix::fs::symlink(&outside, h.ws().join("linkdir")).unwrap();
    h.write("README.md", "r\n");
    let abs = format!("Create {} about README.md", outside.join("n.md").display());
    for goal in [
        "Create ../x.md about README.md",
        abs.as_str(),
        "Create ~/x.md about README.md",
        "Update link.md using README.md",
        "Create linkdir/new.md about README.md",
    ] {
        let why = refused(&h, goal);
        assert!(why.contains("destination"), "{goal}: {why}");
    }
}

#[test]
fn two_destinations_are_an_ambiguous_refusal_naming_both() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("a.md", "a\n")]);
    for goal in ["Create a.md and b.md", "Update a.md. Then create b.md"] {
        let e = refused(&h, goal);
        assert!(
            e.contains("ambiguous destination") && e.contains("a.md") && e.contains("b.md"),
            "{e}"
        );
    }
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn source_only_goal_is_refused_with_candidates() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("README.md", "# r\n")]);
    let e = refused(&h, "Summarise README.md");
    assert!(
        e.contains("no destination named") && e.contains("README.md"),
        "{e}"
    );
    let c = first(&h, "Edit the given README.md ");
    assert_eq!(c.asked, Asked::Edit);
    assert!(
        c.prompt.ends_with(&edit_block("README.md", "# r\n")),
        "{}",
        c.prompt
    );
    let c = first(&h, "Update the reference README.md");
    assert!(
        c.prompt.ends_with(&edit_block("README.md", "# r\n")),
        "{}",
        c.prompt
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn new_document_prompt_states_the_destination() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("README.md", "# r\n")]);
    let p = first(&h, "Create docs/SUMMARY.md about README.md").prompt;
    assert!(p.contains("filename: docs/SUMMARY.md"), "{p}");
}

#[test]
fn rename_is_refused() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("a.md", "a\n")]);
    assert!(refused(&h, "Rename a.md to b.md").contains("rename is not supported"));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn u1_garage_goal_is_an_edit_of_the_named_file() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[]);
    let garage = "# Garage\n\n## Before a trip\n- check oil\n";
    h.write("notes/GARAGE.md", garage);
    let goal = r###"Please add "- check tyre pressure" to the "## Before a trip" section of notes/GARAGE.md."###;
    let c = first(&h, goal);
    assert_eq!(c.asked, Asked::Edit);
    // The edit target, and so the destination, is notes/GARAGE.md.
    assert!(
        c.prompt.ends_with(&edit_block("notes/GARAGE.md", garage)),
        "{}",
        c.prompt
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn summary_of_source_into_destination_is_a_document() {
    let _home = home_guard::home_slot();
    let h = ws_with(&[("README.md", "# r\n")]);
    let c = first(&h, "Write a summary of README.md into docs/SUMMARY.md");
    assert_eq!(c.asked, Asked::Document);
    assert!(!c.prompt.contains("already exists"), "{}", c.prompt);
    assert!(
        c.prompt.ends_with(&new_block("docs/SUMMARY.md")),
        "{}",
        c.prompt
    );
}
