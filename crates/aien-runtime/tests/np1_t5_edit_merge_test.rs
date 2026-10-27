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

//! NEXT-PHASE-1 v7 T5: an edit-mode reply never drops an existing line.
//!
//! v6 T5 FAILed T5-K and Q2: the model (GB10 and the CPU reference agree,
//! reply sha256 cd8e3b30...) answered the edit goal with only the changed
//! part, `filename: CHANGELOG.md` / `## 0.1.0` / `- add contact file`, ending
//! on eos after 16 of 96 tokens, and the Skill handed that fragment on as the
//! complete new file. These tests run the exact v6 seed and reply through
//! `ComposeBridge::run_task` (tests/support/compose_harness.rs) on the CPU.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::check_file_proposal;
use compose_harness::{replies, Asked, Harness};

/// The v6 pre-seed, byte for byte (seed sha256 29e905c6...).
const SEED: &str = include_str!("../../../docs/campaigns/next-phase-1/seed-v6/T5/CHANGELOG.md");
/// The v6 T5 reply, byte for byte (GB10 and CPU reference).
const V6_REPLY: &str = include_str!(
    "../../../docs/campaigns/next-phase-1/replies/cd8e3b30c007e2f0bfe66277688dbec833d55a7bbefc47103203f2615125d094.txt"
);
const GOAL: &str =
    "Add the line \"- add contact file\" under the \"## 0.1.0\" heading in CHANGELOG.md.";

/// Runs `texts` (one per attempt) through the bridge for the T5 goal on the
/// seeded workspace: the proposal (or the refusal) and every attempt.
fn run(texts: &[&str]) -> (Result<String, String>, Vec<aien_runtime::ProposalAttempt>) {
    let h = Harness::new(replies(texts));
    h.write("CHANGELOG.md", SEED);
    let r = h.run(GOAL).unwrap();
    let first = &h.calls()[0];
    assert_eq!(first.asked, Asked::Edit, "edit mode");
    assert!(
        first.prompt.contains(SEED),
        "the model is shown the whole seed"
    );
    assert!(
        first
            .prompt
            .contains("The file CHANGELOG.md already exists."),
        "the edit target is CHANGELOG.md: {}",
        first.prompt
    );
    let out = match (r.proposal, r.proposer_error) {
        (Some(p), _) => Ok(p),
        (None, Some(e)) => Err(e),
        (None, None) => panic!("neither a proposal nor a refusal"),
    };
    (out, r.proposal_attempts)
}

fn lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

/// ACCEPTANCE-v6 T5-K: every non-empty seed line is a line of the content.
/// T5-N: `- add contact file` follows `## 0.1.0` before the next heading.
fn t5_rows(content: &str) -> (bool, bool) {
    let c = lines(content);
    let k = lines(SEED)
        .iter()
        .filter(|l| !l.trim().is_empty())
        .all(|l| c.iter().any(|x| x.trim_end() == l.trim_end()));
    let n = c.iter().position(|l| *l == "## 0.1.0").is_some_and(|h| {
        c[h + 1..]
            .iter()
            .take_while(|l| !l.starts_with('#'))
            .any(|l| *l == "- add contact file")
    });
    (k, n)
}

/// The red-then-green test: the exact v6 reply keeps all three seed lines,
/// byte-identical and in order, with the new line under the heading.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn v6_t5_reply_keeps_every_seed_line() {
    let _home = home_guard::home_slot();
    assert_eq!(SEED, "# Changelog\n\n## 0.1.0\n- initial release\n");
    assert_eq!(
        V6_REPLY,
        "filename: CHANGELOG.md\n## 0.1.0\n- add contact file"
    );
    let (out, attempts) = run(&[V6_REPLY]);
    assert_eq!(attempts.len(), 1, "{attempts:?}");
    assert_eq!(attempts[0].outcome, "parsed");
    // The attempt keeps the model's own reply; the proposal is the merge.
    assert_eq!(attempts[0].text.as_deref(), Some(V6_REPLY));
    let p = check_file_proposal(&out.expect("proposal")).unwrap();
    assert_eq!(p.path, "CHANGELOG.md");
    assert_eq!(
        p.content,
        "# Changelog\n\n## 0.1.0\n- add contact file\n- initial release\n"
    );
    assert_eq!(t5_rows(&p.content), (true, true));
    // Every seed line survives byte-identical, in its original order.
    let c = lines(&p.content);
    let mut at = 0;
    for l in lines(SEED) {
        at += c[at..].iter().position(|x| *x == l).expect(l) + 1;
    }
}

/// A whole-file reply that keeps every line (what the v6 prompt asks for)
/// is proposed byte for byte: the merge adds nothing and drops nothing.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn whole_file_reply_is_kept_byte_for_byte() {
    let _home = home_guard::home_slot();
    let whole =
        "filename: CHANGELOG.md\n# Changelog\n\n## 0.1.0\n- add contact file\n- initial release\n";
    let (out, _) = run(&[whole]);
    assert_eq!(out.unwrap(), whole);
}

/// A fragment that puts the new line above an existing line lands right
/// before that line, not at the top of the file.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn new_line_above_its_anchor_lands_before_it() {
    let _home = home_guard::home_slot();
    let (out, _) = run(&["filename: CHANGELOG.md\n- add contact file\n- initial release\n"]);
    let p = check_file_proposal(&out.unwrap()).unwrap();
    assert_eq!(
        p.content,
        "# Changelog\n\n## 0.1.0\n- add contact file\n- initial release\n"
    );
}

/// A reply sharing no line with the file has no place in it: refused (and
/// retried) instead of replacing the file.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn reply_without_an_anchor_is_refused_and_retried() {
    let _home = home_guard::home_slot();
    let (out, attempts) = run(&["filename: CHANGELOG.md\n- add contact file\n", V6_REPLY]);
    assert_eq!(attempts[0].outcome, "refused", "{attempts:?}");
    assert!(
        attempts[0]
            .reason
            .as_deref()
            .unwrap()
            .contains("shares no line"),
        "{attempts:?}"
    );
    assert_eq!(attempts[1].outcome, "parsed");
    assert_eq!(
        t5_rows(&check_file_proposal(&out.unwrap()).unwrap().content),
        (true, true)
    );
}

/// A reply that only repeats existing lines changes nothing: refused.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn reply_that_changes_nothing_is_refused() {
    let _home = home_guard::home_slot();
    let (out, attempts) = run(&["filename: CHANGELOG.md\n## 0.1.0\n"]);
    assert!(out.is_err());
    assert!(attempts.iter().all(|a| a
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("changes nothing")));
}

/// A reply naming another file keeps the v5 whole-file meaning (it is not
/// merged into the edit target, so the bridge then refuses it for writing
/// another path than the goal named); without an edit target the proposal is
/// the reply.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn other_path_and_no_target_keep_the_v5_proposal() {
    let _home = home_guard::home_slot();
    let other = "filename: NOTES.md\nhello\n";
    let (out, attempts) = run(&[other]);
    assert_eq!(attempts[0].outcome, "parsed", "{attempts:?}");
    assert_eq!(attempts[0].text.as_deref(), Some(other));
    assert_eq!(
        out.unwrap_err(),
        "the proposal writes NOTES.md but the goal named the destination CHANGELOG.md"
    );
    // No edit target: a new CHANGELOG.md, the v6 reply is the proposal byte for byte.
    let h = Harness::new(replies(&[V6_REPLY]));
    let r = h.run("Write a short changelog into CHANGELOG.md.").unwrap();
    assert_eq!(h.calls()[0].asked, Asked::Document);
    assert_eq!(r.proposal.as_deref(), Some(V6_REPLY), "{r:?}");
}

/// Merge unit cases: seed lines keep their exact bytes (trailing spaces
/// included); a seed without a final newline gains one.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn merge_keeps_seed_bytes() {
    let _home = home_guard::home_slot();
    let prior = "a  \nb\nc";
    let goal = "Add the line \"new\" below the line \"b\" in notes.txt.";
    for (reply, want) in [
        ("filename: notes.txt\nb\nnew\n", "a  \nb\nnew\nc\n"),
        ("filename: notes.txt\na\nb\nc\nd\n", "a  \nb\nc\nd\n"),
    ] {
        let h = Harness::new(replies(&[reply]));
        h.write("notes.txt", prior);
        let r = h.run(goal).unwrap();
        assert_eq!(h.calls()[0].asked, Asked::Edit);
        let p = check_file_proposal(r.proposal.as_deref().expect("proposal")).unwrap();
        assert_eq!(p.content, want, "{reply:?}");
    }
}
