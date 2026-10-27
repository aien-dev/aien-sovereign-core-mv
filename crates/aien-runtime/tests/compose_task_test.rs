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

//! NEXT-PHASE-1 cut 1b + 2: RunComposeTask, ComposeNote, ComposeRecall and
//! RecoverComposeHome through omega COMPOSITION-2.
//!
//! With librx_compose.a linked (aien-omega-compose built with
//! AIEN_OMEGA_COMPOSE_DIR, see its README) these run the real composition;
//! in a stub build they check that the bridge reports the missing library.
//!
//! The retry loop, the budgets and the task kind are observed through
//! `ComposeBridge::run_task` (tests/support/compose_harness.rs): the prompt and
//! time limit each model call is handed, which proposer was asked, and the
//! report. The rules under a short injected budget are in
//! compose_budget_test.rs; the budget settings themselves in compose_env_test.rs.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::{ComposeRecallReport, ComposeTaskReport, ControlResponse};
use aien_runtime::{
    check_file_proposal, parse_file_proposal, ComposeBridge, ComposeProposer, Generation,
};
use compose_harness::{reply, Asked, Harness};
use std::sync::Arc;
use std::time::Duration;

/// A goal that names no destination: the model names the file.
const NO_DEST_GOAL: &str = "remember the constraints and propose one change";

fn proposer() -> ComposeProposer {
    Arc::new(|prompt: &str, _limit: std::time::Duration| {
        assert!(prompt.contains("Authorized workspace:"));
        assert!(prompt.contains("filename: <relative path>"));
        Ok(Generation {
            text: "filename: NOTES.md\nconstraint: keep main green\n".to_string(),
            tokens: 12,
            finish_reason: None,
            ..Default::default()
        })
    })
}

fn report(r: ControlResponse) -> ComposeTaskReport {
    match r {
        ControlResponse::ComposeTaskResult(r) => *r,
        other => panic!("expected ComposeTaskResult, got {other:?}"),
    }
}

fn recalled(r: ControlResponse) -> ComposeRecallReport {
    match r {
        ControlResponse::ComposeRecalled(r) => *r,
        other => panic!("expected ComposeRecalled, got {other:?}"),
    }
}

fn snapshot(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap_or_default(),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn compose_task_commits_and_survives_restart() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    let ws = tmp.path().join("workspace");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("README.md"), b"project\n").unwrap();
    let ws_before = snapshot(&ws);

    let bridge = ComposeBridge::new(home.clone(), proposer(), "test:fixed-proposer");
    let resp = bridge.run_task(
        "remember the constraints and propose one change",
        ws.to_str().unwrap(),
    );
    if !aien_omega_compose::LINKED {
        match resp {
            ControlResponse::Error(e) => assert!(e.contains("not linked"), "{e}"),
            other => panic!("stub build must refuse, got {other:?}"),
        }
        return;
    }
    let r1 = report(resp);
    assert!(r1.committed, "{r1:?}");
    assert_eq!(r1.outcome, 1);
    assert_eq!(r1.branch_count, 1);
    assert_eq!(r1.winner, Some(0));
    assert_eq!(r1.aegis_pass_mask & 1, 1);
    assert!(
        r1.cx_goal != 0
            && r1.cx_evidence != 0
            && r1.cx_promotion != 0
            && !r1.cx_candidates.is_empty()
    );
    assert_eq!(
        r1.proposal.as_deref(),
        Some("filename: NOTES.md\nconstraint: keep main green\n")
    );
    assert_eq!(r1.proposal_attempts.len(), 1, "{:?}", r1.proposal_attempts);
    assert_eq!(r1.proposal_attempts[0].outcome, "parsed");
    assert_eq!(r1.proposal_attempts[0].aegis.as_deref(), Some("pass"));
    assert_eq!(r1.proposal_attempts[0].tokens, 12);
    assert_eq!(r1.machine_id.len(), 64);
    assert_eq!(r1.proposal_path.as_deref(), Some("NOTES.md"));
    assert_eq!(
        r1.proposal_content_sha256.as_deref(),
        Some(aien_omega_compose::hex(
            &<sha2::Sha256 as sha2::Digest>::digest(b"constraint: keep main green\n")
        ))
        .as_deref()
    );
    let c = match bridge.note("constraint", "keep main green", &[]) {
        ControlResponse::ComposeNoted(n) => n,
        other => panic!("note: {other:?}"),
    };
    assert!(c.id > r1.cx_promotion);
    assert!(matches!(
        bridge.note("effect", "x", &[1 << 40]),
        ControlResponse::Error(_)
    ));
    assert!(matches!(
        bridge.note("rumour", "x", &[]),
        ControlResponse::Error(_)
    ));
    let before = recalled(bridge.recall(&[r1.cx_promotion], Some(c.id)));
    let cited = [
        r1.cx_goal,
        r1.cx_candidates[0],
        r1.cx_evidence,
        r1.cx_promotion,
    ];
    let digests: Vec<String> = cited
        .iter()
        .map(|&id| bridge.record_digest(id).unwrap())
        .collect();
    drop(bridge);

    // restart: same home, same machine, cited records unchanged
    let bridge = ComposeBridge::new(home.clone(), proposer(), "test:fixed-proposer");
    for (id, d) in cited.iter().zip(digests.iter()) {
        assert_eq!(
            &bridge.record_digest(*id).unwrap(),
            d,
            "record {id} after restart"
        );
    }
    let r2 = report(bridge.run_task("second goal", ws.to_str().unwrap()));
    assert!(r2.committed);
    assert_eq!(
        r2.machine_id, r1.machine_id,
        "same AienMachineId across restart"
    );
    assert!(r2.cx_evidence > r1.cx_evidence);
    let after = recalled(bridge.recall(&[r1.cx_promotion], Some(c.id)));
    assert_eq!(after.machine_id, before.machine_id);
    assert_eq!(
        after.prefix_digest, before.prefix_digest,
        "records 1..=S1 unchanged"
    );
    assert_eq!(after.cited, before.cited);
    let k = after
        .host
        .iter()
        .find(|h| h.id == c.id)
        .expect("constraint recalled after restart");
    assert_eq!(k.note.as_deref(), Some("constraint"));
    assert_eq!(k.text.as_deref(), Some("keep main green"));
    assert!(k.verified);

    // containment: nothing in the workspace changed (this cut executes no effect)
    assert_eq!(snapshot(&ws), ws_before);
    // the home holds only the composition files (no staged-branch leftovers)
    for (name, _) in snapshot(&home) {
        assert!(
            name == "machine.id" || name == "cortex.cx" || name.starts_with("jspace"),
            "unexpected file in compose home: {name}"
        );
    }
}

#[test]
fn failing_model_commits_nothing() {
    let _home = home_guard::home_slot();
    if !aien_omega_compose::LINKED {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let bridge = ComposeBridge::new(
        tmp.path().join("compose"),
        Arc::new(|_: &str, _: std::time::Duration| Err("model unavailable".to_string())),
        "test:failing",
    );
    let r = report(bridge.run_task("goal", ws.to_str().unwrap()));
    assert!(!r.committed, "{r:?}");
    assert_eq!(r.winner, None);
    assert_eq!(r.proposal, None);
}

#[test]
fn bad_workspace_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let bridge = ComposeBridge::new(tmp.path().join("compose"), proposer(), "test");
    match bridge.run_task("goal", tmp.path().join("missing").to_str().unwrap()) {
        ControlResponse::Error(e) => assert!(e.contains("workspace"), "{e}"),
        other => panic!("expected Error, got {other:?}"),
    }
}

#[test]
fn proposal_template_parser() {
    // filename line present
    let p = check_file_proposal(
        "filename: NOTES.md
line one
",
    )
    .unwrap();
    assert_eq!(
        (p.path.as_str(), p.content.as_str()),
        (
            "NOTES.md",
            "line one
"
        )
    );
    let p = check_file_proposal(
        "
**Filename: `docs/plan.txt`**

```
step 1
step 2
```
",
    )
    .unwrap();
    assert_eq!(
        (p.path.as_str(), p.content.as_str()),
        (
            "docs/plan.txt",
            "step 1
step 2
"
        )
    );
    // filename line absent: refused, even when a path appears later
    for t in [
        "Sure, I can help with that!",
        "NOTES.md
line one
",
        "To create NOTES.md:
filename: NOTES.md
x
",
        "",
    ] {
        assert!(check_file_proposal(t).is_err(), "{t:?}");
        assert!(parse_file_proposal(t).is_none());
    }
    assert!(check_file_proposal(
        "NOTES.md
x"
    )
    .unwrap_err()
    .contains("no filename line"));
    // outside the workspace: refused before AEGIS
    for t in [
        "filename: /etc/passwd
root
",
        "filename: ../escape.txt
x
",
        "filename: docs/../../escape.txt
x
",
        "filename: ~/.bashrc
x
",
    ] {
        let e = check_file_proposal(t).unwrap_err();
        assert!(e.contains("outside the workspace"), "{t:?}: {e}");
    }
    assert!(check_file_proposal(
        "filename: a b.txt
x
"
    )
    .is_err());
    assert!(check_file_proposal(
        "filename: NOTES.md

   
"
    )
    .unwrap_err()
    .contains("empty content"));
    // deterministic
    assert_eq!(
        check_file_proposal(
            "filename: a.txt
z
"
        ),
        check_file_proposal(
            "filename: a.txt
z
"
        )
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn retry_is_capped_and_recorded() {
    let _home = home_guard::home_slot();
    let chatty = |text: &'static str, tokens: usize| {
        move |_: usize, _: &str, _: Duration| {
            Ok(Generation {
                text: text.into(),
                tokens,
                finish_reason: None,
                ..Default::default()
            })
        }
    };
    // never parses: exactly 3 calls, all refused
    {
        let h = Harness::new(Arc::new(chatty("Sure!", 2)));
        let r = h.run(NO_DEST_GOAL).unwrap();
        assert!(r
            .proposer_error
            .as_deref()
            .unwrap()
            .contains("no filename line"));
        assert!(r.proposal.is_none() && !r.committed, "{r:?}");
        assert_eq!(h.calls().len(), 3);
        let a = &r.proposal_attempts;
        assert_eq!(a.len(), 3);
        assert!(a
            .iter()
            .all(|x| x.outcome == "refused" && x.tokens == 2 && x.text_sha256.is_some()));
        assert_eq!(
            a.iter().map(|x| x.attempt).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
    // second attempt parses: stops there, and the retry prompt names the refusal
    {
        let h = Harness::new(Arc::new(|n, _: &str, _: Duration| {
            Ok(Generation {
                text: if n == 1 {
                    "Sure!"
                } else {
                    "filename: a.txt\nok\n"
                }
                .into(),
                tokens: 3,
                finish_reason: None,
                ..Default::default()
            })
        }));
        let r = h.run(NO_DEST_GOAL).unwrap();
        assert_eq!(
            r.proposal.as_deref(),
            Some("filename: a.txt\nok\n"),
            "{r:?}"
        );
        let a = &r.proposal_attempts;
        assert_eq!(a.len(), 2);
        assert_eq!(
            (a[0].outcome.as_str(), a[1].outcome.as_str()),
            ("refused", "parsed")
        );
        let p: Vec<String> = h.calls().into_iter().map(|c| c.prompt).collect();
        assert_eq!(p.len(), 2);
        assert!(!p[0].contains("previous answer was refused"), "{}", p[0]);
        assert!(
            p[1].starts_with(&format!("{}\n", p[0]))
                && p[1].contains("previous answer was refused"),
            "{}",
            p[1]
        );
    }
    // The budget rule (attempt 2 does not start when less than the per-attempt
    // budget is left) needs a short budget: compose_budget_test.rs, same name.
    // model errors are recorded, not retried past the cap
    {
        let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
            Err::<Generation, _>("model proposal exceeded 25000 ms".to_string())
        }));
        let r = h.run(NO_DEST_GOAL).unwrap();
        let a = &r.proposal_attempts;
        assert_eq!(h.calls().len(), a.len());
        assert!(
            a.len() <= 3 && a.iter().all(|x| x.outcome == "timeout"),
            "{a:?}"
        );
    }
}

/// ACCEPTANCE-v3 Section 3b at full scale, through the bridge with the default
/// edit budget: a slow (cold) attempt 1 of 16.1 s no longer blocks attempt 2;
/// attempt 2 of 11.6 s leaves too little for 3. Takes about 28 s.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn measured_attempt_budget_admits_a_second_attempt() {
    let _home = home_guard::home_slot();
    // 2 839 + 896 + 47 x 166.6 + 60 ms, the measured full retry attempt, fits
    // the 12 000 ms per-attempt budget of an edit
    let measured: [f64; 4] = [2_839.0, 896.0, 47.0 * 166.6, 60.0];
    assert!(measured.iter().sum::<f64>() <= 12_000.0);
    let h = Harness::new(Arc::new(|n, _: &str, _: Duration| {
        std::thread::sleep(Duration::from_millis(if n == 1 { 16_100 } else { 11_600 }));
        Ok(Generation {
            text: "Goal: echo".into(),
            tokens: 48,
            finish_reason: None,
            ..Default::default()
        })
    }));
    h.write("notes.md", "# notes\n");
    let r = h.run("add a line to notes.md").unwrap();
    let c = h.calls();
    assert_eq!(c.len(), 2, "{r:?}");
    assert!(c.iter().all(|c| c.asked == Asked::Edit));
    // the default (AIEN_COMPOSE_EDIT_BUDGET_MS unset): omega's 30 s wait less 1 s
    assert!(c[0].limit <= Duration::from_secs(29), "{:?}", c[0].limit);
    assert!(c[0].limit > Duration::from_secs(28), "{:?}", c[0].limit);
    assert_eq!(r.proposal_attempts.len(), 2, "{r:?}");
    let reason = r.proposer_error.unwrap();
    assert!(
        reason.contains("no attempt 3") && reason.contains("< 12000 ms per-attempt budget"),
        "{reason}"
    );
}

#[test]
fn unparseable_proposal_fails_the_aegis_contract() {
    let _home = home_guard::home_slot();
    if !aien_omega_compose::LINKED {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let bridge = ComposeBridge::new(
        tmp.path().join("compose"),
        Arc::new(|_: &str, _: std::time::Duration| {
            Ok(Generation {
                text: "Sure, I can help with that!".to_string(),
                tokens: 8,
                finish_reason: None,
                ..Default::default()
            })
        }),
        "test:chatty",
    );
    let r = report(bridge.run_task("goal", ws.to_str().unwrap()));
    assert!(!r.committed, "{r:?}");
    assert_eq!(r.aegis_pass_mask & 1, 0);
    assert_eq!(r.proposal_path, None);
    // Refused by the template parser before AEGIS, retried up to the cap (3).
    assert_eq!(r.proposal_attempts.len(), 3, "{r:?}");
    assert!(r
        .proposal_attempts
        .iter()
        .all(|a| a.outcome == "refused" && a.aegis.is_none()));
    assert!(r
        .proposer_error
        .as_deref()
        .unwrap_or("")
        .contains("no filename line"));
}

#[test]
fn torn_home_is_refused_by_name_then_recovered() {
    let _home = home_guard::home_slot();
    if !aien_omega_compose::LINKED {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let bridge = ComposeBridge::new(home.clone(), proposer(), "test");
    let r1 = report(bridge.run_task("goal", ws.to_str().unwrap()));
    assert!(r1.committed);
    let all = recalled(bridge.recall(&[], Some(r1.cx_promotion)));
    drop(bridge);
    let cx = home.join("cortex.cx");
    let len = std::fs::metadata(&cx).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&cx)
        .unwrap()
        .set_len(len - 5)
        .unwrap();

    let bridge = ComposeBridge::new(home.clone(), proposer(), "test");
    for _ in 0..2 {
        match bridge.note("constraint", "x", &[]) {
            ControlResponse::Error(e) => {
                assert!(e.contains("E_TORN") && e.contains("compose recover"), "{e}")
            }
            other => panic!("torn home must be refused, got {other:?}"),
        }
        assert_eq!(
            std::fs::metadata(&cx).unwrap().len(),
            len - 5,
            "refusal wrote"
        );
    }
    let rep = match bridge.recover() {
        ControlResponse::ComposeRecovered(r) => *r,
        other => panic!("recover: {other:?}"),
    };
    assert!(rep.repaired && rep.tail_torn && rep.opens, "{rep:?}");
    assert_eq!((rep.cause, rep.cut_hi), (-8, len - 5));
    assert!(rep.dropped_records >= 1 && rep.repair_record == rep.records_kept + 1);
    let after = recalled(bridge.recall(&[], Some(r1.cx_promotion)));
    assert_eq!(
        after.prefix_digest, all.prefix_digest,
        "old records unchanged"
    );
    assert_eq!(after.machine_id, all.machine_id);
    assert!(after
        .host
        .iter()
        .any(|h| h.id == rep.repair_record && h.note.as_deref() == Some("repair_tail")));
    let r2 = report(bridge.run_task("goal after recover", ws.to_str().unwrap()));
    assert!(r2.committed);
}

/// ACCEPTANCE-v5 Q3: the proposer's stop reason reaches every attempt record,
/// so the receipt can tell end of sequence from the token limit.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn finish_reason_is_recorded_per_attempt() {
    let _home = home_guard::home_slot();
    let h = Harness::new(Arc::new(|n, _: &str, _: Duration| {
        Ok(if n == 1 {
            Generation {
                text: "Goal: echo".into(),
                tokens: 48,
                finish_reason: Some("max_tokens".into()),
                ..Default::default()
            }
        } else {
            Generation {
                text: "filename: NOTES.md\nkeep changes in the workspace\n".into(),
                tokens: 11,
                finish_reason: Some("eos".into()),
                ..Default::default()
            }
        })
    }));
    let r = h.run("write a short note into NOTES.md").unwrap();
    assert!(r.proposal.is_some(), "{r:?}");
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 2, "{a:?}");
    assert_eq!(a[0].finish_reason.as_deref(), Some("max_tokens"));
    assert_eq!(a[1].finish_reason.as_deref(), Some("eos"));
    // An attempt record without the field (v1..v4 receipts) still reads.
    let old = r#"{"attempt":1,"ms":1,"tokens":48,"outcome":"parsed","reason":null,
        "text_sha256":null,"text":null,"aegis":null}"#;
    let parsed: aien_runtime::ProposalAttempt = serde_json::from_str(old).unwrap();
    assert_eq!(parsed.finish_reason, None);
}

#[test]
fn finish_reason_labels_match_acceptance_v5() {
    use aien_inference_abi::FinishReason;
    assert_eq!(
        aien_runtime::finish_reason_label(&FinishReason::StopToken),
        "eos"
    );
    assert_eq!(
        aien_runtime::finish_reason_label(&FinishReason::LengthLimit),
        "max_tokens"
    );
    assert_eq!(
        aien_runtime::finish_reason_label(&FinishReason::Aborted),
        "aborted"
    );
    assert_eq!(
        aien_runtime::finish_reason_label(&FinishReason::Preempted),
        "preempted"
    );
}

// A file whose own content holds fenced blocks must not be cut at the
// first inner fence (DIAGNOSTIC-LONG-1 task D5: a 649-token FAQ was saved
// as 4 lines).
#[test]
fn proposal_fence_inner_block_kept() {
    let p = check_file_proposal(
        "filename: docs/FAQ.md\n```markdown\n# FAQ\n\nRun:\n```bash\nls -l\n```\nDone.\n```\nTrailing prose.\n",
    )
    .unwrap();
    assert_eq!(p.path, "docs/FAQ.md");
    assert_eq!(p.content, "# FAQ\n\nRun:\n```bash\nls -l\n```\nDone.\n");
}

#[test]
fn proposal_fence_d5_reply_shape() {
    // D5 reply, trimmed to its first two blocks and closed by the outer fence.
    let reply = include_str!("fixtures/d5_reply_trimmed.txt");
    let reply = format!("{reply}```\n");
    let p = check_file_proposal(&reply).unwrap();
    assert_eq!(p.path, "docs/FAQ.md");
    assert_eq!(p.content.matches("```bash").count(), 2, "{}", p.content);
    assert_eq!(p.content.lines().count(), 14, "{}", p.content);
    assert!(p.content.ends_with("```\n"));
    assert!(p.content.contains("sudo curl"));
    assert!(p.content.lines().count() > 4, "old parser kept 4 lines");
}

#[test]
fn proposal_fence_unbalanced_takes_last_bare_fence() {
    // Inner block opened and never closed: the last bare fence ends the file.
    let p = check_file_proposal("filename: a.md\n```md\ntext\n```bash\nls\nmore\n```\n").unwrap();
    assert_eq!(p.content, "text\n```bash\nls\nmore\n");
    // No bare fence at all: nothing to close, whole body kept (as before).
    let p = check_file_proposal("filename: a.md\n```md\ntext\n```bash\nls\n").unwrap();
    assert_eq!(p.content, "text\n```bash\nls\n");
}

#[test]
fn proposal_fence_four_backtick_outer() {
    let p = check_file_proposal(
        "filename: a.md\n````markdown\nintro\n```\nbare inner\n```\n```bash\nls\n```\n````\n",
    )
    .unwrap();
    assert_eq!(p.content, "intro\n```\nbare inner\n```\n```bash\nls\n```\n");
}

#[test]
fn proposal_fence_plain_replies_unchanged() {
    let p = check_file_proposal("filename: a.txt\nx\ny\n").unwrap();
    assert_eq!(p.content, "x\ny\n");
    let p = check_file_proposal("filename: a.txt\n```\nx\n```\nthanks\n").unwrap();
    assert_eq!(p.content, "x\n");
    let p = check_file_proposal("filename: a.txt\n```text\nx\n").unwrap();
    assert_eq!(p.content, "x\n");
}

// Several code examples in one document and normal prose after the last one:
// every byte is kept, in the fenced reply shape and in the unfenced shape.
#[test]
fn proposal_fence_many_examples_and_trailing_text_keep_every_byte() {
    let doc = "# Guide\n\nStep one:\n```bash\nls -l\n```\nStep two:\n```python\nprint(1)\n```\nStep three:\n```\nplain\n```\nNotes after the last example.\nAnd one more line.\n";
    // Outer fence with a label: inner fences nest, nothing after is lost.
    let p = check_file_proposal(&format!("filename: g.md\n```markdown\n{doc}```\n")).unwrap();
    assert_eq!(p.content, doc);
    // Longer outer fence: inner bare and labelled fences are all content.
    let p = check_file_proposal(&format!("filename: g.md\n````markdown\n{doc}````\n")).unwrap();
    assert_eq!(p.content, doc);
    // No outer fence at all: the document is taken whole.
    let p = check_file_proposal(&format!("filename: g.md\n{doc}")).unwrap();
    assert_eq!(p.content, doc);
}

/// A task that names an existing file is an edit (29 s); a task with no
/// existing target creates a new document (120 s). Omega's wait follows by +1 s.
/// The per-attempt budgets (edit 12 s; document 30 s, capped at half a small
/// custom budget) need short budgets: compose_budget_test.rs, same name.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn kind_selects_budget() {
    use std::sync::atomic::Ordering;
    let _home = home_guard::home_slot();
    let waits: Arc<parking_lot::Mutex<Vec<u32>>> = Arc::default();
    let w = waits.clone();
    let h = Harness::new(Arc::new(move |_, p: &str, _: Duration| {
        w.lock()
            .push(aien_omega_compose::LAST_WAIT_MS.load(Ordering::SeqCst));
        reply(if p.contains("The file notes.md already exists") {
            "filename: notes.md\n# notes\nnew line\n"
        } else {
            "filename: faq.md\nq and a\n"
        })
    }));
    h.write("notes.md", "# notes\n");
    let edit = h.run("add a line to notes.md").unwrap();
    let new = h.run("write a new file faq.md").unwrap();
    assert!(
        edit.proposal.is_some() && new.proposal.is_some(),
        "{edit:?} {new:?}"
    );
    let c = h.calls();
    assert_eq!(c.len(), 2, "{c:?}");
    assert_eq!((c[0].asked, c[1].asked), (Asked::Edit, Asked::Document));
    assert!(c[0].limit <= Duration::from_secs(29), "{:?}", c[0].limit);
    assert!(c[0].limit > Duration::from_secs(28), "{:?}", c[0].limit);
    assert!(c[1].limit <= Duration::from_secs(120), "{:?}", c[1].limit);
    assert!(c[1].limit > Duration::from_secs(119), "{:?}", c[1].limit);
    assert_eq!(*waits.lock(), vec![30_000, 121_000]);
}

/// The two ways a long reply can fail leave different, stable reasons: a cut
/// at the token limit is refused ("reply cut at the token limit ...
/// finish_reason max_tokens"), a slow model times out ("model proposal
/// exceeded N ms"). Neither text contains the other's marker.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn token_cut_and_wall_clock_reasons_are_distinct() {
    let _home = home_guard::home_slot();
    let goal = "write a short note into a.md";
    // token-limit cut
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        Ok(Generation {
            text: "filename: a.md\nline\n".into(),
            tokens: 1024,
            finish_reason: Some("max_tokens".into()),
            ..Default::default()
        })
    }));
    let r = h.run(goal).unwrap();
    assert!(
        r.proposal.is_none() && r.uncommitted_proposal.is_none(),
        "a cut reply is never a proposal: {r:?}"
    );
    let why = r.proposer_error.unwrap();
    assert_eq!(r.proposal_attempts[0].outcome, "refused");
    assert!(
        why.contains("reply cut at the token limit after 1024 tokens (finish_reason max_tokens)"),
        "{why}"
    );
    assert!(!why.contains("exceeded"), "{why}");
    drop(h);
    // wall-clock timeout (the text the daemon proposer emits)
    let h = Harness::new(Arc::new(|_, _: &str, limit: Duration| {
        Err(format!("model proposal exceeded {} ms", limit.as_millis()))
    }));
    let r = h.run(goal).unwrap();
    let why = r.proposer_error.expect("timed out");
    assert_eq!(r.proposal_attempts[0].outcome, "timeout");
    assert!(
        why.starts_with("model proposal exceeded ") && why.ends_with(" ms"),
        "{why}"
    );
    assert!(
        !why.contains("token limit") && !why.contains("max_tokens"),
        "{why}"
    );
}
