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

//! sc#429: the model-visible prompt names the workspace by a constant logical
//! label, never by its machine-specific absolute path. Two workspaces with the
//! same logical inputs (goal, files, contents) at different absolute paths
//! (different parent folders, depths and names, as a different host, user or
//! job folder would give) must produce byte-identical prompts and, with a
//! tokenizer, identical prompt token ids. The real path stays in the
//! authorization and evidence layers (the canonical workspace is still
//! resolved, and the task id still hashes it).
//!
//! The prompt is the one `ComposeBridge::run_task` hands the model
//! (tests/support/compose_harness.rs), one harness running both workspaces.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use compose_harness::{silent, Harness};
use std::path::Path;

/// The logical name the prompt gives the workspace.
const WORKSPACE_LABEL: &str = "workspace";

/// The first prompt the bridge hands the model for `goal` in `ws`.
fn prompt(h: &Harness, goal: &str, ws: &Path) -> String {
    h.clear();
    let r = h
        .run_in(goal, ws)
        .unwrap_or_else(|e| panic!("{goal}: refused: {e}"));
    h.calls()
        .first()
        .unwrap_or_else(|| panic!("{goal}: no model call: {r:?}"))
        .prompt
        .clone()
}

fn fill(ws: &Path) {
    std::fs::create_dir_all(ws.join("inbox")).unwrap();
    std::fs::write(ws.join("inbox/tide.txt"), "High tide at 06:40.\n").unwrap();
    std::fs::write(ws.join("inbox/berth.txt"), "Berth 4 is free.\n").unwrap();
    std::fs::write(ws.join("inbox/crane.txt"), "Crane service at 09:00.\n").unwrap();
    std::fs::write(ws.join("report.md"), "# Harbour report\n\n- morning\n").unwrap();
}

/// Two equal workspaces at deliberately different absolute paths.
fn pair() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let a_root = tempfile::tempdir().unwrap();
    let b_root = tempfile::tempdir().unwrap();
    let a = a_root
        .path()
        .join("jobs")
        .join("host-alpha")
        .join("ws-contract");
    let b = b_root
        .path()
        .join("home")
        .join("other-user")
        .join("deeper")
        .join("run")
        .join("ws-contract-2");
    fill(&a);
    fill(&b);
    (a_root, b_root, a, b)
}

const GOALS: &[&str] = &[
    "Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.",
    "Write the file summary.md from the files tide.txt, berth.txt and crane.txt in the inbox folder.",
    "Add the line \"- evening\" under the \"# Harbour report\" heading in report.md.",
];

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn prompt_carries_the_label_and_never_the_path() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    let (_ka, _kb, a, b) = pair();
    let a = std::fs::canonicalize(&a).unwrap();
    let b = std::fs::canonicalize(&b).unwrap();
    assert_ne!(a, b);
    for goal in GOALS {
        let pa = prompt(&h, goal, &a);
        let pb = prompt(&h, goal, &b);
        assert_eq!(pa, pb, "prompt differs between workspaces for goal {goal}");
        assert!(
            pa.starts_with(&format!(
                "Goal: {goal}\nAuthorized workspace: {WORKSPACE_LABEL}\n"
            )),
            "label missing in {pa}"
        );
        for p in [&a, &b] {
            let s = p.display().to_string();
            assert!(!pa.contains(&s), "absolute path {s} leaked into the prompt");
            // not even the leaf folder name
            let leaf = p.file_name().unwrap().to_str().unwrap();
            assert!(
                !pa.contains(leaf),
                "folder name {leaf} leaked into the prompt"
            );
        }
    }
    // The whole prompt of a plain new-document goal (the base template plus
    // its destination line) carries no path separator at all.
    let base = prompt(&h, GOALS[0], &a);
    assert!(
        !base.contains('/'),
        "no path separator in the base prompt: {base}"
    );
}

/// With a tokenizer (`AIEN_TOKENIZER_PATH`), the token ids of the rendered
/// prompt are equal across the two paths, for every chat template the product
/// serves.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn prompt_token_ids_equal_across_paths() {
    let _home = home_guard::home_slot();
    let Ok(path) = std::env::var("AIEN_TOKENIZER_PATH") else {
        eprintln!("AIEN_TOKENIZER_PATH not set: token-id check skipped");
        return;
    };
    use aien_inference_abi::TinyLlamaTokenizer;
    use aien_runtime::control::{format_chat, ChatTurn};
    let tok = TinyLlamaTokenizer::from_file(path).expect("tokenizer");
    let h = Harness::new(silent());
    let (_ka, _kb, a, b) = pair();
    let a = std::fs::canonicalize(&a).unwrap();
    let b = std::fs::canonicalize(&b).unwrap();
    for goal in GOALS {
        let pa = prompt(&h, goal, &a);
        let pb = prompt(&h, goal, &b);
        let ids = |p: &str| {
            let turn = ChatTurn {
                role: "user".into(),
                content: p.into(),
            };
            let text = format_chat(tok.template(), std::slice::from_ref(&turn)).expect("chat");
            tok.encode(&text).expect("encode")
        };
        let ia = ids(&pa);
        let ib = ids(&pb);
        assert_eq!(ia, ib, "token ids differ between paths for goal {goal}");
        eprintln!(
            "goal {:.40}...: {} prompt tokens, equal on both paths",
            goal,
            ia.len()
        );
    }
}
