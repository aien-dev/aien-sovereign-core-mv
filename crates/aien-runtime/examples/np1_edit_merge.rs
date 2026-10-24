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

//! NEXT-PHASE-1 v8 edit-merge re-derivation (ACCEPTANCE-v8 row <launch>-A).
//! Recomputes the proposal the Skill hands to AEGIS for one parsed edit reply,
//! `edit_proposal(reply, Some((path, prior)))` (compose/bridge.rs), from the reply text
//! recorded in the S3 report and the pre-seed file, without a model or a GPU.
//! Prints one JSON object: the proposal's sha256 (compared with the S4
//! authorization's `proposal_sha256`) and the merged content's sha256
//! (compared with its `content_sha256`), or the refusal.
//!
//! usage: np1_edit_merge <path> <prior_file> <reply_file>
use sha2::Digest;

fn hex(b: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(b))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        eprintln!("usage: np1_edit_merge <path> <prior_file> <reply_file>");
        std::process::exit(2);
    }
    let prior = std::fs::read_to_string(&a[2]).expect("prior file");
    let reply = std::fs::read_to_string(&a[3]).expect("reply file");
    let out = match aien_runtime::compose::rederive::edit_proposal(
        &reply,
        Some((a[1].as_str(), prior.as_str())),
    ) {
        Ok(text) => {
            let content = aien_runtime::check_file_proposal(&text)
                .map(|p| p.content)
                .unwrap_or_default();
            serde_json::json!({
                "ok": true, "path": a[1],
                "prior_sha256": hex(prior.as_bytes()), "reply_sha256": hex(reply.as_bytes()),
                "proposal_sha256": hex(text.as_bytes()), "content_sha256": hex(content.as_bytes()),
                "merged": text != reply,
            })
        }
        Err(e) => serde_json::json!({
            "ok": false, "path": a[1],
            "prior_sha256": hex(prior.as_bytes()), "reply_sha256": hex(reply.as_bytes()),
            "error": e,
        }),
    };
    println!("{out}");
}
