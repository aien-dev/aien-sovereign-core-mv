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

//! Provenance link (arch#162), live: the real `aien daemon` process with
//! SmolLM2-1.7B on the CPU reference backend, and the real `aien compose`
//! CLI. `propose` -> `authorize` -> `execute`, then every record of the
//! commit is read back from the ledger: each names the generation record of
//! the model's proposal, whose `model_sha256` is the sha256 of the weights
//! file the daemon loaded (hashed here, independently, from the file).
//!
//! IGNORED, like recovery_matrix_test: it needs a prebuilt compose-linked CPU
//! binary and the model (minutes of CPU inference), so it cannot run in the
//! default suite. Build once, then run:
//!
//! ```text
//! AIEN_OMEGA_COMPOSE_LIB=<librx_compose.a> cargo build --release -p aien-cli
//! AIEN_BIN=target/release/aien-cli cargo test -p aien-cli --test provenance_link_live_test \
//!   -- --ignored --nocapture
//! ```
//!
//! `AIEN_PROPOSER_MODEL` overrides the model directory
//! (default `$HOME/models/SmolLM2-1.7B-Instruct-31b70e2e869a`). The daemon
//! must report the CPU reference backend or the test stops (no GPU use).
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

fn file_sha256(p: &Path) -> String {
    use std::io::Read;
    let mut f = std::fs::File::open(p).unwrap();
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    aien_omega_compose::hex(&h.finalize())
}

struct Rig {
    bin: PathBuf,
    root: PathBuf,
    _tmp: tempfile::TempDir,
}

impl Rig {
    fn env(&self, c: &mut Command) {
        c.current_dir(&self.root)
            .env("HOME", self.root.join("home"))
            .env("NO_COLOR", "1")
            .env("AIEN_RUNTIME_SOCK", self.root.join("s"))
            .env("AIEN_RUNTIME_STATE_DIR", self.root.join("state"))
            .env("AIEN_COMPOSE_DIR", self.root.join("compose"))
            .env("AIEN_PROVENANCE_DIR", self.root.join("prov"))
            .env("AIEN_REQUIRE_CHECKPOINT", "1");
        for k in [
            "AIEN_OMEGA_DIR",
            "AIEN_CUDAPP_DIR",
            "AIEN_OMEGA_GPU_LIB",
            "AIEN_REQUIRE_BLACKWELL",
            "AIEN_GPU_BACKEND",
            "AIEN_MODEL_DIR",
            "AIEN_ALLEN_SUBJECT",
            "AIEN_ALLEN_ADOPT",
            "AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK",
        ] {
            c.env_remove(k);
        }
    }

    fn cli(&self, args: &[&str]) -> Value {
        let mut c = Command::new(&self.bin);
        c.arg("compose").args(args);
        self.env(&mut c);
        let o = c.output().expect("run aien compose");
        let out = String::from_utf8_lossy(&o.stdout).to_string();
        out.lines()
            .rev()
            .find_map(|l| serde_json::from_str::<Value>(l).ok())
            .unwrap_or_else(|| {
                panic!(
                    "no JSON from {args:?}: {out} {}",
                    String::from_utf8_lossy(&o.stderr)
                )
            })
    }

    /// Text of one ledger record, digest-verified.
    fn record(&self, id: u64) -> Value {
        let v = self.cli(&["recall", "--ids", &id.to_string()]);
        let c = &v["recall"]["cited"][0];
        assert_eq!(c["verified"], true, "record {id}: {v}");
        serde_json::from_str(c["text"].as_str().expect("record text")).unwrap()
    }
}

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "live: needs AIEN_BIN (compose-linked CPU build) and SmolLM2 on disk; minutes of CPU"]
fn a_live_model_commit_names_its_generation_record_and_weights_digest() {
    let bin = PathBuf::from(
        std::env::var("AIEN_BIN").expect("AIEN_BIN: the compose-linked CPU aien-cli"),
    );
    let model_dir = std::env::var("AIEN_PROPOSER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap())
                .join("models/SmolLM2-1.7B-Instruct-31b70e2e869a")
        });
    let weights = model_dir.join("model.safetensors");
    let tmp = tempfile::Builder::new()
        .prefix("pl-")
        .tempdir_in("/tmp")
        .unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    for d in ["ws", "state", "home", "prov"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    // sc#328: the daemon requires the approval desk by default.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &root.join("compose"),
    ))
    .unwrap();
    let rig = Rig {
        bin: bin.clone(),
        root: root.clone(),
        _tmp: tmp,
    };
    let log = root.join("daemon.log");
    let f = std::fs::File::create(&log).unwrap();
    let mut c = Command::new(&bin);
    c.arg("daemon")
        .env("AIEN_MODEL_PATH", &weights)
        .env("AIEN_TOKENIZER_PATH", model_dir.join("tokenizer.json"))
        .env("AIEN_COMPOSE_EDIT_BUDGET_MS", "599000")
        .env("AIEN_COMPOSE_DOC_BUDGET_MS", "599000")
        .stdin(Stdio::null())
        .stdout(f.try_clone().unwrap())
        .stderr(f);
    rig.env(&mut c);
    let mut daemon = Daemon(c.spawn().expect("spawn daemon"));
    let mut up = false;
    for _ in 0..2400 {
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            daemon.0.try_wait().unwrap().is_none(),
            "daemon exited: {}",
            std::fs::read_to_string(&log).unwrap_or_default()
        );
        let t = std::fs::read_to_string(&log).unwrap_or_default();
        if root.join("s").exists() && t.contains("Replay reconcile:") {
            let backend = t.lines().find(|l| l.contains("Backend:")).unwrap_or("");
            assert!(
                backend.contains("CPU-reference"),
                "not the CPU backend: {backend}"
            );
            up = true;
            break;
        }
    }
    assert!(up, "daemon did not come up in 20 min");

    let ws = root.join("ws");
    let report_file = root.join("propose.json");
    let propose = rig.cli(&[
        "propose",
        "--goal",
        "Write garden.md in the workspace, a garden plan.",
        "--workspace",
        ws.to_str().unwrap(),
    ]);
    std::fs::write(&report_file, propose.to_string()).unwrap();
    let report = &propose["report"];
    assert_eq!(
        report["committed"], true,
        "the model's proposal did not commit: {propose}"
    );
    let gen_id = report["generation_record"]
        .as_u64()
        .expect("report names the generation record");
    let commit = report["compose_commit"].as_u64().unwrap();
    let auth = rig.cli(&[
        "authorize",
        "--report",
        report_file.to_str().unwrap(),
        "--workspace",
        ws.to_str().unwrap(),
        "--approver",
        "drake",
        "--desk",
        "1",
    ]);
    assert_eq!(auth["ok"], true, "{auth}");
    let grant = auth["authorization"]["id"].as_u64().unwrap();
    let exec = rig.cli(&[
        "execute",
        "--report",
        report_file.to_str().unwrap(),
        "--workspace",
        ws.to_str().unwrap(),
        "--authorization",
        &grant.to_string(),
    ]);
    assert_eq!(exec["state"], "DONE", "{exec}");
    let effects = rig.cli(&["effects"]);
    let intents = effects["ledger"]["intents"].as_array().expect("intents");
    assert_eq!(intents.len(), 1);
    let intent = intents[0]["intent"].as_u64().unwrap();
    let ack = intents[0]["state_record"].as_u64().unwrap();

    let want_model = file_sha256(&weights);
    let want_tok = file_sha256(&model_dir.join("tokenizer.json"));
    for (name, id) in [
        ("compose_commit", commit),
        ("grant", grant),
        ("intent", intent),
        ("ack", ack),
    ] {
        let r = rig.record(id);
        assert_eq!(r["provenance"]["generation_record"], gen_id, "{name}: {r}");
        assert_eq!(r["provenance"]["allen_agent"], "none", "{name}: {r}");
    }
    let g = rig.record(gen_id);
    assert_eq!(g["generation"], 1);
    assert_eq!(g["origin"], "compose_proposal");
    assert_eq!(
        g["model_sha256"], want_model,
        "the digest of the loaded weights file"
    );
    assert_eq!(g["tokenizer_sha256"], want_tok);
    assert_eq!(g["task"], report["task"]);
    let attempt = report["proposal_attempts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["outcome"] == "parsed")
        .expect("the parsed attempt");
    assert_eq!(g["attempt"], attempt["attempt"]);
    assert_eq!(g["prompt_ids_sha256"], attempt["prompt_ids_sha256"]);
    assert_eq!(g["output_text_sha256"], attempt["text_sha256"]);
    println!(
        "LIVE_PROVENANCE generation_record={gen_id} model_sha256={want_model} commit={commit} grant={grant} intent={intent} ack={ack}"
    );
}
