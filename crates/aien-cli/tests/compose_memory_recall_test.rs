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

//! sc#386: `aien compose remember` and `aien compose recall` against ONE real
//! `aien daemon` process, with and without an engaged ALLEN identity.
//!
//!   remember_refuses_when_allen_not_engaged
//!       no identity: `remember` refuses with the NotEngaged text and stores nothing;
//!       `recall` says `not_engaged` instead of looking like an empty success.
//!   remember_then_recall_includes_item_when_engaged
//!       engaged fixture subject: `remember` puts the item into ALLEN scoped memory;
//!       `recall` returns it, also after a SIGKILL and restart.
//!
//! No model is needed. Host CPU only. Needs a compose-linked `aien-cli`:
//!
//! ```text
//! env -u AIEN_OMEGA_DIR AIEN_OMEGA_COMPOSE_LIB=<librx_compose.a> CARGO_TARGET_DIR=<dir> \
//!   cargo test -p aien-cli --test compose_memory_recall_test -- --ignored --nocapture
//! ```
#[path = "../../aien-allen/tests/support/mod.rs"]
mod support;

use serde_json::{json, Value};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const ITEM: &str = "SC386-memory-item: the topic reported on first is harbour";

struct Out {
    code: i32,
    json: Value,
    raw: String,
}

struct Rig {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    bin: PathBuf,
    daemon: Option<Child>,
    nd: u32,
    subject: Option<PathBuf>,
}

impl Rig {
    fn new() -> Rig {
        let bin = PathBuf::from(
            std::env::var("AIEN_BIN").unwrap_or_else(|_| env!("CARGO_BIN_EXE_aien-cli").into()),
        );
        let tmp = tempfile::Builder::new()
            .prefix("cm-")
            .tempdir_in("/tmp")
            .unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        for d in ["state", "home", "prov"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
            &root.join("compose"),
        ))
        .unwrap();
        Rig {
            _tmp: tmp,
            root,
            bin,
            daemon: None,
            nd: 0,
            subject: None,
        }
    }
    fn sock(&self) -> PathBuf {
        self.root.join("s")
    }
    fn cmd_env(&self, c: &mut Command) {
        c.current_dir(&self.root)
            .env("HOME", self.root.join("home"))
            .env("NO_COLOR", "1")
            .env("AIEN_RUNTIME_SOCK", self.sock())
            .env("AIEN_RUNTIME_STATE_DIR", self.root.join("state"))
            .env("AIEN_COMPOSE_DIR", self.root.join("compose"))
            .env("AIEN_PROVENANCE_DIR", self.root.join("prov"))
            .env("AIEN_REQUIRE_CHECKPOINT", "0");
        for k in [
            "AIEN_OMEGA_DIR",
            "AIEN_OMEGA_GPU_LIB",
            "AIEN_REQUIRE_BLACKWELL",
            "AIEN_GPU_BACKEND",
            "AIEN_MODEL_PATH",
            "AIEN_TOKENIZER_PATH",
            "AIEN_MODEL_DIR",
            "AIEN_ALLEN_SUBJECT",
            "AIEN_ALLEN_ADOPT",
            "AIEN_FAULT_HOLD",
            "AIEN_FAULT_HOLD_FILE",
        ] {
            c.env_remove(k);
        }
        if let Some(s) = &self.subject {
            c.env("AIEN_ALLEN_SUBJECT", s);
        }
    }
    fn start(&mut self, adopt: Option<&str>) -> PathBuf {
        assert!(self.daemon.is_none());
        self.nd += 1;
        let log = self.root.join(format!("daemon-{}.log", self.nd));
        let f = std::fs::File::create(&log).unwrap();
        let _ = std::fs::remove_file(self.sock());
        let mut c = Command::new(&self.bin);
        c.arg("daemon");
        self.cmd_env(&mut c);
        if let Some(a) = adopt {
            c.env("AIEN_ALLEN_ADOPT", a);
        }
        c.process_group(0)
            .stdin(Stdio::null())
            .stdout(f.try_clone().unwrap())
            .stderr(f);
        self.daemon = Some(c.spawn().expect("spawn daemon"));
        for _ in 0..1200 {
            std::thread::sleep(Duration::from_millis(50));
            if let Some(st) = self.daemon.as_mut().unwrap().try_wait().unwrap() {
                self.daemon = None;
                panic!(
                    "daemon exited early ({st:?}): {}",
                    std::fs::read_to_string(&log).unwrap_or_default()
                );
            }
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            if self.sock().exists() && text.contains("Replay reconcile:") {
                assert!(
                    text.lines()
                        .any(|l| l.contains("Backend:") && l.contains("CPU-reference")),
                    "daemon is not the CPU reference build"
                );
                return log;
            }
        }
        self.kill();
        panic!("daemon did not come up in 60 s");
    }
    fn kill(&mut self) {
        if let Some(mut c) = self.daemon.take() {
            let _ = c.kill();
            let st = c.wait().unwrap();
            assert_eq!(st.signal(), Some(9), "daemon end: {st:?}");
        }
        let _ = std::fs::remove_file(self.sock());
    }
    fn cli(&self, args: &[&str]) -> Out {
        let mut c = Command::new(&self.bin);
        c.args(args);
        self.cmd_env(&mut c);
        let o = c.output().expect("run aien");
        let raw = String::from_utf8_lossy(&o.stdout).to_string();
        let json = raw
            .lines()
            .rev()
            .find_map(|l| serde_json::from_str::<Value>(l).ok())
            .unwrap_or_else(|| json!({"unparsed": raw}));
        Out {
            code: o.status.code().unwrap_or(-1),
            json,
            raw,
        }
    }
    /// Count of constraint records the compose home holds.
    fn constraints(&self) -> usize {
        let o = self.cli(&["compose", "recall"]);
        assert_eq!(o.code, 0, "recall: {}", o.raw);
        o.json["constraints"].as_array().map_or(0, Vec::len)
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        if let Some(mut c) = self.daemon.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

#[test]
#[ignore = "needs a compose-linked CPU aien-cli (see the file header)"]
fn remember_refuses_when_allen_not_engaged() {
    let mut r = Rig::new();
    r.start(None);
    let before = r.constraints();
    let o = r.cli(&["compose", "remember", "--text", ITEM]);
    assert_ne!(o.code, 0, "remember must refuse without ALLEN: {}", o.raw);
    assert_eq!(o.json["ok"], false, "{}", o.raw);
    let err = o.json["error"].as_str().unwrap_or("");
    assert!(
        err.contains("ALLEN is not engaged"),
        "refusal must say ALLEN is not engaged: {}",
        o.raw
    );
    assert_eq!(r.constraints(), before, "a refused remember stores nothing");
    let rec = r.cli(&["compose", "recall"]);
    assert_eq!(rec.code, 0, "{}", rec.raw);
    assert_eq!(rec.json["memory"]["state"], "not_engaged", "{}", rec.raw);
}

#[test]
#[ignore = "needs a compose-linked CPU aien-cli (see the file header)"]
fn remember_then_recall_includes_item_when_engaged() {
    let mut r = Rig::new();
    // The home must exist so a subject can bind its first record (as the demo S1).
    r.start(None);
    let lin = r.cli(&["compose", "recall", "--ids", "1"]);
    assert_eq!(lin.code, 0, "{}", lin.raw);
    let lineage = aien_allen::unhex32(lin.json["recall"]["cited"][0]["digest"].as_str().unwrap())
        .expect("record 1 digest");
    r.kill();
    let mut f = support::fx("sc386-memory");
    f.cortex = lineage;
    let subj = support::write_head(
        &r.root.join("subject.bin"),
        support::chain(&f, 2).last().unwrap(),
    );
    let agent = aien_allen::hex(&f.agent);
    r.subject = Some(subj);
    let log = r.start(Some(&agent));
    assert!(std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .any(|l| l.starts_with("ALLEN: engaged agent=")));

    let o = r.cli(&["compose", "remember", "--text", ITEM]);
    assert_eq!(o.code, 0, "remember: {}", o.raw);
    assert_eq!(o.json["ok"], true, "{}", o.raw);

    let texts = |r: &Rig| -> Vec<String> {
        let rec = r.cli(&["compose", "recall"]);
        assert_eq!(rec.code, 0, "{}", rec.raw);
        assert_eq!(rec.json["memory"]["state"], "included", "{}", rec.raw);
        rec.json["memory"]["items"]
            .as_array()
            .expect("memory items")
            .iter()
            .filter_map(|i| i["text"].as_str().map(str::to_string))
            .collect()
    };
    assert!(texts(&r).iter().any(|t| t == ITEM), "item recalled");
    // The same item is what a proposal in the `work` context is given.
    let direct = r.cli(&["allen", "memory", "recall", "--context", "work"]);
    assert!(
        direct.raw.contains(ITEM),
        "ALLEN memory holds it: {}",
        direct.raw
    );

    // SIGKILL and restart: the item is still there.
    r.kill();
    r.start(None);
    assert!(
        texts(&r).iter().any(|t| t == ITEM),
        "item recalled after restart"
    );
}
