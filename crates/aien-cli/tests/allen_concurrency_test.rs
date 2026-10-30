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

//! ALLEN under concurrent clients (arch#162): ONE real `aien daemon` process,
//! many real `aien allen ...` CLI processes at the same time over the real
//! socket. No in-process fakes: every call is an OS process, the crash is a
//! real SIGKILL of the daemon in the middle of a write burst.
//!
//! Points proved (ids are used in the assertion messages):
//!   C1  identity: ALLEN is engaged once at daemon boot (not by clients), so this proves
//!       one identity served consistently to concurrent clients: one agent, one pin,
//!       one "ALLEN: engaged" line, unchanged after kill -9 + restart. It does NOT
//!       claim concurrent creation of the identity.
//!   C2  scoped memory: concurrent writers into work and personal; every item
//!       present exactly once in its own scope; recall never crosses scopes.
//!   C3  correction: concurrent corrections of one item end in one coherent
//!       final value and version == 1 + number of acknowledged corrections.
//!   C4  forgetting racing reads: no read that starts after forget returned
//!       shows the canary; the item's key file exists before forget and is gone after
//!       (the ciphertext stays in the immutable log; plaintext is never on disk, so a
//!       plaintext file scan would prove nothing and is not used).
//!   C5  recovery: kill -9 mid write burst, restart: identity unchanged, every
//!       acknowledged write present, nothing half-present, forgotten stays so.
//!   C6  persona: concurrent profile changes with the same expected revision
//!       produce exactly one winner; a permission-looking key is refused.
//!       LIMIT: that persona text cannot widen what compose may do is covered
//!       by the profile crate's own tests and the approval desk, not here.
//!
//! No model is needed: memory commands never call one, the daemon runs with no
//! model path. Host CPU only. Needs a compose-linked `aien-cli` (the same
//! build the recovery matrix uses); run:
//!
//! ```text
//! env -u AIEN_CUDAPP_DIR -u AIEN_OMEGA_DIR AIEN_OMEGA_COMPOSE_LIB=<librx_compose.a> CARGO_TARGET_DIR=<dir> \
//!   cargo test -p aien-cli --test allen_concurrency_test -- --ignored --nocapture
//! ```
#[path = "../../aien-allen/tests/support/mod.rs"]
mod support;

use aien_allen::binding::pin_path;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

const CANARY: &str = "CONC-CANARY-5519-nobody-may-keep-this";

struct Out {
    code: i32,
    json: Value,
    raw: String,
}

impl Out {
    fn ok(&self) -> bool {
        self.code == 0 && self.json["ok"] == true
    }
    fn refusal(&self) -> Option<String> {
        self.json["refused"]["code"].as_str().map(str::to_string)
    }
    fn result(&self) -> &Value {
        &self.json["result"]["result"]
    }
}

struct Rig {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    bin: PathBuf,
    daemon: Option<Child>,
    nd: u32,
    env: Vec<(String, String)>,
}

impl Rig {
    fn new() -> Rig {
        let bin = PathBuf::from(
            std::env::var("AIEN_BIN").unwrap_or_else(|_| env!("CARGO_BIN_EXE_aien-cli").into()),
        );
        // Short path: unix socket paths are limited to 108 bytes.
        let tmp = tempfile::Builder::new()
            .prefix("ac-")
            .tempdir_in("/tmp")
            .unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        for d in ["ws", "state", "home", "prov"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        // sc#328: the daemon refuses to start without the (default) desk key.
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
            env: Vec::new(),
        }
    }
    fn sock(&self) -> PathBuf {
        self.root.join("s")
    }
    fn compose(&self) -> PathBuf {
        self.root.join("compose")
    }
    fn cmd_env(&self, c: &mut Command) {
        c.current_dir(&self.root)
            .env("HOME", self.root.join("home"))
            .env("NO_COLOR", "1")
            .env("AIEN_RUNTIME_SOCK", self.sock())
            .env("AIEN_RUNTIME_STATE_DIR", self.root.join("state"))
            .env("AIEN_COMPOSE_DIR", self.compose())
            .env("AIEN_PROVENANCE_DIR", self.root.join("prov"))
            .env("AIEN_REQUIRE_CHECKPOINT", "0");
        for k in [
            "AIEN_OMEGA_DIR",
            "AIEN_CUDAPP_DIR",
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
        for (k, v) in &self.env {
            c.env(k, v);
        }
    }

    /// Start the daemon; returns its log path. Panics when it does not serve.
    fn start_with(&mut self, once: &[(&str, &str)]) -> PathBuf {
        assert!(self.daemon.is_none());
        self.nd += 1;
        let log = self.root.join(format!("daemon-{}.log", self.nd));
        let f = std::fs::File::create(&log).unwrap();
        let _ = std::fs::remove_file(self.sock());
        let mut c = Command::new(&self.bin);
        c.arg("daemon");
        self.cmd_env(&mut c);
        for (k, v) in once {
            c.env(k, v);
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
                let line = text.lines().find(|l| l.contains("Backend:")).unwrap_or("");
                assert!(
                    line.contains("CPU-reference"),
                    "daemon is not the CPU reference build ({line}); no GPU use here"
                );
                return log;
            }
        }
        self.kill();
        panic!("daemon did not come up in 60 s");
    }

    /// SIGKILL (kill -9) and reap; asserts it really died by signal 9.
    fn kill(&mut self) {
        if let Some(mut c) = self.daemon.take() {
            let _ = c.kill();
            let st = c.wait().unwrap();
            assert_eq!(st.signal(), Some(9), "daemon end: {st:?}");
        }
        let _ = std::fs::remove_file(self.sock());
    }

    fn run(&self, group: &str, args: &[&str]) -> Out {
        let mut c = Command::new(&self.bin);
        c.arg(group).args(args);
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
    fn allen(&self, args: &[&str]) -> Out {
        self.run("allen", args)
    }

    fn rows(&self) -> Vec<Value> {
        let o = self.allen(&["memory", "inspect", "--owner", "1"]);
        assert!(o.ok(), "inspect: {}", o.raw);
        o.result().as_array().expect("inspect rows").clone()
    }
    fn rows_of(&self, ctx: &str) -> Vec<Value> {
        let o = self.allen(&["memory", "inspect", "--context", ctx]);
        assert!(o.ok(), "inspect {ctx}: {}", o.raw);
        o.result().as_array().expect("inspect rows").clone()
    }
    fn recall_texts(&self, ctx: &str) -> (Out, Vec<String>) {
        self.recall_query(ctx, None)
    }
    /// recall is capped (16 items), so racing reads ask for the canary by substring.
    fn recall_query(&self, ctx: &str, q: Option<&str>) -> (Out, Vec<String>) {
        let mut a = vec!["memory", "recall", "--context", ctx];
        if let Some(q) = q {
            a.extend(["--query", q]);
        }
        let o = self.allen(&a);
        let t = o.result()["items"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|r| r["text"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        (o, t)
    }

    /// The store directory `<compose>.allen-memory` (read-only use in this test).
    fn mem_dir(&self) -> PathBuf {
        PathBuf::from(format!("{}.allen-memory", self.compose().display()))
    }
    /// Key files of one item (`keys/<item>-v<n>.key`). The store keeps ciphertext in
    /// its immutable log and destroys KEYS to forget or supersede (store.rs header).
    fn key_files(&self, item: &str) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.mem_dir().join("keys"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with(&format!("{item}-v")) && n.ends_with(".key"))
            .collect();
        v.sort();
        v
    }
    /// Log record files whose bytes name the item (its ciphertext records).
    fn log_records_of(&self, item: &str) -> usize {
        std::fs::read_dir(self.mem_dir().join("log"))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                std::fs::read(e.path())
                    .map(|b| b.windows(item.len()).any(|w| w == item.as_bytes()))
                    .unwrap_or(false)
            })
            .count()
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

fn s(p: &Path) -> String {
    p.to_string_lossy().to_string()
}

fn engaged_lines(log: &Path, agent: &str) -> usize {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("ALLEN: engaged agent="))
        .filter(|l| l.contains(agent))
        .count()
}

fn all_engaged_lines(log: &Path) -> usize {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("ALLEN: engaged agent="))
        .count()
}

/// The set of entries directly under the compose directory (and siblings named
/// compose.*): a second identity or memory system would show up here.
fn compose_entries(r: &Rig) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for d in [r.root.clone(), r.compose()] {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.contains("allen") || n.contains("pin") {
                    set.insert(format!("{}/{n}", d.file_name().unwrap().to_string_lossy()));
                }
            }
        }
    }
    set
}

#[test]
#[ignore = "needs a compose-linked CPU aien-cli (see the file header); host CPU, not GB10"]
fn allen_concurrent_clients_one_daemon() {
    let mut r = Rig::new();

    // ---- setup: a home exists, then ALLEN is adopted once (as the recovery matrix R8/R9).
    r.start_with(&[]);
    let lin = r.run("compose", &["recall", "--ids", "1"]);
    assert_eq!(lin.code, 0, "{}", lin.raw);
    let lineage = aien_allen::unhex32(lin.json["recall"]["cited"][0]["digest"].as_str().unwrap())
        .expect("record 1 digest");
    r.kill();
    let mut f = support::fx("allen-concurrency");
    f.cortex = lineage;
    let subj = support::write_head(
        &r.root.join("subject.bin"),
        support::chain(&f, 2).last().unwrap(),
    );
    let agent = aien_allen::hex(&f.agent);
    r.env.push(("AIEN_ALLEN_SUBJECT".into(), s(&subj)));
    let log1 = r.start_with(&[("AIEN_ALLEN_ADOPT", &agent)]);

    // ---- C1: N concurrent calls (status, put, inspect) all served by the one boot-time identity.
    let mut c1_puts = BTreeMap::new();
    {
        let results = Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for i in 0..9 {
                let (r, results) = (&r, &results);
                sc.spawn(move || {
                    let out = match i % 3 {
                        0 => r.allen(&["status"]),
                        1 => r.allen(&[
                            "memory",
                            "put",
                            "--context",
                            "work",
                            "--text",
                            &format!("first-contact-{i}"),
                        ]),
                        _ => r.allen(&["memory", "inspect", "--owner", "1"]),
                    };
                    results.lock().unwrap().push((i, out));
                });
            }
        });
        for (i, o) in results.into_inner().unwrap() {
            assert!(o.ok(), "C1 concurrent call {i} failed: {}", o.raw);
            if i % 3 == 0 {
                assert_eq!(o.json["result"]["identity"], "engaged", "C1 {}", o.raw);
                assert_eq!(
                    o.json["result"]["fingerprint"].as_str().unwrap(),
                    &agent[..8],
                    "C1 two different identities answered"
                );
            }
            if i % 3 == 1 {
                c1_puts.insert(
                    o.result()["item"].as_str().unwrap().to_string(),
                    format!("first-contact-{i}"),
                );
            }
        }
    }
    assert_eq!(
        all_engaged_lines(&log1),
        1,
        "C1 exactly one engage line in the daemon log"
    );
    assert_eq!(engaged_lines(&log1, &agent), 1, "C1 engaged agent is ours");
    let pin0 = std::fs::read(pin_path(&r.compose())).expect("pin file");
    let entries0 = compose_entries(&r);
    assert_eq!(c1_puts.len(), 3);

    // ---- C2: concurrent writers into work and personal.
    let mut expected: BTreeMap<String, (String, String)> = BTreeMap::new(); // item -> (scope, text)
    for (item, text) in &c1_puts {
        expected.insert(item.clone(), ("work".into(), text.clone()));
    }
    {
        let acks = Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for w in 0..6 {
                let (r, acks) = (&r, &acks);
                sc.spawn(move || {
                    let ctx = if w % 2 == 0 { "work" } else { "personal" };
                    for i in 0..6 {
                        let text = format!("c2-{ctx}-w{w}-n{i}");
                        let o = r.allen(&["memory", "put", "--context", ctx, "--text", &text]);
                        assert!(o.ok(), "C2 put refused: {}", o.raw);
                        acks.lock().unwrap().push((
                            o.result()["item"].as_str().unwrap().to_string(),
                            ctx.to_string(),
                            text,
                        ));
                    }
                });
            }
        });
        for (item, ctx, text) in acks.into_inner().unwrap() {
            assert!(
                expected.insert(item, (ctx, text)).is_none(),
                "C2 duplicate item id"
            );
        }
    }
    for ctx in ["work", "personal"] {
        let rows = r.rows_of(ctx);
        let want: BTreeSet<&String> = expected
            .values()
            .filter(|(c, _)| c == ctx)
            .map(|(_, t)| t)
            .collect();
        let got: Vec<String> = rows
            .iter()
            .filter(|x| x["kind"] != "goal")
            .map(|x| x["text"].as_str().unwrap_or("").to_string())
            .collect();
        assert_eq!(
            got.len(),
            want.len(),
            "C2 {ctx}: item count (exactly once each): {got:?}"
        );
        for t in &want {
            assert_eq!(
                got.iter().filter(|g| g == t).count(),
                1,
                "C2 {ctx}: {t} present exactly once"
            );
        }
        for x in &rows {
            assert_eq!(
                x["scope"], ctx,
                "C2 {ctx} inspect leaked another scope: {x}"
            );
        }
        let (o, texts) = r.recall_texts(ctx);
        assert!(o.ok() && !texts.is_empty(), "C2 recall {ctx}: {}", o.raw);
        for t in &texts {
            assert!(
                want.contains(t),
                "C2 recall in {ctx} returned another scope's text: {t}"
            );
        }
    }

    // ---- C3: concurrent corrections of one item.
    let c3 = r.allen(&[
        "memory",
        "put",
        "--context",
        "work",
        "--text",
        "c3-original",
    ]);
    assert!(c3.ok(), "{}", c3.raw);
    let c3_item = c3.result()["item"].as_str().unwrap().to_string();
    let corr_ok = Mutex::new(Vec::new());
    let corr_refused = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for k in 0..6 {
            let (r, corr_ok, corr_refused, c3_item) = (&r, &corr_ok, &corr_refused, &c3_item);
            sc.spawn(move || {
                let text = format!("c3-corrected-by-{k}");
                let o = r.allen(&[
                    "memory",
                    "correct",
                    "--context",
                    "work",
                    "--item",
                    c3_item,
                    "--text",
                    &text,
                ]);
                if o.ok() {
                    corr_ok
                        .lock()
                        .unwrap()
                        .push((o.result()["version"].as_u64().unwrap(), text));
                } else {
                    // A refusal must be a visible, coded one (conflict), never silence.
                    let code = o
                        .refusal()
                        .unwrap_or_else(|| panic!("C3 unexplained: {}", o.raw));
                    assert!(
                        code == "conflict" || code == "io",
                        "C3 unexpected refusal {code}: {}",
                        o.raw
                    );
                    corr_refused.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    });
    let corr_ok = corr_ok.into_inner().unwrap();
    let versions: BTreeSet<u64> = corr_ok.iter().map(|(v, _)| *v).collect();
    assert_eq!(
        versions.len(),
        corr_ok.len(),
        "C3 two corrections got the same version"
    );
    let hist: Vec<Value> = r
        .rows_of("work")
        .into_iter()
        .filter(|x| x["item"] == c3_item.as_str())
        .collect();
    // One row per version: the superseded ones are "corrected" and show no text.
    assert_eq!(
        hist.len() as u64,
        1 + corr_ok.len() as u64,
        "C3 history length (no lost update)"
    );
    let vs: Vec<u64> = hist
        .iter()
        .map(|x| x["version"].as_u64().unwrap())
        .collect();
    assert_eq!(
        vs,
        (1..=hist.len() as u64).collect::<Vec<_>>(),
        "C3 history gap-free, in order"
    );
    for x in &hist[..hist.len() - 1] {
        assert_eq!(x["state"], "corrected", "C3 {x}");
        assert!(
            x["text"].is_null(),
            "C3 superseded text still readable: {x}"
        );
    }
    let row = hist.last().unwrap().clone();
    assert_eq!(row["state"], "live", "C3 final state");
    assert_eq!(
        row["version"].as_u64().unwrap(),
        1 + corr_ok.len() as u64,
        "C3 final version = 1 + acknowledged corrections (no lost update)"
    );
    assert_eq!(
        versions.iter().copied().collect::<Vec<_>>(),
        (2..2 + corr_ok.len() as u64).collect::<Vec<_>>(),
        "C3 revision history is gap-free"
    );
    let top = corr_ok.iter().max_by_key(|(v, _)| *v).unwrap();
    assert_eq!(
        row["text"].as_str().unwrap(),
        top.1,
        "C3 final text is the highest acknowledged version's text"
    );
    assert_eq!(
        corr_ok.len() + corr_refused.load(Ordering::SeqCst),
        6,
        "C3 every correction was acknowledged or visibly refused"
    );
    assert_eq!(
        r.key_files(&c3_item),
        vec![format!("{c3_item}-v{}.key", 1 + corr_ok.len())],
        "C3 only the final version's key remains (superseded keys destroyed)"
    );
    expected.insert(c3_item, ("work".into(), top.1.clone()));

    // ---- C6: persona. Same expected revision from six clients: exactly one wins.
    {
        let outs = Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for k in 0..6 {
                let (r, outs) = (&r, &outs);
                sc.spawn(move || {
                    let o = r.allen(&["set", "--expect", "0", "--name", &format!("Nova{k}")]);
                    outs.lock().unwrap().push(o);
                });
            }
        });
        let outs = outs.into_inner().unwrap();
        assert_eq!(
            outs.iter().filter(|o| o.ok()).count(),
            1,
            "C6 exactly one winner"
        );
        for o in outs.iter().filter(|o| !o.ok()) {
            assert_eq!(o.refusal().as_deref(), Some("stale_update"), "C6 {}", o.raw);
        }
        let show = r.allen(&["show"]);
        assert_eq!(show.json["result"]["revision"], 1, "C6 {}", show.raw);
        let perm = r.allen(&["set", "--expect", "1", "--pref", "grant_admin=yes"]);
        assert!(
            !perm.ok(),
            "C6 a permission-looking key was accepted: {}",
            perm.raw
        );
        assert_eq!(
            perm.refusal().as_deref(),
            Some("permission_key"),
            "C6 {}",
            perm.raw
        );
        let show2 = r.allen(&["show"]);
        assert_eq!(
            show2.json["result"]["revision"], 1,
            "C6 refused change moved the profile"
        );
    }

    // ---- C4: forgetting racing reads.
    let put = r.allen(&["memory", "put", "--context", "personal", "--text", CANARY]);
    assert!(put.ok(), "{}", put.raw);
    let canary_item = put.result()["item"].as_str().unwrap().to_string();
    // Evidence on the real item, before forget: its key file and its ciphertext record exist.
    let keys_before = r.key_files(&canary_item);
    let recs_before = r.log_records_of(&canary_item);
    assert_eq!(
        keys_before.len(),
        1,
        "C4 before forget the item has its key: {keys_before:?}"
    );
    assert!(
        recs_before >= 1,
        "C4 before forget the item has a log record"
    );
    {
        let forgot = AtomicBool::new(false);
        let stop = AtomicBool::new(false);
        let saw_before = AtomicUsize::new(0);
        let reads_after = AtomicUsize::new(0);
        let violations = Mutex::new(Vec::new());
        std::thread::scope(|sc| {
            for _ in 0..3 {
                sc.spawn(|| {
                    while !stop.load(Ordering::SeqCst) {
                        let started_after = forgot.load(Ordering::SeqCst);
                        let (o, texts) = r.recall_query("personal", Some("CONC-CANARY"));
                        assert!(o.ok(), "C4 read failed: {}", o.raw);
                        let has = o.raw.contains(CANARY) || texts.iter().any(|t| t == CANARY);
                        if started_after {
                            reads_after.fetch_add(1, Ordering::SeqCst);
                            if has {
                                violations.lock().unwrap().push(o.raw.clone());
                            }
                        } else if has {
                            saw_before.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
            }
            let _guard = StopOnDrop(&stop);
            // Wait until readers have really seen the text, so the race is not vacuous.
            for _ in 0..600 {
                if saw_before.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            assert!(
                saw_before.load(Ordering::SeqCst) >= 2,
                "C4 readers never saw the canary"
            );
            let fg = r.allen(&[
                "memory",
                "forget",
                "--context",
                "personal",
                "--item",
                &canary_item,
            ]);
            assert!(fg.ok(), "C4 forget: {}", fg.raw);
            forgot.store(true, Ordering::SeqCst);
            for _ in 0..600 {
                if reads_after.load(Ordering::SeqCst) >= 6 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            stop.store(true, Ordering::SeqCst);
        });
        assert!(
            reads_after.load(Ordering::SeqCst) >= 6,
            "C4 too few reads after forget"
        );
        assert!(
            violations.lock().unwrap().is_empty(),
            "C4 {} read(s) that started after forget returned still showed the canary; first: {}",
            violations.lock().unwrap().len(),
            violations
                .lock()
                .unwrap()
                .first()
                .map(|v| v.chars().take(300).collect::<String>())
                .unwrap_or_default()
        );
    }
    let row = r
        .rows()
        .into_iter()
        .find(|x| x["item"] == canary_item.as_str())
        .expect("forgotten row remains");
    assert!(row["text"].is_null(), "C4 forgotten row shows text: {row}");
    assert!(
        row["state"].as_str().unwrap().starts_with("forgotten"),
        "{row}"
    );
    let exp = r.allen(&["memory", "export", "--owner", "1"]);
    assert!(!exp.raw.contains(CANARY), "C4 export shows the canary");
    // Ground truth (store.rs): forget destroys the key; the ciphertext stays in the immutable log.
    // So the observable change is: key file present before, absent after; record still there.
    assert!(
        r.key_files(&canary_item).is_empty(),
        "C4 key file of the forgotten item still on disk (before: {keys_before:?})"
    );
    assert!(
        r.log_records_of(&canary_item) >= recs_before,
        "C4 the ciphertext record vanished (expected: kept, unreadable without the key)"
    );

    // ---- C5: kill -9 in the middle of a concurrent write burst, then restart.
    let pin_before_kill = std::fs::read(pin_path(&r.compose())).unwrap();
    assert_eq!(pin0, pin_before_kill, "C1 pin changed during the run");
    // The only new directory since C1 is the profile store, created by the C6 `set`.
    let entries1 = compose_entries(&r);
    let added: Vec<&String> = entries1.difference(&entries0).collect();
    assert!(
        entries0.is_subset(&entries1)
            && added.len() == 1
            && added[0].ends_with("compose.allen-profile"),
        "C1 a second identity/memory dir appeared: {added:?}"
    );
    let attempts: Mutex<Vec<(String, String, Option<String>)>> = Mutex::new(Vec::new()); // ctx, text, acked item
    let killing = AtomicBool::new(false);
    let in_flight_unacked = AtomicUsize::new(0);
    let acked = AtomicUsize::new(0);
    let dead = AtomicBool::new(false);
    std::thread::scope(|sc| {
        for w in 0..6 {
            let (r, attempts, acked, dead, killing, in_flight_unacked) =
                (&r, &attempts, &acked, &dead, &killing, &in_flight_unacked);
            sc.spawn(move || {
                let ctx = if w % 2 == 0 { "work" } else { "personal" };
                for i in 0..40 {
                    if dead.load(Ordering::SeqCst) {
                        break;
                    }
                    let text = format!("c5-{ctx}-w{w}-n{i}");
                    let issued_before_kill = !killing.load(Ordering::SeqCst);
                    let o = r.allen(&["memory", "put", "--context", ctx, "--text", &text]);
                    let item = if o.ok() {
                        acked.fetch_add(1, Ordering::SeqCst);
                        Some(o.result()["item"].as_str().unwrap().to_string())
                    } else {
                        dead.store(true, Ordering::SeqCst);
                        if issued_before_kill {
                            in_flight_unacked.fetch_add(1, Ordering::SeqCst);
                        }
                        None
                    };
                    attempts.lock().unwrap().push((ctx.into(), text, item));
                }
            });
        }
        let _guard = StopOnDrop(&dead);
        // Kill once the burst is clearly in flight.
        for _ in 0..1200 {
            if acked.load(Ordering::SeqCst) >= 12 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            acked.load(Ordering::SeqCst) >= 12,
            "C5 burst never got going"
        );
        // kill -9 from the test thread while the writers keep going.
        killing.store(true, Ordering::SeqCst);
        let c = r.daemon.as_ref().expect("daemon").id();
        let st = Command::new("kill")
            .args(["-9", &c.to_string()])
            .status()
            .unwrap();
        assert!(st.success());
    });
    // Reap the killed daemon through the rig (asserts signal 9).
    {
        let mut d = r.daemon.take().unwrap();
        let st = d.wait().unwrap();
        assert_eq!(st.signal(), Some(9), "C5 daemon end: {st:?}");
        let _ = std::fs::remove_file(r.sock());
    }
    let attempts = attempts.into_inner().unwrap();
    let n_acked = attempts.iter().filter(|a| a.2.is_some()).count();
    assert!(n_acked >= 12, "C5 acked {n_acked}");
    let in_flight = in_flight_unacked.load(Ordering::SeqCst);
    assert!(
        in_flight >= 1,
        "C5 no write was in flight (issued before the kill, never acknowledged) when the kill landed"
    );
    let log2 = r.start_with(&[]); // restart over the same state, no ADOPT
    assert_eq!(
        all_engaged_lines(&log2),
        1,
        "C5 one engage line after restart"
    );
    assert_eq!(
        engaged_lines(&log2, &agent),
        1,
        "C5 identity unchanged after restart"
    );
    let st = r.allen(&["status"]);
    assert_eq!(
        st.json["result"]["fingerprint"].as_str().unwrap(),
        &agent[..8],
        "C5"
    );
    assert_eq!(
        pin0,
        std::fs::read(pin_path(&r.compose())).unwrap(),
        "C5 pin changed"
    );
    assert_eq!(
        compose_entries(&r),
        entries1,
        "C5 a second identity appeared"
    );

    let rows = r.rows();
    let ids: Vec<(String, u64)> = rows
        .iter()
        .map(|x| {
            (
                x["item"].as_str().unwrap().to_string(),
                x["version"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        ids.len(),
        ids.iter().collect::<BTreeSet<_>>().len(),
        "C5 duplicate item rows"
    );
    // Every earlier (pre-burst) acknowledged item is still exactly as it was.
    for (item, (ctx, text)) in &expected {
        let x = rows
            .iter()
            .rev()
            .find(|x| x["item"] == item.as_str())
            .unwrap_or_else(|| panic!("C5 pre-burst item {item} lost"));
        assert_eq!(x["scope"], ctx.as_str());
        assert_eq!(
            x["text"].as_str(),
            Some(text.as_str()),
            "C5 pre-burst item changed"
        );
    }
    // Acknowledged burst writes: present, live, right scope, right text.
    for (ctx, text, item) in &attempts {
        let hit: Vec<&Value> = rows.iter().filter(|x| x["text"] == text.as_str()).collect();
        match item {
            Some(id) => {
                assert_eq!(
                    hit.len(),
                    1,
                    "C5 acknowledged write {text} present exactly once"
                );
                assert_eq!(hit[0]["item"], id.as_str());
                assert_eq!(hit[0]["scope"], ctx.as_str());
                assert_eq!(hit[0]["state"], "live", "C5 {text}");
            }
            // Unacknowledged: either fully there or not at all, never twice or damaged.
            None => {
                assert!(
                    hit.len() <= 1,
                    "C5 unacknowledged write {text} appears {} times",
                    hit.len()
                );
                if let Some(h) = hit.first() {
                    assert_eq!(
                        h["state"], "live",
                        "C5 unacknowledged write half-present: {h}"
                    );
                    assert_eq!(h["scope"], ctx.as_str());
                }
            }
        }
    }
    // Nothing half-present anywhere: every row is live/corrected or cleanly forgotten, and
    // every live text is one we wrote.
    let known: BTreeSet<String> = expected
        .values()
        .map(|(_, t)| t.clone())
        .chain(attempts.iter().map(|a| a.1.clone()))
        .collect();
    for x in &rows {
        let state = x["state"].as_str().unwrap();
        if state.starts_with("forgotten") {
            assert_eq!(
                state, "forgotten",
                "C5 forget left pending after restart: {x}"
            );
            assert!(x["text"].is_null(), "C5 forgotten row has text: {x}");
        } else {
            assert!(
                state == "live" || state == "corrected",
                "C5 row in odd state (half-present?): {x}"
            );
            if state == "corrected" {
                assert!(x["text"].is_null(), "C5 superseded text readable: {x}");
                continue;
            }
            let t = x["text"]
                .as_str()
                .unwrap_or_else(|| panic!("C5 live row without text: {x}"));
            assert!(known.contains(t), "C5 text nobody wrote: {t}");
        }
    }
    // Forgotten stays forgotten.
    let cr = rows
        .iter()
        .rev()
        .find(|x| x["item"] == canary_item.as_str())
        .unwrap();
    assert_eq!(cr["state"], "forgotten", "C5 canary row");
    let (o, texts) = r.recall_query("personal", Some("CONC-CANARY"));
    assert!(
        o.ok() && !o.raw.contains(CANARY) && !texts.iter().any(|t| t == CANARY),
        "C5 canary back"
    );
    assert!(
        r.key_files(&canary_item).is_empty(),
        "C5 forgotten item has a key again after restart"
    );
    // Scope isolation still holds after recovery.
    for ctx in ["work", "personal"] {
        for x in r.rows_of(ctx) {
            assert_eq!(x["scope"], ctx, "C5 scope leak after restart");
        }
    }
    println!(
        "CONC_RECEIPT {}",
        json!({"acked_before_kill": n_acked, "in_flight_unacked_at_kill": in_flight, "attempts": attempts.len(),
               "corrections_ok": corr_ok.len(), "agent": &agent[..8]})
    );
}

/// Sets the flag when dropped, so worker threads stop even if the test panics first.
struct StopOnDrop<'a>(&'a AtomicBool);
impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
