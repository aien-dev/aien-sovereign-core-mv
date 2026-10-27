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

//! Test harness that drives the compose pipeline through its interface,
//! `ComposeBridge::run_task`, with scripted proposers.
//!
//! The bridge hands every model call to a proposer: the edit proposer for an
//! edit of an existing file, the document proposer for a new document. Both
//! are scripted here and both record what they were handed (the prompt, the
//! time limit, which of the two was asked), so a test observes exactly what
//! the model would see and the report the caller gets back.
//!
//! A goal the bridge refuses before any model call (a bad destination, a
//! workspace escape) comes back as `Err` and needs no composition library.
//! Everything else needs the real librx_compose.a: mark those tests
//! `#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]`.
//!
//! Every test that opens a home takes `home_guard::home_slot()` first; include
//! both files:
//! `#[path = "support/home_guard.rs"] mod home_guard;`
//! `#[path = "support/compose_harness.rs"] mod compose_harness;`
#![allow(dead_code)]

use aien_runtime::control::{ComposeTaskReport, ControlResponse};
use aien_runtime::{ComposeBridge, ComposeProposer, Generation};
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Which proposer the bridge asked: the edit proposer (an existing file is
/// edited) or the document proposer (a new document is written).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asked {
    Edit,
    Document,
}

/// One model call the bridge made.
#[derive(Debug, Clone)]
pub struct Call {
    /// The full prompt handed to the proposer.
    pub prompt: String,
    /// The time limit handed with it (what is left of the task's budget).
    pub limit: Duration,
    pub asked: Asked,
}

/// The scripted model: called with the 1-based call number (counted over both
/// proposers), the prompt and the limit.
pub type Script = Arc<dyn Fn(usize, &str, Duration) -> Result<Generation, String> + Send + Sync>;

/// A reply the way a model returns one: `tokens` 5, finish reason `eos`.
pub fn reply(text: &str) -> Result<Generation, String> {
    Ok(Generation {
        text: text.to_string(),
        tokens: 5,
        finish_reason: Some("eos".into()),
        ..Default::default()
    })
}

/// A script that answers every call with `text`.
pub fn always(text: &str) -> Script {
    let t = text.to_string();
    Arc::new(move |_, _, _| reply(&t))
}

/// A script that answers call k with `texts[k - 1]` (the last one repeats).
pub fn replies(texts: &[&str]) -> Script {
    let v: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    Arc::new(move |n, _, _| reply(&v[(n - 1).min(v.len() - 1)]))
}

/// A script whose model never answers (every call fails): the bridge still
/// shows what it would send.
pub fn silent() -> Script {
    Arc::new(|_, _, _| Err("harness: no model reply".to_string()))
}

/// One compose home and one workspace in a temporary directory, with a bridge
/// whose two proposers run `script` and record every call.
pub struct Harness {
    pub tmp: tempfile::TempDir,
    pub ws: PathBuf,
    pub bridge: ComposeBridge,
    calls: Arc<Mutex<Vec<Call>>>,
}

fn recording(script: Script, calls: Arc<Mutex<Vec<Call>>>, asked: Asked) -> ComposeProposer {
    Arc::new(move |prompt: &str, limit: Duration| {
        let n = {
            let mut c = calls.lock();
            c.push(Call {
                prompt: prompt.to_string(),
                limit,
                asked,
            });
            c.len()
        };
        script(n, prompt, limit)
    })
}

impl Harness {
    pub fn new(script: Script) -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let ws = std::fs::canonicalize(ws).unwrap();
        let calls: Arc<Mutex<Vec<Call>>> = Arc::default();
        let bridge = ComposeBridge::new(
            tmp.path().join("compose"),
            recording(script.clone(), calls.clone(), Asked::Edit),
            "test:harness",
        )
        .with_doc_proposer(recording(script, calls.clone(), Asked::Document));
        Self {
            tmp,
            ws,
            bridge,
            calls,
        }
    }

    /// Write `content` to `rel` inside the workspace (parents created).
    pub fn write(&self, rel: &str, content: &str) -> &Self {
        let p = self.ws.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
        self
    }

    pub fn ws(&self) -> &Path {
        &self.ws
    }

    /// Run one compose task on this workspace. `Err` is the bridge's refusal text.
    pub fn run(&self, goal: &str) -> Result<ComposeTaskReport, String> {
        self.run_in(goal, &self.ws)
    }

    /// `run` on another workspace directory.
    pub fn run_in(&self, goal: &str, ws: &Path) -> Result<ComposeTaskReport, String> {
        match self.bridge.run_task(goal, ws.to_str().unwrap()) {
            ControlResponse::ComposeTaskResult(r) => Ok(*r),
            ControlResponse::Error(e) => Err(e),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    /// Every call so far, in order.
    pub fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    /// Forget the calls so far (the next run starts a fresh list).
    pub fn clear(&self) {
        self.calls.lock().clear();
    }
}

/// What the bridge sends the model first for `goal` in `ws` (a fresh home, a
/// silent model): the first call, or the refusal text when the bridge refuses
/// the goal before any model call. Also returns the run's report.
pub fn first_call(goal: &str, ws: &Path) -> Result<(Call, ComposeTaskReport), String> {
    let h = Harness::new(silent());
    let r = h.run_in(goal, ws)?;
    let c = h.calls();
    let first = c
        .first()
        .cloned()
        .unwrap_or_else(|| panic!("the bridge made no model call: {r:?}"));
    Ok((first, r))
}
