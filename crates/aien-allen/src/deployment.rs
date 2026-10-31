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

//! Host-produced deployment record `<home>.allen-deployments`: which model
//! artifacts served which logical agent, in order. Append-only JSON lines,
//! each line hash-chained to the previous one. Digests come from hashing
//! files on the host; model-supplied identity never enters (arch#150 s4.4).
//! The subject object is never written here or by a model swap: a swap or a
//! rollback appends a line and nothing else.
//!
//! `placeholder: true` marks synthetic digests used by tests. Such a line
//! does NOT establish real model-swap continuity.
use crate::{hex, unhex32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const GENESIS_PREV: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub seq: u64,
    pub prev: String,
    /// Logical agent id (hex) this deployment serves.
    pub agent: String,
    pub model_sha256: String,
    pub config_sha256: String,
    pub candidate_id: String,
    pub executable_sha256: String,
    pub revision: String,
    pub placeholder: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Line {
    entry: Entry,
    sha256: String,
}

#[derive(Debug, Clone)]
pub struct Input {
    pub model_sha256: String,
    pub config_sha256: String,
    pub candidate_id: String,
    pub executable_sha256: String,
    pub revision: String,
    pub placeholder: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeployRefusal {
    Damaged(String),
    ChainBroken(u64),
    ForgedIdentity(u64),
    ArtifactChanged {
        seq: u64,
        recorded: String,
        actual: String,
    },
    Io(String),
}

impl std::fmt::Display for DeployRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Damaged(w) => write!(f, "deployment record damaged: {w}"),
            Self::ChainBroken(s) => write!(f, "deployment chain broken at line {s}"),
            Self::ForgedIdentity(s) => {
                write!(f, "deployment line {s} names another logical agent")
            }
            Self::ArtifactChanged {
                seq,
                recorded,
                actual,
            } => write!(
                f,
                "artifact bytes changed since deployment {seq}: recorded {recorded}, now {actual}"
            ),
            Self::Io(w) => write!(f, "deployment record io: {w}"),
        }
    }
}

pub fn deployments_path(home: &Path) -> PathBuf {
    let mut n = home.as_os_str().to_owned();
    n.push(".allen-deployments");
    PathBuf::from(n)
}

fn entry_hash(e: &Entry) -> String {
    hex(&Sha256::digest(
        serde_json::to_string(e)
            .expect("entry serializes")
            .as_bytes(),
    ))
}

/// sha256 of a file, hex: the host-side way to name a model artifact.
pub fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Read and verify the whole record for `agent` (hex). Empty/missing = no history.
pub fn verify(path: &Path, agent: &[u8; 32]) -> Result<Vec<Entry>, DeployRefusal> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(DeployRefusal::Damaged(e.to_string())),
    };
    let agent_hex = hex(agent);
    let mut out: Vec<Entry> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let n = i as u64 + 1;
        let line: Line = serde_json::from_str(raw)
            .map_err(|e| DeployRefusal::Damaged(format!("line {n}: {e}")))?;
        let e = line.entry;
        if entry_hash(&e) != line.sha256 || e.seq != n {
            return Err(DeployRefusal::ChainBroken(n));
        }
        let want_prev = out.last().map_or(GENESIS_PREV.to_string(), entry_hash);
        if e.prev != want_prev {
            return Err(DeployRefusal::ChainBroken(n));
        }
        if unhex32(&e.agent).is_none() || e.agent != agent_hex {
            return Err(DeployRefusal::ForgedIdentity(n));
        }
        out.push(e);
    }
    if !text.is_empty() && !text.ends_with('\n') {
        return Err(DeployRefusal::Damaged("truncated final line".into()));
    }
    Ok(out)
}

/// Append one deployment line (verifies the existing record first).
pub fn append(path: &Path, agent: &[u8; 32], input: Input) -> Result<Entry, DeployRefusal> {
    let have = verify(path, agent)?;
    let e = Entry {
        seq: have.len() as u64 + 1,
        prev: have.last().map_or(GENESIS_PREV.to_string(), entry_hash),
        agent: hex(agent),
        model_sha256: input.model_sha256,
        config_sha256: input.config_sha256,
        candidate_id: input.candidate_id,
        executable_sha256: input.executable_sha256,
        revision: input.revision,
        placeholder: input.placeholder,
    };
    let line = Line {
        sha256: entry_hash(&e),
        entry: e.clone(),
    };
    let io = |e: std::io::Error| DeployRefusal::Io(e.to_string());
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io)?;
    let mut s = serde_json::to_string(&line).expect("line serializes");
    s.push('\n');
    f.write_all(s.as_bytes()).map_err(io)?;
    f.sync_all().map_err(io)?;
    Ok(e)
}

/// Does the model file on disk still match deployment `e`?
pub fn check_artifact(e: &Entry, file: &Path) -> Result<(), DeployRefusal> {
    let actual = sha256_file_hex(file).map_err(|x| DeployRefusal::Io(x.to_string()))?;
    if actual != e.model_sha256 {
        return Err(DeployRefusal::ArtifactChanged {
            seq: e.seq,
            recorded: e.model_sha256.clone(),
            actual,
        });
    }
    Ok(())
}
