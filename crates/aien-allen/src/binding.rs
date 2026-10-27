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

//! The host pin record `<home>.allen-binding`: what this machine resolved
//! last. It is sovereign-core's own record, NOT part of the ALLEN subject
//! (ADR 0035 I3: the subject holds no machine id). The machine id is its own
//! field; there is no model field and no KV field.
//!
//! Layout (248 bytes, little-endian): "AIENALB1" | u32 version 1 | u32 0 |
//! agent 32 | root 32 | head object id 32 | u64 head sequence | lineage 32 |
//! machine id 32 | provenance 32 | sha256 of the first 216 bytes.
//! Limits (NOT_PROVED): the sha256 is unkeyed, so a pin deleted and
//! re-adopted, or the pin and the subject rolled back together, are not
//! detected.
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const PIN_LEN: usize = 248;
const MAGIC: &[u8; 8] = b"AIENALB1";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub agent: [u8; 32],
    pub root: [u8; 32],
    pub head_id: [u8; 32],
    pub head_seq: u64,
    pub lineage: [u8; 32],
    pub machine_id: [u8; 32],
    pub provenance: [u8; 32],
}

/// `<home>.allen-binding`, beside `<home>.machine-root` and `<home>.cortex-mark`.
pub fn pin_path(home: &Path) -> PathBuf {
    let mut n = home.as_os_str().to_owned();
    n.push(".allen-binding");
    PathBuf::from(n)
}

impl Pin {
    pub fn encode(&self) -> [u8; PIN_LEN] {
        let mut b = [0u8; PIN_LEN];
        b[..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[16..48].copy_from_slice(&self.agent);
        b[48..80].copy_from_slice(&self.root);
        b[80..112].copy_from_slice(&self.head_id);
        b[112..120].copy_from_slice(&self.head_seq.to_le_bytes());
        b[120..152].copy_from_slice(&self.lineage);
        b[152..184].copy_from_slice(&self.machine_id);
        b[184..216].copy_from_slice(&self.provenance);
        let h = Sha256::digest(&b[..216]);
        b[216..].copy_from_slice(&h);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Pin, String> {
        if b.len() != PIN_LEN {
            return Err(format!("{} bytes, expected {PIN_LEN}", b.len()));
        }
        if &b[..8] != MAGIC {
            return Err("bad magic".into());
        }
        let v = u32::from_le_bytes(b[8..12].try_into().expect("4"));
        if v != VERSION || b[12..16] != [0u8; 4] {
            return Err(format!("unknown version {v}"));
        }
        if Sha256::digest(&b[..216]).as_slice() != &b[216..] {
            return Err("checksum mismatch".into());
        }
        let f = |a: usize| -> [u8; 32] { b[a..a + 32].try_into().expect("32") };
        Ok(Pin {
            agent: f(16),
            root: f(48),
            head_id: f(80),
            head_seq: u64::from_le_bytes(b[112..120].try_into().expect("8")),
            lineage: f(120),
            machine_id: f(152),
            provenance: f(184),
        })
    }
}

/// `Ok(None)` when no pin file exists; `Err` when it exists and is damaged.
pub fn read(path: &Path) -> Result<Option<Pin>, String> {
    match std::fs::read(path) {
        Ok(b) => Pin::decode(&b).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

/// Atomic write: temp file, fsync, rename, fsync of the directory.
pub fn write_atomic(path: &Path, pin: &Pin) -> Result<(), String> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let werr = |what: &str, e: std::io::Error| format!("{what} {}: {e}", path.display());
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).map_err(|e| werr("create dir for", e))?;
    }
    let mut f = std::fs::File::create(&tmp).map_err(|e| werr("create temp for", e))?;
    f.write_all(&pin.encode()).map_err(|e| werr("write", e))?;
    f.sync_all().map_err(|e| werr("sync", e))?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| werr("rename onto", e))?;
    if let Some(p) = path.parent() {
        std::fs::File::open(p)
            .and_then(|d| d.sync_all())
            .map_err(|e| werr("sync dir of", e))?;
    }
    Ok(())
}
