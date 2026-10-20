//! TEST-ONLY subject fixture builder. A second, independent encoder of the
//! kind-24 layout (aienos ADR 0018 s2.2). It lives under tests/ so the
//! production crate gains no provisioning path: real subjects are created
//! only by Aienux `cs_provision`. Anything built here is a fixture.
#![allow(dead_code)]
use aien_allen::subject_v0::object_id;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn h32(tag: &str) -> [u8; 32] {
    Sha256::digest(tag.as_bytes()).into()
}

pub struct Fx {
    pub root: [u8; 32],
    pub agent: [u8; 32],
    pub cortex: [u8; 32],
    pub provenance: [u8; 32],
}

pub fn fx(name: &str) -> Fx {
    Fx {
        root: h32(&format!("root:{name}")),
        agent: h32(&format!("agent:{name}")),
        cortex: h32(&format!("cortex:{name}")),
        provenance: h32(&format!("prov:{name}")),
    }
}

fn intent_id(agent: &[u8; 32], kind: u32, since: u64, p: [u64; 2]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIENOS_INTENT_v0:");
    h.update(agent);
    h.update(kind.to_le_bytes());
    h.update(since.to_le_bytes());
    h.update(p[0].to_le_bytes());
    h.update(p[1].to_le_bytes());
    h.finalize().into()
}

/// One object: sequence `seq`, `previous` (zero for 1), one ACTIVE GOAL_LATENCY
/// intent created at sequence 1 with payload (regime, target ns).
pub fn encode(f: &Fx, seq: u64, previous: [u8; 32], payload: [u64; 2]) -> Vec<u8> {
    let id = intent_id(&f.agent, 1, 1, payload);
    let mut b = Vec::new();
    b.extend_from_slice(b"AIENSUBJ");
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&[0; 4]); // body length, patched below
    b.extend_from_slice(&f.root);
    b.extend_from_slice(&f.agent);
    b.extend_from_slice(&previous);
    b.extend_from_slice(&seq.to_le_bytes());
    b.extend_from_slice(&f.cortex);
    b.extend_from_slice(&f.provenance);
    b.push(1); // origin operator
    b.extend_from_slice(&[0; 7]);
    b.extend_from_slice(&1u16.to_le_bytes()); // n_intents
    b.extend_from_slice(&0u16.to_le_bytes()); // n_knowledge
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&id);
    b.extend_from_slice(&[0; 32]);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.push(1); // ACTIVE
    b.extend_from_slice(&[0; 3]);
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&payload[0].to_le_bytes());
    b.extend_from_slice(&payload[1].to_le_bytes());
    let n = (b.len() - 16) as u32;
    b[12..16].copy_from_slice(&n.to_le_bytes());
    b
}

/// A chain of `n` objects; returns the encoded objects in order.
pub fn chain(f: &Fx, n: u64) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut prev = [0u8; 32];
    for s in 1..=n {
        let o = encode(f, s, prev, [7, 1000]);
        prev = object_id(&o);
        out.push(o);
    }
    out
}

/// Write objects into `dir` as `<id-hex>.bin`.
pub fn write_dir(dir: &Path, objs: &[Vec<u8>]) {
    std::fs::create_dir_all(dir).unwrap();
    for o in objs {
        let name = aien_allen::hex(&object_id(o));
        std::fs::write(dir.join(format!("{name}.bin")), o).unwrap();
    }
}

pub fn write_head(path: &Path, o: &[u8]) -> PathBuf {
    std::fs::write(path, o).unwrap();
    path.to_path_buf()
}
