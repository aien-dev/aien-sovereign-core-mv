//! The worker's signed receipt for one finished job.
//!
//! The receipt is self-attested: the process that ran the job also holds the
//! signing key and reports its own program digest, platform and timings. The
//! signature proves which pinned worker key made the claim and that nothing
//! changed on the way; it does not prove the claimed hardware or program.
//! What the coordinator trusts is its own recompute of the output digest
//! (`task::oracle`). Field names say `claimed_` for that reason.
//!
//! Canonical bytes and key fingerprints are those of the local verification
//! bundle (`aien_test::bundle`), so one canonical JSON encoder signs both.

use aien_test::bundle::{fingerprint, hex_decode, hex_encode, signed_bytes};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde_json::{json, Value};

pub const SCHEMA: &str = "aien.hetjob.receipt/1";
/// Largest receipt a coordinator reads, checked before parsing.
pub const MAX_RECEIPT_BYTES: usize = 8 * 1024;

/// What the worker says about itself. Every field is a claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Platform {
    pub os: String,
    pub arch: String,
    pub cpus: u64,
    pub memory_bytes: u64,
    pub program_digest: [u8; 32],
    pub version: String,
}

/// The facts one receipt binds together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub task_id: [u8; 16],
    pub idempotence_key: [u8; 16],
    pub cause_id: [u8; 32],
    pub kind: String,
    pub input_digest: [u8; 32],
    pub output_digest: [u8; 32],
    pub started_unix_ms: u64,
    pub compute_ns: u64,
    pub threads: u64,
    pub platform: Platform,
}

/// Why a receipt is not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    TooLarge(usize),
    Schema(String),
    /// The signing key is not in the coordinator's roster.
    WrongPeer(String),
    BadSignature,
}

impl ReceiptError {
    pub fn code(&self) -> String {
        match self {
            ReceiptError::TooLarge(n) => format!("RECEIPT_TOO_LARGE:{n}"),
            ReceiptError::Schema(s) => format!("RECEIPT_SCHEMA:{s}"),
            ReceiptError::WrongPeer(fp) => format!("WRONG_PEER:{fp}"),
            ReceiptError::BadSignature => "BAD_SIGNATURE".to_string(),
        }
    }
}

impl Receipt {
    fn body(&self, key: &VerifyingKey) -> Value {
        json!({
            "schema": SCHEMA,
            "attestation": "self-attested-by-worker",
            "job": {
                "task_id": hex_encode(&self.task_id),
                "idempotence_key": hex_encode(&self.idempotence_key),
                "cause_id": hex_encode(&self.cause_id),
            },
            "task": {
                "kind": self.kind,
                "input_digest": hex_encode(&self.input_digest),
            },
            "output_digest": hex_encode(&self.output_digest),
            "timing": {
                "started_unix_ms": self.started_unix_ms,
                "compute_ns": self.compute_ns,
                "threads": self.threads,
            },
            "worker": {
                "pubkey": hex_encode(key.as_bytes()),
                "key_fingerprint": fingerprint(key),
                "claimed_os": self.platform.os,
                "claimed_arch": self.platform.arch,
                "claimed_cpus": self.platform.cpus,
                "claimed_memory_bytes": self.platform.memory_bytes,
                "claimed_program_digest": hex_encode(&self.platform.program_digest),
                "claimed_version": self.platform.version,
            },
        })
    }

    /// Sign and serialise. The signature covers the canonical bytes of every
    /// member except `signature`.
    pub fn sign(&self, key: &SigningKey) -> Vec<u8> {
        let mut v = self.body(&key.verifying_key());
        let bytes = signed_bytes(&v).expect("receipt body is canonical-encodable");
        let sig = key.sign(&bytes);
        v["signature"] = Value::String(hex_encode(&sig.to_bytes()));
        serde_json::to_vec(&v).expect("receipt serialises")
    }

    /// Parse, check the schema, check the key is pinned and the signature
    /// verifies (strict: no small-order keys, canonical s).
    pub fn verify(
        bytes: &[u8],
        roster: &[VerifyingKey],
    ) -> Result<(Receipt, VerifyingKey), ReceiptError> {
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(ReceiptError::TooLarge(bytes.len()));
        }
        let v: Value =
            serde_json::from_slice(bytes).map_err(|e| ReceiptError::Schema(e.to_string()))?;
        let s = |path: &[&str]| -> Result<String, ReceiptError> {
            let mut cur = &v;
            for p in path {
                cur = cur
                    .get(p)
                    .ok_or_else(|| ReceiptError::Schema(format!("missing {}", path.join("."))))?;
            }
            cur.as_str()
                .map(str::to_string)
                .ok_or_else(|| ReceiptError::Schema(format!("{} not a string", path.join("."))))
        };
        let u = |path: &[&str]| -> Result<u64, ReceiptError> {
            let mut cur = &v;
            for p in path {
                cur = cur
                    .get(p)
                    .ok_or_else(|| ReceiptError::Schema(format!("missing {}", path.join("."))))?;
            }
            cur.as_u64()
                .ok_or_else(|| ReceiptError::Schema(format!("{} not an integer", path.join("."))))
        };
        fn fixed<const N: usize>(name: &str, hex: &str) -> Result<[u8; N], ReceiptError> {
            hex_decode(hex)
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| ReceiptError::Schema(format!("{name} is not {N} bytes of hex")))
        }
        if s(&["schema"])? != SCHEMA {
            return Err(ReceiptError::Schema("schema".to_string()));
        }
        let pk: [u8; 32] = fixed("worker.pubkey", &s(&["worker", "pubkey"])?)?;
        let key = roster
            .iter()
            .find(|k| k.as_bytes() == &pk)
            .copied()
            .ok_or_else(|| ReceiptError::WrongPeer(hex_encode(&pk)))?;
        if s(&["worker", "key_fingerprint"])? != fingerprint(&key) {
            return Err(ReceiptError::Schema("worker.key_fingerprint".to_string()));
        }
        let sig: [u8; 64] = fixed("signature", &s(&["signature"])?)?;
        let signed = signed_bytes(&v).map_err(|e| ReceiptError::Schema(format!("{e:?}")))?;
        key.verify_strict(&signed, &Signature::from_bytes(&sig))
            .map_err(|_| ReceiptError::BadSignature)?;
        let r = Receipt {
            task_id: fixed("job.task_id", &s(&["job", "task_id"])?)?,
            idempotence_key: fixed("job.idempotence_key", &s(&["job", "idempotence_key"])?)?,
            cause_id: fixed("job.cause_id", &s(&["job", "cause_id"])?)?,
            kind: s(&["task", "kind"])?,
            input_digest: fixed("task.input_digest", &s(&["task", "input_digest"])?)?,
            output_digest: fixed("output_digest", &s(&["output_digest"])?)?,
            started_unix_ms: u(&["timing", "started_unix_ms"])?,
            compute_ns: u(&["timing", "compute_ns"])?,
            threads: u(&["timing", "threads"])?,
            platform: Platform {
                os: s(&["worker", "claimed_os"])?,
                arch: s(&["worker", "claimed_arch"])?,
                cpus: u(&["worker", "claimed_cpus"])?,
                memory_bytes: u(&["worker", "claimed_memory_bytes"])?,
                program_digest: fixed(
                    "worker.claimed_program_digest",
                    &s(&["worker", "claimed_program_digest"])?,
                )?,
                version: s(&["worker", "claimed_version"])?,
            },
        };
        // The canonical body must be exactly what this key signs for these
        // facts: no extra or renamed members ride along under the signature.
        let mut expect = r.body(&key);
        expect["signature"] = v["signature"].clone();
        if expect != v {
            return Err(ReceiptError::Schema("unexpected members".to_string()));
        }
        Ok((r, key))
    }
}
