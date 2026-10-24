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

//! VerificationBundleV1 (ADR 0033 Decision 6): the signed, per-commit result
//! of AIEN Verification, and the independent verifier GitHub runs.
//!
//! A bundle is one canonical-JSON object (the EvidenceReceiptV1 encoding,
//! `evidence::canonical`) that names the repository, the commit, the graph of
//! checks the planner produced, one entry per check (receipt digest for a run,
//! resolution record for a cache hit, nothing for `not_run`/`blocked`), the
//! commit verdict, the executor, and an ed25519 signature over the canonical
//! bytes of everything but the `signature` member.
//!
//! The verifier does not trust the file: it recomputes the canonical bytes,
//! checks the signature against the public key the repository committed,
//! checks repository and commit against the ones being gated, checks that the
//! executor id is the fingerprint of that key, that every required check is
//! present, and that the declared verdict is what the verdict algebra gives.
//! Any mismatch is a rejection with a code, never a PASS.
//!
//! `plan.scope` (signed with the rest) says what the worker planned: `full`
//! (every gate at HEAD, an Alpha qualification) or `changed` (the gates the
//! diff selects). A bundle without it is `changed`. `verify-bundle
//! --require-full` rejects (NOT_FULL_SCOPE) any bundle whose scope is not
//! `full`.
//!
//! Verdict algebra (Decision 7, applied to a bundle): PASS only if every
//! `required` entry ran or was resolved from cache with verdict PASS; FAIL if
//! any required entry is FAIL; otherwise NOT_RUN. Entries with
//! `required: false` (every ADVISORY check, and optional checks) never change
//! the commit verdict. Nothing promotes NOT_RUN to PASS.
//!
//! Keys: the signing key is a 32-byte seed, hex, in a file the worker user
//! owns with mode 0600 (`keygen`). The public key is 32 bytes, hex, committed
//! at `.aien/verify.pub` in each verified repository. The executor id is
//! `sha256(pubkey bytes)`, hex, so a bundle names the key that must verify it
//! and the verifier can refuse an unknown executor before looking further.

use crate::evidence::{canonical, canonical_digest, sha256_hex};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde_json::{json, Map, Value};
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

pub const SCHEMA: &str = "aien-verification-bundle/1";
pub const SIG_ALG: &str = "ed25519";
pub const RECEIPT_ID_SCHEMES: [&str; 2] = ["sha256-canonical-json", "aien-proof-blake3"];
pub const STATUSES: [&str; 4] = ["run", "cached", "not_run", "blocked"];
pub const VERDICTS: [&str; 6] = [
    "PASS",
    "FAIL",
    "NOT_RUN",
    "BLOCKED_HARDWARE",
    "BLOCKED_OPERATOR",
    "MISSING_IMPLEMENTATION",
];
pub const EXACTNESS: [&str; 4] = ["EXACT", "BOUNDED", "APPROXIMATE", "ADVISORY"];
/// `plan.scope`: every gate at HEAD (an Alpha qualification).
pub const SCOPE_FULL: &str = "full";
/// `plan.scope`: the gates the diff selects. A bundle without the member (or
/// without a `plan`) is this scope.
pub const SCOPE_CHANGED: &str = "changed";
pub const SCOPES: [&str; 2] = [SCOPE_FULL, SCOPE_CHANGED];

/// Why a bundle was refused. The code is what the GitHub log and the operator
/// see; the text says which member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    /// Not a VerificationBundleV1: missing or malformed member.
    Schema(String),
    /// The bundle's public key is not the one the repository committed.
    UnknownKey,
    /// The signature does not verify over the canonical bytes.
    BadSignature,
    /// `repository` is not the repository being gated.
    RepositoryMismatch { bundle: String, expected: String },
    /// `commit` is not the commit being gated.
    CommitMismatch { bundle: String, expected: String },
    /// `executor.id` is not the fingerprint of the signing key.
    UnknownExecutor,
    /// A check the gate requires has no entry.
    MissingRequiredCheck(String),
    /// The declared `verdict` is not what the entries give.
    VerdictInconsistent { declared: String, computed: String },
    /// `--require-full`: the signed plan scope is not `full`.
    NotFullScope(String),
}

impl Reject {
    pub fn code(&self) -> &'static str {
        match self {
            Reject::Schema(_) => "SCHEMA",
            Reject::UnknownKey => "UNKNOWN_KEY",
            Reject::BadSignature => "BAD_SIGNATURE",
            Reject::RepositoryMismatch { .. } => "REPOSITORY_MISMATCH",
            Reject::CommitMismatch { .. } => "COMMIT_MISMATCH",
            Reject::UnknownExecutor => "UNKNOWN_EXECUTOR",
            Reject::MissingRequiredCheck(_) => "MISSING_REQUIRED_CHECK",
            Reject::VerdictInconsistent { .. } => "VERDICT_INCONSISTENT",
            Reject::NotFullScope(_) => "NOT_FULL_SCOPE",
        }
    }
}

impl fmt::Display for Reject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reject::Schema(m) => write!(f, "SCHEMA: {m}"),
            Reject::UnknownKey => {
                write!(f, "UNKNOWN_KEY: bundle key is not the committed verify.pub")
            }
            Reject::BadSignature => write!(
                f,
                "BAD_SIGNATURE: signature does not verify over the canonical bytes"
            ),
            Reject::RepositoryMismatch { bundle, expected } => {
                write!(
                    f,
                    "REPOSITORY_MISMATCH: bundle is for {bundle}, gating {expected}"
                )
            }
            Reject::CommitMismatch { bundle, expected } => {
                write!(
                    f,
                    "COMMIT_MISMATCH: bundle is for {bundle}, gating {expected}"
                )
            }
            Reject::UnknownExecutor => write!(
                f,
                "UNKNOWN_EXECUTOR: executor.id is not the fingerprint of the signing key"
            ),
            Reject::MissingRequiredCheck(g) => write!(f, "MISSING_REQUIRED_CHECK: {g}"),
            Reject::VerdictInconsistent { declared, computed } => {
                write!(
                    f,
                    "VERDICT_INCONSISTENT: bundle says {declared}, entries give {computed}"
                )
            }
            Reject::NotFullScope(scope) => write!(
                f,
                "NOT_FULL_SCOPE: plan scope is {scope}, not a full-scope (Alpha) qualification"
            ),
        }
    }
}

/// What the verifier learned from a bundle it accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    pub verdict: String,
    pub digest: String,
    pub checks: usize,
    pub required: usize,
    pub cached: usize,
    /// The worker's `plan.verdict` member when the bundle carries one (signed
    /// with the rest). None for a bundle without a plan member.
    pub plan_verdict: Option<String>,
    /// The signed `plan.scope`: `full` or `changed` (the default).
    pub scope: String,
}

impl Verified {
    /// What a gate may act on: PASS only when the algebra says PASS, at least
    /// one required check exists (zero required checks is PASS only
    /// vacuously: nothing vouches for the commit), and the plan member, when
    /// present, says PASS too (an UNMAPPED or BLOCKED plan is NOT_RUN).
    pub fn gate_verdict(&self) -> &str {
        match self.verdict.as_str() {
            "PASS" if self.required == 0 => "NOT_RUN",
            "PASS" if self.plan_verdict.as_deref().is_some_and(|p| p != "PASS") => "NOT_RUN",
            v => v,
        }
    }
}

/// What the gate expects the bundle to be about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expect {
    pub repository: String,
    pub commit: String,
    pub pubkey_hex: String,
    /// Gates that must have an entry (the repository's required list). Empty
    /// means "whatever the bundle marks required", which is weaker; the gate
    /// should pass its own list.
    pub required: Vec<String>,
    /// `--require-full`: refuse (NOT_FULL_SCOPE) unless the signed plan scope
    /// is `full`.
    pub require_full: bool,
}

// ---------------------------------------------------------------- hex

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
}

// ---------------------------------------------------------------- keys

/// Executor id: sha256 over the 32 public-key bytes, hex.
pub fn fingerprint(vk: &VerifyingKey) -> String {
    sha256_hex(vk.as_bytes())
}

/// Generate a signing key from the kernel's randomness and write its hex seed
/// to `path` with mode 0600. Refuses to overwrite. Returns the public key hex.
pub fn keygen(path: &Path) -> Result<String, String> {
    let mut seed = [0u8; 32];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut seed))
        .map_err(|e| format!("cannot read /dev/urandom: {e}"))?;
    let sk = SigningKey::from_bytes(&seed);
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    f.write_all(hex_encode(&seed).as_bytes())
        .and_then(|_| f.write_all(b"\n"))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(hex_encode(sk.verifying_key().as_bytes()))
}

pub fn load_signing_key(path: &Path) -> Result<SigningKey, String> {
    let txt =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let bytes = hex_decode(&txt).ok_or_else(|| format!("{} is not hex", path.display()))?;
    let seed: [u8; 32] = bytes
        .try_into()
        .map_err(|_| format!("{} is not a 32-byte seed", path.display()))?;
    Ok(SigningKey::from_bytes(&seed))
}

/// `--pubkey` takes a file (hex inside) or the hex itself.
pub fn load_pubkey_hex(arg: &str) -> Result<String, String> {
    let p = Path::new(arg);
    let txt = if p.is_file() {
        fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?
    } else {
        arg.to_string()
    };
    let t = txt.trim().to_string();
    if !is_hex(&t, 64) {
        return Err("public key must be 64 hex characters".to_string());
    }
    Ok(t)
}

fn verifying_key(hex: &str) -> Option<VerifyingKey> {
    let bytes = hex_decode(hex)?;
    let arr: [u8; 32] = bytes.try_into().ok()?;
    VerifyingKey::from_bytes(&arr).ok()
}

// ---------------------------------------------------------------- shape

fn member<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a Value, Reject> {
    o.get(k)
        .ok_or_else(|| Reject::Schema(format!("missing {k}")))
}

fn text<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a str, Reject> {
    member(o, k)?
        .as_str()
        .ok_or_else(|| Reject::Schema(format!("{k} must be a string")))
}

fn hex_member<'a>(o: &'a Map<String, Value>, k: &str, n: usize) -> Result<&'a str, Reject> {
    let s = text(o, k)?;
    if !is_hex(s, n) {
        return Err(Reject::Schema(format!(
            "{k} must be {n} lower-case hex characters"
        )));
    }
    Ok(s)
}

fn one_of<'a>(o: &'a Map<String, Value>, k: &str, allowed: &[&str]) -> Result<&'a str, Reject> {
    let s = text(o, k)?;
    if !allowed.contains(&s) {
        return Err(Reject::Schema(format!(
            "{k} {s:?} is not one of {allowed:?}"
        )));
    }
    Ok(s)
}

/// Validate one `checks[]` entry and return (required, status, verdict).
fn check_entry(v: &Value, i: usize) -> Result<(String, bool, String, String), Reject> {
    let o = v
        .as_object()
        .ok_or_else(|| Reject::Schema(format!("checks[{i}] must be an object")))?;
    let gate = text(o, "gate")?.to_string();
    hex_member(o, "check_id", 64)?;
    let exactness = one_of(o, "exactness", &EXACTNESS)?;
    let required = member(o, "required")?
        .as_bool()
        .ok_or_else(|| Reject::Schema(format!("checks[{i}].required must be a bool")))?;
    if required && exactness == "ADVISORY" {
        return Err(Reject::Schema(format!(
            "checks[{i}] ({gate}): an ADVISORY check cannot be required"
        )));
    }
    let status = one_of(o, "status", &STATUSES)?.to_string();
    let verdict = one_of(o, "verdict", &VERDICTS)?.to_string();
    text(o, "backend")?;
    match status.as_str() {
        "run" => {
            hex_member(o, "receipt", 64)?;
        }
        "cached" => {
            let r = member(o, "resolution")?.as_object().ok_or_else(|| {
                Reject::Schema(format!("checks[{i}].resolution must be an object"))
            })?;
            hex_member(r, "resolved_from", 64)?;
        }
        _ => {
            if verdict == "PASS" {
                return Err(Reject::Schema(format!(
                    "checks[{i}] ({gate}): status {status} cannot carry verdict PASS"
                )));
            }
        }
    }
    Ok((gate, required, status, verdict))
}

/// The bundle verdict from its entries (Decision 7). Entries are
/// `(required, status, verdict)`.
pub fn compute_verdict(entries: &[(bool, String, String)]) -> &'static str {
    let mut any_fail = false;
    let mut all_pass = true;
    for (required, status, verdict) in entries {
        if !required {
            continue;
        }
        let ran = status == "run" || status == "cached";
        if verdict == "FAIL" {
            any_fail = true;
        }
        if !(ran && verdict == "PASS") {
            all_pass = false;
        }
    }
    if any_fail {
        "FAIL"
    } else if all_pass {
        "PASS"
    } else {
        "NOT_RUN"
    }
}

/// (gate, required, status, verdict) per `checks[]` entry.
type Entries = Vec<(String, bool, String, String)>;

/// Structural validation of an unsigned or signed bundle. Returns the object
/// and the entry summary. Does not look at the signature.
fn validate(bundle: &Value) -> Result<(&Map<String, Value>, Entries), Reject> {
    let o = bundle
        .as_object()
        .ok_or_else(|| Reject::Schema("bundle must be an object".to_string()))?;
    if text(o, "schema")? != SCHEMA {
        return Err(Reject::Schema(format!("schema must be {SCHEMA}")));
    }
    text(o, "repository")?;
    hex_member(o, "commit", 40)?;
    hex_member(o, "tree_digest", 64)?;
    hex_member(o, "graph_digest", 64)?;
    one_of(o, "receipt_id_scheme", &RECEIPT_ID_SCHEMES)?;
    one_of(o, "verdict", &["PASS", "FAIL", "NOT_RUN"])?;
    text(o, "started_utc")?;
    text(o, "finished_utc")?;
    let ex = member(o, "executor")?
        .as_object()
        .ok_or_else(|| Reject::Schema("executor must be an object".to_string()))?;
    hex_member(ex, "id", 64)?;
    text(ex, "class")?;
    hex_member(ex, "machine_digest", 64)?;
    let checks = member(o, "checks")?
        .as_array()
        .ok_or_else(|| Reject::Schema("checks must be an array".to_string()))?;
    let mut out = Vec::with_capacity(checks.len());
    for (i, c) in checks.iter().enumerate() {
        out.push(check_entry(c, i)?);
    }
    let mut gates: Vec<&str> = out.iter().map(|e| e.0.as_str()).collect();
    gates.sort_unstable();
    if gates.windows(2).any(|w| w[0] == w[1]) {
        return Err(Reject::Schema("checks[] lists a gate twice".to_string()));
    }
    // `plan.scope`, when present, is one of the two scopes; anything else is
    // not a bundle this verifier understands.
    if let Some(p) = o.get("plan") {
        let p = p
            .as_object()
            .ok_or_else(|| Reject::Schema("plan must be an object".to_string()))?;
        if p.contains_key("scope") {
            one_of(p, "scope", &SCOPES)?;
        }
    }
    Ok((o, out))
}

/// The bundle's `plan.scope`: `full`, or `changed` when the member (or the
/// whole `plan`) is absent, as in every bundle written before scopes existed.
/// Unvalidated: `verify` refuses an unknown value.
pub fn scope_of(bundle: &Value) -> &str {
    match bundle["plan"]["scope"].as_str() {
        Some(s) if SCOPES.contains(&s) => s,
        _ => SCOPE_CHANGED,
    }
}

// ---------------------------------------------------------------- sign

/// Canonical bytes of the bundle without its `signature` member: what is
/// signed and what is verified.
pub fn signed_bytes(bundle: &Value) -> Result<Vec<u8>, Reject> {
    let mut o = bundle
        .as_object()
        .ok_or_else(|| Reject::Schema("bundle must be an object".to_string()))?
        .clone();
    o.remove("signature");
    canonical(&Value::Object(o))
        .map_err(|e| Reject::Schema(format!("not canonical-encodable: {e}")))
}

/// Sign an unsigned bundle. Refuses a bundle that is not well formed, whose
/// `executor.id` is not this key's fingerprint, or whose declared verdict is
/// not what its entries give: a worker must not be able to sign a lie.
pub fn sign(bundle: &Value, key: &SigningKey) -> Result<Value, Reject> {
    let (o, entries) = validate(bundle)?;
    let vk = key.verifying_key();
    if o["executor"]["id"].as_str() != Some(fingerprint(&vk).as_str()) {
        return Err(Reject::UnknownExecutor);
    }
    let computed = compute_verdict(
        &entries
            .iter()
            .map(|e| (e.1, e.2.clone(), e.3.clone()))
            .collect::<Vec<_>>(),
    );
    let declared = o["verdict"].as_str().unwrap_or("");
    if declared != computed {
        return Err(Reject::VerdictInconsistent {
            declared: declared.to_string(),
            computed: computed.to_string(),
        });
    }
    let bytes = signed_bytes(bundle)?;
    let sig: Signature = key.sign(&bytes);
    let mut out = o.clone();
    out.insert(
        "signature".to_string(),
        json!({ "alg": SIG_ALG, "pubkey": hex_encode(vk.as_bytes()), "sig": hex_encode(&sig.to_bytes()) }),
    );
    Ok(Value::Object(out))
}

// ---------------------------------------------------------------- verify

/// The independent verifier. Order: shape, key known, signature, repository,
/// commit, executor, required checks present, verdict algebra.
pub fn verify(bundle: &Value, expect: &Expect) -> Result<Verified, Reject> {
    let (o, entries) = validate(bundle)?;
    let sig = member(o, "signature")?
        .as_object()
        .ok_or_else(|| Reject::Schema("signature must be an object".to_string()))?;
    if text(sig, "alg")? != SIG_ALG {
        return Err(Reject::Schema(format!("signature.alg must be {SIG_ALG}")));
    }
    let pubkey = hex_member(sig, "pubkey", 64)?;
    let sighex = hex_member(sig, "sig", 128)?;
    if pubkey != expect.pubkey_hex.trim() {
        return Err(Reject::UnknownKey);
    }
    let vk = verifying_key(pubkey).ok_or(Reject::UnknownKey)?;
    let sig_bytes = hex_decode(sighex).ok_or(Reject::BadSignature)?;
    let signature = Signature::from_slice(&sig_bytes).map_err(|_| Reject::BadSignature)?;
    let bytes = signed_bytes(bundle)?;
    vk.verify_strict(&bytes, &signature)
        .map_err(|_| Reject::BadSignature)?;
    let repo = o["repository"].as_str().unwrap_or("");
    if repo != expect.repository {
        return Err(Reject::RepositoryMismatch {
            bundle: repo.to_string(),
            expected: expect.repository.clone(),
        });
    }
    let commit = o["commit"].as_str().unwrap_or("");
    if commit != expect.commit {
        return Err(Reject::CommitMismatch {
            bundle: commit.to_string(),
            expected: expect.commit.clone(),
        });
    }
    if o["executor"]["id"].as_str() != Some(fingerprint(&vk).as_str()) {
        return Err(Reject::UnknownExecutor);
    }
    // The scope is signed: checked only once the signature holds.
    let scope = scope_of(bundle).to_string();
    if expect.require_full && scope != SCOPE_FULL {
        return Err(Reject::NotFullScope(scope));
    }
    for g in &expect.required {
        match entries.iter().find(|e| &e.0 == g) {
            None => return Err(Reject::MissingRequiredCheck(g.clone())),
            Some(e) if !e.1 => {
                return Err(Reject::Schema(format!(
                    "{g} is required by the gate but marked optional in the bundle"
                )))
            }
            _ => {}
        }
    }
    let computed = compute_verdict(
        &entries
            .iter()
            .map(|e| (e.1, e.2.clone(), e.3.clone()))
            .collect::<Vec<_>>(),
    );
    let declared = o["verdict"].as_str().unwrap_or("");
    if declared != computed {
        return Err(Reject::VerdictInconsistent {
            declared: declared.to_string(),
            computed: computed.to_string(),
        });
    }
    let digest = canonical_digest(bundle).map_err(|e| Reject::Schema(format!("{e}")))?;
    Ok(Verified {
        verdict: computed.to_string(),
        digest,
        checks: entries.len(),
        required: entries.iter().filter(|e| e.1).count(),
        cached: entries.iter().filter(|e| e.2 == "cached").count(),
        plan_verdict: o
            .get("plan")
            .map(|p| p["verdict"].as_str().unwrap_or("NOT_RUN").to_string()),
        scope,
    })
}

// ---------------------------------------------------------------- CLI

pub const USAGE: &str = "usage:
  aien-test keygen KEY_FILE
  aien-test bundle-sign BUNDLE.json --key KEY_FILE [--out SIGNED.json]
  aien-test verify-bundle BUNDLE.json --repo github.com/OWNER/REPO --commit SHA40 --pubkey FILE_OR_HEX [--require GATE[,GATE...]] [--require-full]
  --require-full: REJECTED unless the signed plan scope is full (an Alpha qualification)
exit: 0 PASS, 1 FAIL, 2 NOT_RUN or usage, 3 REJECTED";

/// Flags of the bundle commands that take no value.
const BOOL_FLAGS: [&str; 1] = ["--require-full"];

pub fn is_bundle_command(name: &str) -> bool {
    matches!(name, "keygen" | "bundle-sign" | "verify-bundle")
}

fn flag(argv: &[String], name: &str) -> Result<Option<String>, String> {
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it
                .next()
                .cloned()
                .map(Some)
                .ok_or_else(|| format!("{name} needs a value"));
        }
    }
    Ok(None)
}

fn read_json(path: &str) -> Result<Value, String> {
    let txt = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    serde_json::from_str(&txt).map_err(|e| format!("{path} is not JSON: {e}"))
}

/// `aien-test keygen|bundle-sign|verify-bundle ...`. Returns the exit code.
pub fn cli_main(argv: &[String]) -> i32 {
    match cli_inner(argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("aien-test: {e}");
            eprintln!("{USAGE}");
            2
        }
    }
}

fn cli_inner(argv: &[String]) -> Result<i32, String> {
    let name = argv.first().map(|s| s.as_str()).unwrap_or("");
    let positional: Vec<&String> = argv[1..]
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            !a.starts_with("--")
                && (*i == 0
                    || !argv[*i].starts_with("--")
                    || BOOL_FLAGS.contains(&argv[*i].as_str()))
        })
        .map(|(_, a)| a)
        .collect();
    match name {
        "keygen" => {
            let path = positional.first().ok_or("keygen needs KEY_FILE")?;
            let pk = keygen(Path::new(path))?;
            println!("pubkey {pk}");
            println!("key written to {path} (mode 0600); commit the pubkey as .aien/verify.pub, never the key");
            Ok(0)
        }
        "bundle-sign" => {
            let path = positional.first().ok_or("bundle-sign needs BUNDLE.json")?;
            let key_path = flag(&argv[1..], "--key")?.ok_or("bundle-sign needs --key KEY_FILE")?;
            let out = flag(&argv[1..], "--out")?;
            let bundle = read_json(path)?;
            let key = load_signing_key(Path::new(&key_path))?;
            let signed = sign(&bundle, &key).map_err(|r| format!("refusing to sign: {r}"))?;
            let bytes = canonical(&signed).map_err(|e| format!("{e}"))?;
            let digest = sha256_hex(&bytes);
            let target = out.unwrap_or_else(|| path.to_string());
            fs::write(&target, &bytes).map_err(|e| format!("cannot write {target}: {e}"))?;
            println!("signed {target} digest {digest}");
            Ok(0)
        }
        "verify-bundle" => {
            let path = positional
                .first()
                .ok_or("verify-bundle needs BUNDLE.json")?;
            let repository = flag(&argv[1..], "--repo")?.ok_or("verify-bundle needs --repo")?;
            let commit = flag(&argv[1..], "--commit")?.ok_or("verify-bundle needs --commit")?;
            let pubkey_arg = flag(&argv[1..], "--pubkey")?.ok_or("verify-bundle needs --pubkey")?;
            let required = flag(&argv[1..], "--require")?
                .map(|s| {
                    s.split(',')
                        .filter(|g| !g.is_empty())
                        .map(|g| g.to_string())
                        .collect()
                })
                .unwrap_or_default();
            let pubkey_hex = load_pubkey_hex(&pubkey_arg)?;
            let bundle = read_json(path)?;
            let expect = Expect {
                repository,
                commit,
                pubkey_hex,
                required,
                require_full: argv[1..].iter().any(|a| a == "--require-full"),
            };
            match verify(&bundle, &expect) {
                Ok(v) => {
                    let why = match (v.verdict.as_str(), v.gate_verdict()) {
                        ("PASS", "NOT_RUN") if v.required == 0 => {
                            " (no required checks: PASS only vacuously)".to_string()
                        }
                        ("PASS", "NOT_RUN") => {
                            format!(" (plan {})", v.plan_verdict.as_deref().unwrap_or("?"))
                        }
                        _ => String::new(),
                    };
                    println!(
                        "AIEN Verification: {}{why} ({} checks, {} required, {} from cache) bundle {}",
                        v.gate_verdict(), v.checks, v.required, v.cached, v.digest
                    );
                    Ok(match v.gate_verdict() {
                        "PASS" => 0,
                        "FAIL" => 1,
                        _ => 2,
                    })
                }
                Err(r) => {
                    println!("AIEN Verification: REJECTED {}", r);
                    Ok(3)
                }
            }
        }
        _ => Err(format!("unknown command {name:?}")),
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    const REPO: &str = "github.com/aien-dev/aien-architecture";
    const COMMIT: &str = "d0f0b95d344b4c6628c3d868346932fd247ce168";
    const H: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn entry(gate: &str, exactness: &str, required: bool, status: &str, verdict: &str) -> Value {
        let mut e = json!({ "gate": gate, "check_id": H, "exactness": exactness, "required": required,
            "status": status, "verdict": verdict, "backend": "spark" });
        match status {
            "run" => {
                e["receipt"] = json!(H);
            }
            "cached" => {
                e["resolution"] =
                    json!({ "resolved_from": H, "resolved_utc": "2026-10-04T10:00:00Z" });
            }
            _ => {}
        }
        e
    }

    fn unsigned(sk: &SigningKey, checks: Vec<Value>, verdict: &str) -> Value {
        json!({
            "schema": SCHEMA, "repository": REPO, "commit": COMMIT, "tree_digest": H, "graph_digest": H,
            "receipt_id_scheme": "sha256-canonical-json", "checks": checks, "verdict": verdict,
            "executor": { "id": fingerprint(&sk.verifying_key()), "class": "spark", "machine_digest": H },
            "started_utc": "2026-10-04T10:00:00Z", "finished_utc": "2026-10-04T10:05:00Z"
        })
    }

    fn good() -> (SigningKey, Value, Expect) {
        let sk = key(7);
        let b = unsigned(
            &sk,
            vec![
                entry("crumb.verify", "EXACT", true, "run", "PASS"),
                entry("crypto.native", "EXACT", true, "cached", "PASS"),
                entry("search.order", "ADVISORY", false, "run", "FAIL"),
            ],
            "PASS",
        );
        let signed = sign(&b, &sk).expect("sign");
        let expect = Expect {
            repository: REPO.to_string(),
            commit: COMMIT.to_string(),
            pubkey_hex: hex_encode(sk.verifying_key().as_bytes()),
            required: vec!["crumb.verify".to_string(), "crypto.native".to_string()],
            require_full: false,
        };
        (sk, signed, expect)
    }

    #[test]
    fn good_bundle_verifies_pass_and_advisory_fail_does_not_count() {
        let (_, b, e) = good();
        let v = verify(&b, &e).expect("verifies");
        assert_eq!(v.verdict, "PASS");
        assert_eq!((v.checks, v.required, v.cached), (3, 2, 1));
    }

    #[test]
    fn key_order_in_the_file_does_not_matter() {
        let (_, b, e) = good();
        // Re-encode with serde_json's own (unsorted, pretty) output: the verifier canonicalizes.
        let pretty = serde_json::to_string_pretty(&b).unwrap();
        let again: Value = serde_json::from_str(&pretty).unwrap();
        assert!(verify(&again, &e).is_ok());
    }

    #[test]
    fn t11_tampered_verdict_is_rejected() {
        let (_, mut b, e) = good();
        b["checks"][0]["verdict"] = json!("FAIL");
        b["verdict"] = json!("FAIL");
        assert_eq!(verify(&b, &e).unwrap_err(), Reject::BadSignature);
    }

    #[test]
    fn t11_tampered_single_byte_in_commit_is_rejected() {
        let (_, mut b, mut e) = good();
        let other = format!("e{}", &COMMIT[1..]);
        b["commit"] = json!(other);
        e.commit = other;
        assert_eq!(verify(&b, &e).unwrap_err(), Reject::BadSignature);
    }

    #[test]
    fn t12_bundle_from_another_commit_is_rejected() {
        let (_, b, mut e) = good();
        e.commit = format!("e{}", &COMMIT[1..]);
        assert!(matches!(
            verify(&b, &e).unwrap_err(),
            Reject::CommitMismatch { .. }
        ));
    }

    #[test]
    fn t13_bundle_from_another_repository_is_rejected() {
        let (_, b, mut e) = good();
        e.repository = "github.com/aien-dev/omega".to_string();
        assert!(matches!(
            verify(&b, &e).unwrap_err(),
            Reject::RepositoryMismatch { .. }
        ));
    }

    #[test]
    fn t18_forged_bundle_signed_with_another_key_is_rejected() {
        let (_, _, e) = good();
        let forger = key(9);
        let b = unsigned(
            &forger,
            vec![
                entry("crumb.verify", "EXACT", true, "run", "PASS"),
                entry("crypto.native", "EXACT", true, "run", "PASS"),
            ],
            "PASS",
        );
        let forged = sign(&b, &forger).unwrap();
        assert_eq!(verify(&forged, &e).unwrap_err(), Reject::UnknownKey);
        // Even if the forger pastes the real pubkey into the signature block, the signature fails.
        let mut pasted = forged.clone();
        pasted["signature"]["pubkey"] = json!(e.pubkey_hex);
        assert_eq!(verify(&pasted, &e).unwrap_err(), Reject::BadSignature);
    }

    #[test]
    fn t18_executor_id_must_be_the_key_fingerprint() {
        let sk = key(7);
        let mut b = unsigned(
            &sk,
            vec![entry("crumb.verify", "EXACT", true, "run", "PASS")],
            "PASS",
        );
        b["executor"]["id"] = json!(H);
        assert_eq!(sign(&b, &sk).unwrap_err(), Reject::UnknownExecutor);
    }

    #[test]
    fn unsigned_bundle_is_rejected() {
        let (sk, _, e) = good();
        let b = unsigned(
            &sk,
            vec![
                entry("crumb.verify", "EXACT", true, "run", "PASS"),
                entry("crypto.native", "EXACT", true, "run", "PASS"),
            ],
            "PASS",
        );
        assert!(matches!(verify(&b, &e).unwrap_err(), Reject::Schema(_)));
    }

    #[test]
    fn t5_missing_required_check_blocks_pass() {
        let (sk, _, e) = good();
        let b = sign(
            &unsigned(
                &sk,
                vec![entry("crumb.verify", "EXACT", true, "run", "PASS")],
                "PASS",
            ),
            &sk,
        )
        .unwrap();
        assert_eq!(
            verify(&b, &e).unwrap_err(),
            Reject::MissingRequiredCheck("crypto.native".to_string())
        );
    }

    #[test]
    fn t6_fail_propagates_to_the_bundle_verdict() {
        let entries = vec![
            (true, "run".to_string(), "PASS".to_string()),
            (true, "run".to_string(), "FAIL".to_string()),
        ];
        assert_eq!(compute_verdict(&entries), "FAIL");
        let (sk, _, _) = good();
        let b = unsigned(
            &sk,
            vec![
                entry("crumb.verify", "EXACT", true, "run", "PASS"),
                entry("crypto.native", "EXACT", true, "run", "FAIL"),
            ],
            "PASS",
        );
        assert!(matches!(
            sign(&b, &sk).unwrap_err(),
            Reject::VerdictInconsistent { .. }
        ));
    }

    #[test]
    fn t7_not_run_never_becomes_pass() {
        let entries = vec![
            (true, "run".to_string(), "PASS".to_string()),
            (true, "not_run".to_string(), "NOT_RUN".to_string()),
        ];
        assert_eq!(compute_verdict(&entries), "NOT_RUN");
        let blocked = vec![(true, "blocked".to_string(), "BLOCKED_HARDWARE".to_string())];
        assert_eq!(compute_verdict(&blocked), "NOT_RUN");
        // A not_run entry claiming PASS is malformed on its face.
        let (sk, _, _) = good();
        let b = unsigned(
            &sk,
            vec![entry("kernel.qemu", "EXACT", true, "not_run", "PASS")],
            "PASS",
        );
        assert!(matches!(sign(&b, &sk).unwrap_err(), Reject::Schema(_)));
        // And a worker cannot sign PASS over a NOT_RUN graph.
        let b = unsigned(
            &sk,
            vec![entry("kernel.qemu", "EXACT", true, "not_run", "NOT_RUN")],
            "PASS",
        );
        assert!(matches!(
            sign(&b, &sk).unwrap_err(),
            Reject::VerdictInconsistent { .. }
        ));
    }

    #[test]
    fn advisory_cannot_be_required_and_optional_entries_never_decide() {
        let (sk, _, _) = good();
        let b = unsigned(
            &sk,
            vec![entry("search.order", "ADVISORY", true, "run", "PASS")],
            "PASS",
        );
        assert!(matches!(sign(&b, &sk).unwrap_err(), Reject::Schema(_)));
        assert_eq!(
            compute_verdict(&[(false, "run".to_string(), "FAIL".to_string())]),
            "PASS"
        );
    }

    #[test]
    fn gate_required_list_overrides_a_bundle_that_marks_it_optional() {
        let (sk, _, e) = good();
        let b = sign(
            &unsigned(
                &sk,
                vec![
                    entry("crumb.verify", "EXACT", true, "run", "PASS"),
                    entry("crypto.native", "EXACT", false, "run", "FAIL"),
                ],
                "PASS",
            ),
            &sk,
        )
        .unwrap();
        assert!(matches!(verify(&b, &e).unwrap_err(), Reject::Schema(_)));
    }

    #[test]
    fn keygen_writes_0600_and_refuses_overwrite() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("aien-test-keygen-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("key");
        let pk = keygen(&p).unwrap();
        assert!(is_hex(&pk, 64));
        assert_eq!(
            fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(keygen(&p).is_err());
        let sk = load_signing_key(&p).unwrap();
        assert_eq!(hex_encode(sk.verifying_key().as_bytes()), pk);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hex_roundtrip_and_rejects_odd_or_bad() {
        assert_eq!(
            hex_decode(&hex_encode(&[0, 1, 254, 255])).unwrap(),
            vec![0, 1, 254, 255]
        );
        assert!(hex_decode("abc").is_none());
        assert!(hex_decode("zz").is_none());
    }

    #[test]
    fn gate_verdict_is_never_a_vacuous_pass() {
        let v = |verdict: &str, required: usize, plan: Option<&str>| Verified {
            verdict: verdict.to_string(),
            digest: String::new(),
            checks: required,
            required,
            cached: 0,
            plan_verdict: plan.map(str::to_string),
            scope: SCOPE_CHANGED.to_string(),
        };
        assert_eq!(v("PASS", 1, None).gate_verdict(), "PASS");
        assert_eq!(v("PASS", 1, Some("PASS")).gate_verdict(), "PASS");
        // Nothing required: PASS only vacuously.
        assert_eq!(v("PASS", 0, None).gate_verdict(), "NOT_RUN");
        assert_eq!(v("PASS", 0, Some("PASS")).gate_verdict(), "NOT_RUN");
        // The worker's plan says UNMAPPED or BLOCKED.
        assert_eq!(v("PASS", 2, Some("NOT_RUN")).gate_verdict(), "NOT_RUN");
        assert_eq!(
            v("PASS", 2, Some("BLOCKED_OPERATOR")).gate_verdict(),
            "NOT_RUN"
        );
        assert_eq!(v("FAIL", 2, Some("NOT_RUN")).gate_verdict(), "FAIL");
        assert_eq!(v("NOT_RUN", 2, Some("PASS")).gate_verdict(), "NOT_RUN");
    }

    /// `good()`'s required checks with a `plan` member of `scope` (None: a
    /// plan written before scopes existed).
    fn scoped(sk: &SigningKey, scope: Option<&str>) -> Value {
        let mut b = unsigned(
            sk,
            vec![
                entry("crumb.verify", "EXACT", true, "run", "PASS"),
                entry("crypto.native", "EXACT", true, "cached", "PASS"),
            ],
            "PASS",
        );
        b["plan"] = json!({ "verdict": "PASS", "reason": "", "unmapped": [], "removed": [], "blocked": null });
        if let Some(s) = scope {
            b["plan"]["scope"] = json!(s);
        }
        sign(&b, sk).expect("sign")
    }

    #[test]
    fn a_bundle_without_a_scope_is_changed_scope() {
        let (sk, old, e) = good();
        // No `plan` at all, and a `plan` written before scopes existed.
        assert_eq!(scope_of(&old), SCOPE_CHANGED);
        assert_eq!(verify(&old, &e).unwrap().scope, "changed");
        let b = scoped(&sk, None);
        assert_eq!(scope_of(&b), SCOPE_CHANGED);
        let v = verify(&b, &e).unwrap();
        assert_eq!((v.scope.as_str(), v.gate_verdict()), ("changed", "PASS"));
        let v = verify(&scoped(&sk, Some("full")), &e).unwrap();
        assert_eq!((v.scope.as_str(), v.gate_verdict()), ("full", "PASS"));
        // Any other scope is not a bundle this verifier reads.
        let mut bad = unsigned(&sk, vec![], "PASS");
        bad["plan"] = json!({ "verdict": "PASS", "scope": "everything" });
        assert!(matches!(sign(&bad, &sk).unwrap_err(), Reject::Schema(_)));
    }

    #[test]
    fn the_scope_is_signed() {
        let (sk, _, e) = good();
        let mut b = scoped(&sk, Some("changed"));
        b["plan"]["scope"] = json!("full");
        assert_eq!(verify(&b, &e).unwrap_err(), Reject::BadSignature);
        let mut b = scoped(&sk, None);
        b["plan"]["scope"] = json!("full");
        assert_eq!(verify(&b, &e).unwrap_err(), Reject::BadSignature);
        let mut b = scoped(&sk, Some("full"));
        b["plan"]["scope"] = json!("changed");
        assert_eq!(verify(&b, &e).unwrap_err(), Reject::BadSignature);
    }

    #[test]
    fn require_full_rejects_a_changed_scope_pass_with_exit_3() {
        let (sk, old, e) = good();
        let dir = crate::testutil::TempDir::new("verify-bundle-full");
        let write = |name: &str, b: &Value| {
            let p = dir.path().join(name);
            fs::write(&p, canonical(b).unwrap()).unwrap();
            p.display().to_string()
        };
        let changed = write("changed.json", &scoped(&sk, Some("changed")));
        let unscoped = write("unscoped.json", &old);
        let full = write("full.json", &scoped(&sk, Some("full")));
        let run = |path: &str, require_full: bool| {
            let mut argv = vec!["verify-bundle".to_string()];
            // The bool flag before the file must not swallow it.
            if require_full {
                argv.push("--require-full".to_string());
            }
            argv.extend(
                [
                    path,
                    "--repo",
                    REPO,
                    "--commit",
                    COMMIT,
                    "--pubkey",
                    &e.pubkey_hex,
                ]
                .map(str::to_string),
            );
            cli_main(&argv)
        };
        assert_eq!(run(&changed, false), 0);
        assert_eq!(run(&changed, true), 3);
        assert_eq!(run(&unscoped, true), 3);
        assert_eq!(run(&full, true), 0);
        assert_eq!(run(&full, false), 0);
        let mut strict = e.clone();
        strict.require_full = true;
        let r = verify(&scoped(&sk, Some("changed")), &strict).unwrap_err();
        assert_eq!(r, Reject::NotFullScope("changed".to_string()));
        assert_eq!(r.code(), "NOT_FULL_SCOPE");
        assert_eq!(
            format!("AIEN Verification: REJECTED {r}"),
            "AIEN Verification: REJECTED NOT_FULL_SCOPE: plan scope is changed, not a full-scope (Alpha) qualification"
        );
        assert_eq!(
            verify(&scoped(&sk, Some("full")), &strict).unwrap().scope,
            "full"
        );
    }
}
