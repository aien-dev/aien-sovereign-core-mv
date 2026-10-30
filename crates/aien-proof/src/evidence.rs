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

//! Canonical evidence receipts (EvidenceReceiptV1).
//!
//! A receipt is a structured proof object: exact source, exact artifacts,
//! exact machine, structured assertions, and links to earlier receipts, to
//! Crumb ledger events, and to the resource lease that protected the run.
//!
//! Identity rule: the receipt ID is the BLAKE3 hash of an explicit canonical
//! byte encoding defined in [`canonical_bytes`]. The human readable JSON
//! rendering is not canonical. Map ordering, local paths, string timestamp
//! formats, and Rust struct layout never affect the ID, because the canonical
//! form uses fixed field order, length prefixed strings, and big endian
//! integers throughout.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const SCHEMA: &str = "EvidenceReceiptV1";
pub const SCHEMA_VERSION: u32 = 1;
/// Domain separator for the canonical encoding.
pub const DOMAIN: &[u8] = b"AIEN_EVIDENCE_RECEIPT_V1";
/// Features a receipt may require. Verification rejects anything else.
const KNOWN_FEATURES: &[&str] = &["v1-base"];
/// Subdirectory of the proof board root holding receipts.
pub const RECEIPT_DIR: &str = "receipts";

/// Final result of a receipt. Missing data never collapses into `Pass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Pass,
    Fail,
    Blocked,
    Incomplete,
    Skipped,
}

impl Verdict {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "PASS" => Some(Verdict::Pass),
            "FAIL" => Some(Verdict::Fail),
            "BLOCKED" => Some(Verdict::Blocked),
            "INCOMPLETE" => Some(Verdict::Incomplete),
            "SKIPPED" => Some(Verdict::Skipped),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Blocked => "BLOCKED",
            Verdict::Incomplete => "INCOMPLETE",
            Verdict::Skipped => "SKIPPED",
        }
    }

    fn code(self) -> u8 {
        match self {
            Verdict::Pass => 0,
            Verdict::Fail => 1,
            Verdict::Blocked => 2,
            Verdict::Incomplete => 3,
            Verdict::Skipped => 4,
        }
    }

    fn from_code(b: u8) -> Option<Self> {
        match b {
            0 => Some(Verdict::Pass),
            1 => Some(Verdict::Fail),
            2 => Some(Verdict::Blocked),
            3 => Some(Verdict::Incomplete),
            4 => Some(Verdict::Skipped),
            _ => None,
        }
    }
}

/// Qualification tier. A QEMU proof is never physical qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    #[serde(rename = "TEST_ONLY_TRUST")]
    TestOnlyTrust,
    #[serde(rename = "HOST_TEST")]
    HostTest,
    #[serde(rename = "QEMU")]
    Qemu,
    #[serde(rename = "QEMU_SECURITY")]
    QemuSecurity,
    #[serde(rename = "MACHINE1_READ_ONLY")]
    Machine1ReadOnly,
    #[serde(rename = "MACHINE1_ATTENDED")]
    Machine1Attended,
    #[serde(rename = "MACHINE1_MUTATING")]
    Machine1Mutating,
    #[serde(rename = "PRODUCTION")]
    Production,
}

impl Tier {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "TEST_ONLY_TRUST" => Some(Tier::TestOnlyTrust),
            "HOST_TEST" => Some(Tier::HostTest),
            "QEMU" => Some(Tier::Qemu),
            "QEMU_SECURITY" => Some(Tier::QemuSecurity),
            "MACHINE1_READ_ONLY" => Some(Tier::Machine1ReadOnly),
            "MACHINE1_ATTENDED" => Some(Tier::Machine1Attended),
            "MACHINE1_MUTATING" => Some(Tier::Machine1Mutating),
            "PRODUCTION" => Some(Tier::Production),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Tier::TestOnlyTrust => "TEST_ONLY_TRUST",
            Tier::HostTest => "HOST_TEST",
            Tier::Qemu => "QEMU",
            Tier::QemuSecurity => "QEMU_SECURITY",
            Tier::Machine1ReadOnly => "MACHINE1_READ_ONLY",
            Tier::Machine1Attended => "MACHINE1_ATTENDED",
            Tier::Machine1Mutating => "MACHINE1_MUTATING",
            Tier::Production => "PRODUCTION",
        }
    }

    /// Rank used for minimum tier checks. Higher means stronger evidence.
    pub fn rank(self) -> u8 {
        match self {
            Tier::TestOnlyTrust => 0,
            Tier::HostTest => 1,
            Tier::Qemu => 2,
            Tier::QemuSecurity => 3,
            Tier::Machine1ReadOnly => 4,
            Tier::Machine1Attended => 5,
            Tier::Machine1Mutating => 6,
            Tier::Production => 7,
        }
    }

    pub fn is_hardware(self) -> bool {
        matches!(
            self,
            Tier::Machine1ReadOnly | Tier::Machine1Attended | Tier::Machine1Mutating
        )
    }

    pub fn is_physical(self) -> bool {
        self.is_hardware() || self == Tier::Production
    }
}

/// Hardware mutation classification. Descriptive, not authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mutation {
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "VOLATILE_ONLY")]
    VolatileOnly,
    #[serde(rename = "REMOVABLE_MEDIA_ONLY")]
    RemovableMediaOnly,
    #[serde(rename = "ONE_TIME_BOOT_SELECTION")]
    OneTimeBootSelection,
    #[serde(rename = "BOUNDED_TEST_REGION_WRITE")]
    BoundedTestRegionWrite,
    #[serde(rename = "BOOT_CONFIGURATION_CHANGE")]
    BootConfigurationChange,
    #[serde(rename = "TRUST_ROOT_CHANGE")]
    TrustRootChange,
    #[serde(rename = "TPM_POLICY_CHANGE")]
    TpmPolicyChange,
    #[serde(rename = "DESTRUCTIVE_STORAGE")]
    DestructiveStorage,
}

impl Mutation {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "NONE" => Some(Mutation::None),
            "VOLATILE_ONLY" => Some(Mutation::VolatileOnly),
            "REMOVABLE_MEDIA_ONLY" => Some(Mutation::RemovableMediaOnly),
            "ONE_TIME_BOOT_SELECTION" => Some(Mutation::OneTimeBootSelection),
            "BOUNDED_TEST_REGION_WRITE" => Some(Mutation::BoundedTestRegionWrite),
            "BOOT_CONFIGURATION_CHANGE" => Some(Mutation::BootConfigurationChange),
            "TRUST_ROOT_CHANGE" => Some(Mutation::TrustRootChange),
            "TPM_POLICY_CHANGE" => Some(Mutation::TpmPolicyChange),
            "DESTRUCTIVE_STORAGE" => Some(Mutation::DestructiveStorage),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mutation::None => "NONE",
            Mutation::VolatileOnly => "VOLATILE_ONLY",
            Mutation::RemovableMediaOnly => "REMOVABLE_MEDIA_ONLY",
            Mutation::OneTimeBootSelection => "ONE_TIME_BOOT_SELECTION",
            Mutation::BoundedTestRegionWrite => "BOUNDED_TEST_REGION_WRITE",
            Mutation::BootConfigurationChange => "BOOT_CONFIGURATION_CHANGE",
            Mutation::TrustRootChange => "TRUST_ROOT_CHANGE",
            Mutation::TpmPolicyChange => "TPM_POLICY_CHANGE",
            Mutation::DestructiveStorage => "DESTRUCTIVE_STORAGE",
        }
    }

    fn severity(self) -> u8 {
        match self {
            Mutation::None => 0,
            Mutation::VolatileOnly => 1,
            Mutation::RemovableMediaOnly => 2,
            Mutation::OneTimeBootSelection => 3,
            Mutation::BoundedTestRegionWrite => 4,
            Mutation::BootConfigurationChange => 5,
            Mutation::TrustRootChange => 6,
            Mutation::TpmPolicyChange => 6,
            Mutation::DestructiveStorage => 7,
        }
    }

    fn code(self) -> u8 {
        match self {
            Mutation::None => 0,
            Mutation::VolatileOnly => 1,
            Mutation::RemovableMediaOnly => 2,
            Mutation::OneTimeBootSelection => 3,
            Mutation::BoundedTestRegionWrite => 4,
            Mutation::BootConfigurationChange => 5,
            Mutation::TrustRootChange => 6,
            Mutation::TpmPolicyChange => 7,
            Mutation::DestructiveStorage => 8,
        }
    }

    fn from_code(b: u8) -> Option<Self> {
        match b {
            0 => Some(Mutation::None),
            1 => Some(Mutation::VolatileOnly),
            2 => Some(Mutation::RemovableMediaOnly),
            3 => Some(Mutation::OneTimeBootSelection),
            4 => Some(Mutation::BoundedTestRegionWrite),
            5 => Some(Mutation::BootConfigurationChange),
            6 => Some(Mutation::TrustRootChange),
            7 => Some(Mutation::TpmPolicyChange),
            8 => Some(Mutation::DestructiveStorage),
            _ => None,
        }
    }
}

/// One structured assertion. Console lines may exist, but this record is the
/// proof: stable ID, expected and observed values, outcome, and source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assertion {
    pub id: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub observed: String,
    pub pass: bool,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub note: String,
}

/// Reference to a Crumb ledger event recorded for the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerRef {
    pub index: u64,
    /// Hex of the ledger event hash.
    pub hash: String,
}

/// Binding to the exclusive resource lease that protected a hardware run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseRef {
    /// Hold ID (hex of the hold audit event hash).
    pub hold_id: String,
    /// Resource the lease covered, for example `machine-1`.
    pub resource: String,
}

/// Canonical evidence receipt, version 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    pub version: u32,
    /// BLAKE3 hex of the canonical encoding. Must recompute exactly.
    pub id: String,
    /// Free form procedure class, for example `seed-0b-qemu`.
    pub kind: String,
    pub tier: Tier,
    pub result: Verdict,
    /// Unix seconds. Metadata only; stored as an integer so no string
    /// clock format can affect identity.
    pub timestamp: u64,
    pub repo: String,
    /// Exact source commit, lowercase hex, 40 or 64 digits.
    pub commit: String,
    /// True when the source tree had uncommitted changes.
    pub dirty: bool,
    /// Compiler and flag identity, for example `rustc -vV` output.
    pub toolchain: String,
    /// Command or procedure identity, for example the joined command line.
    pub procedure: String,
    /// Machine or resource identity, for example `machine-1`.
    pub machine: String,
    /// Environment classification. Must equal the tier string, so a QEMU
    /// run can never be labeled as physical qualification.
    pub env_class: String,
    pub input_artifacts: Vec<String>,
    pub output_artifacts: Vec<String>,
    pub assertions: Vec<Assertion>,
    /// Immutable receipt IDs (hashes) this receipt depends on.
    pub dependencies: Vec<String>,
    pub declared_mutation: Mutation,
    pub observed_mutation: Mutation,
    #[serde(default)]
    pub authority: String,
    /// BLAKE3 hex of the complete captured output.
    pub output_digest: String,
    #[serde(default)]
    pub external_refs: Vec<String>,
    #[serde(default)]
    pub ledger: Option<LedgerRef>,
    #[serde(default)]
    pub lease: Option<LeaseRef>,
    /// Features the verifier must understand. Unknown entries fail closed.
    #[serde(default)]
    pub required_features: Vec<String>,
    /// Must stay empty. Any content fails verification.
    #[serde(default)]
    pub reserved: String,
}

/// Verification outcome for one receipt or chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub status: Verdict,
    pub reasons: Vec<String>,
}

impl Report {
    pub fn pass(reason: &str) -> Self {
        Report {
            status: Verdict::Pass,
            reasons: vec![reason.to_string()],
        }
    }

    pub fn fail(reasons: Vec<String>) -> Self {
        Report {
            status: Verdict::Fail,
            reasons,
        }
    }

    pub fn closed(status: Verdict, reasons: Vec<String>) -> Self {
        Report { status, reasons }
    }

    pub fn ok(&self) -> bool {
        self.status == Verdict::Pass
    }
}

fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_lower_hex(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn hex_to_bytes(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !is_hex(s) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = char::from(bytes[i]).to_digit(16)?;
        let lo = char::from(bytes[i + 1]).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

pub fn check_commit(commit: &str) -> Result<(), String> {
    if (commit.len() == 40 || commit.len() == 64) && is_lower_hex(commit) {
        Ok(())
    } else {
        Err(format!(
            "commit must be 40 or 64 lowercase hex digits, got {commit:?}"
        ))
    }
}

fn check_digest_list(name: &str, items: &[String]) -> Result<(), String> {
    for item in items {
        if let Some((algo, hex)) = item.split_once(':') {
            match algo {
                "blake3" | "sha256" | "sha512" => {}
                _ => return Err(format!("{name}: unknown digest algo in {item:?}")),
            }
            if hex.len() < 32 || hex.len() % 2 != 0 || !is_hex(hex) {
                return Err(format!("{name}: bad digest hex in {item:?}"));
            }
        } else if item.len() == 64 && is_hex(item) {
            // Bare 64 digit hex, read as BLAKE3.
        } else {
            return Err(format!(
                "{name}: digest must be <algo>:<hex> or 64 hex digits, got {item:?}"
            ));
        }
    }
    Ok(())
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn sorted_unique(items: &[String]) -> Vec<String> {
    let set: BTreeSet<&str> = items.iter().map(String::as_str).collect();
    set.into_iter().map(str::to_string).collect()
}

/// Build the canonical byte encoding. Every format violation is an error,
/// so malformed input fails closed before any hash is compared.
pub fn canonical_bytes(r: &Receipt) -> Result<Vec<u8>, String> {
    if r.schema != SCHEMA {
        return Err(format!("unknown receipt schema {:?}", r.schema));
    }
    if r.version != SCHEMA_VERSION {
        return Err(format!("unsupported receipt version {}", r.version));
    }
    for f in &r.required_features {
        if !KNOWN_FEATURES.contains(&f.as_str()) {
            return Err(format!("unknown required receipt feature {f:?}"));
        }
    }
    if !r.reserved.is_empty() {
        return Err("reserved field must be empty".to_string());
    }
    if r.kind.is_empty() {
        return Err("receipt kind must not be empty".to_string());
    }
    if r.repo.is_empty() {
        return Err("repo must not be empty".to_string());
    }
    check_commit(&r.commit)?;
    if r.procedure.is_empty() {
        return Err("procedure identity must not be empty".to_string());
    }
    if r.machine.is_empty() {
        return Err("machine identity must not be empty".to_string());
    }
    if r.env_class != r.tier.as_str() {
        return Err(format!(
            "env_class {:?} contradicts tier {:?}",
            r.env_class,
            r.tier.as_str()
        ));
    }
    check_digest_list("input_artifacts", &r.input_artifacts)?;
    check_digest_list("output_artifacts", &r.output_artifacts)?;
    if r.output_digest.len() != 64 || !is_hex(&r.output_digest) {
        return Err("output_digest must be 64 hex digits (BLAKE3)".to_string());
    }
    let output_digest = hex_to_bytes(&r.output_digest).ok_or("bad output_digest")?;
    if r.assertions.is_empty() && r.result == Verdict::Pass {
        return Err("PASS receipts must carry at least one assertion".to_string());
    }
    let mut seen_assertions = BTreeSet::new();
    for a in &r.assertions {
        if a.id.is_empty() {
            return Err("assertion id must not be empty".to_string());
        }
        if !seen_assertions.insert(a.id.clone()) {
            return Err(format!("duplicate assertion id {:?}", a.id));
        }
    }
    for d in &r.dependencies {
        if d.len() != 64 || !is_hex(d) {
            return Err(format!("dependency {d:?} is not a 64 digit receipt hash"));
        }
    }
    if r.observed_mutation.severity() > r.declared_mutation.severity() {
        return Err(format!(
            "observed mutation {} exceeds declared mutation {}",
            r.observed_mutation.as_str(),
            r.declared_mutation.as_str()
        ));
    }
    check_tier_mutation(r.tier, r.observed_mutation)?;
    if r.tier == Tier::Production {
        let low = r.authority.to_ascii_lowercase();
        if r.authority.trim().is_empty() {
            return Err("PRODUCTION receipts require an authority reference".to_string());
        }
        if low.contains("test-only") || low.contains("test_only") || low.contains("testonly") {
            return Err("a TEST_ONLY signer cannot produce a PRODUCTION receipt".to_string());
        }
    }
    if r.tier.is_hardware() && r.lease.is_none() {
        // Not a canonical encoding failure; verification reports INCOMPLETE.
    }
    if let Some(ledger) = &r.ledger {
        if ledger.hash.len() != 64 || !is_hex(&ledger.hash) {
            return Err("ledger hash must be 64 hex digits".to_string());
        }
    }
    if let Some(lease) = &r.lease {
        if lease.hold_id.len() != 64 || !is_hex(&lease.hold_id) {
            return Err("lease hold_id must be 64 hex digits".to_string());
        }
        if lease.resource.trim().is_empty() {
            return Err("lease resource must not be empty".to_string());
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(DOMAIN);
    push_u32(&mut out, SCHEMA_VERSION);
    push_str(&mut out, &r.kind);
    push_str(&mut out, r.tier.as_str());
    out.push(r.result.code());
    push_str(&mut out, &r.repo);
    push_str(&mut out, &r.commit);
    out.push(u8::from(r.dirty));
    push_str(&mut out, &r.toolchain);
    push_str(&mut out, &r.procedure);
    push_str(&mut out, &r.machine);
    push_str(&mut out, &r.env_class);
    let inputs = sorted_unique(&r.input_artifacts);
    push_u32(&mut out, inputs.len() as u32);
    for item in &inputs {
        push_str(&mut out, item);
    }
    let outputs = sorted_unique(&r.output_artifacts);
    push_u32(&mut out, outputs.len() as u32);
    for item in &outputs {
        push_str(&mut out, item);
    }
    let mut assertions = r.assertions.clone();
    assertions.sort_by(|a, b| a.id.cmp(&b.id));
    push_u32(&mut out, assertions.len() as u32);
    for a in &assertions {
        push_str(&mut out, &a.id);
        push_str(&mut out, &a.expected);
        push_str(&mut out, &a.observed);
        out.push(u8::from(a.pass));
        push_str(&mut out, &a.source);
        push_str(&mut out, &a.note);
    }
    let deps = sorted_unique(&r.dependencies);
    push_u32(&mut out, deps.len() as u32);
    for d in &deps {
        let raw = hex_to_bytes(d).ok_or("bad dependency hash")?;
        out.extend_from_slice(&raw);
    }
    out.push(r.declared_mutation.code());
    out.push(r.observed_mutation.code());
    push_str(&mut out, &r.authority);
    out.extend_from_slice(&output_digest);
    let refs = sorted_unique(&r.external_refs);
    push_u32(&mut out, refs.len() as u32);
    for item in &refs {
        push_str(&mut out, item);
    }
    match &r.ledger {
        Some(ledger) => {
            out.push(1);
            push_u64(&mut out, ledger.index);
            let raw = hex_to_bytes(&ledger.hash).ok_or("bad ledger hash")?;
            out.extend_from_slice(&raw);
        }
        None => out.push(0),
    }
    match &r.lease {
        Some(lease) => {
            out.push(1);
            let normalized = lease.hold_id.to_ascii_lowercase();
            let raw = hex_to_bytes(&normalized).ok_or("bad hold id")?;
            out.extend_from_slice(&raw);
            push_str(&mut out, &lease.resource);
        }
        None => out.push(0),
    }
    push_u64(&mut out, r.timestamp);
    let feats = sorted_unique(&r.required_features);
    push_u32(&mut out, feats.len() as u32);
    for f in &feats {
        push_str(&mut out, f);
    }
    push_u32(&mut out, 0);
    Ok(out)
}

/// Maximum observed hardware mutation allowed per tier.
fn check_tier_mutation(tier: Tier, observed: Mutation) -> Result<(), String> {
    let max = match tier {
        Tier::Machine1ReadOnly => Mutation::RemovableMediaOnly.severity(),
        Tier::Machine1Attended => Mutation::BootConfigurationChange.severity(),
        Tier::Machine1Mutating | Tier::Production => u8::MAX,
        _ => Mutation::BoundedTestRegionWrite.severity(),
    };
    if observed.severity() > max {
        return Err(format!(
            "tier {} cannot claim observed mutation {}",
            tier.as_str(),
            observed.as_str()
        ));
    }
    if tier == Tier::Machine1ReadOnly
        && matches!(
            observed,
            Mutation::BoundedTestRegionWrite
                | Mutation::BootConfigurationChange
                | Mutation::TrustRootChange
                | Mutation::TpmPolicyChange
                | Mutation::DestructiveStorage
        )
    {
        return Err(format!(
            "a read-only Machine 1 run cannot claim mutation {}",
            observed.as_str()
        ));
    }
    Ok(())
}

/// Compute the canonical receipt ID.
pub fn receipt_id(r: &Receipt) -> Result<String, String> {
    let bytes = canonical_bytes(r)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.saturating_add(n);
        if end > self.bytes.len() {
            return Err("canonical encoding truncated".to_string());
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32_be(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64_be(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn text(&mut self) -> Result<String, String> {
        let len = self.u64_be()? as usize;
        if len > 16 * 1024 * 1024 {
            return Err("canonical string too long".to_string());
        }
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| "canonical string not UTF-8".to_string())
    }

    fn count(&mut self) -> Result<usize, String> {
        let n = self.u32_be()? as usize;
        if n > 100_000 {
            return Err("canonical list too long".to_string());
        }
        Ok(n)
    }

    fn hash_hex(&mut self) -> Result<String, String> {
        let raw = self.take(32)?;
        Ok(raw.iter().map(|b| format!("{b:02x}")).collect())
    }
}

/// Strictly decode canonical bytes back to a receipt. Anything malformed,
/// truncated, extended with trailing bytes, carrying unknown enum codes,
/// unknown features, or nonzero reserved content fails closed. The decoded
/// receipt's ID is recomputed from the bytes.
pub fn decode_canonical(bytes: &[u8]) -> Result<Receipt, String> {
    let mut c = Cursor { bytes, pos: 0 };
    if c.take(DOMAIN.len())? != DOMAIN {
        return Err("bad canonical domain separator".to_string());
    }
    if c.u32_be()? != SCHEMA_VERSION {
        return Err("unsupported canonical version".to_string());
    }
    let kind = c.text()?;
    let tier = Tier::parse(&c.text()?).ok_or("unknown tier in canonical encoding")?;
    let result = Verdict::from_code(c.byte()?).ok_or("unknown result in canonical encoding")?;
    let repo = c.text()?;
    let commit = c.text()?;
    let dirty = match c.byte()? {
        0 => false,
        1 => true,
        _ => return Err("bad dirty flag in canonical encoding".to_string()),
    };
    let toolchain = c.text()?;
    let procedure = c.text()?;
    let machine = c.text()?;
    let env_class = c.text()?;
    let mut input_artifacts = Vec::new();
    for _ in 0..c.count()? {
        input_artifacts.push(c.text()?);
    }
    let mut output_artifacts = Vec::new();
    for _ in 0..c.count()? {
        output_artifacts.push(c.text()?);
    }
    let mut assertions = Vec::new();
    for _ in 0..c.count()? {
        assertions.push(Assertion {
            id: c.text()?,
            expected: c.text()?,
            observed: c.text()?,
            pass: match c.byte()? {
                0 => false,
                1 => true,
                _ => return Err("bad assertion result in canonical encoding".to_string()),
            },
            source: c.text()?,
            note: c.text()?,
        });
    }
    let mut dependencies = Vec::new();
    for _ in 0..c.count()? {
        dependencies.push(c.hash_hex()?);
    }
    let declared_mutation =
        Mutation::from_code(c.byte()?).ok_or("unknown mutation in canonical encoding")?;
    let observed_mutation =
        Mutation::from_code(c.byte()?).ok_or("unknown mutation in canonical encoding")?;
    let authority = c.text()?;
    let output_digest = c.hash_hex()?;
    let mut external_refs = Vec::new();
    for _ in 0..c.count()? {
        external_refs.push(c.text()?);
    }
    let ledger = match c.byte()? {
        0 => None,
        1 => Some(LedgerRef {
            index: c.u64_be()?,
            hash: c.hash_hex()?,
        }),
        _ => return Err("bad ledger flag in canonical encoding".to_string()),
    };
    let lease = match c.byte()? {
        0 => None,
        1 => {
            let hold_id = c.hash_hex()?;
            let resource = c.text()?;
            Some(LeaseRef { hold_id, resource })
        }
        _ => return Err("bad lease flag in canonical encoding".to_string()),
    };
    let timestamp = c.u64_be()?;
    let mut required_features = Vec::new();
    for _ in 0..c.count()? {
        required_features.push(c.text()?);
    }
    if c.u32_be()? != 0 {
        return Err("reserved field must be zero".to_string());
    }
    if c.pos != bytes.len() {
        return Err("trailing bytes after canonical encoding".to_string());
    }
    let mut receipt = Receipt {
        schema: SCHEMA.to_string(),
        version: SCHEMA_VERSION,
        id: String::new(),
        kind,
        tier,
        result,
        timestamp,
        repo,
        commit,
        dirty,
        toolchain,
        procedure,
        machine,
        env_class,
        input_artifacts,
        output_artifacts,
        assertions,
        dependencies,
        declared_mutation,
        observed_mutation,
        authority,
        output_digest,
        external_refs,
        ledger,
        lease,
        required_features,
        reserved: String::new(),
    };
    // Re-encode to confirm the bytes were canonical and to mint the ID.
    // Canonical encoding also re-applies every semantic check.
    let reencoded = canonical_bytes(&receipt)?;
    if reencoded != bytes {
        return Err("canonical encoding not in canonical order".to_string());
    }
    receipt.id = blake3::hash(bytes).to_hex().to_string();
    Ok(receipt)
}

/// Parse receipt JSON and check identity plus internal consistency.
/// Ledger and lease stores are checked separately by [`verify_with_store`].
pub fn verify_bytes(text: &[u8]) -> Result<(Receipt, Report), String> {
    let receipt: Receipt =
        serde_json::from_slice(text).map_err(|e| format!("receipt JSON invalid: {e}"))?;
    let computed = receipt_id(&receipt)?;
    if receipt.id.to_ascii_lowercase() != computed {
        return Err(format!(
            "receipt id mismatch: file claims {}, canonical identity is {}",
            receipt.id, computed
        ));
    }
    Ok((receipt.clone(), check_consistency(&receipt)))
}

/// Consistency rules that need no store access.
fn check_consistency(r: &Receipt) -> Report {
    let mut problems = Vec::new();
    if r.result == Verdict::Pass {
        if r.assertions.iter().any(|a| !a.pass) {
            problems.push("result is PASS but a structured assertion failed".to_string());
        }
        if r.tier.is_hardware() && r.lease.is_none() {
            return Report::closed(
                Verdict::Incomplete,
                vec![
                    "hardware receipt has no resource lease binding; run under `aien-proof hold` and reference the hold".to_string(),
                ],
            );
        }
    }
    if r.result == Verdict::Blocked && r.tier.is_physical() {
        problems.push("a run marked BLOCKED cannot close a gate".to_string());
    }
    if problems.is_empty() {
        match r.result {
            Verdict::Pass => Report::pass("identity matches, assertions all pass"),
            Verdict::Fail => {
                Report::closed(Verdict::Fail, vec!["receipt records FAIL".to_string()])
            }
            Verdict::Blocked => Report::closed(
                Verdict::Blocked,
                vec!["receipt records BLOCKED".to_string()],
            ),
            Verdict::Incomplete => Report::closed(
                Verdict::Incomplete,
                vec!["receipt records INCOMPLETE".to_string()],
            ),
            Verdict::Skipped => Report::closed(
                Verdict::Skipped,
                vec!["receipt records SKIPPED".to_string()],
            ),
        }
    } else {
        Report::fail(problems)
    }
}

/// Full single receipt verification against a proof store root:
/// identity, consistency, ledger event, and lease binding.
pub fn verify_with_store(text: &[u8], store: &Path) -> Result<(Receipt, Report), String> {
    let (receipt, mut report) = verify_bytes(text)?;
    if report.status == Verdict::Fail {
        return Ok((receipt, report));
    }
    if let Some(problem) = crate::ledger_bind::check_ledger_ref(&receipt, store)? {
        return Ok((receipt, Report::fail(vec![problem])));
    }
    if let Some(report2) = crate::lease::check_lease_ref(&receipt, store)? {
        if report2.status != Verdict::Pass {
            return Ok((receipt, report2));
        }
    }
    if report.status == Verdict::Pass {
        // Recompute the consistency PASS line with store checks included.
        report = Report::pass("identity, ledger, and lease bindings check out");
        if receipt.result != Verdict::Pass {
            report = check_consistency(&receipt);
        }
    }
    Ok((receipt, report))
}

/// Load a receipt from the store by ID. A file whose bytes hash to a
/// different ID is a duplicate ID collision and fails closed.
pub fn load_receipt(store: &Path, id: &str) -> Result<Receipt, String> {
    let normalized = id.to_ascii_lowercase();
    if normalized.len() != 64 || !is_hex(&normalized) {
        return Err(format!("receipt id {id:?} is not a 64 digit hash"));
    }
    let path = store.join(RECEIPT_DIR).join(format!("{normalized}.json"));
    let text = fs::read(&path).map_err(|_| format!("missing dependency receipt {normalized}"))?;
    let (receipt, _) = verify_bytes(&text)?;
    if receipt.id.to_ascii_lowercase() != normalized {
        return Err(format!(
            "receipt file {normalized} holds bytes for receipt {}",
            receipt.id
        ));
    }
    Ok(receipt)
}

/// Write a receipt into the store under its canonical ID.
pub fn store_receipt(store: &Path, receipt: &Receipt) -> Result<String, String> {
    let computed = receipt_id(receipt)?;
    if receipt.id.to_ascii_lowercase() != computed {
        return Err(format!(
            "refusing to store: id {} does not match canonical {computed}",
            receipt.id
        ));
    }
    let dir = store.join(RECEIPT_DIR);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{computed}.json"));
    let mut owned = receipt.clone();
    owned.id = computed.clone();
    let text = serde_json::to_string_pretty(&owned).map_err(|e| e.to_string())?;
    fs::write(&path, format!("{text}\n")).map_err(|e| e.to_string())?;
    Ok(computed)
}

/// List all receipt IDs present in the store.
pub fn list_receipts(store: &Path) -> Vec<String> {
    let mut ids = Vec::new();
    let dir = store.join(RECEIPT_DIR);
    let entries = fs::read_dir(dir).into_iter().flatten().flatten();
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(id) = name.strip_suffix(".json") {
            if id.len() == 64 && is_hex(id) {
                ids.push(id.to_ascii_lowercase());
            }
        }
    }
    ids.sort();
    ids
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    pub fn sample() -> Receipt {
        Receipt {
            schema: SCHEMA.to_string(),
            version: SCHEMA_VERSION,
            id: String::new(),
            kind: "seed-0b-qemu".to_string(),
            tier: Tier::Qemu,
            result: Verdict::Pass,
            timestamp: 1_786_000_000,
            repo: "https://github.com/aien-dev/aienux".to_string(),
            commit: "a".repeat(40),
            dirty: false,
            toolchain: "rustc 1.90.0".to_string(),
            procedure: "scripts/qemu_seed0b.sh".to_string(),
            machine: "qemu-virt-aarch64".to_string(),
            env_class: "QEMU".to_string(),
            input_artifacts: vec!["blake3:".to_string() + &"b".repeat(64)],
            output_artifacts: vec![],
            assertions: vec![Assertion {
                id: "artifact_signature_valid".to_string(),
                expected: "valid".to_string(),
                observed: "valid".to_string(),
                pass: true,
                source: "qemu_seed0b.sh".to_string(),
                note: String::new(),
            }],
            dependencies: vec![],
            declared_mutation: Mutation::None,
            observed_mutation: Mutation::None,
            authority: String::new(),
            output_digest: "c".repeat(64),
            external_refs: vec![],
            ledger: None,
            lease: None,
            required_features: vec![],
            reserved: String::new(),
        }
    }

    pub fn sealed(mut r: Receipt) -> Receipt {
        let id = receipt_id(&r).unwrap();
        r.id = id;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{sample, sealed};
    use super::*;

    #[test]
    fn identity_is_stable_and_order_independent() {
        let a = sealed(sample());
        let mut b = a.clone();
        b.input_artifacts
            .push("blake3:".to_string() + &"a".repeat(64));
        b.input_artifacts.swap(0, 1);
        b.assertions.push(Assertion {
            id: "aaa_first".to_string(),
            expected: "x".to_string(),
            observed: "x".to_string(),
            pass: true,
            source: "s".to_string(),
            note: String::new(),
        });
        let mut c = b.clone();
        c.assertions.swap(0, 1);
        // Different assertion sets differ; same set in any order matches.
        assert_ne!(receipt_id(&a).unwrap(), receipt_id(&b).unwrap());
        assert_eq!(receipt_id(&b).unwrap(), receipt_id(&c).unwrap());
        let _ = a;
    }

    #[test]
    fn altered_bytes_fail_closed() {
        let r = sealed(sample());
        let mut text = serde_json::to_vec(&r).unwrap();
        text[60] ^= 0x01;
        assert!(verify_bytes(&text).is_err());
    }

    #[test]
    fn changed_commit_changes_identity() {
        let mut r = sample();
        let before = receipt_id(&r).unwrap();
        r.commit = "d".repeat(40);
        assert_ne!(receipt_id(&r).unwrap(), before);
    }

    #[test]
    fn changed_artifact_hash_changes_identity() {
        let mut r = sample();
        let before = receipt_id(&r).unwrap();
        r.input_artifacts = vec!["blake3:".to_string() + &"d".repeat(64)];
        assert_ne!(receipt_id(&r).unwrap(), before);
    }

    #[test]
    fn changed_assertion_result_fails_pass_claim() {
        let mut r = sealed(sample());
        r.assertions[0].pass = false;
        // Identity changed, so reseal, then consistency must fail the PASS.
        let r = sealed(r);
        let text = serde_json::to_vec(&r).unwrap();
        let (_, report) = verify_bytes(&text).unwrap();
        assert_eq!(report.status, Verdict::Fail);
    }

    #[test]
    fn pass_without_assertions_is_rejected() {
        let mut r = sample();
        r.assertions.clear();
        assert!(receipt_id(&r).is_err());
    }

    #[test]
    fn unknown_required_feature_fails_closed() {
        let mut r = sample();
        r.required_features = vec!["v9-teleport".to_string()];
        assert!(receipt_id(&r).unwrap_err().contains("unknown required"));
    }

    #[test]
    fn reserved_content_fails_closed() {
        let mut r = sample();
        r.reserved = "x".to_string();
        assert!(receipt_id(&r).unwrap_err().contains("reserved"));
    }

    #[test]
    fn malformed_canonical_inputs_fail() {
        let mut r = sample();
        r.commit = "not-a-sha".to_string();
        assert!(receipt_id(&r).is_err());
        let mut r = sample();
        r.output_digest = "zz".to_string();
        assert!(receipt_id(&r).is_err());
        let mut r = sample();
        r.dependencies = vec!["nope".to_string()];
        assert!(receipt_id(&r).is_err());
    }

    #[test]
    fn test_only_signer_cannot_mint_production() {
        let mut r = sample();
        r.tier = Tier::Production;
        r.env_class = "PRODUCTION".to_string();
        r.authority = "test-only lab signer".to_string();
        assert!(receipt_id(&r).unwrap_err().contains("TEST_ONLY"));
        r.authority = String::new();
        assert!(receipt_id(&r).is_err());
    }

    #[test]
    fn observed_beyond_declared_fails() {
        let mut r = sample();
        r.tier = Tier::Machine1Mutating;
        r.env_class = "MACHINE1_MUTATING".to_string();
        r.declared_mutation = Mutation::VolatileOnly;
        r.observed_mutation = Mutation::BootConfigurationChange;
        assert!(receipt_id(&r).unwrap_err().contains("exceeds declared"));
    }

    #[test]
    fn read_only_run_cannot_claim_mutation() {
        let mut r = sample();
        r.tier = Tier::Machine1ReadOnly;
        r.env_class = "MACHINE1_READ_ONLY".to_string();
        r.declared_mutation = Mutation::BootConfigurationChange;
        r.observed_mutation = Mutation::BootConfigurationChange;
        assert!(receipt_id(&r).is_err());
    }

    #[test]
    fn hardware_receipt_without_lease_is_incomplete() {
        let mut r = sample();
        r.tier = Tier::Machine1Attended;
        r.env_class = "MACHINE1_ATTENDED".to_string();
        let r = sealed(r);
        let text = serde_json::to_vec(&r).unwrap();
        let (_, report) = verify_bytes(&text).unwrap();
        assert_eq!(report.status, Verdict::Incomplete);
    }

    #[test]
    fn duplicate_id_with_different_bytes_fails() {
        let r = sealed(sample());
        let mut tampered = r.clone();
        tampered.assertions[0].observed = "tampered".to_string();
        // Keep the original ID on different bytes.
        let text = serde_json::to_vec(&tampered).unwrap();
        // Rewriting the ID field back to the original still fails identity.
        let mut value: serde_json::Value = serde_json::from_slice(&text).unwrap();
        value["id"] = serde_json::Value::String(r.id.clone());
        let forged = serde_json::to_vec(&value).unwrap();
        assert!(verify_bytes(&forged).is_err());
    }

    #[test]
    fn env_tier_contradiction_rejected() {
        let mut r = sample();
        r.env_class = "MACHINE1_MUTATING".to_string();
        assert!(receipt_id(&r).unwrap_err().contains("contradicts"));
    }

    #[test]
    fn canonical_roundtrip() {
        let r = sealed(sample());
        let bytes = canonical_bytes(&r).unwrap();
        let back = decode_canonical(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn malformed_canonical_encoding_fails() {
        let r = sealed(sample());
        let bytes = canonical_bytes(&r).unwrap();
        // Truncated.
        assert!(decode_canonical(&bytes[..bytes.len() - 5]).is_err());
        // Bad domain.
        let mut bad = bytes.clone();
        bad[0] ^= 0xFF;
        assert!(decode_canonical(&bad).is_err());
        // Trailing bytes.
        let mut long = bytes.clone();
        long.push(0);
        assert!(decode_canonical(&long).unwrap_err().contains("trailing"));
        // Corruption inside a free form string still decodes structurally,
        // but the identity changes, so tampering is always visible.
        let mut kind_hit = bytes.clone();
        kind_hit[40] ^= 0x09;
        let altered = decode_canonical(&kind_hit).unwrap();
        assert_ne!(altered.id, r.id);
        assert_ne!(altered.kind, r.kind);
    }

    #[test]
    fn canonical_rejects_nonzero_reserved() {
        let r = sealed(sample());
        let mut bytes = canonical_bytes(&r).unwrap();
        // Reserved is the final u32; set it nonzero.
        let n = bytes.len();
        bytes[n - 1] = 1;
        assert!(decode_canonical(&bytes).unwrap_err().contains("reserved"));
    }
}
