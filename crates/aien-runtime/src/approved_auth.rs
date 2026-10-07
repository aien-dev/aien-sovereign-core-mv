//! Approval authentication for the proposer hook (#249, part A).
//!
//! An [`crate::approved::ApprovedProposal`] carries `approval_mac`: an
//! HMAC-SHA256, keyed by the approval desk key, over the canonical binding of
//! every field the approval covers ([`ApprovalIdentity`]). The daemon holds
//! the same key (`<compose dir>/approval-desk.key`) and recomputes the MAC;
//! any field the caller changes (trace, request, approval id, approver, path,
//! content, approved proposal digest) or a MAC made with another key is
//! refused `PROPOSAL_REFUSED Unauthenticated`.
//!
//! What the MAC proves: the holder of the desk key approved exactly these
//! fields. It does not prove which human pressed approve; the desk (the
//! INTERPLANE host's approval continuation, host-only) names the approver and
//! the MAC stops anyone without the key from changing that name. Caller text
//! (model output, tool arguments, request JSON) is never approval: it cannot
//! carry a valid MAC without the key.
//!
//! Why a key and not the existing operator `authorization` grant: grant
//! records are `ComposeNote`s any same-user socket caller can append, so
//! "bound to a grant" would be caller text in two steps; the aien-mcp
//! `ApprovalGrant` lives in the INTERPLANE process memory and the daemon
//! cannot check it. The key is the smallest thing that separates "the desk
//! approved" from "the caller says it was approved".
//!
//! Key rules (checked on every load, so a replaced key takes effect at once):
//! a regular file (never a symlink, opened `O_NOFOLLOW`), owned by this
//! process's effective uid, no group or other permission bits, 64 hex digits.
//! The key is never printed, logged, recorded or reported; only
//! [`DeskKey::id`] (first 16 hex digits of sha256 of the key) is.
//! The compose home must not overlap the workspace the proposal targets (the
//! model-facing tools read inside the workspace), else `Confinement`.
//!
//! Rotation: replacing the key file makes every MAC made with the old key
//! fail (`Unauthenticated`), including approvals issued but not yet
//! submitted; the desk must approve again. Records already written keep the
//! old `desk_key_id`. Loss of the key: the daemon refuses every approved
//! proposal (`NoDesk`) until an operator creates a new one
//! (`aien compose desk-key`).
//!
//! Limit: the OS user stays the outer boundary. A process running as the
//! daemon's user that can read the compose home can read the key.
use aien_omega_compose::hex;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Version tag inside every binding.
pub const APPROVAL_BINDING_VERSION: &str = "aien.approval.v2";
/// File name of the desk key inside the compose home.
pub const DESK_KEY_FILE: &str = "approval-desk.key";

const REFUSED: &str = crate::approved::REFUSED;

fn refuse(name: &str, why: impl std::fmt::Display) -> String {
    format!("{REFUSED} {name}: {why}")
}

/// The fields one approval covers. `desk_key_id` names the key that
/// authenticated it; it is not caller input (the daemon fills it from the key
/// it holds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalIdentity {
    pub trace_id: String,
    pub request_id: String,
    pub approval_id: String,
    pub approver: String,
    pub path: String,
    pub content_sha256: String,
    pub approved_proposal_sha256: String,
    pub desk_key_id: String,
    /// The canonical absolute workspace the approval is for (476ca4 c17): the
    /// effect can land only here. The daemon canonicalises the submitted
    /// workspace and binds that, so a valid approval presented with another
    /// workspace fails the MAC (Unauthenticated, nothing consumed).
    pub workspace: String,
}

impl ApprovalIdentity {
    /// The identity of `p` for the canonical `workspace` under the desk key
    /// `desk_key_id`.
    pub fn of(p: &crate::approved::ApprovedProposal, workspace: &str, desk_key_id: &str) -> Self {
        Self {
            trace_id: p.trace_id.clone(),
            request_id: p.request_id.clone(),
            approval_id: p.approval_id.clone(),
            approver: p.approver.clone(),
            path: p.path.clone(),
            content_sha256: p.content_sha256.clone(),
            approved_proposal_sha256: p.approved_proposal_sha256.clone(),
            desk_key_id: desk_key_id.to_string(),
            workspace: workspace.to_string(),
        }
    }
}

/// The canonical absolute workspace path (symlinks resolved), as bound.
pub fn canonical_workspace(workspace: &Path) -> Result<String, String> {
    let c = std::fs::canonicalize(workspace)
        .map_err(|e| format!("workspace {}: {e}", workspace.display()))?;
    c.to_str()
        .map(str::to_string)
        .ok_or_else(|| format!("workspace {} is not UTF-8", c.display()))
}

/// Canonical binding: compact JSON, keys in sorted order, of the nine
/// identity fields plus `"v": APPROVAL_BINDING_VERSION`.
pub fn binding_bytes(id: &ApprovalIdentity) -> Vec<u8> {
    // serde_json's Map is a BTreeMap here (no preserve_order feature), so
    // keys serialize sorted; inserted sorted anyway.
    let mut m = Map::new();
    for (k, v) in [
        ("approval_id", &id.approval_id),
        ("approved_proposal_sha256", &id.approved_proposal_sha256),
        ("approver", &id.approver),
        ("content_sha256", &id.content_sha256),
        ("desk_key_id", &id.desk_key_id),
        ("path", &id.path),
        ("request_id", &id.request_id),
        ("trace_id", &id.trace_id),
        ("workspace", &id.workspace),
    ] {
        m.insert(k.into(), Value::String(v.clone()));
    }
    m.insert("v".into(), Value::String(APPROVAL_BINDING_VERSION.into()));
    Value::Object(m).to_string().into_bytes()
}

/// The durable replay key of an approval: sha256 (hex) of its binding.
pub fn approval_key(id: &ApprovalIdentity) -> String {
    hex(&Sha256::digest(binding_bytes(id)))
}

/// HMAC-SHA256 (RFC 2104) over sha2; no extra dependency.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const B: usize = 64;
    let mut k = [0u8; B];
    if key.len() > B {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; B];
    let mut opad = [0x5cu8; B];
    for i in 0..B {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let inner = Sha256::new()
        .chain_update(ipad)
        .chain_update(msg)
        .finalize();
    let out = Sha256::new()
        .chain_update(opad)
        .chain_update(inner)
        .finalize();
    let mut r = [0u8; 32];
    r.copy_from_slice(&out);
    r
}

/// Equal length and equal bytes, without an early exit.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim_end_matches('\n');
    // Canonical form only: 64 lowercase hex digits.
    if s.len() != 64
        || !s
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Path of the desk key in a compose home.
pub fn desk_key_path(compose_dir: &Path) -> PathBuf {
    compose_dir.join(DESK_KEY_FILE)
}

/// The approval desk key. `Debug` shows only the id; the bytes are zeroed on drop.
pub struct DeskKey {
    key: [u8; 32],
    id: String,
}

impl std::fmt::Debug for DeskKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeskKey {{ id: {} }}", self.id)
    }
}

impl Drop for DeskKey {
    fn drop(&mut self) {
        for b in self.key.iter_mut() {
            // SAFETY: a valid, aligned &mut u8.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
    }
}

fn euid() -> u32 {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

impl DeskKey {
    fn from_bytes(key: [u8; 32]) -> Self {
        let id = hex(&Sha256::digest(key))[..16].to_string();
        Self { key, id }
    }

    /// Load and check the key file (see the module rules). Errors are
    /// `PROPOSAL_REFUSED NoDesk: ...` and never contain key bytes.
    pub fn load(path: &Path) -> Result<Self, String> {
        let nodesk = |why: String| refuse("NoDesk", format!("{}: {why}", path.display()));
        let lmeta = std::fs::symlink_metadata(path).map_err(|e| nodesk(e.to_string()))?;
        if lmeta.file_type().is_symlink() {
            return Err(nodesk("is a symlink".into()));
        }
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .map_err(|e| nodesk(e.to_string()))?;
        // Checks on the opened file, not the path (no swap in between).
        let m = f.metadata().map_err(|e| nodesk(e.to_string()))?;
        if !m.file_type().is_file() {
            return Err(nodesk("not a regular file".into()));
        }
        if m.uid() != euid() {
            return Err(nodesk(format!(
                "owned by uid {}, not this process's uid {}",
                m.uid(),
                euid()
            )));
        }
        if m.mode() & 0o077 != 0 {
            return Err(nodesk(format!(
                "mode {:o} gives group or other access (must be 0600 or stricter)",
                m.mode() & 0o777
            )));
        }
        let mut s = String::new();
        f.take(256)
            .read_to_string(&mut s)
            .map_err(|e| nodesk(e.to_string()))?;
        let key = unhex32(&s).ok_or_else(|| nodesk("not 64 hex digits".into()))?;
        s.replace_range(.., &"0".repeat(s.len()));
        Ok(Self::from_bytes(key))
    }

    /// Create a new key file (exclusive create, mode 0600, `O_NOFOLLOW`) from
    /// /dev/urandom. Refuses when the file exists: rotation is delete + create
    /// by the operator, never an overwrite here.
    pub fn create(path: &Path) -> Result<Self, String> {
        let mut key = [0u8; 32];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut r| r.read_exact(&mut key))
            .map_err(|e| format!("desk key: /dev/urandom: {e}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("desk key: {}: {e}", parent.display()))?;
        }
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .map_err(|e| format!("desk key: create {}: {e}", path.display()))?;
        // umask can only remove bits; set 0600 explicitly anyway.
        f.set_permissions(std::fs::Permissions::from_mode(0o600))
            .and_then(|_| f.write_all(format!("{}\n", hex(&key)).as_bytes()))
            .and_then(|_| f.sync_all())
            .map_err(|e| format!("desk key: write {}: {e}", path.display()))?;
        Ok(Self::from_bytes(key))
    }

    /// First 16 hex digits of sha256(key): names the key, reveals nothing usable.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The MAC (hex) the desk attaches to `id`'s binding.
    pub fn mac(&self, id: &ApprovalIdentity) -> String {
        hex(&hmac_sha256(&self.key, &binding_bytes(id)))
    }

    /// Desk side: the MAC for a proposal under this key, for `workspace`
    /// (canonicalised here as the daemon will).
    pub fn sign(&self, p: &crate::approved::ApprovedProposal, workspace: &Path) -> String {
        let ws = canonical_workspace(workspace).unwrap_or_else(|_| workspace.display().to_string());
        self.mac(&ApprovalIdentity::of(p, &ws, &self.id))
    }

    /// Daemon side: Ok(identity) only when `p.approval_mac` is this key's MAC
    /// over `p`'s binding; else `PROPOSAL_REFUSED Unauthenticated`.
    pub fn authenticate(
        &self,
        p: &crate::approved::ApprovedProposal,
        workspace: &Path,
    ) -> Result<ApprovalIdentity, String> {
        let ws = canonical_workspace(workspace).map_err(|e| refuse("Unauthenticated", e))?;
        let id = ApprovalIdentity::of(p, &ws, &self.id);
        let want = hmac_sha256(&self.key, &binding_bytes(&id));
        let got = unhex32(&p.approval_mac);
        match got {
            Some(g) if ct_eq(&g, &want) => Ok(id),
            _ => Err(refuse(
                "Unauthenticated",
                format!(
                    "approval_mac is not the approval desk's (key {}) MAC over these fields",
                    self.id
                ),
            )),
        }
    }
}

/// Refuse when the compose home and the workspace overlap (either inside the
/// other): the model-facing tools read inside the workspace, and the desk key
/// and the journal must stay out of their reach.
pub fn check_confinement(compose_dir: &Path, workspace: &Path) -> Result<(), String> {
    let c = std::fs::canonicalize(compose_dir).map_err(|e| {
        refuse(
            "Confinement",
            format!("compose home {}: {e}", compose_dir.display()),
        )
    })?;
    let w = std::fs::canonicalize(workspace).map_err(|e| {
        refuse(
            "Confinement",
            format!("workspace {}: {e}", workspace.display()),
        )
    })?;
    if c.starts_with(&w) || w.starts_with(&c) {
        return Err(refuse(
            "Confinement",
            format!(
                "compose home {} and workspace {} overlap; the approval desk refuses to run",
                c.display(),
                w.display()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_matches_rfc4231_case_2() {
        // RFC 4231 test case 2.
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex(&mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn binding_is_sorted_compact_json() {
        let id = ApprovalIdentity {
            trace_id: "t".into(),
            request_id: "r".into(),
            approval_id: "a".into(),
            approver: "p".into(),
            path: "N.md".into(),
            content_sha256: "c".into(),
            approved_proposal_sha256: "s".into(),
            desk_key_id: "k".into(),
            workspace: "/w".into(),
        };
        assert_eq!(
            String::from_utf8(binding_bytes(&id)).unwrap(),
            r#"{"approval_id":"a","approved_proposal_sha256":"s","approver":"p","content_sha256":"c","desk_key_id":"k","path":"N.md","request_id":"r","trace_id":"t","v":"aien.approval.v2","workspace":"/w"}"#
        );
    }

    #[test]
    fn ct_eq_and_unhex() {
        assert!(ct_eq(b"ab", b"ab"));
        assert!(!ct_eq(b"ab", b"ac"));
        assert!(!ct_eq(b"ab", b"abc"));
        assert!(unhex32(&"0".repeat(63)).is_none());
        assert!(unhex32(&"zz".repeat(32)).is_none());
        assert_eq!(unhex32(&"ff".repeat(32)), Some([0xff; 32]));
        assert!(unhex32(&"FF".repeat(32)).is_none());
        assert!(unhex32(&format!(" {}", "f".repeat(63))).is_none());
    }
}
