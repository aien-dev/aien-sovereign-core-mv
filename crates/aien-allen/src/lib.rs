//! ALLEN identity gate (aien-architecture ADR 0035, aienos ADR 0018).
//!
//! RESOLVE ONLY. This crate reads a kind-24 SubjectState object that Aienux
//! provisioned and exported; it never creates, encodes or edits one (I8: ALLEN
//! is never created implicitly). The logical agent identity, the machine id
//! (host pin record) and the model binding (deployment record) are three
//! separate records; KV state is in none of them.
//!
//! Opt-in: engaged only when `AIEN_ALLEN_SUBJECT` is set. When engaged every
//! refusal is fatal (see [`EXIT_REFUSED`]); there is never a silent fallback.
pub mod binding;
pub mod deployment;
pub mod resolve;
pub mod subject_v0;

pub use resolve::{gate, Context, Engagement, Gate, Refusal, Resolved, SubjectSource};

/// Environment variable naming the exported subject (a head object file, or a
/// directory of raw objects named `<object-id-hex>.bin`).
pub const ENV_SUBJECT: &str = "AIEN_ALLEN_SUBJECT";
/// One-time operator adoption: the expected logical agent id (64 hex).
pub const ENV_ADOPT: &str = "AIEN_ALLEN_ADOPT";
/// Process exit code when an engaged ALLEN gate refuses (EX_CONFIG).
pub const EXIT_REFUSED: i32 = 78;
/// The one log line when the gate is not engaged.
pub const NOT_ENGAGED_LINE: &str =
    "ALLEN: not engaged (AIEN_ALLEN_SUBJECT unset); no identity claim";

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}
