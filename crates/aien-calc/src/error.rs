//! Error types. A *syntax error* means the request was malformed (no receipt). A *refusal* means
//! the request was well formed but exact arithmetic declines to answer (a receipt is still issued,
//! with `exactness: refused`).

use std::fmt;

/// Why a request could not even be posed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalcError {
    Syntax { pos: usize, msg: String },
    Invalid(String),
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcError::Syntax { pos, msg } => write!(f, "syntax error at offset {pos}: {msg}"),
            CalcError::Invalid(msg) => write!(f, "invalid request: {msg}"),
        }
    }
}

impl std::error::Error for CalcError {}

/// An honest "no": the reason text is what the receipt records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal(pub String);

impl Refusal {
    pub fn new(msg: impl Into<String>) -> Self {
        Refusal(msg.into())
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

pub type Res<T> = Result<T, Refusal>;
