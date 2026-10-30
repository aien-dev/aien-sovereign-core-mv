use std::fmt;

/// Every failure is terminal for the ring (fail closed).
#[derive(Debug)]
pub enum HeartbeatError {
    Io {
        context: String,
        source: std::io::Error,
    },
    ShortRead {
        expert: (u16, u16),
        wanted: usize,
        got: usize,
    },
    DigestMismatch {
        expert: (u16, u16),
    },
    Manifest(String),
    RingFailed,
    SlotBusy(usize),
    PayloadTooLarge {
        slot: usize,
        need: usize,
        cap: usize,
    },
}

impl fmt::Display for HeartbeatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "I/O error ({context}): {source}"),
            Self::ShortRead {
                expert,
                wanted,
                got,
            } => {
                write!(
                    f,
                    "short read for expert {expert:?}: wanted {wanted} got {got}"
                )
            }
            Self::DigestMismatch { expert } => write!(f, "digest mismatch for expert {expert:?}"),
            Self::Manifest(m) => write!(f, "manifest: {m}"),
            Self::RingFailed => write!(f, "ring has failed closed"),
            Self::SlotBusy(s) => write!(f, "slot {s} is already being filled"),
            Self::PayloadTooLarge { slot, need, cap } => {
                write!(f, "slot {slot}: need {need} bytes, capacity {cap}")
            }
        }
    }
}

impl std::error::Error for HeartbeatError {}

impl HeartbeatError {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}
