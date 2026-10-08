//! Every way a memory operation can say no, in plain words.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryRefusal {
    /// ALLEN is not engaged: there is no identity to attach memory to.
    NotEngaged,
    /// A stored record names another agent or root than the engaged identity.
    ForeignMemory(String),
    /// A stored file is damaged, has an unknown schema or field, is oversize, or breaks the chain.
    Damaged(String),
    /// A requested value is not allowed.
    Invalid(String),
    UnknownItem(String),
    /// The item was forgotten (or a forget is pending); it cannot be changed.
    ItemForgotten(String),
    NotAGoal(String),
    /// Nothing matched a forget request.
    NothingToForget,
    /// The item's key is missing or damaged, so its content cannot be read.
    KeyUnavailable(String),
    /// The ciphertext or the record data it is bound to was changed.
    Tampered(String),
    /// Another writer appended first.
    Conflict,
    Io(String),
    /// Test seam only (feature `fault-injection`): the write was stopped on purpose.
    Crashed(String),
}

impl MemoryRefusal {
    pub fn code(&self) -> &'static str {
        use MemoryRefusal::*;
        match self {
            NotEngaged => "not_engaged",
            ForeignMemory(_) => "foreign_memory",
            Damaged(_) => "damaged",
            Invalid(_) => "invalid",
            UnknownItem(_) => "unknown_item",
            ItemForgotten(_) => "item_forgotten",
            NotAGoal(_) => "not_a_goal",
            NothingToForget => "nothing_to_forget",
            KeyUnavailable(_) => "key_unavailable",
            Tampered(_) => "tampered",
            Conflict => "conflict",
            Io(_) => "io",
            Crashed(_) => "crashed",
        }
    }
}

impl fmt::Display for MemoryRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use MemoryRefusal::*;
        match self {
            NotEngaged => write!(f, "ALLEN is not engaged on this machine, so there is no identity to attach memory to. Nothing was changed."),
            ForeignMemory(w) => write!(f, "The saved memory belongs to a different ALLEN identity ({w}). It was not used or changed."),
            Damaged(w) => write!(f, "The saved memory is damaged or not understood ({w}). It was not replaced; ask the operator to inspect the memory folder."),
            Invalid(w) => write!(f, "Not allowed: {w}. Nothing was changed."),
            UnknownItem(i) => write!(f, "There is no memory item {i}. Nothing was changed."),
            ItemForgotten(i) => write!(f, "Item {i} was forgotten (or is being forgotten) and cannot be changed."),
            NotAGoal(i) => write!(f, "Item {i} is not a goal. Nothing was changed."),
            NothingToForget => write!(f, "No live memory matched. Nothing was forgotten."),
            KeyUnavailable(i) => write!(f, "The key for item {i} is missing or damaged, so its content cannot be read."),
            Tampered(i) => write!(f, "Item {i} failed its integrity check (ciphertext or bound data changed). It was not used."),
            Conflict => write!(f, "Another writer changed the memory first. Nothing was changed; try again."),
            Io(w) => write!(f, "Could not read or write the memory store: {w}."),
            Crashed(w) => write!(f, "write stopped on purpose (test): {w}"),
        }
    }
}

impl std::error::Error for MemoryRefusal {}
