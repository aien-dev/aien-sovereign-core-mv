//! ALLEN scoped memory store (aien-architecture#159).
//!
//! A SEPARATE record store `<home>.allen-memory/`, bound to the engaged ALLEN
//! identity (`aien_allen::Resolved`: agent + root; no second identity exists
//! here). Content is encrypted per item version; `forget` destroys the key.
//!
//! ENFORCED in code: scope separation at recall (a [`ScopeGrant`] names exactly
//! one scope and there is no scope-string path), the identity binding,
//! crash-safe appends, key destruction order, refusal of damaged or foreign
//! stores. NOT claimed: erasure everywhere. See RETENTION.md.
//!
//! Standing goals here are "host goal records, not subject intents": the host
//! never writes SubjectState (that needs an aienos ADR 0018 amendment).
pub mod refusal;
pub mod schema;
pub mod store;

pub use refusal::MemoryRefusal;
pub use schema::{Kind, Scope, GOAL_LABEL, MEMORY_SCHEMA};
pub use store::{
    memory_dir, Authority, ForgetTarget, GoalList, GoalView, InspectAll, ItemStatus, ItemView,
    Memory, Recall, RecallLimits, ScopeGrant, Step,
};
