//! The Omegese guardrail: a [`Constraint`](aien_inference_abi::generation::Constraint) for the
//! generation controller (ADR 0004 section 2.3) that keeps a model's reply a prefix of a program
//! `omegese emit` accepts.
//!
//! A separately tested feature (Drake's guardrail decision, ADR 0004): nothing depends on this
//! crate by default, the controller with no constraint is unchanged, and the sparse output head
//! stays the default gather. The legal-token source is any [`ItemOracle`] (the contract is omega
//! `docs/one_layer/OMEGESE.md` section 6.2); [`SubprocessOracle`] runs `omegese next --machine`.
//! [`TokenTable`] holds the model vocabulary as bytes (Qwen's `tokenizer.json`, or a synthetic
//! table in tests). [`OmegeseConstraint`] forces the opening fence and the entry header, masks the
//! body, and maps `eof` to the closing fence plus the stop token; each part is switchable
//! ([`Framing`]).

pub mod constraint;
pub mod linked;
pub mod oracle;
pub mod signature;
pub mod table;
pub mod word;

pub use constraint::{Framing, OmegeseConstraint, Stats};
pub use linked::LinkedOracle;
pub use oracle::{parse_machine, Answer, Item, ItemOracle, SubprocessOracle, DEFAULT_TIMEOUT};
pub use signature::Signature;
pub use table::{Shape, TokenTable};
