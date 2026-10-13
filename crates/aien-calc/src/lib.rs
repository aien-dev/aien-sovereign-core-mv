//! aien-calc: an exact-arithmetic calculator for AI sessions. Every answer is an exact rational
//! (never a float) and carries a receipt whose id is the BLAKE3 hash of its canonical bytes.
//!
//! The library has no I/O. See `docs/` in the design handoff for the language (`calc-lang v1`)
//! and receipt (`aien-calc-receipt` v1) contracts.

pub mod error;
pub mod eval;
pub mod format;
pub mod json;
pub mod lexer;
pub mod parser;
pub mod ratutil;
pub mod receipt;
pub mod render;
pub mod store;
pub mod sweep;
pub mod unit_eval;
pub mod units;
pub mod value;
pub mod verify;

pub use error::{CalcError, Refusal};
pub use json::Json;
pub use receipt::{eval, EvalOutcome, Receipt, BUILD_INFO, ENGINE_VERSION};
pub use value::Value;
