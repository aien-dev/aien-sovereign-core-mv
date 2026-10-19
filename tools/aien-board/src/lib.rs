//! aien-board: repo agents and the change board (ledger L088, design in docs/REPO_AGENTS.md).
//!
//! - `measure`: what the Spark judge spends on gates a change cannot reach (evidence for L088).
//! - `contract`: a consumer's declaration of the producer interface it uses and the check that proves it.
//! - `board`: the change board, an append-only TSV on the Spark; producers post, consumers answer.
//! - `agent`: one repo agent pass: post own changes, answer other repos' posts with own checks only.
pub mod measure;
pub mod agent;
pub mod board;
pub mod contract;
pub mod probe;
