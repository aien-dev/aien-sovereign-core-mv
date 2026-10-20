//! The engine (ledger L312, 2026-10-20). Drake's picture: a car engine with two separate machines in one body,
//! prewired before the key turns.
//!
//! * [`schedule`]: the camshaft. One static table built once per model + placement plan: which part lands in
//!   which buffer at which beat, which buffer each beat fires on. Checked complete and acyclic on the host before
//!   anything runs. Both engines read it; nothing about placement is decided at run time.
//! * [`movement`]: the movement engine (intake). Delivers bytes along the schedule through the heartbeat ring
//!   (`aien-heartbeat`: O_DIRECT reads, BLAKE3-verified slots, fail-closed). It never computes.
//! * [`piston`]: the power engine (pistons). Fires on a buffer that is already full. The host piston is the exact
//!   per-expert reference forward (the same arithmetic as the resident reference, L190 gate 64/64); the chip piston
//!   is the next cut. It never fetches.
//! * [`timing`]: the timing diagram. Per beat: fill issued, fill done, fire start, fire done; the ignition gap is
//!   fire start minus fill done. Text table and a PNG, no outside tools.
pub mod movement;
pub mod piston;
pub mod plan;
pub mod schedule;
pub mod timing;

/// Qwen3-30B-A3B shapes (the interim model; the 35B differs: 40 layers, 256 experts, inter 512, documented in the plan).
pub const HIDDEN: usize = 2048;
pub const INTER: usize = 768;
pub const BLOCK: usize = 128;
/// Routed experts per layer of the 30B.
pub const EXPERTS: usize = 128;
