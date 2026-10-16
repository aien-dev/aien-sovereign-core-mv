//! swarm-bridge: the thin live bridge between the synapse#83 swarm board and
//! running Claude Code sessions (sc#587).
//!
//! Claude Code runs `swarm-bridge hook` at SessionStart, SubagentStart,
//! UserPromptSubmit and PostToolUse. The hook reads the session identity from
//! stdin, takes the next bounded, deduplicated batch of board events
//! addressed to the session's tag or matching its subscriptions, and returns
//! it as `hookSpecificOutput.additionalContext` under an explicit untrusted
//! header. A batch counts as delivered once its batch tag is found in the
//! session transcript at the next boundary; only then does the board cursor
//! (ack) move. Agents post, ask, answer, ack and park from Bash with the same
//! binary. Notification never spends a model token; reading a delivered
//! message does.

#![forbid(unsafe_code)]

pub mod board;
pub mod bridge;
pub mod memory;
pub mod render;
pub mod state;

pub use board::{Board, BridgeError, Event, Kind, Page, Post, Ref, RefKind, Sub};
pub use bridge::{Bridge, Delivery, HookConfig, HookInput, PostReq, ReadOut, WatchEnd};
