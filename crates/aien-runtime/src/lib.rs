//! AIEN Runtime: Canonical Single-Process Host for Persistent Branching Intelligence

pub mod allen_memory;
pub mod approved;
pub mod approved_auth;
pub mod approved_replay;
pub mod client;
pub mod compose;
pub mod compute_config;
pub mod context;
pub mod control;
pub mod cortex_mark;
pub mod destination;
pub mod effects;
pub mod generation;
pub mod kv_residency;
pub mod objective;
pub mod persona;
/// Headstart M3 (#537): opt-in provisional preparation and final commit fence.
#[cfg(feature = "provisional-prep")]
pub mod provisional;
pub mod requirements;
mod requirements_extract;
pub mod rsi;
pub mod sequence;
pub mod server;
pub mod shared_kv;
pub mod spine;
pub mod swarm;
pub mod trace_observer;
pub mod world;

/// Test-only: one slot per process for tests that open a compose home (omega allows 4 handles).
#[cfg(test)]
#[path = "../tests/support/home_guard.rs"]
mod home_guard;

pub use aien_kv_cache::create_shared_kv_manager;
pub use aien_scheduler::SchedulerConfig;
pub use client::*;
pub use compose::{
    authorize_desk_at_start_from_env, authorize_desk_setting, authorize_requires_desk_from_env,
    check_file_proposal, compose_budgets_from_env, compose_dir_from_env,
    compose_doc_max_tokens_from_env, parse_file_proposal, token_ids_sha256, ComposeBridge,
    ComposeBudgets, ComposeProposer, FileProposal, Generation, AUTHORIZE_DESK_ENV,
};
pub use context::*;
pub use control::*;
pub use rsi::*;
pub use sequence::*;
pub use server::*;
pub use spine::*;
pub use swarm::*;
pub use world::*;
