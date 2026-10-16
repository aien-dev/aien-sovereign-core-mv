//! AIEN-TEST (ADR 0028), Slices A to C: gate manifests, verdict derivation,
//! immutable evidence receipts, a runner for one gate or a whole graph of
//! gates with resource pools, GPU locks, dependency handling and `why`, and
//! the digest cache that reuses a result only for an identical experiment.
//!
//! Tests report observations; this crate derives the verdict and writes an
//! immutable, content-addressed receipt. See `manifest`, `verdict`,
//! `evidence`, `process`, `runner`, `graph`, `resources`, `why`, `cache`.
//!
//! # Manifest v2 and check identity (Slice D, ADR 0033 Decisions 2 and 3)
//!
//! `manifest_version: 2` adds optional keys; a v1 manifest parses to the same
//! values as before and may not use them. Defaults in brackets.
//!
//! - `exactness`: `EXACT` [default], `BOUNDED`, `APPROXIMATE`, `ADVISORY`.
//! - `requires`: v1 values plus `portable`, `aarch64`, `nvrm`, `bare_metal`,
//!   `analog_eligible` [`host`; optional in v2]. One of host, qemu, gb10,
//!   nvrm, portable, aarch64 is needed.
//! - `tolerance` (child lines `metric`, `abs`, `rel_ppm`, integers only):
//!   required for `BOUNDED`, refused otherwise.
//! - `backends` (children `eligible: [..]`, `canonical: X`) from `portable,
//!   spark, qemu, gb10, analog_sim, analog_device`. [eligible implied by
//!   `requires`; canonical is spark, or portable when `requires` is only
//!   portable, or the first digital eligible one when spark is not eligible].
//! - `fallback` [`["spark"]`], `cache_scope`: `portable|arch|machine|hardware`
//!   [`machine`], `oracle` (child `golden: PATH` or `compare: GATE`).
//! - Refusals: tolerance without BOUNDED (and BOUNDED without tolerance);
//!   `analog_sim`/`analog_device` eligible, or `analog_eligible` required, for
//!   an EXACT check; an analog canonical backend for EXACT or BOUNDED; a
//!   canonical backend not in `eligible`; any unknown enum value or key.
//!
//! `identity::check_id` is sha256 over the canonical object
//! `{manifest_digest, input_blob_ids, compiler_identity, flags,
//! dependency_check_ids, golden_digests, scope_binding}`. `aien-test identity
//! GATE [--compiler-identity S] [--flag F]... [--hardware-identity S]` prints
//! `CHECK_ID` and the `OBJECT` it hashed (and the ids of the gates it rests
//! on).

pub mod bundle;
pub mod cache;
pub mod cli;
pub mod evidence;
pub mod graph;
pub mod identity;
pub mod idle;
pub mod manifest;
pub mod plan;
pub mod process;
pub mod resources;
pub mod runner;
pub mod sandbox;
pub mod status;
pub mod verdict;
pub mod verify;
pub mod why;
pub mod worker;

#[cfg(test)]
pub mod testutil;
