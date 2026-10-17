// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Switchboard core (ADR 0030): one typed, deterministic router over our own
//! AI accounts.
//!
//! What lives here: the account list, the job description, a usage gauge per
//! account, the rule table the operator edits as a plain TOML file, the pure
//! routing decision, and the `Runner` trait with a scripted `FakeRunner`.
//!
//! What does not live here: login tokens (the crate never reads, copies or
//! stores any; an account only names a settings folder or a key reference
//! name), network calls, or process spawning. Real program adapters are a
//! later cut.
//!
//! All times are Unix seconds passed in by the caller, so every decision is a
//! pure function of its inputs.

pub mod account;
pub mod cli;
pub mod dispatch;
pub mod doctor;
pub mod gauge;
pub mod job;
pub mod launch;
pub mod router;
pub mod rules;
pub mod runner;

pub use account::{Account, AccountId, Provider};
pub use dispatch::{run_job, Completed, DispatchError};
pub use doctor::{doctor, FakeProbe, Finding, Level, Report, VersionProbe};
pub use gauge::{Gauge, Gauges, Window};
pub use job::{Job, JobKind, JobSize};
pub use launch::{launch, plan_launch, Env, FakeSpawner, LaunchError, LaunchPlan, Spawner};
pub use router::{route, Decision};
pub use rules::{RuleTable, RulesError};
pub use runner::{FakeRunner, RunResult, Runner};
