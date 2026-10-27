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

//! The population mixer (sealed side).
//!
//! Crumbs are drawn from five secret populations with configurable weights
//! (initial v1 schedule 55/20/10/10/5, loaded from data, not hard-coded
//! behaviour). The mix adapts to evidence: as capabilities are demonstrated,
//! weight moves from acquisition to composition. Within a population the
//! mixer balances new capabilities, retention retests, previously failed
//! capabilities and random draws. Consecutive draws never repeat a family or
//! a capability, so curriculum order carries no ontology.
//!
//! The learner never sees a population, a family, or the draw reason.

use crate::gen::rng::Rng;
use crate::gen::{registry, FamilySpec};
use crate::provenance::clean_room_check;
use crate::sealed::{CapabilityId, Population, SourceFamily};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixerConfig {
    /// Percent weights per population. Need not sum to 100.
    pub populations: BTreeMap<Population, u32>,
    /// Within the fundamental population: weights for new / retention / uncertain / random.
    pub select_new: u32,
    pub select_retention: u32,
    pub select_uncertain: u32,
    pub select_random: u32,
    /// Fraction (percent) of acquisition weight moved to composition when every capability is demonstrated.
    pub adapt_pct: u32,
    /// Only families that pass the clean-room policy.
    pub clean_room: bool,
    /// Restrict capability-bearing families to these capability pool indices (None = all).
    pub capability_subset: Option<Vec<u16>>,
    /// Restrict composition families to these family keys (None = all ready ones).
    pub composition_keys: Option<Vec<String>>,
}

impl Default for MixerConfig {
    fn default() -> Self {
        let mut populations = BTreeMap::new();
        populations.insert(Population::Fundamental, 55);
        populations.insert(Population::Composition, 20);
        populations.insert(Population::Ambiguous, 10);
        populations.insert(Population::RabbitHole, 10);
        populations.insert(Population::Frontier, 5);
        MixerConfig {
            populations,
            select_new: 40,
            select_retention: 20,
            select_uncertain: 20,
            select_random: 20,
            adapt_pct: 50,
            clean_room: false,
            capability_subset: None,
            composition_keys: None,
        }
    }
}

impl MixerConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Evidence about one capability, updated from sealed evaluations.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CapabilityState {
    pub attempts: u32,
    pub verified: u32,
    pub last_verified: bool,
}

impl CapabilityState {
    pub fn demonstrated(&self) -> bool {
        self.verified > 0 && self.last_verified
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawReason {
    NewCapability,
    Retention,
    Uncertain,
    Random,
    Composition,
    Fallback,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Draw {
    pub seq: u64,
    pub family_id: u32,
    pub seed: u64,
    pub population: Population,
    pub reason: DrawReason,
}

pub struct Mixer {
    pub cfg: MixerConfig,
    rng: Rng,
    run_seed: u64,
    seq: u64,
    pub caps: HashMap<CapabilityId, CapabilityState>,
    last_family: Option<u32>,
    last_caps: Vec<CapabilityId>,
    /// Draws whose population had nothing eligible (empty, or only back-to-back repeats) and were redistributed.
    pub redistributed_draws: BTreeMap<Population, u32>,
}

fn population_of(f: &FamilySpec) -> Option<Population> {
    Some(match f.source_family {
        SourceFamily::Capability => Population::Fundamental,
        SourceFamily::Composition => Population::Composition,
        SourceFamily::Ambiguous => Population::Ambiguous,
        SourceFamily::RabbitHole => Population::RabbitHole,
        SourceFamily::Frontier => Population::Frontier,
        SourceFamily::CleanRoomPhysics => return None,
    })
}

impl Mixer {
    pub fn new(cfg: MixerConfig, run_seed: u64) -> Self {
        Mixer {
            cfg,
            rng: Rng::named("mixer", &[run_seed]),
            run_seed,
            seq: 0,
            caps: HashMap::new(),
            last_family: None,
            last_caps: vec![],
            redistributed_draws: BTreeMap::new(),
        }
    }

    fn family_caps(f: &FamilySpec) -> Vec<CapabilityId> {
        let pool = crate::gen::capability_pool();
        f.params.caps.iter().map(|&c| pool[c as usize].id).collect()
    }

    fn eligible(&self, f: &FamilySpec) -> bool {
        if self.cfg.clean_room && clean_room_check(&f.provenance).is_err() {
            return false;
        }
        if let Some(sub) = &self.cfg.capability_subset {
            if !f.params.caps.is_empty() && !f.params.caps.iter().all(|c| sub.contains(c)) {
                return false;
            }
        }
        if f.source_family == SourceFamily::Composition {
            if let Some(keys) = &self.cfg.composition_keys {
                if !keys.contains(&f.key) {
                    return false;
                }
            }
        }
        true
    }

    fn adjacent(&self, f: &FamilySpec) -> bool {
        if self.last_family == Some(f.id) {
            return true;
        }
        let caps = Self::family_caps(f);
        caps.iter().any(|c| self.last_caps.contains(c))
    }

    /// Current population weights after evidence-driven adaptation.
    pub fn weights(&self) -> BTreeMap<Population, u32> {
        let mut w = self.cfg.populations.clone();
        let pool = crate::gen::capability_pool();
        let tracked: Vec<_> = pool
            .iter()
            .filter(|c| {
                self.cfg
                    .capability_subset
                    .as_ref()
                    .is_none_or(|s| s.contains(&c.index))
            })
            .collect();
        if !tracked.is_empty() {
            let demo = tracked
                .iter()
                .filter(|c| self.caps.get(&c.id).is_some_and(|s| s.demonstrated()))
                .count();
            let fund = *w.get(&Population::Fundamental).unwrap_or(&0);
            let moved = fund as u64 * self.cfg.adapt_pct as u64 * demo as u64
                / (100 * tracked.len() as u64);
            *w.entry(Population::Fundamental).or_insert(0) -= moved as u32;
            *w.entry(Population::Composition).or_insert(0) += moved as u32;
        }
        w
    }

    fn pick_weighted<T: Copy>(&mut self, items: &[(T, u32)]) -> Option<T> {
        let total: u64 = items.iter().map(|x| x.1 as u64).sum();
        if total == 0 {
            return None;
        }
        let mut r = self.rng.below(total);
        for &(t, w) in items {
            if r < w as u64 {
                return Some(t);
            }
            r -= w as u64;
        }
        None
    }

    fn candidates(&self, pop: Population) -> Vec<&'static FamilySpec> {
        registry()
            .families
            .iter()
            .filter(|f| population_of(f) == Some(pop) && self.eligible(f))
            .collect()
    }

    fn cap_state(&self, id: &CapabilityId) -> CapabilityState {
        self.caps.get(id).copied().unwrap_or_default()
    }

    /// Choose a family within a population, or None if the population is empty.
    fn choose_in(&mut self, pop: Population) -> Option<(&'static FamilySpec, DrawReason)> {
        let all = self.candidates(pop);
        if all.is_empty() {
            return None;
        }
        let pools: Vec<(DrawReason, Vec<&'static FamilySpec>)> = match pop {
            Population::Fundamental => {
                let with_cap =
                    |pred: &dyn Fn(CapabilityState) -> bool| -> Vec<&'static FamilySpec> {
                        all.iter()
                            .copied()
                            .filter(|f| {
                                f.params.caps.len() == 1
                                    && pred(self.cap_state(&Self::family_caps(f)[0]))
                            })
                            .collect()
                    };
                vec![
                    (DrawReason::NewCapability, with_cap(&|s| s.attempts == 0)),
                    (DrawReason::Retention, with_cap(&|s| s.demonstrated())),
                    (
                        DrawReason::Uncertain,
                        with_cap(&|s| s.attempts > 0 && !s.demonstrated()),
                    ),
                    (DrawReason::Random, all.clone()),
                ]
            }
            Population::Composition => {
                let ready: Vec<_> = all
                    .iter()
                    .copied()
                    .filter(|f| {
                        Self::family_caps(f)
                            .iter()
                            .all(|c| self.cap_state(c).demonstrated())
                    })
                    .collect();
                vec![(DrawReason::Composition, ready)]
            }
            _ => vec![(DrawReason::Random, all.clone())],
        };
        // Never repeat a family or a capability back to back.
        let pools: Vec<(DrawReason, Vec<&'static FamilySpec>)> = pools
            .into_iter()
            .map(|(r, v)| (r, v.into_iter().filter(|f| !self.adjacent(f)).collect()))
            .collect();
        let weights = [
            self.cfg.select_new,
            self.cfg.select_retention,
            self.cfg.select_uncertain,
            self.cfg.select_random,
        ];
        let options: Vec<(usize, u32)> = pools
            .iter()
            .enumerate()
            .filter(|(_, (_, v))| !v.is_empty())
            .map(|(i, _)| (i, if pools.len() == 4 { weights[i] } else { 1 }))
            .collect();
        let i = self.pick_weighted(&options)?;
        let (reason, fams) = &pools[i];
        let f = fams[self.rng.below(fams.len() as u64) as usize];
        Some((f, *reason))
    }

    pub fn draw(&mut self) -> Draw {
        let seq = self.seq;
        self.seq += 1;
        let weights: Vec<(Population, u32)> = self.weights().into_iter().collect();
        let mut pop = self
            .pick_weighted(&weights)
            .unwrap_or(Population::Fundamental);
        let mut chosen = self.choose_in(pop);
        if chosen.is_none() {
            // Empty population (e.g. no v1 frontier sources, or nothing ready to
            // compose): record it and redistribute over the populations that can draw.
            *self.redistributed_draws.entry(pop).or_insert(0) += 1;
            let rest: Vec<(Population, u32)> =
                weights.iter().copied().filter(|(p, _)| *p != pop).collect();
            for _ in 0..8 {
                pop = self.pick_weighted(&rest).unwrap_or(Population::Fundamental);
                chosen = self.choose_in(pop);
                if chosen.is_some() {
                    break;
                }
            }
        }
        let (f, reason) = chosen.unwrap_or_else(|| (&registry().families[0], DrawReason::Fallback));
        self.last_family = Some(f.id);
        self.last_caps = Self::family_caps(f);
        let seed = Rng::named("draw-seed", &[self.run_seed, seq]).next_u64();
        Draw {
            seq,
            family_id: f.id,
            seed,
            population: pop,
            reason,
        }
    }

    /// Feed back a sealed verdict for a drawn family.
    pub fn record(&mut self, family: &FamilySpec, verified: bool) {
        if family.params.caps.len() == 1 {
            let id = Self::family_caps(family)[0];
            let s = self.caps.entry(id).or_default();
            s.attempts += 1;
            if verified {
                s.verified += 1;
            }
            s.last_verified = verified;
        }
    }
}
