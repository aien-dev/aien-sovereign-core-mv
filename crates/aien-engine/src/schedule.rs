//! The camshaft: one static table, built once from (manifest, placement plan), read by both engines.
//!
//! Beat `b` fires on buffer `b % B` holding head(layer_b); during beat `b` the movement engine fills buffer
//! `(b + B - 1) % B` with head(layer_{b+B-1}) for beat `b + B - 1` (the pistons alternate when B = 2). Before beat 0 the
//! first `B - 1` buffers are primed. Every dependency edge points forward in beat order, so the graph is acyclic by
//! construction; [`Schedule::check`] proves it on the host and refuses a table that is incomplete, colliding or cyclic.
use crate::plan::PlacementPlan;
use aien_heartbeat::manifest::{ExpertEntry, Manifest, Segment};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Part {
    pub id: u32,
    pub layer: u16,
    pub expert: u16,
    pub bytes: u64,
    pub segments: Vec<Segment>,
    pub digest: String,
    /// Tensor name -> (offset within the expert payload, length).
    pub tensors: Vec<(String, u64, u64)>,
}

impl Part {
    pub fn tensor_offset(&self, name: &str) -> Result<u64, String> {
        self.tensors.iter().find(|t| t.0 == name).map(|t| t.1).ok_or_else(|| format!("{name}: not in part {}", self.id))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fill {
    /// Beat the filled buffer will be fired on.
    pub for_beat: u32,
    pub buffer: u32,
    /// Part ids in slot order: part `i` lands in slot `buffer * slots + i`.
    pub parts: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Beat {
    pub b: u32,
    pub token: u32,
    pub layer: u16,
    pub fire_buffer: u32,
    pub fire_parts: Vec<u32>,
    /// Fill issued during this beat (None at the tail, when nothing is left to deliver).
    pub fill: Option<Fill>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub schema: String,
    pub model: String,
    pub synthetic_plan: bool,
    pub tokens: u32,
    pub layers: Vec<u16>,
    pub buffers: u32,
    /// Slots per buffer = the largest head.
    pub slots: u32,
    pub slot_bytes: u64,
    pub parts: Vec<Part>,
    pub prime: Vec<Fill>,
    pub beats: Vec<Beat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckReport {
    pub beats: usize,
    pub parts: usize,
    pub fills: usize,
    pub bytes_total: u64,
    pub bytes_per_beat_min: u64,
    pub bytes_per_beat_max: u64,
    pub complete: bool,
    pub acyclic: bool,
    pub no_collision: bool,
    pub problems: Vec<String>,
}

impl CheckReport {
    pub fn pass(&self) -> bool {
        self.complete && self.acyclic && self.no_collision && self.problems.is_empty()
    }
}

fn entry<'m>(m: &'m Manifest, layer: u16, expert: u16) -> Result<&'m ExpertEntry, String> {
    m.experts.iter().find(|e| e.layer == layer && e.expert == expert).ok_or(format!("layer {layer} expert {expert}: not in the manifest"))
}

impl Schedule {
    /// Build the table: `tokens` passes over `plan.layers`, B = plan.buffers_in_flight (>= 1).
    pub fn build(plan: &PlacementPlan, m: &Manifest, tokens: u32) -> Result<Self, String> {
        Self::build_swap(plan, plan, m, tokens, tokens)
    }

    /// H4 head swap: tokens before `swap_token` fire on `plan`'s heads, tokens from `swap_token` on `next`'s. Same layers
    /// and buffers; the parts table holds both heads. `build` is the no-swap case.
    pub fn build_swap(plan: &PlacementPlan, next: &PlacementPlan, m: &Manifest, tokens: u32, swap_token: u32) -> Result<Self, String> {
        if next.layers != plan.layers || next.buffers_in_flight != plan.buffers_in_flight {
            return Err("swap: both plans must have the same layers and buffers".into());
        }
        let bcount = plan.buffers_in_flight.max(1) as u32;
        let k = plan.layers.len() as u32;
        if k == 0 || tokens == 0 {
            return Err("empty plan or zero tokens".into());
        }
        let mut parts: Vec<Part> = Vec::new();
        let mut id_of: HashMap<(u16, u16), u32> = HashMap::new();
        let mut slots = 0u32;
        for (pl, l) in [plan, next].iter().flat_map(|pl| pl.layers.iter().map(move |l| (*pl, l))) {
            let h = pl.head(*l).ok_or(format!("layer {l}: no head in the plan"))?;
            slots = slots.max(h.experts.len() as u32);
            for &x in &h.experts {
                if id_of.contains_key(&(*l, x)) {
                    continue;
                }
                let e = entry(m, *l, x)?;
                id_of.insert((*l, x), parts.len() as u32);
                parts.push(Part { id: parts.len() as u32, layer: *l, expert: x, bytes: e.len, segments: e.segments.clone(), digest: e.digest.clone(), tensors: e.tensors.iter().map(|t| (t.name.clone(), t.offset, t.len)).collect() });
            }
        }
        let total = tokens * k;
        let head_parts = |b: u32| -> Vec<u32> {
            let l = plan.layers[(b % k) as usize];
            let pl = if b / k < swap_token { plan } else { next };
            pl.head(l).unwrap().experts.iter().map(|x| id_of[&(l, *x)]).collect()
        };
        let prime: Vec<Fill> = (0..(bcount - 1).min(total)).map(|b| Fill { for_beat: b, buffer: b % bcount, parts: head_parts(b) }).collect();
        let beats = (0..total)
            .map(|b| {
                let ahead = b + bcount - 1;
                let fill = (ahead < total && (bcount > 1 || true)).then(|| Fill { for_beat: ahead, buffer: ahead % bcount, parts: head_parts(ahead) });
                Beat { b, token: b / k, layer: plan.layers[(b % k) as usize], fire_buffer: b % bcount, fire_parts: head_parts(b), fill }
            })
            .collect();
        let slot_bytes = parts.iter().map(|p| p.bytes).max().unwrap_or(0);
        Ok(Self {
            schema: "aien-engine-schedule/1".into(),
            model: plan.model.clone(),
            synthetic_plan: plan.synthetic,
            tokens,
            layers: plan.layers.clone(),
            buffers: bcount,
            slots,
            slot_bytes,
            parts,
            prime,
            beats,
        })
    }

    /// Host proof. complete: every part a beat fires on was delivered to that buffer by an earlier fill and not
    /// overwritten since. acyclic: every fill happens strictly before the beat it feeds (prime counts as beat -1) and
    /// beats are numbered 0..n in order. no_collision: a fill never targets the buffer the same beat fires on (B = 1
    /// fails this by construction: one cylinder cannot be filled while it fires).
    pub fn check(&self) -> CheckReport {
        let mut r = CheckReport { beats: self.beats.len(), parts: self.parts.len(), complete: true, acyclic: true, no_collision: true, ..Default::default() };
        // buffer -> (fill issued at beat (i64; -1 = prime), parts)
        let mut state: HashMap<u32, (i64, Vec<u32>)> = HashMap::new();
        let mut min = u64::MAX;
        let mut max = 0u64;
        let mut fills = 0usize;
        let mut apply = |f: &Fill, at: i64, state: &mut HashMap<u32, (i64, Vec<u32>)>, r: &mut CheckReport| {
            fills += 1;
            if (f.for_beat as i64) <= at {
                r.acyclic = false;
                r.problems.push(format!("fill for beat {} issued at beat {at}: not before the beat it feeds", f.for_beat));
            }
            if f.parts.len() as u32 > self.slots {
                r.problems.push(format!("fill for beat {}: {} parts > {} slots", f.for_beat, f.parts.len(), self.slots));
            }
            let bytes: u64 = f.parts.iter().map(|&p| self.parts[p as usize].bytes).sum();
            r.bytes_total += bytes;
            min = min.min(bytes);
            max = max.max(bytes);
            state.insert(f.buffer, (at, f.parts.clone()));
        };
        for f in &self.prime {
            apply(f, -1, &mut state, &mut r);
        }
        for (i, beat) in self.beats.iter().enumerate() {
            if beat.b as usize != i {
                r.acyclic = false;
                r.problems.push(format!("beat {} at position {i}: out of order", beat.b));
            }
            match state.get(&beat.fire_buffer) {
                Some((at, parts)) => {
                    if *at >= beat.b as i64 {
                        r.acyclic = false;
                        r.problems.push(format!("beat {}: buffer {} filled at beat {at}, not earlier", beat.b, beat.fire_buffer));
                    }
                    for p in &beat.fire_parts {
                        if !parts.contains(p) {
                            r.complete = false;
                            r.problems.push(format!("beat {}: part {p} (layer {} expert {}) not in buffer {}", beat.b, self.parts[*p as usize].layer, self.parts[*p as usize].expert, beat.fire_buffer));
                        }
                    }
                }
                None => {
                    r.complete = false;
                    r.problems.push(format!("beat {}: buffer {} never filled", beat.b, beat.fire_buffer));
                }
            }
            if let Some(f) = &beat.fill {
                if f.buffer == beat.fire_buffer {
                    r.no_collision = false;
                    r.problems.push(format!("beat {}: fill targets buffer {} while it fires", beat.b, f.buffer));
                }
                apply(f, beat.b as i64, &mut state, &mut r);
            }
        }
        r.fills = fills;
        r.bytes_per_beat_min = if min == u64::MAX { 0 } else { min };
        r.bytes_per_beat_max = max;
        if r.problems.len() > 20 {
            let n = r.problems.len();
            r.problems.truncate(20);
            r.problems.push(format!("... {} problems in all", n));
        }
        r
    }

    pub fn part(&self, id: u32) -> &Part {
        &self.parts[id as usize]
    }

    pub fn slot(&self, buffer: u32, i: usize) -> usize {
        (buffer * self.slots) as usize + i
    }

    pub fn load(p: &Path) -> Result<Self, String> {
        serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())
    }

    pub fn save(&self, p: &Path) -> Result<(), String> {
        std::fs::write(p, serde_json::to_vec(self).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", p.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Head, Tier};

    fn toy_manifest(layers: u16, experts: u16) -> Manifest {
        let mut m = Manifest {
            schema: "toy".into(),
            model_dir: "/nonexistent".into(),
            shards: vec!["s0".into()],
            num_layers: layers,
            experts_per_layer: experts,
            experts: vec![],
        };
        for l in 0..layers {
            for e in 0..experts {
                let start = (l as u64 * experts as u64 + e as u64) * 1000;
                m.experts.push(ExpertEntry { layer: l, expert: e, segments: vec![Segment { shard: 0, start, len: 1000 }], len: 1000, tensors: vec![], digest: "0".repeat(64) });
            }
        }
        m
    }

    fn toy_plan(k: u16, head: &[u16], buffers: usize) -> PlacementPlan {
        PlacementPlan {
            schema: "aien-engine-plan/1".into(),
            model: "toy".into(),
            model_dir: "/nonexistent".into(),
            synthetic: true,
            source: "test".into(),
            layers: (0..k).collect(),
            heads: (0..k).map(|l| Head { layer: l, experts: head.to_vec() }).collect(),
            head_tier: Tier::Nvme,
            buffers_in_flight: buffers,
        }
    }

    #[test]
    fn two_buffers_is_complete_acyclic_and_collision_free() {
        let m = toy_manifest(4, 8);
        let s = Schedule::build(&toy_plan(4, &[1, 2, 3], 2), &m, 5).unwrap();
        let r = s.check();
        assert!(r.pass(), "{:?}", r.problems);
        assert_eq!(r.beats, 20);
        assert_eq!(r.fills, 20); // prime 1 + 19 in-beat fills
        assert_eq!(r.bytes_per_beat_min, 3000);
        assert_eq!(r.bytes_per_beat_max, 3000);
        assert_eq!(s.prime.len(), 1);
        assert_eq!(s.beats[0].fill.as_ref().unwrap().for_beat, 1);
        assert!(s.beats[19].fill.is_none());
    }

    #[test]
    fn one_buffer_collides_by_construction() {
        let m = toy_manifest(2, 4);
        let s = Schedule::build(&toy_plan(2, &[0], 1), &m, 3).unwrap();
        let r = s.check();
        assert!(!r.no_collision);
        assert!(!r.pass());
    }

    #[test]
    fn missing_part_is_caught() {
        let m = toy_manifest(2, 4);
        let mut s = Schedule::build(&toy_plan(2, &[0, 1], 2), &m, 2).unwrap();
        // sabotage: the fill for beat 2 forgets a part
        s.beats[1].fill.as_mut().unwrap().parts.pop();
        let r = s.check();
        assert!(!r.complete);
        assert!(r.problems.iter().any(|p| p.contains("not in buffer")));
    }

    #[test]
    fn backwards_fill_is_a_cycle() {
        let m = toy_manifest(2, 4);
        let mut s = Schedule::build(&toy_plan(2, &[0], 2), &m, 2).unwrap();
        s.beats[2].fill = Some(Fill { for_beat: 1, buffer: 1, parts: vec![0] });
        let r = s.check();
        assert!(!r.acyclic);
    }

    #[test]
    fn three_buffers_fill_two_ahead() {
        let m = toy_manifest(3, 4);
        let s = Schedule::build(&toy_plan(3, &[0, 1], 3), &m, 2).unwrap();
        assert_eq!(s.prime.len(), 2);
        assert_eq!(s.beats[0].fill.as_ref().unwrap().for_beat, 2);
        assert!(s.check().pass());
    }
}

/// The dependency graph of a schedule, by edge kind. Drake (2026-10-20): only the true dependencies are edges;
/// every other order is left to the engine that wants it.
/// * `fuel`: fill(for_beat = b) -> fire(b). The buffer must be full before the beat fires.
/// * `residual`: fire(b) -> fire(b + 1) when both beats are layers of the same token (layer L+1 reads what L wrote).
/// * `token`: fire(last layer of token t) -> fire(first layer of token t + 1) of the same sequence.
/// * `release`: fire(b') -> fill issued at beat b for the buffer b' fired on (the buffer is free again). The fill
///   issued during beat b targets a buffer whose last fire was at a beat < b (the buffer beat b fires on is a violation).
/// * `free_parts`: the parts of one fill carry no order between them (sum over fills): the movement engine lands
///   them in any order, on any worker.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct EdgeReport {
    pub fuel: usize,
    pub residual: usize,
    pub token: usize,
    pub release: usize,
    pub free_parts: usize,
    /// Edges that point backwards or sideways in beat order (must be 0) and fills that would overwrite a buffer
    /// before its last fire (must be 0).
    pub violations: Vec<String>,
}

impl Schedule {
    /// Enumerate the edges and prove there are exactly these: fuel = beats, residual = (layers - 1) x tokens,
    /// token = tokens - 1, release = in-beat fills whose buffer was fired on before; no edge points backwards.
    pub fn edges(&self) -> EdgeReport {
        let mut r = EdgeReport::default();
        let mut last_fire: HashMap<u32, u32> = HashMap::new();
        for (i, beat) in self.beats.iter().enumerate() {
            r.fuel += 1;
            if i > 0 {
                let prev = &self.beats[i - 1];
                if prev.token == beat.token {
                    r.residual += 1;
                } else if prev.token + 1 == beat.token {
                    r.token += 1;
                } else {
                    r.violations.push(format!("beat {}: token {} follows token {}", beat.b, beat.token, prev.token));
                }
            }
            // record this beat's fire first: a fill issued during beat b that targets the buffer b fires on is a violation
            last_fire.insert(beat.fire_buffer, beat.b);
            if let Some(f) = &beat.fill {
                if let Some(&lf) = last_fire.get(&f.buffer) {
                    if lf >= beat.b {
                        r.violations.push(format!("beat {}: fill targets buffer {} fired at beat {lf} (not released)", beat.b, f.buffer));
                    }
                    r.release += 1;
                }
                r.free_parts += f.parts.len().saturating_sub(1);
            }
        }
        for f in &self.prime {
            r.free_parts += f.parts.len().saturating_sub(1);
        }
        r
    }

    /// The expected edge counts for a table with this many tokens, layers and fills: the host test for
    /// "exactly those edges and no others".
    pub fn expected_edges(&self) -> EdgeReport {
        let k = self.layers.len();
        let t = self.tokens as usize;
        let in_beat_fills = self.beats.iter().filter(|b| b.fill.is_some()).count();
        EdgeReport {
            fuel: self.beats.len(),
            residual: (k - 1) * t,
            token: t.saturating_sub(1),
            // after the B - 1 prime fills, only the first in-beat fill (for beat B - 1, buffer B - 1) targets a buffer
            // never fired on; every later in-beat fill at beat b refills the buffer fired on at beat b - 1
            release: in_beat_fills.saturating_sub(1),
            free_parts: self.beats.iter().filter_map(|b| b.fill.as_ref()).chain(self.prime.iter()).map(|f| f.parts.len().saturating_sub(1)).sum(),
            violations: vec![],
        }
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;
    use crate::plan::{Head, PlacementPlan, Tier};

    fn toy(k: u16, experts: u16, buffers: usize, tokens: u32) -> Schedule {
        let mut m = Manifest { schema: "toy".into(), model_dir: "/nonexistent".into(), shards: vec!["s0".into()], num_layers: k, experts_per_layer: experts, experts: vec![] };
        for l in 0..k {
            for e in 0..experts {
                m.experts.push(ExpertEntry { layer: l, expert: e, segments: vec![Segment { shard: 0, start: 0, len: 10 }], len: 10, tensors: vec![], digest: "0".repeat(64) });
            }
        }
        let plan = PlacementPlan { schema: "p".into(), model: "toy".into(), model_dir: "/nonexistent".into(), synthetic: true, source: "t".into(), layers: (0..k).collect(), heads: (0..k).map(|l| Head { layer: l, experts: (0..experts).collect() }).collect(), head_tier: Tier::Nvme, buffers_in_flight: buffers };
        Schedule::build(&plan, &m, tokens).unwrap()
    }

    #[test]
    fn exactly_the_true_edges_two_buffers() {
        let s = toy(4, 3, 2, 5);
        let got = s.edges();
        assert_eq!(got, s.expected_edges(), "{got:?}");
        assert_eq!(got.fuel, 20);
        assert_eq!(got.residual, 15);
        assert_eq!(got.token, 4);
        assert_eq!(got.release, 18); // 19 in-beat fills, the first targets buffer 1 never fired on
        assert_eq!(got.free_parts, 40); // 20 fills x (3 - 1)
    }

    #[test]
    fn exactly_the_true_edges_three_buffers() {
        let s = toy(3, 2, 3, 4);
        let got = s.edges();
        assert_eq!(got, s.expected_edges(), "{got:?}");
        assert!(got.violations.is_empty());
    }

    #[test]
    fn unreleased_buffer_is_a_violation() {
        let mut s = toy(2, 2, 2, 3);
        // sabotage: the fill issued at beat 2 targets the buffer beat 2 itself fires on
        let fb = s.beats[2].fire_buffer;
        s.beats[2].fill.as_mut().unwrap().buffer = fb;
        assert!(!s.edges().violations.is_empty());
    }
}
