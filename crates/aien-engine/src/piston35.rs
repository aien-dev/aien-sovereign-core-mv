//! The 35B host piston (Drake 2026-10-20: the motor is the 35B). L306's exact 35B forward (`Qwen38Reference`, DeltaNet +
//! 256-expert MoE, host oracle ppl 1.18) does all the arithmetic unchanged; the only thing that changes is where its
//! expert weights come from. [`RingSource`] is a `TensorSource` that serves every routed-expert tensor from the
//! heartbeat ring (the cylinder the movement engines fill along the schedule) and everything else (trunk, DeltaNet,
//! attention, router, lm_head) from the checkpoint shards. A routed expert that is not leased in the ring is a misfire
//! and fails the forward closed; nothing is ever read from the disk on the fast path. Beats are detected inside the
//! source: the first expert read of a (token, layer) pair is the fire start of that beat, so the timing diagram is the
//! same one the 30B host piston drew, now around the 35B's own forward.
use crate::piston::{tensor_from_lease, Crank};
use crate::plan::RouteTrace;
use crate::schedule::Schedule;
use crate::timing::BeatTrace;
use aien_heartbeat::ring::Ring;
use aien_inference_abi::qwen3_moe_omega::{ShardSource, TensorSource};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// What the driver (the generation loop) and the source share: which token is being fired, the beats drawn so far,
/// and the beat still open when the forward returns.
#[derive(Default)]
pub struct Shared {
    /// Token (schedule pass) the next forward fires; set by the driver before each forward.
    pub token: AtomicU32,
    /// Expert reads that happened before `armed` (at open): the lazy-expert claim is false if this is not 0.
    pub eager_expert_reads: AtomicU32,
    pub armed: AtomicBool,
    pub traces: Mutex<Vec<BeatTrace>>,
    open: Mutex<Option<BeatTrace>>,
}

impl Shared {
    /// Close the beat still open after the last forward: its fire ends when the forward returned.
    pub fn close(&self, now_ns: i64) {
        if let Some(mut t) = self.open.lock().unwrap().take() {
            t.fire_done_ns = now_ns;
            self.traces.lock().unwrap().push(t);
        }
    }
}

/// `...layers.L.mlp.experts.E.<rest>` -> (L, E); anything else -> None.
pub fn expert_of(name: &str) -> Option<(u16, u16)> {
    let i = name.find("layers.")? + "layers.".len();
    let rest = &name[i..];
    let dot = rest.find('.')?;
    let layer: u16 = rest[..dot].parse().ok()?;
    let rest = rest[dot..].strip_prefix(".mlp.experts.")?;
    let dot = rest.find('.')?;
    let expert: u16 = rest[..dot].parse().ok()?;
    Some((layer, expert))
}

pub struct RingSource {
    shards: ShardSource,
    ring: Arc<Ring>,
    sched: Arc<Schedule>,
    part_of: HashMap<(u16, u16), u32>,
    layer_index: HashMap<u16, u32>,
    crank: Crank,
    shared: Arc<Shared>,
    cur_beat: Option<u32>,
}

impl RingSource {
    pub fn open(dir: &Path, ring: Arc<Ring>, sched: Arc<Schedule>, crank: Crank, shared: Arc<Shared>) -> Result<Self, String> {
        let shards = ShardSource::open(dir)?;
        let part_of = sched.parts.iter().map(|p| ((p.layer, p.expert), p.id)).collect();
        let layer_index = sched.layers.iter().enumerate().map(|(j, &l)| (l, j as u32)).collect();
        Ok(Self { shards, ring, sched, part_of, layer_index, crank, shared, cur_beat: None })
    }

    /// Prime the fills the schedule wants before the first beat (B-1 ahead).
    pub fn prime(&self) -> Result<(), String> {
        self.crank.prime()
    }

    /// First expert read of a beat: close the previous beat, issue this beat's fill-ahead, wait for this beat's fuel.
    fn enter_beat(&mut self, beat: u32) -> Result<(), String> {
        if self.cur_beat == Some(beat) {
            return Ok(());
        }
        let now = self.crank.now_ns();
        self.shared.close(now);
        let b = self.sched.beats.get(beat as usize).ok_or(format!("beat {beat}: past the schedule ({} beats)", self.sched.beats.len()))?;
        if let Some(f) = &b.fill {
            self.crank.issue(f)?;
        }
        let stamp = self.crank.wait_fill(beat)?;
        let fire_start = self.crank.now_ns();
        *self.shared.open.lock().unwrap() = Some(BeatTrace { b: beat, token: b.token, layer: b.layer, buffer: b.fire_buffer, bytes: stamp.bytes, fill_issue_ns: stamp.issue_ns, fill_done_ns: stamp.done_ns, fire_start_ns: fire_start, fire_done_ns: fire_start, misfire: false });
        self.cur_beat = Some(beat);
        Ok(())
    }
}

impl TensorSource for RingSource {
    fn read_into(&mut self, name: &str, dtype: &str, shape: &[usize], dst: &mut [u8]) -> Result<(), String> {
        let Some((layer, expert)) = expert_of(name) else {
            return self.shards.read_into(name, dtype, shape, dst);
        };
        if !self.shared.armed.load(Ordering::Acquire) {
            // the reference is still opening: an expert read here means experts are not lazy; counted, served from disk
            self.shared.eager_expert_reads.fetch_add(1, Ordering::Relaxed);
            return self.shards.read_into(name, dtype, shape, dst);
        }
        let j = *self.layer_index.get(&layer).ok_or(format!("{name}: layer {layer} is not in the schedule"))?;
        let beat = self.shared.token.load(Ordering::Acquire) * self.sched.layers.len() as u32 + j;
        self.enter_beat(beat)?;
        let part = *self.part_of.get(&(layer, expert)).ok_or_else(|| {
            if let Some(t) = self.shared.open.lock().unwrap().as_mut() {
                t.misfire = true;
            }
            format!("beat {beat}: layer {layer} expert {expert} is not in the plan (misfire, fail closed)")
        })?;
        tensor_from_lease(&self.ring, &self.sched, part, name, dst)
    }

    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        self.shards.dtype_of(name)
    }
}

/// L306's golden (`evidence/q36-host-20261020/gen-p*.json`: prompt_ids, generated_ids, routes[step][layer] = the ids the
/// reference routed to, all prompt rows at once for step 0) as a [`RouteTrace`]: token 0 = the prefill (its head is the
/// union over the prompt rows), token i = the decode step that fed generated_ids[i-1]; argmax[i] = generated_ids[i].
pub fn golden_to_trace(golden: &Value, model: &str) -> Result<RouteTrace, String> {
    let ids = |k: &str| -> Result<Vec<u32>, String> {
        golden.get(k).and_then(|v| v.as_array()).ok_or(format!("golden: no {k}"))?.iter().map(|x| x.as_u64().map(|x| x as u32).ok_or(format!("golden: {k} has a non-integer"))).collect()
    };
    let prompt_tokens = ids("prompt_ids")?;
    let argmax = ids("generated_ids")?;
    let steps = golden.get("routes").and_then(|v| v.as_array()).ok_or("golden: no routes")?;
    let mut routes = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let layers = step.as_array().ok_or(format!("golden: routes[{i}] is not an array"))?;
        let mut per_layer = Vec::with_capacity(layers.len());
        for (l, row) in layers.iter().enumerate() {
            let mut v: Vec<u16> = row.as_array().ok_or(format!("golden: routes[{i}][{l}] is not an array"))?.iter().map(|x| x.as_u64().and_then(|x| u16::try_from(x).ok()).ok_or(format!("golden: routes[{i}][{l}]: bad id"))).collect::<Result<_, _>>()?;
            v.sort_unstable();
            v.dedup();
            per_layer.push(v);
        }
        routes.push(per_layer);
    }
    let n = routes.len().min(argmax.len());
    if n == 0 {
        return Err("golden: no steps".into());
    }
    Ok(RouteTrace { schema: "aien-engine-trace/1".into(), model: model.into(), prompt_tokens, argmax: argmax[..n].to_vec(), routes: routes[..n].to_vec(), host_ns_per_token: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_names_parse_and_others_do_not() {
        assert_eq!(expert_of("model.language_model.layers.12.mlp.experts.207.down_proj.weight_scale_inv"), Some((12, 207)));
        assert_eq!(expert_of("model.language_model.layers.3.mlp.experts.0.gate_proj.weight"), Some((3, 0)));
        assert_eq!(expert_of("model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"), None);
        assert_eq!(expert_of("model.language_model.layers.3.linear_attn.in_proj_qkvz.weight"), None);
        assert_eq!(expert_of("lm_head.weight"), None);
    }

    #[test]
    fn golden_becomes_a_trace_with_the_prefill_union_first() {
        let g: Value = serde_json::from_str(r#"{"prompt_ids":[1,2],"generated_ids":[7,8,9],"routes":[[[5,1,5,3],[2,2]],[[4],[0]],[[6],[1]]]}"#).unwrap();
        let t = golden_to_trace(&g, "m").unwrap();
        assert_eq!(t.prompt_tokens, vec![1, 2]);
        assert_eq!(t.argmax, vec![7, 8, 9]);
        assert_eq!(t.routes[0], vec![vec![1, 3, 5], vec![2]]);
        assert_eq!(t.routes[2], vec![vec![6], vec![1]]);
    }
}
