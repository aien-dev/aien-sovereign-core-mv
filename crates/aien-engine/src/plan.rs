//! Placement plan: the input to the schedule. L308 owns the real one (per-domain heads, measured tiers); until it
//! exists this module builds a synthetic plan and says so in the file (`synthetic: true`).
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

/// Where the schedule expects a part to live before the beat. On the GB10 host RAM and the GPU region are one
/// LPDDR5X pool, so `Ram` means "already resident, no movement"; `Nvme` means the movement engine reads it per beat.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Nvme,
    Ram,
}

/// One type head: the expert set that must be in the cylinder when the beat for `layer` fires.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Head {
    pub layer: u16,
    pub experts: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub schema: String,
    pub model: String,
    pub model_dir: String,
    /// true until L308's measured plan replaces it.
    pub synthetic: bool,
    pub source: String,
    /// Layers in beat order.
    pub layers: Vec<u16>,
    pub heads: Vec<Head>,
    pub head_tier: Tier,
    pub buffers_in_flight: usize,
}

/// Route trace: `routes[token][layer] = routed expert ids` from a teacher-forced pass (the model's own routing).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouteTrace {
    pub schema: String,
    pub model: String,
    pub prompt_tokens: Vec<u32>,
    /// Token id the reference produced (argmax) after each position; the identity gate compares against it.
    pub argmax: Vec<u32>,
    pub routes: Vec<Vec<Vec<u16>>>,
    pub host_ns_per_token: u64,
}

impl PlacementPlan {
    /// Synthetic plan: the first `k` layers, head(layer) = union of the experts the trace routed to in that layer.
    /// With the trace's own tokens replayed this head covers every routed expert, so output is identical by construction.
    pub fn synthetic_from_trace(model: &str, model_dir: &Path, trace: &RouteTrace, k: usize, buffers: usize) -> Self {
        let layers: Vec<u16> = (0..k as u16).collect();
        let heads = layers
            .iter()
            .map(|&l| {
                let set: BTreeSet<u16> = trace.routes.iter().flat_map(|t| t[l as usize].iter().copied()).collect();
                Head { layer: l, experts: set.into_iter().collect() }
            })
            .collect();
        Self {
            schema: "aien-engine-plan/1".into(),
            model: model.into(),
            model_dir: model_dir.display().to_string(),
            synthetic: true,
            source: format!("union of routed experts per layer over {} traced tokens", trace.routes.len()),
            layers,
            heads,
            head_tier: Tier::Nvme,
            buffers_in_flight: buffers,
        }
    }

    /// Synthetic plan with a fixed head: experts `0..e` in every one of the first `k` layers (movement-only runs,
    /// where no token is produced and the head only has to be a realistic number of bytes).
    pub fn synthetic_fixed(model: &str, model_dir: &Path, k: usize, e: u16, buffers: usize) -> Self {
        let layers: Vec<u16> = (0..k as u16).collect();
        let heads = layers.iter().map(|&l| Head { layer: l, experts: (0..e).collect() }).collect();
        Self {
            schema: "aien-engine-plan/1".into(),
            model: model.into(),
            model_dir: model_dir.display().to_string(),
            synthetic: true,
            source: format!("fixed head: experts 0..{e} in each of the first {k} layers"),
            layers,
            heads,
            head_tier: Tier::Nvme,
            buffers_in_flight: buffers,
        }
    }

    pub fn head(&self, layer: u16) -> Option<&Head> {
        self.heads.iter().find(|h| h.layer == layer)
    }

    pub fn load(p: &Path) -> Result<Self, String> {
        serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())
    }

    pub fn save(&self, p: &Path) -> Result<(), String> {
        std::fs::write(p, serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", p.display()))
    }
}

impl RouteTrace {
    pub fn load(p: &Path) -> Result<Self, String> {
        serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())
    }
    pub fn save(&self, p: &Path) -> Result<(), String> {
        std::fs::write(p, serde_json::to_vec(self).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", p.display()))
    }
}

impl PlacementPlan {
    /// Synthetic plan over several traces (H6: N heads on one stroke): head(layer) = union of the experts any of
    /// the traces routed to in that layer, so every head's own routing is covered and bytes per beat do not depend
    /// on how many heads ride the stroke.
    pub fn synthetic_from_traces(model: &str, model_dir: &Path, traces: &[RouteTrace], k: usize, buffers: usize) -> Self {
        let layers: Vec<u16> = (0..k as u16).collect();
        let heads = layers
            .iter()
            .map(|&l| {
                let set: BTreeSet<u16> = traces.iter().flat_map(|tr| tr.routes.iter().flat_map(move |t| t[l as usize].iter().copied())).collect();
                Head { layer: l, experts: set.into_iter().collect() }
            })
            .collect();
        Self {
            schema: "aien-engine-plan/1".into(),
            model: model.into(),
            model_dir: model_dir.display().to_string(),
            synthetic: true,
            source: format!("union of routed experts per layer over {} traces, {} positions", traces.len(), traces.iter().map(|t| t.routes.len()).sum::<usize>()),
            layers,
            heads,
            head_tier: Tier::Nvme,
            buffers_in_flight: buffers,
        }
    }
}
