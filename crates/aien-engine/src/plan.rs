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

    /// L308's measured heads file (`evidence/typewriter/heads-0.90.json` on omega feat/typewriter-experts):
    /// `{mode, heads: {DOMAIN: {layers: [[expert ids] per layer, in layer order]}}}`. One domain becomes one plan:
    /// layer l's head = that domain's expert list for layer l, every layer in order (or the first `k` when `k` > 0).
    /// This is the measured plan, so `synthetic` is false; the source names the file and the domain.
    pub fn from_l308_heads(model: &str, model_dir: &Path, heads_path: &Path, domain: &str, k: usize, buffers: usize) -> Result<Self, String> {
        let text = std::fs::read_to_string(heads_path).map_err(|e| format!("{}: {e}", heads_path.display()))?;
        Self::from_l308_heads_str(model, model_dir, &text, &heads_path.display().to_string(), domain, k, buffers)
    }

    pub fn from_l308_heads_str(model: &str, model_dir: &Path, text: &str, name: &str, domain: &str, k: usize, buffers: usize) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{name}: {e}"))?;
        let dom = v.get("heads").and_then(|h| h.get(domain)).ok_or_else(|| {
            let have: Vec<String> = v.get("heads").and_then(|h| h.as_object()).map(|o| o.keys().cloned().collect()).unwrap_or_default();
            format!("{name}: no head for domain {domain:?} (have {have:?})")
        })?;
        let layers_v = dom.get("layers").and_then(|l| l.as_array()).ok_or(format!("{name}: heads.{domain}.layers is not an array"))?;
        if layers_v.is_empty() {
            return Err(format!("{name}: heads.{domain}.layers is empty"));
        }
        let n = if k > 0 { k.min(layers_v.len()) } else { layers_v.len() };
        let mut heads = Vec::with_capacity(n);
        for (l, row) in layers_v.iter().take(n).enumerate() {
            let mut experts: Vec<u16> = row
                .as_array()
                .ok_or(format!("{name}: layer {l} is not an array"))?
                .iter()
                .map(|x| x.as_u64().and_then(|x| u16::try_from(x).ok()).ok_or(format!("{name}: layer {l}: expert id {x} is not a small integer")))
                .collect::<Result<_, _>>()?;
            experts.sort_unstable();
            experts.dedup();
            if experts.is_empty() {
                return Err(format!("{name}: layer {l} has no experts"));
            }
            heads.push(Head { layer: l as u16, experts });
        }
        let mode = v.get("mode").and_then(|m| m.as_str()).unwrap_or("?");
        Ok(Self {
            schema: "aien-engine-plan/1".into(),
            model: model.into(),
            model_dir: model_dir.display().to_string(),
            synthetic: false,
            source: format!("L308 measured heads {name}, domain {domain}, mode {mode}, {} of {} layers", n, layers_v.len()),
            layers: (0..n as u16).collect(),
            heads,
            head_tier: Tier::Nvme,
            buffers_in_flight: buffers,
        })
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

#[cfg(test)]
mod l308_tests {
    use super::*;

    const HEADS: &str = r#"{"mode":"coverage-0.90","heads":{"prose":{"layers":[[5,1,3,1],[7],[2,0]]},"code":{"layers":[[9],[8],[6]]}}}"#;

    #[test]
    fn l308_heads_become_a_measured_plan() {
        let p = PlacementPlan::from_l308_heads_str("m", Path::new("/models/m"), HEADS, "heads.json", "prose", 0, 2).unwrap();
        assert!(!p.synthetic);
        assert_eq!(p.layers, vec![0, 1, 2]);
        assert_eq!(p.head(0).unwrap().experts, vec![1, 3, 5]); // sorted, duplicates dropped
        assert_eq!(p.head(1).unwrap().experts, vec![7]);
        assert_eq!(p.head(2).unwrap().experts, vec![0, 2]);
        assert!(p.source.contains("prose") && p.source.contains("coverage-0.90"));
        let first2 = PlacementPlan::from_l308_heads_str("m", Path::new("/models/m"), HEADS, "heads.json", "code", 2, 2).unwrap();
        assert_eq!(first2.layers, vec![0, 1]);
        assert_eq!(first2.heads.len(), 2);
    }

    #[test]
    fn l308_heads_unknown_domain_or_bad_rows_fail_closed() {
        let e = PlacementPlan::from_l308_heads_str("m", Path::new("/m"), HEADS, "h", "math", 0, 2).unwrap_err();
        assert!(e.contains("math") && e.contains("prose"), "{e}");
        let bad = r#"{"heads":{"x":{"layers":[[1],[]]}}}"#;
        assert!(PlacementPlan::from_l308_heads_str("m", Path::new("/m"), bad, "h", "x", 0, 2).unwrap_err().contains("no experts"));
        let big = r#"{"heads":{"x":{"layers":[[70000]]}}}"#;
        assert!(PlacementPlan::from_l308_heads_str("m", Path::new("/m"), big, "h", "x", 0, 2).is_err());
    }
}
