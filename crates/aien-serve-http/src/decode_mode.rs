//! Decode path of the Omega door (OWN-30B release candidate 2, W-RC2 2026-10-14).
//!
//! Both switches are ON by default:
//!   * one-list decode (`set_decode_graph`): each decode token is one omega command list (all 48 layers,
//!     final norm, lm_head, argmax) instead of ~290 separate launches.
//!   * resident decode chain (omega `omega_gpu_decode_graph_set_resident`, grid 128, row pairs): inside that
//!     list, [o_proj + add] and the MoE chain run as one resident launch each (436 -> 197 launches per token).
//!
//! Full-model gate (sc own30b/resident, evidence own30b-L2h-resident-20261014T0340Z): golden tokens
//! IDENTICAL on p1..p4 in every run; decode 41.7-45.1 tok/s with both on vs 36.5-38.9 with the one-list
//! decode off (the RC 1 path).
//!
//! Off: `--decode-graph off` / `AIEN_SERVE_DECODE_GRAPH=0` (the launch-at-a-time decode of RC 1; the
//! resident chain lives inside the list, so it goes off too), `--resident-decode off` /
//! `AIEN_SERVE_RESIDENT_DECODE=0` (the one-list decode without the resident chain). Flag beats
//! environment beats default; anything but on/off/1/0/true/false/yes/no is refused. The door sets both
//! explicitly, so omega's own `OWN30B_DECODE_GRAPH` / `AIEN_OMEGA_DGRAPH_RESIDENT*` variables do not
//! change what it runs.

use crate::omega_engine::parse_switch;

/// Environment switch for [`DecodeMode::graph`].
pub const DECODE_GRAPH_ENV: &str = "AIEN_SERVE_DECODE_GRAPH";
/// Environment switch for [`DecodeMode::resident`].
pub const RESIDENT_ENV: &str = "AIEN_SERVE_RESIDENT_DECODE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMode {
    pub graph: bool,
    pub resident: bool,
}

impl Default for DecodeMode {
    fn default() -> Self {
        Self { graph: true, resident: true }
    }
}

impl DecodeMode {
    /// Defaults overridden by the environment (`None` = unset).
    pub fn from_env(graph: Option<&str>, resident: Option<&str>) -> Result<Self, String> {
        let mut m = Self::default();
        if let Some(v) = graph {
            m.graph = parse_switch(DECODE_GRAPH_ENV, v)?;
        }
        if let Some(v) = resident {
            m.resident = parse_switch(RESIDENT_ENV, v)?;
        }
        Ok(m)
    }

    /// What the engine runs: the resident chain only exists inside the one-list decode.
    pub fn effective(self) -> Self {
        Self { graph: self.graph, resident: self.graph && self.resident }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_mode_defaults_on_and_env_switches_off() {
        assert_eq!(DecodeMode::default(), DecodeMode { graph: true, resident: true });
        assert_eq!(DecodeMode::from_env(None, None).unwrap(), DecodeMode::default());
        let m = DecodeMode::from_env(Some("0"), None).unwrap();
        assert!(!m.graph && m.resident);
        assert_eq!(m.effective(), DecodeMode { graph: false, resident: false });
        let m = DecodeMode::from_env(Some("on"), Some("off")).unwrap();
        assert_eq!(m.effective(), DecodeMode { graph: true, resident: false });
        assert!(DecodeMode::from_env(Some("onn"), None).is_err());
        assert!(DecodeMode::from_env(None, Some("2")).is_err());
    }
}
