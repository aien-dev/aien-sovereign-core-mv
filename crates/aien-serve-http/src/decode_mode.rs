//! Decode path of the Omega door (OWN-30B release candidate 2, W-RC2 2026-10-14; RC3 adds `spec`).
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
//!
//! RC3 (W-L4e): speculative decoding (guess-ahead), OFF by default: `--spec on` / `AIEN_SERVE_SPEC=1`.
//! Greedy requests then draft from a suffix index of this request and past ones and check the guesses in one
//! exact verify pass (every row bit-identical to the launch-at-a-time `step`); rounds without a draft, and
//! the pauses of the break-even back-off, are ordinary decode steps (the one-list token when it is on).
//! Sampled requests always decode plainly. The door sets it explicitly, so the legacy `AIEN_OWN30B_SPEC`
//! variable does not change what it runs.

use crate::omega_engine::parse_switch;

/// Environment switch for [`DecodeMode::graph`].
pub const DECODE_GRAPH_ENV: &str = "AIEN_SERVE_DECODE_GRAPH";
/// Environment switch for [`DecodeMode::resident`].
pub const RESIDENT_ENV: &str = "AIEN_SERVE_RESIDENT_DECODE";
/// Environment switch for [`DecodeMode::spec`] (RC3).
pub const SPEC_ENV: &str = "AIEN_SERVE_SPEC";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeMode {
    pub graph: bool,
    pub resident: bool,
    /// RC3: speculative decoding for greedy requests. OFF by default.
    pub spec: bool,
}

impl Default for DecodeMode {
    fn default() -> Self {
        Self { graph: true, resident: true, spec: false }
    }
}

impl DecodeMode {
    /// Defaults overridden by the environment (`None` = unset).
    pub fn from_env(graph: Option<&str>, resident: Option<&str>, spec: Option<&str>) -> Result<Self, String> {
        let mut m = Self::default();
        if let Some(v) = graph {
            m.graph = parse_switch(DECODE_GRAPH_ENV, v)?;
        }
        if let Some(v) = resident {
            m.resident = parse_switch(RESIDENT_ENV, v)?;
        }
        if let Some(v) = spec {
            m.spec = parse_switch(SPEC_ENV, v)?;
        }
        Ok(m)
    }

    /// What the engine runs: the resident chain only exists inside the one-list decode. Speculation composes
    /// with either decode path (its plain steps are that path's steps).
    pub fn effective(self) -> Self {
        Self { graph: self.graph, resident: self.graph && self.resident, spec: self.spec }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_mode_defaults_on_and_env_switches_off() {
        assert_eq!(DecodeMode::default(), DecodeMode { graph: true, resident: true, spec: false });
        assert_eq!(DecodeMode::from_env(None, None, None).unwrap(), DecodeMode::default());
        let m = DecodeMode::from_env(Some("0"), None, None).unwrap();
        assert!(!m.graph && m.resident);
        assert_eq!(m.effective(), DecodeMode { graph: false, resident: false, spec: false });
        let m = DecodeMode::from_env(Some("on"), Some("off"), None).unwrap();
        assert_eq!(m.effective(), DecodeMode { graph: true, resident: false, spec: false });
        assert!(DecodeMode::from_env(Some("onn"), None, None).is_err());
        assert!(DecodeMode::from_env(None, Some("2"), None).is_err());
    }

    #[test]
    fn spec_defaults_off_env_turns_it_on_and_typos_are_refused() {
        assert!(!DecodeMode::default().spec, "guess-ahead is off unless asked for");
        let m = DecodeMode::from_env(None, None, Some("1")).unwrap();
        assert!(m.spec && m.graph && m.resident, "spec does not change the decode path");
        // Composes with the launch-at-a-time decode too.
        let m = DecodeMode::from_env(Some("off"), None, Some("on")).unwrap().effective();
        assert_eq!(m, DecodeMode { graph: false, resident: false, spec: true });
        assert!(!DecodeMode::from_env(None, None, Some(" OFF ")).unwrap().spec);
        let e = DecodeMode::from_env(None, None, Some("onn")).unwrap_err();
        assert!(e.contains(SPEC_ENV), "{e}");
        assert!(DecodeMode::from_env(None, None, Some("2")).is_err());
        assert!(DecodeMode::from_env(None, None, Some("")).is_err());
    }
}
