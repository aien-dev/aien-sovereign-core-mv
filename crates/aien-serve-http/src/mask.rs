//! Per-request decode constraints at the door (ADR 0004 section 2.3): the request field
//! `"constraint": "omegese"` keeps the reply inside the Omegese next-word mask
//! (`aien-omegese-constraint`, the reader linked as a library). Without the field nothing here
//! runs and every engine takes its own loop, byte for byte.
//!
//! An engine that takes a constraint asks [`Mask::rule`] once per step, before it reads the step's
//! logits: `Force` and `Finish` need none, so an engine skips the logits read and the sampler
//! there (the forward pass that stores the previous token still runs, and so does the output
//! head where it is fused into the step, as on the 27B door). `Any` and `Allow`
//! are answered by [`Mask::choose`] with the engine's own sampler on the logits, masked to the
//! allowed rows under `Allow`.

use std::time::Instant;

use aien_inference_abi::generation::{Constraint, StepRule};
use serde::Serialize;

/// What a request asked for (the `constraint` field).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintRequest {
    /// The Omegese mask, the reply framed as one fenced code block (the first fenced block is
    /// what the exp445 reader takes). `header`: the entry function's header line, forced after
    /// the fence when given (`constraint_header`).
    Omegese { header: Option<String> },
}

impl ConstraintRequest {
    /// The request fields `constraint` (absent or null: none) and `constraint_header`.
    pub fn parse(
        constraint: Option<&serde_json::Value>,
        header: Option<&str>,
    ) -> Result<Option<Self>, String> {
        match constraint {
            None | Some(serde_json::Value::Null) => {
                if header.is_some() {
                    return Err("constraint_header needs \"constraint\": \"omegese\"".into());
                }
                Ok(None)
            }
            Some(serde_json::Value::String(s)) if s == "omegese" => Ok(Some(Self::Omegese {
                header: header.map(str::to_string),
            })),
            Some(v) => Err(format!(
                "unknown constraint {v}: the one constraint is \"omegese\""
            )),
        }
    }
}

/// The record of one constrained reply, returned as `usage.aien_constraint`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MaskRecord {
    pub kind: String,
    /// Steps that asked the constraint (every generated token, plus a final `Finish` or forced
    /// stop).
    pub steps: u64,
    /// Steps with exactly one legal token: taken without sampling, and without reading the
    /// logits where the engine can skip that.
    pub forced: u64,
    /// Of `forced`, the steps whose logits the engine neither read nor sampled (on the 27B door
    /// the output head still runs inside the step: it is fused into the chip graph).
    pub logits_skipped: u64,
    /// Steps sampled from the masked logits.
    pub allowed: u64,
    /// Steps sampled with no mask (`Any`).
    pub any: u64,
    /// Wall time inside the constraint, microseconds: total and the slowest step.
    pub rule_us: u64,
    pub rule_us_max: u64,
    /// The program the constraint saw (the Omegese text, fences and stop excluded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
}

/// One request's constraint.
pub struct Mask {
    c: Box<dyn Constraint + Send>,
    program: Option<Box<dyn Fn() -> String + Send>>,
    masked: Vec<f32>,
    rec: MaskRecord,
}

impl Mask {
    /// Any constraint (tests use a scripted one).
    pub fn new(kind: &str, c: Box<dyn Constraint + Send>) -> Self {
        Self {
            c,
            program: None,
            masked: Vec::new(),
            rec: MaskRecord {
                kind: kind.into(),
                ..Default::default()
            },
        }
    }

    /// The rule for the step after `generated`; asked once per step, before its logits.
    pub fn rule(&mut self, prompt: &[u32], generated: &[u32]) -> Result<StepRule, String> {
        let t = Instant::now();
        let r = self.c.rule(prompt, generated);
        let us = t.elapsed().as_micros() as u64;
        self.rec.steps += 1;
        self.rec.rule_us += us;
        self.rec.rule_us_max = self.rec.rule_us_max.max(us);
        match &r {
            Ok(StepRule::Force(_)) => self.rec.forced += 1,
            Ok(StepRule::Allow(_)) => self.rec.allowed += 1,
            Ok(StepRule::Any) => self.rec.any += 1,
            _ => {}
        }
        r
    }

    /// Counts a forced step whose logits the engine did not read.
    pub fn note_logits_skipped(&mut self) {
        self.rec.logits_skipped += 1;
    }

    /// The token for an `Any` or `Allow` step: `sample` (the engine's sampler) on the logits,
    /// masked to the allowed rows under `Allow`. A token outside the set is an error.
    pub fn choose(
        &mut self,
        rule: &StepRule,
        logits: &[f32],
        sample: &mut dyn FnMut(&[f32]) -> u32,
    ) -> Result<u32, String> {
        match rule {
            StepRule::Any => Ok(sample(logits)),
            StepRule::Allow(rows) => {
                if rows.is_empty() {
                    return Err("constraint allowed no token".into());
                }
                self.masked.clear();
                self.masked.resize(logits.len(), f32::NEG_INFINITY);
                for &r in rows {
                    let r = r as usize;
                    if r >= logits.len() {
                        return Err(format!(
                            "constraint allowed token {r}, vocabulary is {}",
                            logits.len()
                        ));
                    }
                    self.masked[r] = logits[r];
                }
                let t = sample(&self.masked);
                if rows.binary_search(&t).is_err() {
                    return Err(format!(
                        "the sampler picked token {t}, outside the {} allowed",
                        rows.len()
                    ));
                }
                Ok(t)
            }
            StepRule::Force(t) => Ok(*t),
            StepRule::Finish => Err("choose called on a Finish step".into()),
        }
    }

    /// The record for the reply.
    pub fn record(&self) -> MaskRecord {
        let mut r = self.rec.clone();
        r.program = self.program.as_ref().map(|p| p());
        r
    }
}

/// The Omegese mask for one request (`omegese` feature).
#[cfg(feature = "omegese")]
pub mod omegese {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};

    use aien_omegese_constraint::{Framing, LinkedOracle, OmegeseConstraint, Signature, TokenTable};

    use super::{ConstraintRequest, Mask};

    /// The vocabulary as bytes, loaded once from the model's `tokenizer.json`.
    pub struct Setup {
        table: Arc<TokenTable>,
        stop: u32,
    }

    static TOKENIZER: OnceLock<PathBuf> = OnceLock::new();
    static SETUP: OnceLock<Result<Setup, String>> = OnceLock::new();

    /// The model's `tokenizer.json` (main sets it from `--model-dir`); read on the first
    /// constrained request.
    pub fn set_tokenizer(path: PathBuf) {
        let _ = TOKENIZER.set(path);
    }

    /// The setup, loaded once.
    pub fn setup() -> Result<&'static Setup, String> {
        SETUP
            .get_or_init(|| {
                let p = TOKENIZER
                    .get()
                    .ok_or("no tokenizer.json for the Omegese mask (the engine has no --model-dir)")?;
                Setup::load(p)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    impl Setup {
        /// The reply ends on `<|im_end|>`.
        pub fn load(tokenizer_json: &Path) -> Result<Self, String> {
            LinkedOracle::new()?;
            let table = TokenTable::from_tokenizer_json(tokenizer_json)?;
            let stop = table
                .special_id("<|im_end|>")
                .ok_or("the tokenizer has no <|im_end|> token")?;
            Ok(Self {
                table: Arc::new(table),
                stop,
            })
        }

        pub fn stop(&self) -> u32 {
            self.stop
        }

        pub fn mask(&self, req: &ConstraintRequest) -> Result<Mask, String> {
            let ConstraintRequest::Omegese { header } = req;
            let header = header
                .as_deref()
                .map(Signature::from_header)
                .transpose()?;
            let framing = Framing {
                open_fence: Some("```\n".into()),
                header,
                mask_body: true,
                close_fence: "```".into(),
                stop_token: self.stop,
            };
            let c = OmegeseConstraint::new(self.table.clone(), LinkedOracle::new()?, framing)?;
            // The constraint lives in the mask; the record reads its program through a shared
            // handle.
            let shared = Arc::new(Mutex::new(c));
            let reader = Arc::clone(&shared);
            let mut m = Mask::new("omegese", Box::new(Shared(shared)));
            m.program = Some(Box::new(move || {
                let c = reader.lock().unwrap_or_else(|p| p.into_inner());
                String::from_utf8_lossy(c.program()).into_owned()
            }));
            Ok(m)
        }
    }

    struct Shared(Arc<Mutex<OmegeseConstraint<LinkedOracle>>>);

    impl aien_inference_abi::generation::Constraint for Shared {
        fn rule(
            &mut self,
            prompt: &[u32],
            generated: &[u32],
        ) -> Result<aien_inference_abi::generation::StepRule, String> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .rule(prompt, generated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Script(Vec<StepRule>);
    impl Constraint for Script {
        fn rule(&mut self, _p: &[u32], g: &[u32]) -> Result<StepRule, String> {
            Ok(self.0.get(g.len()).cloned().unwrap_or(StepRule::Finish))
        }
    }

    #[test]
    fn the_request_field_parses() {
        let o = serde_json::json!("omegese");
        assert_eq!(ConstraintRequest::parse(None, None), Ok(None));
        assert_eq!(
            ConstraintRequest::parse(Some(&serde_json::Value::Null), None),
            Ok(None)
        );
        assert_eq!(
            ConstraintRequest::parse(Some(&o), Some("fn f")),
            Ok(Some(ConstraintRequest::Omegese {
                header: Some("fn f".into())
            }))
        );
        assert!(ConstraintRequest::parse(Some(&serde_json::json!("json")), None).is_err());
        assert!(ConstraintRequest::parse(None, Some("fn f")).is_err());
    }

    #[test]
    fn allow_masks_force_takes_any_samples() {
        let mut m = Mask::new(
            "script",
            Box::new(Script(vec![
                StepRule::Allow(vec![1, 3]),
                StepRule::Force(2),
                StepRule::Any,
            ])),
        );
        let logits = [5.0, 1.0, 9.0, 2.0];
        let mut argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32
        };
        let r = m.rule(&[], &[]).unwrap();
        assert_eq!(m.choose(&r, &logits, &mut argmax), Ok(3), "the best allowed row");
        let r = m.rule(&[], &[3]).unwrap();
        assert_eq!(r, StepRule::Force(2));
        let r = m.rule(&[], &[3, 2]).unwrap();
        assert_eq!(m.choose(&r, &logits, &mut argmax), Ok(2), "unmasked");
        assert_eq!(m.rule(&[], &[3, 2, 2]).unwrap(), StepRule::Finish);
        let rec = m.record();
        assert_eq!((rec.steps, rec.forced, rec.allowed, rec.any), (4, 1, 1, 1));
        let mut bad = |_: &[f32]| 0u32;
        assert!(m
            .choose(&StepRule::Allow(vec![1, 3]), &logits, &mut bad)
            .unwrap_err()
            .contains("outside"));
    }
}
