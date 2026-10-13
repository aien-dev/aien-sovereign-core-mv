//! Values the evaluator produces.

use crate::format::{Class, Fields, Format, Policy};
use crate::ratutil::RoundMode;
use crate::units::Dim;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Num(BigRational),
    Bool(bool),
    /// Not-a-number: carries no value, every arithmetic use refuses.
    NaN,
    Inf {
        negative: bool,
    },
    /// Negative zero (arithmetic treats it as 0; only bit encodings see the sign).
    NegZero,
    Fmt(Format),
    Class(Class),
    Mode(RoundMode),
    Policy(Policy),
    Fields(Fields),
    Qty(Quantity),
    /// Dimensions of `m x n`, only valid as the argument of `params()`.
    Shape(Vec<BigInt>),
}

/// A number with a dimension. `v` is always in base units (B, s, tok, param, launch, FLOP);
/// `shown` records the unit requested by `->` for display only and is ignored by equality.
#[derive(Clone, Debug)]
pub struct Quantity {
    pub v: BigRational,
    pub dim: Dim,
    pub shown: Option<(BigRational, String)>,
}

impl PartialEq for Quantity {
    fn eq(&self, o: &Self) -> bool {
        self.v == o.v && self.dim == o.dim
    }
}

impl Eq for Quantity {}

impl Quantity {
    /// The number as displayed, and its unit text.
    pub fn display_parts(&self) -> (BigRational, String) {
        match &self.shown {
            Some((factor, text)) => (&self.v / factor, text.clone()),
            None => (self.v.clone(), self.dim.name()),
        }
    }
}

/// `p/q` always, even for integers (`7/1`): the receipt and test vocabulary.
pub fn rat_canonical(r: &BigRational) -> String {
    format!("{}/{}", r.numer(), r.denom())
}

/// Human spelling: integers print bare.
pub fn rat_human(r: &BigRational) -> String {
    if r.denom().is_one() {
        r.numer().to_string()
    } else {
        rat_canonical(r)
    }
}

impl Value {
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Num(_) | Value::NegZero => "a number",
            Value::Bool(_) => "a boolean",
            Value::NaN => "NaN",
            Value::Inf { .. } => "an infinity",
            Value::Fmt(_) => "a format",
            Value::Class(_) => "a class",
            Value::Mode(_) => "a rounding mode",
            Value::Policy(_) => "an overflow policy",
            Value::Fields(_) => "a field breakdown",
            Value::Qty(_) => "a quantity with a unit",
            Value::Shape(_) => "a shape",
        }
    }

    pub fn canonical(&self) -> String {
        match self {
            Value::Num(r) => rat_canonical(r),
            Value::Bool(b) => b.to_string(),
            Value::NaN => "nan".to_string(),
            Value::Inf { negative } => if *negative { "-inf" } else { "+inf" }.to_string(),
            Value::NegZero => "-0".to_string(),
            Value::Fmt(f) => f.name().to_string(),
            Value::Class(c) => c.name().to_string(),
            Value::Mode(m) => m.name().to_string(),
            Value::Policy(p) => p.name().to_string(),
            Value::Fields(f) => format!("sign={} exp={} mant={}", f.sign, f.exp, f.mant),
            Value::Qty(q) => {
                let (n, unit) = q.display_parts();
                format!("{} {unit}", rat_canonical(&n))
            }
            Value::Shape(d) => d
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(" x "),
        }
    }

    pub fn human(&self) -> String {
        match self {
            Value::Num(r) => rat_human(r),
            Value::Qty(q) => {
                let (n, unit) = q.display_parts();
                format!("{} {unit}", rat_human(&n))
            }
            other => other.canonical(),
        }
    }
}
