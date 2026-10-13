//! Values the evaluator produces.

use crate::format::{Class, Fields, Format, Policy};
use crate::ratutil::RoundMode;
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
        }
    }

    pub fn human(&self) -> String {
        match self {
            Value::Num(r) => rat_human(r),
            other => other.canonical(),
        }
    }
}
