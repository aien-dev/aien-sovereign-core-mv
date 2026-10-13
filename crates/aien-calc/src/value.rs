//! Values the evaluator produces.

use num_rational::BigRational;
use num_traits::One;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Num(BigRational),
    Bool(bool),
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
            Value::Num(_) => "a number",
            Value::Bool(_) => "a boolean",
        }
    }

    pub fn canonical(&self) -> String {
        match self {
            Value::Num(r) => rat_canonical(r),
            Value::Bool(b) => b.to_string(),
        }
    }

    pub fn human(&self) -> String {
        match self {
            Value::Num(r) => rat_human(r),
            Value::Bool(b) => b.to_string(),
        }
    }
}
