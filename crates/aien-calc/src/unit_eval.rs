//! Evaluation of quantities (numbers with units) and the shape helpers.

use crate::error::{Refusal, Res};
use crate::eval::{checked, int};
use crate::parser::BinOp;
use crate::units::{Dim, UnitExpr};
use crate::value::{rat_human, Quantity, Value};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn dim_label(d: &Dim) -> String {
    if d.is_zero() {
        "a plain number".to_string()
    } else {
        d.name()
    }
}

fn mismatch(a: &Dim, b: &Dim) -> Refusal {
    Refusal::new(format!(
        "unit mismatch: {} and {}",
        dim_label(a),
        dim_label(b)
    ))
}

pub(crate) fn mk(v: BigRational, dim: Dim) -> Res<Value> {
    let v = checked(v, "quantity")?;
    if dim.is_zero() {
        Ok(Value::Num(v))
    } else {
        Ok(Value::Qty(Quantity {
            v,
            dim,
            shown: None,
        }))
    }
}

/// (value in base units, dimension) of a number or quantity.
fn parts(v: &Value) -> Option<(BigRational, Dim)> {
    match v {
        Value::Num(r) => Some((r.clone(), Dim::default())),
        Value::NegZero => Some((BigRational::zero(), Dim::default())),
        Value::Qty(q) => Some((q.v.clone(), q.dim)),
        _ => None,
    }
}

fn plain(v: &Value, what: &str) -> Res<BigRational> {
    match v {
        Value::Num(r) => Ok(r.clone()),
        Value::NegZero => Ok(BigRational::zero()),
        Value::Qty(q) => Err(Refusal::new(format!(
            "{what} needs a plain number, found a quantity in {}",
            q.dim.name()
        ))),
        other => Err(Refusal::new(format!(
            "{what} needs a plain number, found {}",
            other.kind()
        ))),
    }
}

/// Binary operators where at least one side is a quantity or shape, or the operator is `x`.
pub(crate) fn binary_units(op: BinOp, a: &Value, b: &Value) -> Res<Value> {
    use BinOp::*;
    if op == By {
        let mut dims = Vec::new();
        for side in [a, b] {
            match side {
                Value::Shape(d) => dims.extend(d.iter().cloned()),
                Value::Num(r) => dims.push(int(r, "shape dimension")?),
                other => {
                    return Err(Refusal::new(format!(
                        "a shape dimension must be an integer, found {}",
                        other.kind()
                    )))
                }
            }
        }
        return Ok(Value::Shape(dims));
    }
    if matches!(a, Value::Shape(_)) || matches!(b, Value::Shape(_)) {
        return Err(Refusal::new(
            "a shape (m x n) is only valid as the argument of params()",
        ));
    }
    let (Some((x, dx)), Some((y, dy))) = (parts(a), parts(b)) else {
        return Err(Refusal::new(format!(
            "operator {} needs numbers or quantities",
            op.symbol()
        )));
    };
    match op {
        Add | Sub => {
            if dx != dy {
                return Err(mismatch(&dx, &dy));
            }
            mk(if op == Add { x + y } else { x - y }, dx)
        }
        Mul => mk(x * y, dx.mul(&dy)),
        Div => {
            if y.is_zero() {
                return Err(Refusal::new("division by zero"));
            }
            mk(x / y, dx.div(&dy))
        }
        Pow => {
            if !dy.is_zero() {
                return Err(Refusal::new("an exponent needs a plain number"));
            }
            let e = int(&y, "exponent")?
                .to_i32()
                .filter(|e| e.abs() <= 64)
                .ok_or_else(|| Refusal::new("budget: unit exponent magnitude exceeds 64"))?;
            if x.is_zero() && e < 0 {
                return Err(Refusal::new("division by zero (0 to a negative power)"));
            }
            mk(
                checked(num_traits::pow::Pow::pow(&x, e), "power")?,
                dx.pow(e),
            )
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            if dx != dy {
                return Err(mismatch(&dx, &dy));
            }
            Ok(Value::Bool(match op {
                Eq => x == y,
                Ne => x != y,
                Lt => x < y,
                Le => x <= y,
                Gt => x > y,
                _ => x >= y,
            }))
        }
        other => Err(Refusal::new(format!(
            "operator {} needs plain numbers, found a quantity with a unit",
            other.symbol()
        ))),
    }
}

pub(crate) fn negate(q: &Quantity) -> Value {
    Value::Qty(Quantity {
        v: -q.v.clone(),
        dim: q.dim,
        shown: None,
    })
}

/// `value -> unit`.
pub(crate) fn convert(v: &Value, u: &UnitExpr) -> Res<Value> {
    match v {
        Value::Qty(q) => {
            if q.dim != u.dim {
                return Err(Refusal::new(format!(
                    "unit mismatch: cannot convert {} to {} ({})",
                    q.dim.name(),
                    u.text,
                    dim_label(&u.dim)
                )));
            }
            Ok(Value::Qty(Quantity {
                v: q.v.clone(),
                dim: q.dim,
                shown: Some((u.factor.clone(), u.text.clone())),
            }))
        }
        Value::Num(_) | Value::NegZero => Err(Refusal::new(format!(
            "cannot convert a plain number to {}: attach a unit to it first",
            u.text
        ))),
        other => Err(Refusal::new(format!(
            "cannot convert {} to {}",
            other.kind(),
            u.text
        ))),
    }
}

/// abs / min / max when an argument is a quantity.
pub(crate) fn qty_call(name: &str, args: &[Value]) -> Res<Option<Value>> {
    if !matches!(name, "abs" | "min" | "max") || !args.iter().any(|a| matches!(a, Value::Qty(_))) {
        return Ok(None);
    }
    let mut it = args.iter().map(|a| {
        parts(a).ok_or_else(|| Refusal::new(format!("{name} needs numbers or quantities")))
    });
    let (mut best, dim) = it
        .next()
        .ok_or_else(|| Refusal::new(format!("{name} expects arguments")))??;
    if name == "abs" {
        if args.len() != 1 {
            return Err(Refusal::new("abs expects 1 arguments"));
        }
        return Ok(Some(mk(best.abs(), dim)?));
    }
    for p in it {
        let (v, d) = p?;
        if d != dim {
            return Err(mismatch(&dim, &d));
        }
        if (name == "min" && v < best) || (name == "max" && v > best) {
            best = v;
        }
    }
    Ok(Some(mk(best, dim)?))
}

fn arity(name: &str, args: &[Value], n: usize) -> Res<()> {
    if args.len() != n {
        return Err(Refusal::new(format!(
            "{name} expects {n} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

fn positive_int(v: &Value, what: &str) -> Res<BigInt> {
    let r = plain(v, what)?;
    let n = int(&r, what)?;
    if !n.is_positive() {
        return Err(Refusal::new(format!("{what} must be positive, got {n}")));
    }
    Ok(n)
}

const PARAM: usize = 3;
const TOK: usize = 2;

fn dim1(i: usize) -> Dim {
    let mut d = Dim::default();
    d.0[i] = 1;
    d
}

/// A count that is either a plain number or a quantity of exactly `unit_dim`.
fn count(v: &Value, unit_dim: Dim, what: &str) -> Res<BigRational> {
    match v {
        Value::Num(r) => Ok(r.clone()),
        Value::Qty(q) if q.dim == unit_dim => Ok(q.v.clone()),
        other => Err(Refusal::new(format!(
            "{what} count must be a plain number or a number of {}, found {}",
            unit_dim.name(),
            other.kind()
        ))),
    }
}

fn dtype(v: &Value) -> Res<BigRational> {
    match v {
        Value::Fmt(f) => Ok(BigRational::new(BigInt::from(f.width()), BigInt::from(8))),
        other => Err(Refusal::new(format!(
            "expected a data type (bf16, f16, f32, e4m3, int8, int4, ...), found {}",
            other.kind()
        ))),
    }
}

/// params / matmul_flops / bytes / kv_bytes.
pub(crate) fn shape_call(name: &str, args: &[Value]) -> Option<Res<Value>> {
    let run = || -> Res<Value> {
        match name {
            "params" => {
                arity(name, args, 1)?;
                let dims = match &args[0] {
                    Value::Shape(d) => d.clone(),
                    other => vec![positive_int(other, "params count")?],
                };
                let mut n = BigInt::one();
                for d in &dims {
                    if !d.is_positive() {
                        return Err(Refusal::new(format!(
                            "shape dimensions must be positive integers, got {d}"
                        )));
                    }
                    n *= d;
                }
                mk(BigRational::from_integer(n), dim1(PARAM))
            }
            "matmul_flops" => {
                arity(name, args, 3)?;
                let m = positive_int(&args[0], "m")?;
                let k = positive_int(&args[1], "k")?;
                let n = positive_int(&args[2], "n")?;
                mk(
                    BigRational::from_integer(BigInt::from(2) * m * k * n),
                    dim1(5),
                )
            }
            "bytes" => {
                arity(name, args, 2)?;
                let c = count(&args[0], dim1(PARAM), "bytes")?;
                mk(c * dtype(&args[1])?, dim1(0))
            }
            "kv_bytes" => {
                arity(name, args, 5)?;
                let layers = positive_int(&args[0], "layers")?;
                let heads = positive_int(&args[1], "kv_heads")?;
                let head_dim = positive_int(&args[2], "head_dim")?;
                let per = dtype(&args[3])?;
                let tokens = count(&args[4], dim1(TOK), "kv_bytes tokens")?;
                if !tokens.is_integer() || tokens.is_negative() {
                    return Err(Refusal::new(format!(
                        "tokens must be a non-negative integer, got {}",
                        rat_human(&tokens)
                    )));
                }
                let elems = BigRational::from_integer(BigInt::from(2) * layers * heads * head_dim);
                mk(elems * per * tokens, dim1(0))
            }
            _ => unreachable!("guarded by the caller"),
        }
    };
    matches!(name, "params" | "matmul_flops" | "bytes" | "kv_bytes").then(run)
}
