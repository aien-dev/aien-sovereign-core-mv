//! Exact evaluator for calc-lang v1. Every intermediate is a rational; there is no floating point.

use crate::error::{Refusal, Res};
use crate::format::{
    class_of, decode, encode, fields_of, flush, neighbour, round_to, ulp, Class, Format, Policy,
};
use crate::lexer::MAX_BITS;
use crate::parser::{Ast, BinOp};
use crate::ratutil::{round_signed, RoundMode};
use crate::value::{rat_human, Value};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use std::collections::BTreeMap;

/// Variables bound for the duration of one evaluation (`$b`, `$x` inside a sweep).
#[derive(Default, Clone, Debug)]
pub struct Env {
    pub vars: BTreeMap<String, Value>,
}

fn budget(what: &str) -> Refusal {
    Refusal::new(format!(
        "budget: {what} exceeds the {MAX_BITS}-bit size limit"
    ))
}

fn within(r: &BigRational) -> bool {
    r.numer().bits() <= MAX_BITS && r.denom().bits() <= MAX_BITS
}

fn checked(r: BigRational, what: &str) -> Res<BigRational> {
    if within(&r) {
        Ok(r)
    } else {
        Err(budget(what))
    }
}

pub(crate) fn num(v: &Value) -> Res<BigRational> {
    match v {
        Value::Num(r) => Ok(r.clone()),
        Value::NegZero => Ok(BigRational::zero()),
        other => Err(Refusal::new(format!(
            "expected a number, found {}",
            other.kind()
        ))),
    }
}

pub(crate) fn int(r: &BigRational, what: &str) -> Res<BigInt> {
    if r.is_integer() {
        Ok(r.to_integer())
    } else {
        Err(Refusal::new(format!(
            "{what} must be an integer, got {}",
            rat_human(r)
        )))
    }
}

fn arity(name: &str, args: &[Value], min: usize, max: usize) -> Res<()> {
    if args.len() < min || args.len() > max {
        let want = if min == max {
            format!("{min}")
        } else {
            format!("{min} to {max}")
        };
        return Err(Refusal::new(format!(
            "{name} expects {want} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

fn shift_amount(r: &BigRational) -> Res<usize> {
    let n = int(r, "shift amount")?;
    if n.is_negative() {
        return Err(Refusal::new("shift amount must not be negative"));
    }
    match n.to_u64() {
        Some(k) if k <= MAX_BITS => Ok(k as usize),
        _ => Err(budget("shift amount")),
    }
}

fn pow(base: &BigRational, exp: &BigRational) -> Res<BigRational> {
    let e = int(exp, "exponent")?;
    let e = match e.to_i64() {
        Some(e) if e.unsigned_abs() <= MAX_BITS => e,
        _ => return Err(budget("exponent magnitude")),
    };
    if base.is_zero() {
        return if e < 0 {
            Err(Refusal::new("division by zero (0 to a negative power)"))
        } else {
            Ok(BigRational::from_integer(BigInt::from((e == 0) as u8)))
        };
    }
    let bits = base.numer().bits().max(base.denom().bits());
    if bits.saturating_mul(e.unsigned_abs()) > MAX_BITS {
        return Err(budget("power result"));
    }
    checked(base.pow(e as i32), "power result")
}

fn binary(op: BinOp, a: &Value, b: &Value) -> Res<Value> {
    use BinOp::*;
    if matches!(op, Eq | Ne) {
        let same_kind = matches!(
            (a, b),
            (Value::Bool(_), Value::Bool(_))
                | (Value::Class(_), Value::Class(_))
                | (Value::Fmt(_), Value::Fmt(_))
                | (Value::Mode(_), Value::Mode(_))
                | (Value::Policy(_), Value::Policy(_))
                | (Value::Inf { .. }, Value::Inf { .. })
        );
        if same_kind {
            return Ok(Value::Bool((a == b) == (op == Eq)));
        }
    }
    let x = num(a)?;
    let y = num(b)?;
    Ok(match op {
        Add => Value::Num(checked(&x + &y, "sum")?),
        Sub => Value::Num(checked(&x - &y, "difference")?),
        Mul => Value::Num(checked(&x * &y, "product")?),
        Div => {
            if y.is_zero() {
                return Err(Refusal::new("division by zero"));
            }
            Value::Num(checked(&x / &y, "quotient")?)
        }
        Rem => {
            if y.is_zero() {
                return Err(Refusal::new("division by zero (remainder)"));
            }
            let q = (&x / &y).trunc();
            Value::Num(checked(&x - &y * q, "remainder")?)
        }
        Pow => Value::Num(pow(&x, &y)?),
        Shl => {
            let n = int(&x, "left operand of <<")?;
            let k = shift_amount(&y)?;
            Value::Num(checked(BigRational::from_integer(n << k), "shift result")?)
        }
        Shr => {
            let n = int(&x, "left operand of >>")?;
            let k = shift_amount(&y)?;
            Value::Num(BigRational::from_integer(n >> k))
        }
        And | Or | Xor => {
            let l = int(&x, "bitwise operand")?;
            let r = int(&y, "bitwise operand")?;
            let v = match op {
                And => l & r,
                Or => l | r,
                _ => l ^ r,
            };
            Value::Num(BigRational::from_integer(v))
        }
        Eq => Value::Bool(x == y),
        Ne => Value::Bool(x != y),
        Lt => Value::Bool(x < y),
        Le => Value::Bool(x <= y),
        Gt => Value::Bool(x > y),
        Ge => Value::Bool(x >= y),
    })
}

fn is_pow2_int(n: &BigInt) -> Option<u64> {
    let mag = n.magnitude();
    let tz = mag.trailing_zeros()?;
    (mag.bits() == tz + 1).then_some(tz)
}

fn call(name: &str, args: &[Value]) -> Res<Value> {
    if let Some(r) = format_call(name, args) {
        return r;
    }
    let n = |i: usize| num(&args[i]);
    let out = match name {
        "abs" => {
            arity(name, args, 1, 1)?;
            n(0)?.abs()
        }
        "floor" => {
            arity(name, args, 1, 1)?;
            n(0)?.floor()
        }
        "ceil" => {
            arity(name, args, 1, 1)?;
            n(0)?.ceil()
        }
        "trunc" => {
            arity(name, args, 1, 1)?;
            n(0)?.trunc()
        }
        "round_half_even" => {
            arity(name, args, 1, 1)?;
            BigRational::from_integer(round_signed(&n(0)?, RoundMode::Rne))
        }
        "gcd" | "lcm" => {
            arity(name, args, 2, 2)?;
            let a = int(&n(0)?, "gcd/lcm argument")?;
            let b = int(&n(1)?, "gcd/lcm argument")?;
            let v = if name == "gcd" { a.gcd(&b) } else { a.lcm(&b) };
            BigRational::from_integer(v)
        }
        "min" | "max" => {
            arity(name, args, 2, usize::MAX)?;
            let mut best = n(0)?;
            for i in 1..args.len() {
                let v = n(i)?;
                if (name == "min" && v < best) || (name == "max" && v > best) {
                    best = v;
                }
            }
            best
        }
        "sqrt" => {
            arity(name, args, 1, 1)?;
            let x = n(0)?;
            if x.is_negative() {
                return Err(Refusal::new("sqrt of a negative number is not real"));
            }
            let (p, q) = (x.numer().sqrt(), x.denom().sqrt());
            if &p * &p == *x.numer() && &q * &q == *x.denom() {
                BigRational::new(p, q)
            } else {
                return Err(Refusal::new(format!(
                    "sqrt({}) is not rational: exact sqrt needs a perfect square (refused, not approximated)",
                    rat_human(&x)
                )));
            }
        }
        "log2" => {
            arity(name, args, 1, 1)?;
            let x = n(0)?;
            if !x.is_positive() {
                return Err(Refusal::new("log2 needs a positive argument"));
            }
            let k = if x.denom() == &BigInt::from(1u8) {
                is_pow2_int(x.numer()).map(|k| k as i64)
            } else if x.numer() == &BigInt::from(1u8) {
                is_pow2_int(x.denom()).map(|k| -(k as i64))
            } else {
                None
            };
            match k {
                Some(k) => BigRational::from_integer(BigInt::from(k)),
                None => {
                    return Err(Refusal::new(format!(
                        "log2({}) is not exact: log2 is defined only for exact powers of two (refused, not approximated)",
                        rat_human(&x)
                    )))
                }
            }
        }
        "relerr" => {
            arity(name, args, 2, 2)?;
            let (a, b) = (n(0)?, n(1)?);
            if b.is_zero() {
                return Err(Refusal::new("division by zero (relerr reference is 0)"));
            }
            checked(((&a - &b) / &b).abs(), "relative error")?
        }
        other => return Err(Refusal::new(format!("unknown function `{other}`"))),
    };
    Ok(Value::Num(checked(out, "result")?))
}

fn ident(name: &str) -> Res<Value> {
    match name {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        "saturate" => Ok(Value::Policy(Policy::Saturate)),
        "refuse" => Ok(Value::Policy(Policy::Refuse)),
        other => {
            if let Some(f) = Format::from_name(other) {
                Ok(Value::Fmt(f))
            } else if let Some(c) = Class::from_name(other) {
                Ok(Value::Class(c))
            } else if let Some(m) = RoundMode::from_name(other) {
                Ok(Value::Mode(m))
            } else {
                Err(Refusal::new(format!("unknown name `{other}`")))
            }
        }
    }
}

/// Evaluate `ast` in `env`.
pub fn evaluate(ast: &Ast, env: &Env) -> Res<Value> {
    match ast {
        Ast::Num(r) => Ok(Value::Num(r.clone())),
        Ast::Ident(s) => ident(s),
        Ast::Var(v) => env.vars.get(v).cloned().ok_or_else(|| {
            Refusal::new(format!(
                "unbound variable ${v} (only defined inside a sweep)"
            ))
        }),
        Ast::Neg(a) => Ok(Value::Num(-num(&evaluate(a, env)?)?)),
        Ast::Bin(op, a, b) => binary(*op, &evaluate(a, env)?, &evaluate(b, env)?),
        Ast::Call(name, args) => {
            let vals = args
                .iter()
                .map(|a| evaluate(a, env))
                .collect::<Res<Vec<_>>>()?;
            call(name, &vals)
        }
    }
}

// ---- format lab functions (cut 2) ----

fn fmt_arg(v: &Value) -> Res<Format> {
    match v {
        Value::Fmt(f) => Ok(*f),
        other => Err(Refusal::new(format!("expected a format (e4m3, e5m2, f16, bf16, f32, f64, int8, int4, uint8, uint4), found {}", other.kind()))),
    }
}

fn code_arg(fmt: Format, v: &Value) -> Res<u64> {
    let r = num(v)?;
    let n = int(&r, "bit pattern")?;
    match n.to_u64() {
        Some(c) if fmt.width() == 64 || c >> fmt.width() == 0 => Ok(c),
        _ => Err(Refusal::new(format!(
            "code {n} is out of range for {} ({} bits)",
            fmt.name(),
            fmt.width()
        ))),
    }
}

/// Optional trailing rounding mode and overflow policy, in either order, each at most once.
fn options(rest: &[Value]) -> Res<(Option<RoundMode>, Option<Policy>)> {
    let (mut mode, mut policy) = (None, None);
    for v in rest {
        match v {
            Value::Mode(m) => {
                if mode.replace(*m).is_some() {
                    return Err(Refusal::new("rounding mode given twice"));
                }
            }
            Value::Policy(_) | Value::Class(Class::Inf) => {
                let p = match v {
                    Value::Policy(p) => *p,
                    _ => Policy::Inf,
                };
                if policy.replace(p).is_some() {
                    return Err(Refusal::new("overflow policy given twice"));
                }
            }
            other => {
                return Err(Refusal::new(format!(
                    "expected a rounding mode (rne rtz rup rdown rna) or overflow policy (saturate inf refuse), found {}",
                    other.kind()
                )))
            }
        }
    }
    Ok((mode, policy))
}

fn format_call(name: &str, args: &[Value]) -> Option<Res<Value>> {
    let run = || -> Res<Value> {
        match name {
            "val" => {
                arity(name, args, 2, 2)?;
                let f = fmt_arg(&args[0])?;
                decode(f, code_arg(f, &args[1])?)
            }
            "bits" => {
                arity(name, args, 2, 4)?;
                let f = fmt_arg(&args[0])?;
                let (mode, policy) = options(&args[2..])?;
                let code = encode(f, &args[1], mode, policy)?;
                Ok(Value::Num(BigRational::from_integer(BigInt::from(code))))
            }
            "round" => {
                arity(name, args, 2, 4)?;
                let f = fmt_arg(&args[0])?;
                let (mode, policy) = options(&args[2..])?;
                round_to(f, &args[1], mode, policy)
            }
            "class" => {
                arity(name, args, 2, 2)?;
                Ok(Value::Class(class_of(fmt_arg(&args[0])?, &args[1])?))
            }
            "next" | "prev" => {
                arity(name, args, 2, 2)?;
                neighbour(fmt_arg(&args[0])?, &args[1], name == "next")
            }
            "ulp" => {
                arity(name, args, 2, 2)?;
                ulp(fmt_arg(&args[0])?, &args[1])
            }
            "ftz" | "daz" => {
                arity(name, args, 2, 2)?;
                flush(fmt_arg(&args[0])?, &args[1])
            }
            "fields" => {
                arity(name, args, 2, 2)?;
                let f = fmt_arg(&args[0])?;
                Ok(Value::Fields(fields_of(f, code_arg(f, &args[1])?)?))
            }
            _ => unreachable!("guarded by the caller"),
        }
    };
    matches!(
        name,
        "val" | "bits" | "round" | "class" | "next" | "prev" | "ulp" | "ftz" | "daz" | "fields"
    )
    .then(run)
}
