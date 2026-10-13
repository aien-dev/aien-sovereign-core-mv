//! Number-format lab: exact codecs for the small formats, hand-written from the bit layouts.
//!
//! Every code decodes to an exact rational (all finite floats are dyadic rationals) or to one of
//! the specials `NaN`, `Inf`, `NegZero`. Encoding rounds an exact rational by an explicit mode and
//! overflow policy using integer arithmetic only; Rust's `f32` arithmetic is used only by the tests
//! as an oracle.

use crate::error::{Refusal, Res};
use crate::ratutil::{floor_log2, pow2, round_mag, RoundMode};
use crate::value::Value;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// OCP E4M3FN: bias 7, no infinities, only `S.1111.111` is NaN, max 448.
    E4m3,
    /// IEEE-like 1-5-2, bias 15, with infinities and NaNs.
    E5m2,
    /// IEEE binary16.
    F16,
    /// bfloat16: the top 16 bits of binary32.
    Bf16,
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    Int8,
    Int4,
    Uint8,
    Uint4,
}

/// IEEE class of a value relative to a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Zero,
    Subnormal,
    Normal,
    Inf,
    Nan,
}

/// What to do when a value rounds past the largest finite value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Clamp to the largest finite value.
    Saturate,
    /// Go to infinity (formats with no infinity refuse).
    Inf,
    /// Refuse.
    Refuse,
}

/// Bit fields of a float code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fields {
    pub sign: u64,
    pub exp: u64,
    pub mant: u64,
}

/// Layout of a float format.
#[derive(Clone, Copy, Debug)]
pub struct FloatSpec {
    pub ebits: u32,
    pub mbits: u32,
    pub bias: i64,
    /// IEEE-style top exponent (inf and NaN) versus E4M3FN (only the all-ones code is NaN).
    pub has_inf: bool,
}

impl FloatSpec {
    fn width(&self) -> u32 {
        1 + self.ebits + self.mbits
    }

    fn emin(&self) -> i64 {
        1 - self.bias
    }

    fn exp_max_field(&self) -> u64 {
        (1u64 << self.ebits) - 1
    }

    /// Largest finite value as an integer significand times a power of two.
    fn max_finite(&self) -> BigRational {
        let (sig, exp) = if self.has_inf {
            (
                (1u64 << (self.mbits + 1)) - 1,
                self.exp_max_field() as i64 - 1 - self.bias,
            )
        } else {
            (
                (1u64 << (self.mbits + 1)) - 2,
                self.exp_max_field() as i64 - self.bias,
            )
        };
        BigRational::from_integer(BigInt::from(sig)) * pow2(exp - self.mbits as i64)
    }

    fn min_normal(&self) -> BigRational {
        pow2(self.emin())
    }

    fn min_subnormal(&self) -> BigRational {
        pow2(self.emin() - self.mbits as i64)
    }

    fn split(&self, code: u64) -> Fields {
        Fields {
            sign: (code >> (self.ebits + self.mbits)) & 1,
            exp: (code >> self.mbits) & self.exp_max_field(),
            mant: code & ((1u64 << self.mbits) - 1),
        }
    }

    fn is_nan(&self, f: &Fields) -> bool {
        f.exp == self.exp_max_field()
            && if self.has_inf {
                f.mant != 0
            } else {
                f.mant == (1u64 << self.mbits) - 1
            }
    }

    fn is_inf(&self, f: &Fields) -> bool {
        self.has_inf && f.exp == self.exp_max_field() && f.mant == 0
    }

    /// Code of the (positive, exactly representable, finite, nonzero) value `v`.
    fn code_of_exact(&self, v: &BigRational) -> u64 {
        if *v < self.min_normal() {
            let m = v / self.min_subnormal();
            debug_assert!(m.is_integer());
            return m.to_integer().to_u64().unwrap_or(0);
        }
        let e2 = floor_log2(v);
        let m = (v / pow2(e2) - BigRational::one()) * pow2(self.mbits as i64);
        debug_assert!(m.is_integer());
        let e = (e2 + self.bias) as u64;
        (e << self.mbits) | m.to_integer().to_u64().unwrap_or(0)
    }
}

impl Format {
    pub const ALL: [Format; 10] = [
        Format::E4m3,
        Format::E5m2,
        Format::F16,
        Format::Bf16,
        Format::F32,
        Format::F64,
        Format::Int8,
        Format::Int4,
        Format::Uint8,
        Format::Uint4,
    ];

    pub fn from_name(s: &str) -> Option<Format> {
        Format::ALL.into_iter().find(|f| f.name() == s)
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::E4m3 => "e4m3",
            Format::E5m2 => "e5m2",
            Format::F16 => "f16",
            Format::Bf16 => "bf16",
            Format::F32 => "f32",
            Format::F64 => "f64",
            Format::Int8 => "int8",
            Format::Int4 => "int4",
            Format::Uint8 => "uint8",
            Format::Uint4 => "uint4",
        }
    }

    /// Total bits in a code.
    pub fn width(self) -> u32 {
        match self.kind() {
            Kind::Float(s) => s.width(),
            Kind::Int { bits, .. } => bits,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self.kind(), Kind::Float(_))
    }

    fn kind(self) -> Kind {
        let f = |ebits, mbits, bias, has_inf| {
            Kind::Float(FloatSpec {
                ebits,
                mbits,
                bias,
                has_inf,
            })
        };
        match self {
            Format::E4m3 => f(4, 3, 7, false),
            Format::E5m2 => f(5, 2, 15, true),
            Format::F16 => f(5, 10, 15, true),
            Format::Bf16 => f(8, 7, 127, true),
            Format::F32 => f(8, 23, 127, true),
            Format::F64 => f(11, 52, 1023, true),
            Format::Int8 => Kind::Int {
                bits: 8,
                signed: true,
            },
            Format::Int4 => Kind::Int {
                bits: 4,
                signed: true,
            },
            Format::Uint8 => Kind::Int {
                bits: 8,
                signed: false,
            },
            Format::Uint4 => Kind::Int {
                bits: 4,
                signed: false,
            },
        }
    }

    fn spec(self, op: &str) -> Res<FloatSpec> {
        match self.kind() {
            Kind::Float(s) => Ok(s),
            Kind::Int { .. } => Err(Refusal::new(format!(
                "{op} applies to float formats; {} is an integer format",
                self.name()
            ))),
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Float(FloatSpec),
    Int { bits: u32, signed: bool },
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Zero => "zero",
            Class::Subnormal => "subnormal",
            Class::Normal => "normal",
            Class::Inf => "inf",
            Class::Nan => "nan",
        }
    }

    pub fn from_name(s: &str) -> Option<Class> {
        Some(match s {
            "zero" => Class::Zero,
            "subnormal" => Class::Subnormal,
            "normal" => Class::Normal,
            "inf" => Class::Inf,
            "nan" => Class::Nan,
            _ => return None,
        })
    }
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::Saturate => "saturate",
            Policy::Inf => "inf",
            Policy::Refuse => "refuse",
        }
    }
}

fn big(n: u64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

/// Decode `code` of `fmt` to an exact value.
pub fn decode(fmt: Format, code: u64) -> Res<Value> {
    let width = fmt.width();
    if width < 64 && code >> width != 0 {
        return Err(Refusal::new(format!(
            "code {code:#x} is out of range for {} ({width} bits)",
            fmt.name()
        )));
    }
    match fmt.kind() {
        Kind::Int { bits, signed } => {
            let v = BigInt::from(code);
            let v = if signed && code >> (bits - 1) == 1 {
                v - (BigInt::one() << bits as usize)
            } else {
                v
            };
            Ok(Value::Num(BigRational::from_integer(v)))
        }
        Kind::Float(s) => {
            let f = s.split(code);
            if s.is_nan(&f) {
                return Ok(Value::NaN);
            }
            if s.is_inf(&f) {
                return Ok(Value::Inf {
                    negative: f.sign == 1,
                });
            }
            let mag = if f.exp == 0 {
                big(f.mant) * s.min_subnormal()
            } else {
                big((1u64 << s.mbits) | f.mant) * pow2(f.exp as i64 - s.bias - s.mbits as i64)
            };
            Ok(match (f.sign == 1, mag.is_zero()) {
                (true, true) => Value::NegZero,
                (true, false) => Value::Num(-mag),
                (false, _) => Value::Num(mag),
            })
        }
    }
}

fn sign_bit(s: &FloatSpec) -> u64 {
    1u64 << (s.width() - 1)
}

/// Encode `v` into `fmt`. `mode` defaults to round-to-nearest-even. `policy` defaults to IEEE
/// behaviour for formats with infinity (the mode decides between infinity and the largest finite
/// value) and to saturation for formats without (E4M3FN and the integers).
pub fn encode(fmt: Format, v: &Value, mode: Option<RoundMode>, policy: Option<Policy>) -> Res<u64> {
    let mode = mode.unwrap_or(RoundMode::Rne);
    let r = match v {
        Value::NaN => {
            return Err(Refusal::new(format!(
                "cannot convert NaN to {}: refused, not approximated",
                fmt.name()
            )))
        }
        Value::Inf { negative } => {
            return match fmt.kind() {
                Kind::Float(s) if s.has_inf => {
                    Ok((s.exp_max_field() << s.mbits) | if *negative { sign_bit(&s) } else { 0 })
                }
                _ => Err(Refusal::new(format!(
                    "{} has no infinity to hold {}",
                    fmt.name(),
                    v.canonical()
                ))),
            };
        }
        Value::NegZero => {
            return Ok(match fmt.kind() {
                Kind::Float(s) => sign_bit(&s),
                Kind::Int { .. } => 0,
            })
        }
        Value::Num(r) => r.clone(),
        other => {
            return Err(Refusal::new(format!(
                "expected a number to encode, found {}",
                other.kind()
            )))
        }
    };
    match fmt.kind() {
        Kind::Int { bits, signed } => encode_int(fmt, &r, mode, policy, bits, signed),
        Kind::Float(s) => encode_float(fmt, &s, &r, mode, policy),
    }
}

fn encode_int(
    fmt: Format,
    r: &BigRational,
    mode: RoundMode,
    policy: Option<Policy>,
    bits: u32,
    signed: bool,
) -> Res<u64> {
    let neg = r.is_negative();
    let k = round_mag(&r.abs(), mode, neg);
    let i = if neg { -k } else { k };
    let (lo, hi) = if signed {
        (
            -(BigInt::one() << (bits as usize - 1)),
            (BigInt::one() << (bits as usize - 1)) - 1,
        )
    } else {
        (BigInt::zero(), (BigInt::one() << bits as usize) - 1)
    };
    let clamped = if i < lo {
        Some(lo.clone())
    } else if i > hi {
        Some(hi.clone())
    } else {
        None
    };
    let i = match clamped {
        None => i,
        Some(edge) => match policy.unwrap_or(Policy::Saturate) {
            Policy::Saturate => edge,
            Policy::Refuse => {
                return Err(Refusal::new(format!(
                    "overflow: {} is outside {} range [{lo}, {hi}]",
                    crate::value::rat_human(r),
                    fmt.name()
                )))
            }
            Policy::Inf => return Err(Refusal::new(format!("{} has no infinity", fmt.name()))),
        },
    };
    let modulus = BigInt::one() << bits as usize;
    let code = ((i % &modulus) + &modulus) % &modulus;
    Ok(code.to_u64().unwrap_or(0))
}

fn encode_float(
    fmt: Format,
    s: &FloatSpec,
    r: &BigRational,
    mode: RoundMode,
    policy: Option<Policy>,
) -> Res<u64> {
    let neg = r.is_negative();
    let sign = if neg { sign_bit(s) } else { 0 };
    if r.is_zero() {
        return Ok(0);
    }
    let a = r.abs();
    let e2 = floor_log2(&a);
    let q = e2.max(s.emin()) - s.mbits as i64;
    let k = round_mag(&(&a / pow2(q)), mode, neg);
    if k.is_zero() {
        return Ok(sign); // underflow to a signed zero, as IEEE does
    }
    let rounded = BigRational::from_integer(k) * pow2(q);
    let max = s.max_finite();
    if rounded <= max {
        return Ok(sign | s.code_of_exact(&rounded));
    }
    // Overflow.
    let max_code = sign | s.code_of_exact(&max);
    let inf_code = sign | (s.exp_max_field() << s.mbits);
    let ieee_to_inf = match mode {
        RoundMode::Rne | RoundMode::Rna => true,
        RoundMode::Rtz => false,
        RoundMode::Rup => !neg,
        RoundMode::Rdown => neg,
    };
    match policy {
        Some(Policy::Refuse) => Err(Refusal::new(format!(
            "overflow: {} exceeds the largest finite {} value {}",
            crate::value::rat_human(r),
            fmt.name(),
            crate::value::rat_human(&max)
        ))),
        Some(Policy::Saturate) => Ok(max_code),
        Some(Policy::Inf) => {
            if s.has_inf {
                Ok(inf_code)
            } else {
                Err(Refusal::new(format!("{} has no infinity", fmt.name())))
            }
        }
        None => Ok(if s.has_inf && ieee_to_inf {
            inf_code
        } else {
            max_code
        }),
    }
}

/// The exact code of `r` in `fmt`, or `None` when `r` is not exactly representable.
fn exact_code(fmt: Format, r: &BigRational) -> Option<u64> {
    let code = encode(
        fmt,
        &Value::Num(r.clone()),
        Some(RoundMode::Rtz),
        Some(Policy::Refuse),
    )
    .ok()?;
    match decode(fmt, code).ok()? {
        Value::Num(back) if back == *r => Some(code),
        _ => None,
    }
}

fn number_of(v: &Value) -> Res<Option<BigRational>> {
    Ok(match v {
        Value::Num(r) => Some(r.clone()),
        Value::NegZero => Some(BigRational::zero()),
        Value::NaN | Value::Inf { .. } => None,
        other => {
            return Err(Refusal::new(format!(
                "expected a number, found {}",
                other.kind()
            )))
        }
    })
}

/// `class(fmt, x)`: zero / subnormal / normal / inf / nan. A finite `x` must be exactly
/// representable in `fmt` (use `round` first otherwise).
pub fn class_of(fmt: Format, v: &Value) -> Res<Class> {
    let Some(r) = number_of(v)? else {
        return Ok(if *v == Value::NaN {
            Class::Nan
        } else {
            Class::Inf
        });
    };
    if r.is_zero() {
        return Ok(Class::Zero);
    }
    if exact_code(fmt, &r).is_none() {
        return Err(Refusal::new(format!(
            "{} is not representable in {}; apply round() first",
            crate::value::rat_human(&r),
            fmt.name()
        )));
    }
    Ok(match fmt.kind() {
        Kind::Int { .. } => Class::Normal,
        Kind::Float(s) => {
            if r.abs() < s.min_normal() {
                Class::Subnormal
            } else {
                Class::Normal
            }
        }
    })
}

/// Round `v` into `fmt` and decode the result back to an exact value.
pub fn round_to(
    fmt: Format,
    v: &Value,
    mode: Option<RoundMode>,
    policy: Option<Policy>,
) -> Res<Value> {
    decode(fmt, encode(fmt, v, mode, policy)?)
}

/// Spacing of `fmt` values in the binade containing `|x|` (the gap above `x`).
pub fn ulp(fmt: Format, v: &Value) -> Res<Value> {
    let r = number_of(v)?.ok_or_else(|| Refusal::new("ulp of NaN or infinity is undefined"))?;
    match fmt.kind() {
        Kind::Int { .. } => Ok(Value::Num(BigRational::one())),
        Kind::Float(s) => {
            let a = r.abs();
            if a > s.max_finite() {
                return Err(Refusal::new(format!(
                    "{} is outside the finite range of {}",
                    crate::value::rat_human(&r),
                    fmt.name()
                )));
            }
            let e2 = if a.is_zero() {
                s.emin()
            } else {
                floor_log2(&a)
            };
            Ok(Value::Num(pow2(e2.max(s.emin()) - s.mbits as i64)))
        }
    }
}

fn int_bounds(bits: u32, signed: bool) -> (BigInt, BigInt) {
    if signed {
        (
            -(BigInt::one() << (bits as usize - 1)),
            (BigInt::one() << (bits as usize - 1)) - 1,
        )
    } else {
        (BigInt::zero(), (BigInt::one() << bits as usize) - 1)
    }
}

/// Next representable value above `v` (`up`) or below it.
pub fn neighbour(fmt: Format, v: &Value, up: bool) -> Res<Value> {
    let word = if up { "next" } else { "prev" };
    match fmt.kind() {
        Kind::Int { bits, signed } => {
            let r = number_of(v)?.ok_or_else(|| Refusal::new("NaN has no neighbours"))?;
            exact_code(fmt, &r).ok_or_else(|| {
                Refusal::new(format!(
                    "{} is not representable in {}",
                    crate::value::rat_human(&r),
                    fmt.name()
                ))
            })?;
            let (lo, hi) = int_bounds(bits, signed);
            let i = r.to_integer();
            if up {
                if i == hi {
                    return Err(Refusal::new(format!(
                        "{} is the largest {} value",
                        i,
                        fmt.name()
                    )));
                }
                Ok(Value::Num(BigRational::from_integer(i + 1)))
            } else {
                if i == lo {
                    return Err(Refusal::new(format!(
                        "{} is the smallest {} value",
                        i,
                        fmt.name()
                    )));
                }
                Ok(Value::Num(BigRational::from_integer(i - 1)))
            }
        }
        Kind::Float(s) => {
            let code = value_code(fmt, v)?;
            // Mirror so that "prev" is "next" of the negated code.
            let code = if up { code } else { code ^ sign_bit(&s) };
            let next = next_code(fmt, &s, code, word)?;
            let next = if up { next } else { next ^ sign_bit(&s) };
            decode(fmt, next)
        }
    }
}

/// Code of the smallest value strictly above the value at `code`.
fn next_code(fmt: Format, s: &FloatSpec, code: u64, word: &str) -> Res<u64> {
    let f = s.split(code);
    let mag = code & !sign_bit(s);
    if s.is_inf(&f) {
        if f.sign == 1 {
            return Ok(sign_bit(s) | s.code_of_exact(&s.max_finite()));
        }
        return Err(Refusal::new(format!(
            "{word}: nothing lies beyond +infinity"
        )));
    }
    if mag == 0 {
        return Ok(1); // +-0 -> smallest positive subnormal
    }
    if f.sign == 0 {
        let max_code = s.code_of_exact(&s.max_finite());
        if mag == max_code {
            return if s.has_inf {
                Ok(s.exp_max_field() << s.mbits)
            } else {
                Err(Refusal::new(format!(
                    "{word}: {} has no infinity and this is its largest value",
                    fmt.name()
                )))
            };
        }
        Ok(mag + 1)
    } else {
        Ok(sign_bit(s) | (mag - 1)) // toward zero; code with mag 0 is -0
    }
}

/// Flush a subnormal (relative to `fmt`) to a signed zero; everything else is unchanged.
pub fn flush(fmt: Format, v: &Value) -> Res<Value> {
    let s = fmt.spec("ftz/daz")?;
    match v {
        Value::Num(r) if !r.is_zero() && r.abs() < s.min_normal() => Ok(if r.is_negative() {
            Value::NegZero
        } else {
            Value::Num(BigRational::zero())
        }),
        other => Ok(other.clone()),
    }
}

/// Split a code into sign / exponent / mantissa fields.
pub fn fields_of(fmt: Format, code: u64) -> Res<Fields> {
    let s = fmt.spec("fields")?;
    decode(fmt, code)?; // range check
    Ok(s.split(code))
}

/// The exact code of a value that must be representable in float format `fmt`.
fn value_code(fmt: Format, v: &Value) -> Res<u64> {
    let s = fmt.spec("next/prev")?;
    match v {
        Value::NaN => Err(Refusal::new("next/prev of NaN is undefined")),
        Value::Inf { .. } => encode(fmt, v, None, None),
        Value::NegZero => Ok(sign_bit(&s)),
        Value::Num(r) if r.is_zero() => Ok(0),
        Value::Num(r) => exact_code(fmt, r).ok_or_else(|| {
            Refusal::new(format!(
                "{} is not representable in {}",
                crate::value::rat_human(r),
                fmt.name()
            ))
        }),
        other => Err(Refusal::new(format!(
            "expected a number, found {}",
            other.kind()
        ))),
    }
}
