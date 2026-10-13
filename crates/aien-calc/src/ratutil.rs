//! Exact helpers on rationals: powers of two, binary exponent, integer rounding by mode.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// IEEE-style rounding directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundMode {
    /// Nearest, ties to even.
    Rne,
    /// Toward zero.
    Rtz,
    /// Toward +infinity.
    Rup,
    /// Toward -infinity.
    Rdown,
    /// Nearest, ties away from zero.
    Rna,
}

impl RoundMode {
    pub fn name(self) -> &'static str {
        match self {
            RoundMode::Rne => "rne",
            RoundMode::Rtz => "rtz",
            RoundMode::Rup => "rup",
            RoundMode::Rdown => "rdown",
            RoundMode::Rna => "rna",
        }
    }

    pub fn from_name(s: &str) -> Option<RoundMode> {
        Some(match s {
            "rne" => RoundMode::Rne,
            "rtz" => RoundMode::Rtz,
            "rup" => RoundMode::Rup,
            "rdown" => RoundMode::Rdown,
            "rna" => RoundMode::Rna,
            _ => return None,
        })
    }
}

/// 2^k as an exact rational (k may be negative).
pub fn pow2(k: i64) -> BigRational {
    if k >= 0 {
        BigRational::from_integer(BigInt::one() << (k as usize))
    } else {
        BigRational::new(BigInt::one(), BigInt::one() << ((-k) as usize))
    }
}

/// floor(log2(r)) for r > 0.
pub fn floor_log2(r: &BigRational) -> i64 {
    debug_assert!(r.is_positive());
    let est = r.numer().bits() as i64 - r.denom().bits() as i64;
    if *r < pow2(est) {
        est - 1
    } else {
        est
    }
}

/// Round the non-negative rational `mag` to an integer. `negative` says the original value was
/// negative (directed modes act on the signed value, `mag` is its absolute value).
pub fn round_mag(mag: &BigRational, mode: RoundMode, negative: bool) -> BigInt {
    debug_assert!(!mag.is_negative());
    let (floor, rem) = mag.numer().div_mod_floor(mag.denom());
    let up = &floor + BigInt::one();
    if rem.is_zero() {
        return floor;
    }
    // Compare rem/denom with 1/2 via 2*rem vs denom.
    let twice = &rem * BigInt::from(2u8);
    let half_cmp = twice.cmp(mag.denom());
    match mode {
        RoundMode::Rtz => floor,
        RoundMode::Rup => {
            if negative {
                floor
            } else {
                up
            }
        }
        RoundMode::Rdown => {
            if negative {
                up
            } else {
                floor
            }
        }
        RoundMode::Rna => {
            if half_cmp == std::cmp::Ordering::Less {
                floor
            } else {
                up
            }
        }
        RoundMode::Rne => match half_cmp {
            std::cmp::Ordering::Less => floor,
            std::cmp::Ordering::Greater => up,
            std::cmp::Ordering::Equal => {
                if floor.is_even() {
                    floor
                } else {
                    up
                }
            }
        },
    }
}

/// Round a signed rational to an integer.
pub fn round_signed(r: &BigRational, mode: RoundMode) -> BigInt {
    let neg = r.is_negative();
    let k = round_mag(&r.abs(), mode, neg);
    if neg {
        -k
    } else {
        k
    }
}
