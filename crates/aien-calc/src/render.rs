//! Decimal rendering of exact rationals: rounded only here, only at the end, to a stated number
//! of significant digits (half-even), with an exactness flag.

use crate::ratutil::{round_mag, RoundMode};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn pow10(k: u32) -> BigInt {
    num_traits::pow(BigInt::from(10u8), k as usize)
}

fn rat_pow10(k: i64) -> BigRational {
    if k >= 0 {
        BigRational::from_integer(pow10(k as u32))
    } else {
        BigRational::new(BigInt::one(), pow10((-k) as u32))
    }
}

/// floor(log10(a)) for a > 0.
fn floor_log10(a: &BigRational) -> i64 {
    // log10(2) ~ 0.30103: integer estimate from the binary exponent, then corrected exactly.
    let e2 = crate::ratutil::floor_log2(a);
    let mut k = (e2 * 30103).div_euclid(100_000);
    while rat_pow10(k + 1) <= *a {
        k += 1;
    }
    while rat_pow10(k) > *a {
        k -= 1;
    }
    k
}

/// Render `q` with `sig` significant digits. Returns the text and whether it equals `q` exactly.
/// Plain notation for decimal exponents in `[-5, 21)`, scientific (`1.175e-38`) outside.
pub fn decimal(q: &BigRational, sig: u32) -> (String, bool) {
    if q.is_zero() {
        return ("0".to_string(), true);
    }
    let neg = q.is_negative();
    let a = q.abs();
    let mut k = floor_log10(&a);
    let scaled = &a * rat_pow10(sig as i64 - 1 - k);
    let exact = scaled.denom().is_one();
    let mut n = round_mag(&scaled, RoundMode::Rne, false);
    // Rounding may carry to 10^sig (e.g. 9.99 -> 10.0): renormalise.
    if n == pow10(sig) {
        n /= BigInt::from(10u8);
        k += 1;
    }
    let all = n.to_string();
    let digits = all.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-5..21).contains(&k) {
        if k >= 0 {
            let int_len = (k + 1) as usize;
            if digits.len() <= int_len {
                out.push_str(digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-k - 1) as usize));
            out.push_str(digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!("e{k}"));
    }
    (out, exact)
}
