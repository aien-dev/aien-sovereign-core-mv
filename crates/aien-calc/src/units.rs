//! Units and dimensions (calc-lang v1, spec section 2.3).
//!
//! Base dimensions: byte, second, token, param, launch, FLOP. `bit` is 1/8 byte and `Hz` is 1/second.
//! Prefixes are strict: `k K M G T` are powers of ten (a bare `GB` is decimal), `Ki Mi Gi Ti` are powers
//! of two and only exist on `B` and `bit`, and `m u n` exist only on `s`. Compound units are read left to
//! right: `B/param*tok` is `(B/param)*tok`.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use std::collections::BTreeSet;

pub const BASE_NAMES: [&str; 6] = ["B", "s", "tok", "param", "launch", "FLOP"];

/// Exponent of each base dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Dim(pub [i32; 6]);

impl Dim {
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|e| *e == 0)
    }

    pub fn mul(&self, o: &Dim) -> Dim {
        let mut d = *self;
        for (a, b) in d.0.iter_mut().zip(o.0.iter()) {
            *a += b;
        }
        d
    }

    pub fn div(&self, o: &Dim) -> Dim {
        let mut d = *self;
        for (a, b) in d.0.iter_mut().zip(o.0.iter()) {
            *a -= b;
        }
        d
    }

    pub fn pow(&self, n: i32) -> Dim {
        let mut d = *self;
        for a in d.0.iter_mut() {
            *a *= n;
        }
        d
    }

    /// `B/tok`, `FLOP/s`, `1/s`, `B*B`; empty for a dimensionless value.
    pub fn name(&self) -> String {
        let part = |positive: bool| -> Vec<String> {
            let mut v = Vec::new();
            for (i, e) in self.0.iter().enumerate() {
                let m = if positive { *e } else { -*e };
                for _ in 0..m.max(0) {
                    v.push(BASE_NAMES[i].to_string());
                }
            }
            v
        };
        let (num, den) = (part(true), part(false));
        match (num.is_empty(), den.is_empty()) {
            (true, true) => String::new(),
            (false, true) => num.join("*"),
            (true, false) => format!("1/{}", den.join("/")),
            (false, false) => format!("{}/{}", num.join("*"), den.join("/")),
        }
    }
}

fn pow_int(base: u32, exp: i64) -> BigRational {
    let p = num_traits::pow(BigInt::from(base), exp.unsigned_abs() as usize);
    if exp >= 0 {
        BigRational::from_integer(p)
    } else {
        BigRational::new(BigInt::one(), p)
    }
}

/// A parsed unit expression: its scale to base units, its dimension and the prefixes it used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitExpr {
    /// Canonical text, atoms joined by `/` and `*` with no spaces.
    pub text: String,
    pub factor: BigRational,
    pub dim: Dim,
    pub prefixes: BTreeSet<String>,
}

struct Atom {
    factor: BigRational,
    dim: Dim,
    prefix: Option<String>,
}

fn base_dim(i: usize) -> Dim {
    let mut d = Dim::default();
    d.0[i] = 1;
    d
}

/// Base unit name -> (factor, dim, is information unit).
fn base_unit(name: &str) -> Option<(BigRational, Dim, bool)> {
    let one = BigRational::one();
    Some(match name {
        "B" => (one, base_dim(0), true),
        "bit" => (
            BigRational::new(BigInt::one(), BigInt::from(8)),
            base_dim(0),
            true,
        ),
        "s" => (one, base_dim(1), false),
        "tok" => (one, base_dim(2), false),
        "param" => (one, base_dim(3), false),
        "launch" => (one, base_dim(4), false),
        "FLOP" => (one, base_dim(5), false),
        "Hz" => (one, base_dim(1).pow(-1), false),
        _ => return None,
    })
}

fn atom(name: &str) -> Option<Atom> {
    if let Some((factor, dim, _)) = base_unit(name) {
        return Some(Atom {
            factor,
            dim,
            prefix: None,
        });
    }
    let binary: [(&str, i64); 4] = [("Ki", 10), ("Mi", 20), ("Gi", 30), ("Ti", 40)];
    for (p, bits) in binary {
        if let Some(rest) = name.strip_prefix(p) {
            if let Some((factor, dim, info)) = base_unit(rest) {
                if info {
                    let f = factor * BigRational::from_integer(BigInt::one() << bits as usize);
                    return Some(Atom {
                        factor: f,
                        dim,
                        prefix: Some(format!("{name}:binary(2^{bits})")),
                    });
                }
            }
        }
    }
    let decimal: [(char, i64); 5] = [('k', 3), ('K', 3), ('M', 6), ('G', 9), ('T', 12)];
    for (p, exp) in decimal {
        if let Some(rest) = name.strip_prefix(p) {
            if let Some((factor, dim, _)) = base_unit(rest) {
                return Some(Atom {
                    factor: factor * pow_int(10, exp),
                    dim,
                    prefix: Some(format!("{name}:decimal(10^{exp})")),
                });
            }
        }
    }
    let small: [(char, i64); 3] = [('m', -3), ('u', -6), ('n', -9)];
    for (p, exp) in small {
        if let Some(rest) = name.strip_prefix(p) {
            if rest == "s" {
                let (factor, dim, _) = base_unit("s")?;
                return Some(Atom {
                    factor: factor * pow_int(10, exp),
                    dim,
                    prefix: Some(format!("{name}:decimal(10^{exp})")),
                });
            }
        }
    }
    None
}

/// Is `name` a unit atom (`GB`, `ms`, `tok`, ...)?
pub fn is_unit_name(name: &str) -> bool {
    atom(name).is_some()
}

impl UnitExpr {
    /// A single atom as a unit expression.
    pub fn from_atom(name: &str) -> Option<UnitExpr> {
        UnitExpr::from_parts(&[(name.to_string(), '*')])
    }

    /// `parts` are (atom, operator-before-it); the first operator is ignored.
    pub fn from_parts(parts: &[(String, char)]) -> Option<UnitExpr> {
        let mut factor = BigRational::one();
        let mut dim = Dim::default();
        let mut prefixes = BTreeSet::new();
        let mut text = String::new();
        for (i, (name, op)) in parts.iter().enumerate() {
            let a = atom(name)?;
            if i == 0 || *op == '*' {
                factor *= a.factor;
                dim = dim.mul(&a.dim);
            } else {
                factor /= a.factor;
                dim = dim.div(&a.dim);
            }
            if let Some(p) = a.prefix {
                prefixes.insert(p);
            }
            if i > 0 {
                text.push(*op);
            }
            text.push_str(name);
        }
        if parts.is_empty() {
            return None;
        }
        Some(UnitExpr {
            text,
            factor,
            dim,
            prefixes,
        })
    }

    /// Parse canonical unit text such as `GB/s` or `B/param*tok`.
    pub fn parse_text(s: &str) -> Option<UnitExpr> {
        let mut parts = Vec::new();
        let mut cur = String::new();
        let mut op = '*';
        for c in s.chars() {
            if c == '/' || c == '*' {
                parts.push((std::mem::take(&mut cur), op));
                op = c;
            } else {
                cur.push(c);
            }
        }
        parts.push((cur, op));
        UnitExpr::from_parts(&parts)
    }
}
