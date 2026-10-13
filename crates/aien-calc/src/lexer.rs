//! Lexer for calc-lang v1.

use crate::error::CalcError;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

/// Largest numerator or denominator, in bits, that any value may reach (see [`crate::eval`]).
pub const MAX_BITS: u64 = 1 << 17;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Num(BigRational),
    Ident(String),
    Var(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
    Eof,
}

fn err(pos: usize, msg: impl Into<String>) -> CalcError {
    CalcError::Syntax {
        pos,
        msg: msg.into(),
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Digits of `radix` with single underscores allowed strictly between digits.
fn read_digits(chars: &[char], i: &mut usize, radix: u32) -> Result<String, CalcError> {
    let start = *i;
    let mut out = String::new();
    let mut last_underscore = false;
    while *i < chars.len() {
        let c = chars[*i];
        if c == '_' {
            if out.is_empty() || last_underscore {
                return Err(err(*i, "misplaced `_` in number"));
            }
            last_underscore = true;
        } else if c.is_digit(radix) {
            out.push(c);
            last_underscore = false;
        } else {
            break;
        }
        *i += 1;
    }
    if last_underscore {
        return Err(err(*i - 1, "number may not end with `_`"));
    }
    if out.is_empty() {
        return Err(err(start, "expected digits"));
    }
    Ok(out)
}

fn pow10(exp: u64) -> BigInt {
    num_traits::pow(BigInt::from(10u8), exp as usize)
}

fn lex_number(chars: &[char], i: &mut usize) -> Result<BigRational, CalcError> {
    let start = *i;
    if chars[*i] == '0' && *i + 1 < chars.len() && matches!(chars[*i + 1], 'x' | 'X' | 'b' | 'B') {
        let radix = if matches!(chars[*i + 1], 'x' | 'X') {
            16
        } else {
            2
        };
        *i += 2;
        let digits = read_digits(chars, i, radix)?;
        let n = BigInt::parse_bytes(digits.as_bytes(), radix)
            .ok_or_else(|| err(start, "bad number"))?;
        return Ok(BigRational::from_integer(n));
    }
    let int_part = read_digits(chars, i, 10)?;
    let mut frac_part = String::new();
    if *i < chars.len() && chars[*i] == '.' {
        *i += 1;
        frac_part = read_digits(chars, i, 10).map_err(|_| err(*i, "expected digits after `.`"))?;
    }
    let mut exp10: i64 = 0;
    if *i < chars.len() && matches!(chars[*i], 'e' | 'E') {
        *i += 1;
        let mut neg = false;
        if *i < chars.len() && matches!(chars[*i], '+' | '-') {
            neg = chars[*i] == '-';
            *i += 1;
        }
        let digits =
            read_digits(chars, i, 10).map_err(|_| err(*i, "expected exponent digits after `e`"))?;
        let mag: i64 = digits.parse().unwrap_or(i64::MAX);
        exp10 = if neg { -mag } else { mag };
    }
    let mantissa = BigInt::parse_bytes(format!("{int_part}{frac_part}").as_bytes(), 10)
        .ok_or_else(|| err(start, "bad number"))?;
    let scale = frac_part.len() as i64;
    let net = exp10.saturating_sub(scale);
    // 10^k needs about 3.33*k bits; refuse before building anything enormous.
    if net.unsigned_abs() > MAX_BITS / 4 {
        return Err(err(start, "number literal exceeds the size budget"));
    }
    if mantissa.is_zero() {
        return Ok(BigRational::zero());
    }
    let r = if net >= 0 {
        BigRational::from_integer(mantissa * pow10(net as u64))
    } else {
        BigRational::new(mantissa, pow10(net.unsigned_abs()))
    };
    Ok(r)
}

/// Tokenise `src`; each token carries its character offset.
pub fn lex(src: &str) -> Result<Vec<(Tok, usize)>, CalcError> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let pos = i;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() {
            let n = lex_number(&chars, &mut i)?;
            if i < chars.len() && (is_ident_char(chars[i]) || chars[i] == '.') {
                return Err(err(i, format!("unexpected `{}` after number", chars[i])));
            }
            toks.push((Tok::Num(n), pos));
        } else if is_ident_start(c) {
            let mut s = String::new();
            while i < chars.len() && is_ident_char(chars[i]) {
                s.push(chars[i]);
                i += 1;
            }
            toks.push((Tok::Ident(s), pos));
        } else if c == '$' {
            i += 1;
            let mut s = String::new();
            while i < chars.len() && is_ident_char(chars[i]) {
                s.push(chars[i]);
                i += 1;
            }
            if s.is_empty() || !is_ident_start(s.chars().next().unwrap_or('0')) {
                return Err(err(pos, "`$` must be followed by a variable name"));
            }
            toks.push((Tok::Var(s), pos));
        } else {
            let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
            let op2: Option<&'static str> = match two.as_str() {
                "<<" => Some("<<"),
                ">>" => Some(">>"),
                "==" => Some("=="),
                "!=" => Some("!="),
                "<=" => Some("<="),
                ">=" => Some(">="),
                _ => None,
            };
            if let Some(op) = op2 {
                toks.push((Tok::Op(op), pos));
                i += 2;
                continue;
            }
            let t = match c {
                '+' => Tok::Op("+"),
                '-' => Tok::Op("-"),
                '*' => Tok::Op("*"),
                '/' => Tok::Op("/"),
                '%' => Tok::Op("%"),
                '^' => Tok::Op("^"),
                '&' => Tok::Op("&"),
                '|' => Tok::Op("|"),
                '<' => Tok::Op("<"),
                '>' => Tok::Op(">"),
                '(' => Tok::LParen,
                ')' => Tok::RParen,
                ',' => Tok::Comma,
                other => return Err(err(pos, format!("unexpected character `{other}`"))),
            };
            toks.push((t, pos));
            i += 1;
        }
    }
    toks.push((Tok::Eof, chars.len()));
    Ok(toks)
}
