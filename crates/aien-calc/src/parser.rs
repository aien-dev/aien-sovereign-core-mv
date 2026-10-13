//! Pratt parser for calc-lang v1 and the canonical (prefix) form used in receipts.
//!
//! Precedence, loosest to tightest (all left associative unless noted):
//! comparison (non-associative) < `|` < `xor` < `&` < `<< >>` < `+ -` < `* / %` < unary `-` < `^` (right).
//! This is the Python ordering: `x & 7 == 7` means `(x & 7) == 7`, and `-2^2` is `-(2^2)`.

use crate::error::CalcError;
use crate::lexer::{lex, Tok};
use crate::units::{is_unit_name, UnitExpr};
use num_rational::BigRational;
use num_traits::One;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `m x n`: a shape, only meaningful inside `params()`.
    By,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Pow => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "xor",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::By => "x",
        }
    }

    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ast {
    Num(BigRational),
    Ident(String),
    Var(String),
    Neg(Box<Ast>),
    Bin(BinOp, Box<Ast>, Box<Ast>),
    Call(String, Vec<Ast>),
    /// A number with a unit: `273 GB/s`.
    Qty(BigRational, UnitExpr),
    /// `expr -> unit`.
    Convert(Box<Ast>, UnitExpr),
}

impl Ast {
    /// Prefixes used by unit literals and `->` targets, e.g. `GiB:binary(2^30)`.
    pub fn prefixes(&self) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        self.collect_prefixes(&mut out);
        out
    }

    fn collect_prefixes(&self, out: &mut std::collections::BTreeSet<String>) {
        match self {
            Ast::Qty(_, u) => out.extend(u.prefixes.iter().cloned()),
            Ast::Convert(a, u) => {
                a.collect_prefixes(out);
                out.extend(u.prefixes.iter().cloned());
            }
            Ast::Neg(a) => a.collect_prefixes(out),
            Ast::Bin(_, a, b) => {
                a.collect_prefixes(out);
                b.collect_prefixes(out);
            }
            Ast::Call(_, args) => args.iter().for_each(|a| a.collect_prefixes(out)),
            Ast::Num(_) | Ast::Ident(_) | Ast::Var(_) => {}
        }
    }

    /// Canonical prefix form: whitespace and literal spelling (`0x7f`, `127`) disappear, structure stays.
    /// Numbers print as `n` or `p/q`; operators and functions as `(head arg ...)`.
    pub fn canonical(&self) -> String {
        match self {
            Ast::Num(r) => {
                if r.denom().is_one() {
                    r.numer().to_string()
                } else {
                    format!("{}/{}", r.numer(), r.denom())
                }
            }
            Ast::Ident(s) => s.clone(),
            Ast::Var(s) => format!("${s}"),
            Ast::Neg(a) => format!("(neg {})", a.canonical()),
            Ast::Qty(r, u) => format!("(qty {} {})", Ast::Num(r.clone()).canonical(), u.text),
            Ast::Convert(a, u) => format!("(-> {} {})", a.canonical(), u.text),
            Ast::Bin(op, a, b) => format!("({} {} {})", op.symbol(), a.canonical(), b.canonical()),
            Ast::Call(name, args) => {
                let mut s = format!("({name}");
                for a in args {
                    s.push(' ');
                    s.push_str(&a.canonical());
                }
                s.push(')');
                s
            }
        }
    }
}

const MAX_DEPTH: usize = 128;

struct Parser {
    toks: Vec<(Tok, usize)>,
    i: usize,
    depth: usize,
}

fn err(pos: usize, msg: impl Into<String>) -> CalcError {
    CalcError::Syntax {
        pos,
        msg: msg.into(),
    }
}

/// (left binding power, right binding power, operator)
fn infix(tok: &Tok) -> Option<(u8, u8, BinOp)> {
    let op = match tok {
        Tok::Op(s) => match *s {
            "==" => (1, 2, BinOp::Eq),
            "!=" => (1, 2, BinOp::Ne),
            "<" => (1, 2, BinOp::Lt),
            "<=" => (1, 2, BinOp::Le),
            ">" => (1, 2, BinOp::Gt),
            ">=" => (1, 2, BinOp::Ge),
            "|" => (3, 4, BinOp::Or),
            "&" => (7, 8, BinOp::And),
            "<<" => (9, 10, BinOp::Shl),
            ">>" => (9, 10, BinOp::Shr),
            "+" => (11, 12, BinOp::Add),
            "-" => (11, 12, BinOp::Sub),
            "*" => (13, 14, BinOp::Mul),
            "/" => (13, 14, BinOp::Div),
            "%" => (13, 14, BinOp::Rem),
            "^" => (17, 16, BinOp::Pow),
            _ => return None,
        },
        Tok::Ident(s) if s == "xor" => (5, 6, BinOp::Xor),
        Tok::Ident(s) if s == "x" => (13, 14, BinOp::By),
        _ => return None,
    };
    Some(op)
}

const UNARY_BP: u8 = 15;

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.i].0
    }

    fn pos(&self) -> usize {
        self.toks[self.i].1
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.i].0.clone();
        if self.i + 1 < self.toks.len() {
            self.i += 1;
        }
        t
    }

    fn expr(&mut self, min_bp: u8) -> Result<Ast, CalcError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(err(self.pos(), "expression nests too deeply"));
        }
        let mut lhs = self.primary()?;
        let mut compared = false;
        while let Some((lbp, rbp, op)) = infix(self.peek()) {
            if lbp < min_bp {
                break;
            }
            if op.is_comparison() {
                if compared {
                    return Err(err(
                        self.pos(),
                        "comparisons do not chain; parenthesise one side",
                    ));
                }
                compared = true;
            }
            self.bump();
            let rhs = self.expr(rbp)?;
            lhs = Ast::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        while min_bp == 0 && matches!(self.peek(), Tok::Op("->")) {
            self.bump();
            let unit = self.unit_expr()?;
            lhs = Ast::Convert(Box::new(lhs), unit);
        }
        self.depth -= 1;
        Ok(lhs)
    }

    /// Is the token after the current one a unit atom?
    fn next_is_unit(&self, offset: usize) -> bool {
        matches!(self.toks.get(self.i + offset), Some((Tok::Ident(n), _)) if is_unit_name(n))
    }

    /// `GB`, `GB/s`, `B/param*tok`: atoms joined by `/` or `*`, read left to right.
    fn unit_expr(&mut self) -> Result<UnitExpr, CalcError> {
        let pos = self.pos();
        let mut parts: Vec<(String, char)> = Vec::new();
        match self.bump() {
            Tok::Ident(n) if is_unit_name(&n) => parts.push((n, '*')),
            _ => return Err(err(pos, "expected a unit")),
        }
        while let Tok::Op(o @ ("/" | "*")) = self.peek().clone() {
            if !self.next_is_unit(1) {
                break;
            }
            self.bump();
            if let Tok::Ident(n) = self.bump() {
                parts.push((n, o.chars().next().unwrap_or('*')));
            }
        }
        UnitExpr::from_parts(&parts).ok_or_else(|| err(pos, "bad unit"))
    }

    fn primary(&mut self) -> Result<Ast, CalcError> {
        let pos = self.pos();
        match self.bump() {
            Tok::Num(r) => {
                if self.next_is_unit(0) {
                    let u = self.unit_expr()?;
                    Ok(Ast::Qty(r, u))
                } else {
                    Ok(Ast::Num(r))
                }
            }
            Tok::Var(v) => Ok(Ast::Var(v)),
            Tok::Op("-") => {
                let inner = self.expr(UNARY_BP)?;
                Ok(Ast::Neg(Box::new(inner)))
            }
            Tok::LParen => {
                let inner = self.expr(0)?;
                match self.bump() {
                    Tok::RParen => Ok(inner),
                    _ => Err(err(self.pos(), "expected `)`")),
                }
            }
            Tok::Ident(name) => {
                if name == "xor" {
                    return Err(err(pos, "`xor` is an operator and needs a left operand"));
                }
                if matches!(self.peek(), Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if matches!(self.peek(), Tok::RParen) {
                        self.bump();
                        return Ok(Ast::Call(name, args));
                    }
                    loop {
                        args.push(self.expr(0)?);
                        match self.bump() {
                            Tok::Comma => continue,
                            Tok::RParen => break,
                            _ => return Err(err(self.pos(), "expected `,` or `)` in call")),
                        }
                    }
                    Ok(Ast::Call(name, args))
                } else {
                    Ok(Ast::Ident(name))
                }
            }
            Tok::Eof => Err(err(pos, "unexpected end of expression")),
            Tok::RParen => Err(err(pos, "unexpected `)`")),
            Tok::Comma => Err(err(pos, "unexpected `,`")),
            Tok::Op(s) => Err(err(pos, format!("unexpected operator `{s}`"))),
        }
    }
}

/// Parse a full expression; trailing tokens are an error.
pub fn parse(src: &str) -> Result<Ast, CalcError> {
    let toks = lex(src)?;
    let mut p = Parser {
        toks,
        i: 0,
        depth: 0,
    };
    let ast = p.expr(0)?;
    if !matches!(p.peek(), Tok::Eof) {
        return Err(err(p.pos(), "unexpected token after expression"));
    }
    Ok(ast)
}
