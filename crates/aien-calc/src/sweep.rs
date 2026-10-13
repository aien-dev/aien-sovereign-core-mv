//! The `sweep` tool: evaluate `expect` and `got` (and optionally `flag`) for every code of a small
//! format, with `$b` bound to the code and `$x` to its exact decoded value.
//!
//! Result comparison is exact: numbers compare as rationals (a signed zero equals zero), and
//! NaN equals NaN as a sweep result. Per-code refusals are counted and listed, never dropped.
//! A BLAKE3 digest over the full per-code table is part of the receipt.

use crate::error::{CalcError, Refusal};
use crate::eval::{evaluate_top as evaluate, Env};
use crate::format::{decode, Format};
use crate::json::Json;
use crate::parser::{parse, Ast};
use crate::receipt::{method_block, Receipt, ENGINE_VERSION, SCHEMA};
use crate::value::Value;

/// Largest sweep, in codes (16-bit formats). f32 and f64 are refused.
pub const MAX_SWEEP_CODES: u64 = 1 << 16;
/// Entries listed in a receipt per list; counts are always exact.
pub const LIST_CAP: usize = 1024;
/// Mismatches printed in the human text.
const TEXT_MISMATCHES: usize = 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub code: u64,
    pub expect: String,
    pub got: String,
}

#[derive(Clone, Debug)]
pub struct SweepOutcome {
    receipt: Receipt,
    text: String,
    total: u64,
    considered: u64,
    equal: u64,
    mismatches: Vec<Mismatch>,
    flagged: Vec<u64>,
    refused: Vec<(u64, String)>,
    table_digest: String,
}

impl SweepOutcome {
    pub fn receipt(&self) -> &Receipt {
        &self.receipt
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Codes in the format.
    pub fn total(&self) -> u64 {
        self.total
    }
    /// Codes that passed `where`.
    pub fn considered(&self) -> u64 {
        self.considered
    }
    pub fn equal(&self) -> u64 {
        self.equal
    }
    pub fn mismatch_count(&self) -> u64 {
        self.mismatches.len() as u64
    }
    pub fn mismatches(&self) -> &[Mismatch] {
        &self.mismatches
    }
    /// Considered codes where `flag` evaluated to true, ascending.
    pub fn flagged(&self) -> &[u64] {
        &self.flagged
    }
    /// Considered codes where `expect`, `got` or `flag` refused, with the reason.
    pub fn refused(&self) -> &[(u64, String)] {
        &self.refused
    }
    /// BLAKE3 hex of the full per-code table.
    pub fn table_digest(&self) -> &str {
        &self.table_digest
    }
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::NegZero, Value::NegZero) => true,
        (Value::NegZero, Value::Num(y)) | (Value::Num(y), Value::NegZero) => {
            num_traits::Zero::is_zero(y)
        }
        _ => a == b,
    }
}

fn hex(code: u64, digits: usize) -> String {
    format!("{code:#0width$x}", width = digits + 2)
}

/// Compress ascending codes into `0x01..0x07, 0x09` text (runs of three or more become ranges).
fn ranges(codes: &[u64], digits: usize) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < codes.len() {
        let mut j = i;
        while j + 1 < codes.len() && codes[j + 1] == codes[j] + 1 {
            j += 1;
        }
        if j - i >= 2 {
            parts.push(format!(
                "{}..{}",
                hex(codes[i], digits),
                hex(codes[j], digits)
            ));
        } else {
            for c in &codes[i..=j] {
                parts.push(hex(*c, digits));
            }
        }
        i = j + 1;
    }
    parts.join(", ")
}

fn parse_opt(src: Option<&str>) -> Result<Option<Ast>, CalcError> {
    src.map(parse).transpose()
}

fn shown(r: &Result<Value, Refusal>) -> String {
    match r {
        Ok(v) => v.canonical(),
        Err(e) => format!("!{}", e.0),
    }
}

fn mismatch_json(m: &Mismatch, digits: usize) -> Json {
    Json::obj(vec![
        ("code", Json::Str(hex(m.code, digits))),
        ("expect", Json::Str(m.expect.clone())),
        ("got", Json::Str(m.got.clone())),
    ])
}

/// Run a sweep. `format` must be a format of at most 16 bits.
pub fn sweep(
    format: &str,
    expect: &str,
    got: &str,
    flag: Option<&str>,
    where_: Option<&str>,
) -> Result<SweepOutcome, CalcError> {
    let fmt = Format::from_name(format)
        .ok_or_else(|| CalcError::Invalid(format!("unknown format `{format}`")))?;
    let total = 1u64 << fmt.width().min(63);
    if fmt.width() > 16 {
        return Err(CalcError::Invalid(format!(
            "sweep of {} refused: {total} codes exceeds the {MAX_SWEEP_CODES}-code limit (use where ranges or next/prev walks)",
            fmt.name()
        )));
    }
    sweep_ast(
        format,
        &parse(expect)?,
        &parse(got)?,
        parse_opt(flag)?.as_ref(),
        parse_opt(where_)?.as_ref(),
    )
}

/// [`sweep`] on already parsed expressions (also how `verify` recomputes a stored receipt).
pub fn sweep_ast(
    format: &str,
    expect_ast: &Ast,
    got_ast: &Ast,
    flag_ast: Option<&Ast>,
    where_ast: Option<&Ast>,
) -> Result<SweepOutcome, CalcError> {
    let fmt = Format::from_name(format)
        .ok_or_else(|| CalcError::Invalid(format!("unknown format `{format}`")))?;
    let total = 1u64 << fmt.width().min(63);
    if fmt.width() > 16 {
        return Err(CalcError::Invalid(format!(
            "sweep of {} refused: {total} codes exceeds the {MAX_SWEEP_CODES}-code limit (use where ranges or next/prev walks)",
            fmt.name()
        )));
    }
    let digits = (fmt.width() as usize).div_ceil(4);

    let mut table = format!("aien-calc-sweep-table v1 {}\n", fmt.name());
    let (mut considered, mut equal) = (0u64, 0u64);
    let mut mismatches = Vec::new();
    let mut flagged = Vec::new();
    let mut refused: Vec<(u64, String)> = Vec::new();
    for code in 0..total {
        let x = decode(fmt, code).map_err(|e| CalcError::Invalid(e.0))?;
        let mut env = Env::default();
        env.vars.insert(
            "b".into(),
            Value::Num(num_rational::BigRational::from_integer(
                num_bigint::BigInt::from(code),
            )),
        );
        env.vars.insert("x".into(), x);
        if let Some(w) = &where_ast {
            match evaluate(w, &env) {
                Ok(Value::Bool(true)) => {}
                Ok(Value::Bool(false)) => {
                    table.push_str(&format!("{}\texcluded\n", hex(code, digits)));
                    continue;
                }
                Ok(other) => {
                    return Err(CalcError::Invalid(format!(
                        "where must be true or false, got {} at code {}",
                        other.kind(),
                        hex(code, digits)
                    )))
                }
                Err(e) => {
                    return Err(CalcError::Invalid(format!(
                        "where refused at code {}: {}",
                        hex(code, digits),
                        e.0
                    )))
                }
            }
        }
        considered += 1;
        let e = evaluate(expect_ast, &env);
        let g = evaluate(got_ast, &env);
        let f = flag_ast.as_ref().map(|a| match evaluate(a, &env) {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(Refusal::new(format!(
                "flag must be true or false, got {}",
                other.kind()
            ))),
            Err(e) => Err(e),
        });
        let status = match (&e, &g, &f) {
            (Err(r), _, _) | (_, Err(r), _) | (_, _, Some(Err(r))) => {
                refused.push((code, r.0.clone()));
                "refused"
            }
            (Ok(ev), Ok(gv), _) => {
                if matches!(f, Some(Ok(true))) {
                    flagged.push(code);
                }
                if same(ev, gv) {
                    equal += 1;
                    "equal"
                } else {
                    mismatches.push(Mismatch {
                        code,
                        expect: ev.canonical(),
                        got: gv.canonical(),
                    });
                    "mismatch"
                }
            }
        };
        let flag_cell = match &f {
            None => "-".to_string(),
            Some(Ok(b)) => b.to_string(),
            Some(Err(r)) => format!("!{}", r.0),
        };
        table.push_str(&format!(
            "{}\t{status}\t{}\t{}\t{flag_cell}\n",
            hex(code, digits),
            shown(&e),
            shown(&g)
        ));
    }
    let table_digest = blake3::hash(table.as_bytes()).to_hex().to_string();

    let mut warnings = Vec::new();
    let capped = |n: usize, what: &str, warnings: &mut Vec<Json>| {
        if n > LIST_CAP {
            warnings.push(Json::Str(format!("{what} list truncated to {LIST_CAP} of {n} entries; counts are exact and the table digest covers all rows")));
        }
        n > LIST_CAP
    };
    let mism_trunc = capped(mismatches.len(), "mismatch", &mut warnings);
    let flag_trunc = capped(flagged.len(), "flag_true", &mut warnings);
    let ref_trunc = capped(refused.len(), "refused", &mut warnings);

    let canon = |a: &Option<&Ast>| {
        a.as_ref()
            .map(|x| Json::Str(x.canonical()))
            .unwrap_or(Json::Null)
    };
    let body = Json::obj(vec![
        ("calc_version", Json::str(ENGINE_VERSION)),
        ("exactness", Json::str("exact")),
        (
            "inputs",
            Json::obj(vec![
                ("expect", Json::Str(expect_ast.canonical())),
                ("flag", canon(&flag_ast)),
                ("format", Json::str(fmt.name())),
                ("got", Json::Str(got_ast.canonical())),
                ("where", canon(&where_ast)),
            ]),
        ),
        ("method", method_block(None)),
        (
            "result",
            Json::obj(vec![
                ("considered", Json::Int(considered as i64)),
                ("equal", Json::Int(equal as i64)),
                ("excluded", Json::Int((total - considered) as i64)),
                (
                    "flag_true",
                    Json::Arr(
                        flagged
                            .iter()
                            .take(LIST_CAP)
                            .map(|c| Json::Str(hex(*c, digits)))
                            .collect(),
                    ),
                ),
                ("flag_true_count", Json::Int(flagged.len() as i64)),
                ("flag_true_truncated", Json::Bool(flag_trunc)),
                ("kind", Json::str("sweep")),
                ("mismatch_count", Json::Int(mismatches.len() as i64)),
                (
                    "mismatches",
                    Json::Arr(
                        mismatches
                            .iter()
                            .take(LIST_CAP)
                            .map(|m| mismatch_json(m, digits))
                            .collect(),
                    ),
                ),
                ("mismatches_truncated", Json::Bool(mism_trunc)),
                (
                    "refused",
                    Json::Arr(
                        refused
                            .iter()
                            .take(LIST_CAP)
                            .map(|(c, r)| {
                                Json::obj(vec![
                                    ("code", Json::Str(hex(*c, digits))),
                                    ("reason", Json::Str(r.clone())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("refused_count", Json::Int(refused.len() as i64)),
                ("refused_truncated", Json::Bool(ref_trunc)),
                ("table_digest", Json::Str(table_digest.clone())),
                ("total", Json::Int(total as i64)),
            ]),
        ),
        ("schema", Json::str(SCHEMA)),
        ("tool", Json::str("sweep")),
        ("version", Json::Int(1)),
        ("warnings", Json::Arr(warnings)),
    ]);
    let receipt = Receipt::new(body);

    let mut text = format!(
        "sweep   {} ({total} codes)\nexpect  {}\ngot     {}\n",
        fmt.name(),
        expect_ast.canonical(),
        got_ast.canonical()
    );
    if let Some(a) = &flag_ast {
        text.push_str(&format!("flag    {}\n", a.canonical()));
    }
    if let Some(a) = &where_ast {
        text.push_str(&format!("where   {}\n", a.canonical()));
    }
    text.push_str(&format!(
        "codes   {total} total, {} excluded by where, {considered} considered\n",
        total - considered
    ));
    text.push_str(&format!("equal   {equal}/{considered}\n"));
    if mismatches.is_empty() {
        text.push_str("mismatches none\n");
    } else {
        text.push_str(&format!("mismatches {}\n", mismatches.len()));
        for m in mismatches.iter().take(TEXT_MISMATCHES) {
            text.push_str(&format!(
                "  {} expect {} got {}\n",
                hex(m.code, digits),
                m.expect,
                m.got
            ));
        }
        if mismatches.len() > TEXT_MISMATCHES {
            text.push_str(&format!(
                "  ... and {} more (all are in the receipt up to {LIST_CAP})\n",
                mismatches.len() - TEXT_MISMATCHES
            ));
        }
    }
    if flag_ast.is_some() {
        text.push_str(&format!(
            "flag true: {} codes [{}]\n",
            flagged.len(),
            ranges(&flagged, digits)
        ));
    }
    if refused.is_empty() {
        text.push_str("refused none\n");
    } else {
        text.push_str(&format!("refused {}\n", refused.len()));
        for (c, r) in refused.iter().take(TEXT_MISMATCHES) {
            text.push_str(&format!("  {} {r}\n", hex(*c, digits)));
        }
    }
    text.push_str(&format!(
        "table   {table_digest}\n{}receipt {}",
        crate::receipt::provenance_lines(),
        receipt.id()
    ));
    Ok(SweepOutcome {
        receipt,
        text,
        total,
        considered,
        equal,
        mismatches,
        flagged,
        refused,
        table_digest,
    })
}
