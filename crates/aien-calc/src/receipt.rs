//! Receipts: canonical JSON body, BLAKE3 id, and the `eval` tool.

use crate::error::CalcError;
use crate::eval::{evaluate, Env};
use crate::json::Json;
use crate::parser::parse;
use crate::render::decimal;
use crate::value::Value;

/// Crate version plus the git commit the binary was built from.
pub const CALC_VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), "+", env!("AIEN_CALC_GIT_COMMIT"));
pub const LANGUAGE: &str = "calc-lang v1";
pub const FORMAT_TABLE: &str = "v1";
pub const SCHEMA: &str = "aien-calc-receipt";
pub const MAX_PRECISION: u32 = 1000;

/// A receipt is its canonical body; the id is the BLAKE3 hex of those exact bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    body: Json,
    bytes: Vec<u8>,
}

impl Receipt {
    pub fn new(body: Json) -> Receipt {
        let bytes = body.to_canonical().into_bytes();
        Receipt { body, bytes }
    }

    pub fn body(&self) -> &Json {
        &self.body
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    pub fn id_hex(&self) -> String {
        blake3::hash(&self.bytes).to_hex().to_string()
    }

    /// `calc:<64 hex>`
    pub fn id(&self) -> String {
        format!("calc:{}", self.id_hex())
    }
}

/// Fixed `method` block shared by every tool.
pub(crate) fn method_block(precision: u32) -> Json {
    Json::obj(vec![
        ("decimal_rounding", Json::str("half-even")),
        ("format_table", Json::str(FORMAT_TABLE)),
        ("language", Json::str(LANGUAGE)),
        ("precision_significant_digits", Json::Int(precision as i64)),
        ("rounding_default", Json::str("rne")),
    ])
}

pub(crate) fn check_precision(precision: u32) -> Result<(), CalcError> {
    if precision == 0 || precision > MAX_PRECISION {
        return Err(CalcError::Invalid(format!(
            "precision must be between 1 and {MAX_PRECISION} digits"
        )));
    }
    Ok(())
}

/// JSON for a successful value, plus its human spelling.
pub(crate) fn value_json(v: &Value, precision: u32) -> (Json, String) {
    match v {
        Value::Num(r) => {
            let (dec, exact) = decimal(r, precision);
            let note = if exact {
                "exact".to_string()
            } else {
                format!("rounded to {precision} significant digits")
            };
            (
                Json::obj(vec![
                    ("decimal", Json::Str(dec.clone())),
                    ("decimal_exact", Json::Bool(exact)),
                    ("kind", Json::str("rational")),
                    ("value", Json::Str(v.canonical())),
                ]),
                format!("value   {}\ndecimal {dec} ({note})", v.human()),
            )
        }
        Value::Bool(b) => (
            Json::obj(vec![("kind", Json::str("bool")), ("value", Json::Bool(*b))]),
            format!("value   {b}"),
        ),
    }
}

/// Everything `eval` returns: the receipt, the human text, and the answer or the refusal.
#[derive(Clone, Debug)]
pub struct EvalOutcome {
    receipt: Receipt,
    text: String,
    value: Option<Value>,
    refusal: Option<String>,
}

impl EvalOutcome {
    pub fn receipt(&self) -> &Receipt {
        &self.receipt
    }

    /// Human output; the last line is `receipt calc:<hex>`.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    pub fn canonical_value(&self) -> Option<String> {
        self.value.as_ref().map(Value::canonical)
    }

    pub fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }
}

/// The `eval` tool: parse, evaluate exactly, and issue a receipt. Malformed input is an `Err`
/// (no receipt); a well-formed question that exact arithmetic declines is an `Ok` with a refusal.
pub fn eval(expr: &str, precision: u32) -> Result<EvalOutcome, CalcError> {
    check_precision(precision)?;
    let ast = parse(expr)?;
    let canonical = ast.canonical();
    let result = evaluate(&ast, &Env::default());
    let (result_json, exactness, body_text, value, refusal) = match result {
        Ok(v) => {
            let (j, t) = value_json(&v, precision);
            (j, "exact", t, Some(v), None)
        }
        Err(r) => (
            Json::obj(vec![
                ("kind", Json::str("refusal")),
                ("reason", Json::Str(r.0.clone())),
            ]),
            "refused",
            format!("refused: {}", r.0),
            None,
            Some(r.0),
        ),
    };
    let body = Json::obj(vec![
        ("calc_version", Json::str(CALC_VERSION)),
        ("exactness", Json::str(exactness)),
        (
            "inputs",
            Json::obj(vec![
                ("expr", Json::Str(canonical.clone())),
                ("precision", Json::Int(precision as i64)),
            ]),
        ),
        ("method", method_block(precision)),
        ("result", result_json),
        ("schema", Json::str(SCHEMA)),
        ("tool", Json::str("eval")),
        ("version", Json::Int(1)),
        ("warnings", Json::Arr(vec![])),
    ]);
    let receipt = Receipt::new(body);
    let text = format!("expr    {canonical}\n{body_text}\nreceipt {}", receipt.id());
    Ok(EvalOutcome {
        receipt,
        text,
        value,
        refusal,
    })
}
