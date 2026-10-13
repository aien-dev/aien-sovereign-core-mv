//! Minimal canonical JSON (the receipt wire format). Hand-written on purpose: the bytes must be
//! reproducible by an independent implementation (Omega, a shell script), so the rules are tiny:
//!
//! * objects: keys sorted by their UTF-8 bytes, no duplicates, no whitespace;
//! * integers: plain decimal when `|n| <= 2^53`, otherwise a JSON string of the decimal digits;
//! * strings: only `"` `\` and bytes below 0x20 are escaped (`\"`, `\\`, `\u00xx` lowercase hex);
//!   everything else is emitted as UTF-8 verbatim;
//! * no floating point anywhere.

/// A JSON value restricted to what receipts need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered pairs; [`Json::to_canonical`] sorts them.
    Obj(Vec<(String, Json)>),
}

const SAFE_INT: i64 = 1 << 53;

impl Json {
    /// Build an object. Panics on a duplicate key (a programming error, never user input).
    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        let mut seen: Vec<&str> = Vec::with_capacity(pairs.len());
        for (k, _) in &pairs {
            assert!(!seen.contains(k), "duplicate key `{k}` in JSON object");
            seen.push(k);
        }
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn str(s: &str) -> Json {
        Json::Str(s.to_string())
    }

    /// Canonical serialisation (see module docs).
    pub fn to_canonical(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(n) => {
                if (-SAFE_INT..=SAFE_INT).contains(n) {
                    out.push_str(&n.to_string());
                } else {
                    write_string(&n.to_string(), out);
                }
            }
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_canonical(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                let mut sorted: Vec<&(String, Json)> = pairs.iter().collect();
                sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
                out.push('{');
                for (i, (k, v)) in sorted.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(k, out);
                    out.push(':');
                    v.write_canonical(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
