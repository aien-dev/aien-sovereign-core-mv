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
    /// Parse JSON text. Integers only (no fractions or exponents), no duplicate keys, nothing after
    /// the value. Accepts any valid spelling; [`Json::to_canonical`] decides whether it was canonical.
    pub fn parse(s: &str) -> Result<Json, String> {
        let mut p = JsonParser {
            b: s.as_bytes(),
            i: 0,
        };
        let v = p.value(0)?;
        if p.i != p.b.len() {
            return Err(format!("trailing characters at byte {}", p.i));
        }
        Ok(v)
    }

    /// Field of an object (`None` for other kinds or a missing key).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

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

const MAX_JSON_DEPTH: usize = 64;

struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!("expected `{lit}` at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_JSON_DEPTH {
            return Err("JSON nested too deeply".into());
        }
        self.ws();
        match self.b.get(self.i) {
            None => Err("unexpected end of JSON".into()),
            Some(b'n') => self.eat("null").map(|_| Json::Null),
            Some(b't') => self.eat("true").map(|_| Json::Bool(true)),
            Some(b'f') => self.eat("false").map(|_| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected `,` or `]` at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut pairs: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(pairs));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(format!("expected a key at byte {}", self.i));
                    }
                    let k = self.string()?;
                    if pairs.iter().any(|(e, _)| *e == k) {
                        return Err(format!("duplicate key `{k}`"));
                    }
                    self.ws();
                    self.eat(":")?;
                    let v = self.value(depth + 1)?;
                    pairs.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(pairs));
                        }
                        _ => return Err(format!("expected `,` or `}}` at byte {}", self.i)),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.int(),
            Some(c) => Err(format!("unexpected `{}` at byte {}", *c as char, self.i)),
        }
    }

    fn int(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let digits = self.i;
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.i == digits {
            return Err(format!("bad number at byte {start}"));
        }
        if self.b[digits] == b'0' && self.i - digits > 1 {
            return Err(format!("leading zero at byte {start}"));
        }
        if matches!(self.b.get(self.i), Some(b'.' | b'e' | b'E')) {
            return Err(format!(
                "non-integer number at byte {start} (receipts contain no floats)"
            ));
        }
        std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|t| t.parse::<i64>().ok())
            .map(Json::Int)
            .ok_or_else(|| format!("integer out of range at byte {start}"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let t = std::str::from_utf8(h).map_err(|_| "bad \\u escape")?;
        if !t.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("bad \\u escape at byte {}", self.i));
        }
        self.i += 4;
        u32::from_str_radix(t, 16).map_err(|e| e.to_string())
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|_| "string is not valid UTF-8".to_string())
                }
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.eat("\\u")?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("unpaired surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(cp).ok_or("invalid code point")?
                        }
                        _ => return Err(format!("bad escape `\\{}`", e as char)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return Err("raw control character in string".into()),
                c => out.push(c),
            }
        }
    }
}
