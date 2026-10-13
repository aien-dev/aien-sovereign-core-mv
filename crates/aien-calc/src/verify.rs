//! Receipt verification: re-read a stored receipt, recompute it from its own recorded inputs with
//! this engine, and report every field that differs. A hash proves the file was not edited; the
//! recomputation proves the numbers are what the engine says.

use crate::json::Json;
use crate::parser::Ast;
use crate::receipt::{eval_ast, ENGINE_VERSION, SCHEMA};
use crate::sweep::sweep_ast;

#[derive(Clone, Debug)]
pub struct Verdict {
    pub matches: bool,
    /// `calc:<blake3 of the bytes>`.
    pub id: String,
    /// Human-readable differences; empty exactly when `matches`.
    pub diffs: Vec<String>,
}

fn short(j: &Json) -> String {
    let s = j.to_canonical();
    if s.chars().count() > 80 {
        format!("{}...", s.chars().take(77).collect::<String>())
    } else {
        s
    }
}

fn diff_json(path: &str, stored: &Json, fresh: &Json, out: &mut Vec<String>) {
    match (stored, fresh) {
        (Json::Obj(a), Json::Obj(b)) => {
            let mut keys: Vec<&String> = a.iter().chain(b.iter()).map(|(k, _)| k).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                let p = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                match (stored.get(k), fresh.get(k)) {
                    (Some(x), Some(y)) => diff_json(&p, x, y, out),
                    (Some(x), None) => out.push(format!(
                        "{p}: stored {} but the engine produces no such field",
                        short(x)
                    )),
                    (None, Some(y)) => {
                        out.push(format!("{p}: missing (engine produces {})", short(y)))
                    }
                    (None, None) => {}
                }
            }
        }
        (Json::Arr(a), Json::Arr(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                diff_json(&format!("{path}[{i}]"), x, y, out);
            }
        }
        _ if stored != fresh => out.push(format!(
            "{path}: stored {}, recomputed {}",
            short(stored),
            short(fresh)
        )),
        _ => {}
    }
}

fn str_field<'a>(j: &'a Json, key: &str) -> Option<&'a str> {
    match j.get(key) {
        Some(Json::Str(s)) => Some(s),
        _ => None,
    }
}

fn ast_field(inputs: &Json, key: &str) -> Result<Ast, String> {
    let s =
        str_field(inputs, key).ok_or_else(|| format!("inputs.{key} missing or not a string"))?;
    Ast::from_canonical(s).map_err(|e| format!("inputs.{key}: {e}"))
}

fn opt_ast_field(inputs: &Json, key: &str) -> Result<Option<Ast>, String> {
    match inputs.get(key) {
        Some(Json::Null) => Ok(None),
        _ => ast_field(inputs, key).map(Some),
    }
}

fn recompute(body: &Json) -> Result<Json, String> {
    let inputs = body.get("inputs").ok_or("inputs missing")?;
    match str_field(body, "tool") {
        Some("eval") => {
            let ast = ast_field(inputs, "expr")?;
            let precision = match inputs.get("precision") {
                Some(Json::Int(p)) if *p >= 1 && *p <= u32::MAX as i64 => *p as u32,
                _ => return Err("inputs.precision missing or out of range".into()),
            };
            let out = eval_ast(&ast, precision).map_err(|e| e.to_string())?;
            Ok(out.receipt().body().clone())
        }
        Some("sweep") => {
            let format = str_field(inputs, "format").ok_or("inputs.format missing")?;
            let out = sweep_ast(
                format,
                &ast_field(inputs, "expect")?,
                &ast_field(inputs, "got")?,
                opt_ast_field(inputs, "flag")?.as_ref(),
                opt_ast_field(inputs, "where")?.as_ref(),
            )
            .map_err(|e| e.to_string())?;
            Ok(out.receipt().body().clone())
        }
        Some(other) => Err(format!("unknown tool `{other}`")),
        None => Err("tool missing".into()),
    }
}

/// Verify receipt bytes. `file_hex` is the hash in the file's name, when it has one: it must equal
/// the BLAKE3 of the bytes.
pub fn verify_bytes(bytes: &[u8], file_hex: Option<&str>) -> Verdict {
    let hex = blake3::hash(bytes).to_hex().to_string();
    let mut v = Verdict {
        matches: false,
        id: format!("calc:{hex}"),
        diffs: Vec::new(),
    };
    if let Some(named) = file_hex {
        if named != hex {
            v.diffs.push(format!(
                "file name says calc:{named} but the content hashes to calc:{hex}"
            ));
        }
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        v.diffs.push("receipt is not valid UTF-8".into());
        return v;
    };
    let body = match Json::parse(text) {
        Ok(b) => b,
        Err(e) => {
            v.diffs.push(format!("receipt is not valid JSON: {e}"));
            return v;
        }
    };
    if body.to_canonical() != text {
        v.diffs
            .push("receipt bytes are not in canonical form (sorted keys, no whitespace)".into());
    }
    if str_field(&body, "schema") != Some(SCHEMA) {
        v.diffs.push(format!(
            "schema: stored {}, expected \"{SCHEMA}\"",
            body.get("schema").map_or("missing".into(), short)
        ));
        return v;
    }
    if body.get("version") != Some(&Json::Int(1)) {
        v.diffs.push(format!(
            "version: stored {}, this engine reads 1",
            body.get("version").map_or("missing".into(), short)
        ));
        return v;
    }
    if str_field(&body, "calc_version") != Some(ENGINE_VERSION) {
        v.diffs.push(format!(
            "calc_version: stored {}, this engine is \"{ENGINE_VERSION}\" (recompute with the matching engine)",
            body.get("calc_version").map_or("missing".into(), short)
        ));
        return v;
    }
    match recompute(&body) {
        Ok(fresh) => {
            let before = v.diffs.len();
            diff_json("", &body, &fresh, &mut v.diffs);
            if v.diffs.len() == before
                && fresh.to_canonical() != text
                && body.to_canonical() == text
            {
                v.diffs.push("recomputed bytes differ".into());
            }
        }
        Err(e) => v.diffs.push(format!("cannot recompute: {e}")),
    }
    v.matches = v.diffs.is_empty();
    v
}
