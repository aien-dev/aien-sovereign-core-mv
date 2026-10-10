//! sovereign-core #380 (lane L6, acceptance step E2 and CTRL-E2): the
//! operator's objective is recorded, with its identity, before any work.
//!
//! `aien-cli objective record --text TEXT [--effect-class CLASS]` writes
//! `objectives/<objective_id>.json` under `AIEN_PROVENANCE_DIR` (the same
//! directory the effect receipts use). The operator declares the effect class;
//! nothing here classifies natural language. A declared
//! `EXTERNAL_IRREVERSIBLE` (the class name in `aien-mcp` `ToolEffects`) is
//! refused before anything is written. `compose propose --objective-id` ties
//! a proposal to a record by the sha256 of the goal text.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

pub const DOMAIN_TAG: &str = "AIEN_E2E_OBJECTIVE_V1";
pub const MAX_TEXT_BYTES: usize = 2000;
pub const FORBIDDEN_CLASS: &str = "EXTERNAL_IRREVERSIBLE";
pub const DEFAULT_CLASS: &str = "WORKSPACE_WRITE";

/// A named refusal: `OBJECTIVE_REFUSED <name>: <detail>`.
#[derive(Debug)]
pub struct Refusal {
    pub name: &'static str,
    pub detail: String,
}

impl Refusal {
    fn new(name: &'static str, detail: impl Into<String>) -> Self {
        Refusal {
            name,
            detail: detail.into(),
        }
    }
    pub fn line(&self) -> String {
        format!("OBJECTIVE_REFUSED {}: {}", self.name, self.detail)
    }
}

fn sha256_hex(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

pub fn objective_id(text: &str) -> String {
    sha256_hex(format!("{DOMAIN_TAG}\n{text}").as_bytes())
}

fn dir() -> PathBuf {
    std::env::var("AIEN_PROVENANCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/aien-provenance"))
        .join("objectives")
}

fn valid_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Validate and write the record. Refuses before touching the disk.
pub fn record(text: &str, class: Option<&str>) -> Result<Value, Refusal> {
    let class = class.unwrap_or(DEFAULT_CLASS).trim().to_ascii_uppercase();
    if class == FORBIDDEN_CLASS {
        return Err(Refusal::new(
            "ForbiddenEffectClass",
            format!("the declared effect class {FORBIDDEN_CLASS} is never executed"),
        ));
    }
    if text.trim().is_empty() {
        return Err(Refusal::new("Empty", "the objective text is empty"));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(Refusal::new(
            "TooLong",
            format!("{} bytes, at most {MAX_TEXT_BYTES}", text.len()),
        ));
    }
    let id = objective_id(text);
    let text_sha = sha256_hex(text.as_bytes());
    let utc = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let rec = json!({
        "version": 1,
        "objective_id": id,
        "objective_text": text,
        "objective_text_sha256": text_sha,
        "effect_class": class,
        "recorded_before_work": true,
        "operator_surface": "aien-cli",
        "utc": utc,
    });
    let d = dir();
    std::fs::create_dir_all(&d)
        .map_err(|e| Refusal::new("RecordUnwritable", format!("{}: {e}", d.display())))?;
    let path = d.join(format!("{id}.json"));
    let bytes = serde_json::to_vec_pretty(&rec).expect("record is JSON");
    std::fs::write(&path, bytes)
        .map_err(|e| Refusal::new("RecordUnwritable", format!("{}: {e}", path.display())))?;
    Ok(json!({
        "ok": true,
        "objective_id": id,
        "path": path.display().to_string(),
        "objective_text_sha256": text_sha,
        "recorded_before_work": true,
        "operator_surface": "aien-cli",
        "utc": utc,
    }))
}

pub fn load(id: &str) -> Result<Value, Refusal> {
    if !valid_id(id) {
        return Err(Refusal::new(
            "ObjectiveMissing",
            format!("{id:?} is not a 64 character lowercase hex objective id"),
        ));
    }
    let path = dir().join(format!("{id}.json"));
    let bytes = std::fs::read(&path).map_err(|_| {
        Refusal::new(
            "ObjectiveMissing",
            format!("no objective record {}", path.display()),
        )
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Refusal::new("ObjectiveMissing", format!("{}: {e}", path.display())))
}

/// `compose propose --objective-id`: the record must exist and the goal must
/// be the recorded text. Returns the id to carry in the report.
pub fn check_binding(id: &str, goal: &str) -> Result<String, Refusal> {
    let rec = load(id)?;
    let want = rec["objective_text_sha256"].as_str().unwrap_or("");
    let got = sha256_hex(goal.as_bytes());
    if want != got {
        return Err(Refusal::new(
            "ObjectiveMismatch",
            format!("sha256 of --goal is {got}, the record {id} holds {want}"),
        ));
    }
    Ok(id.to_string())
}

/// Add the objective id to a propose report.
pub fn attach(mut report: Value, id: &str) -> Value {
    report["objective_id"] = json!(id);
    report
}

fn refuse(r: Refusal) -> ! {
    eprintln!("{}", r.line());
    let v = json!({"ok": false, "refusal": r.name, "error": r.line()});
    println!("{v}");
    std::process::exit(1);
}

pub fn handle_objective_command(args: &[String]) {
    let usage = || -> ! {
        println!(
            "{}",
            json!({"ok": false, "error": "usage: aien-cli objective record --text TEXT [--effect-class CLASS] | objective show ID"})
        );
        std::process::exit(1);
    };
    match args.first().map(String::as_str) {
        Some("record") => {
            let (mut text, mut class) = (None, None);
            let mut i = 1;
            while i < args.len() {
                let Some(v) = args.get(i + 1) else { usage() };
                match args[i].as_str() {
                    "--text" => text = Some(v.clone()),
                    "--effect-class" => class = Some(v.clone()),
                    _ => usage(),
                }
                i += 2;
            }
            let Some(text) = text else { usage() };
            match record(&text, class.as_deref()) {
                Ok(v) => println!("{v}"),
                Err(r) => refuse(r),
            }
        }
        Some("show") if args.len() == 2 => match load(&args[1]) {
            Ok(mut v) => {
                v["ok"] = json!(true);
                println!("{v}");
            }
            Err(r) => refuse(r),
        },
        _ => usage(),
    }
}

pub fn refuse_propose(r: Refusal) -> String {
    eprintln!("{}", r.line());
    r.line()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_objective_id_is_accepted_and_carried() {
        let t = tempfile::tempdir().unwrap();
        std::env::set_var("AIEN_PROVENANCE_DIR", t.path());
        let r = record("do the thing", None).unwrap();
        let id = r["objective_id"].as_str().unwrap();
        assert_eq!(check_binding(id, "do the thing").unwrap(), id);
        assert_eq!(
            check_binding(id, "other").unwrap_err().name,
            "ObjectiveMismatch"
        );
        let rep = attach(json!({"step": "S3"}), id);
        assert_eq!(rep["objective_id"], id);
        std::env::remove_var("AIEN_PROVENANCE_DIR");
    }
}
