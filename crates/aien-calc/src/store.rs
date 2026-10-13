//! Receipt store: one file per receipt plus an append-only ledger.
//!
//! Layout under the store root (`$AIEN_CALC_HOME`, else `$HOME/.aien/calc`):
//!
//! * `receipts/<64 hex>.json`: the canonical receipt bytes, verbatim, no trailing newline, so
//!   `b3sum receipts/<hex>.json` prints the receipt id;
//! * `ledger.jsonl`: one line per save, `{"argv":[..],"cwd":"..","id":"calc:..","ts":<unix secs>}`.

use crate::json::Json;
use crate::receipt::Receipt;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Shortest accepted id prefix, in hex digits.
pub const MIN_PREFIX: usize = 12;

pub struct Store {
    root: PathBuf,
}

/// `calc:<hex>` or bare hex, lowercase, as an unvalidated-length hex string.
fn bare_hex(id: &str) -> Result<&str, String> {
    let h = id.strip_prefix("calc:").unwrap_or(id);
    if h.is_empty() || !h.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!("`{id}` is not a receipt id (expected calc:<hex>)"));
    }
    Ok(h)
}

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into() }
    }

    /// The store named by `AIEN_CALC_HOME`, else `$HOME/.aien/calc`.
    pub fn from_env() -> Result<Store, String> {
        if let Some(h) = std::env::var_os("AIEN_CALC_HOME").filter(|h| !h.is_empty()) {
            return Ok(Store::new(h));
        }
        match std::env::var_os("HOME").filter(|h| !h.is_empty()) {
            Some(h) => Ok(Store::new(PathBuf::from(h).join(".aien").join("calc"))),
            None => Err("neither AIEN_CALC_HOME nor HOME is set".into()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn receipt_path(&self, hex: &str) -> PathBuf {
        self.root.join("receipts").join(format!("{hex}.json"))
    }

    pub fn ledger_path(&self) -> PathBuf {
        self.root.join("ledger.jsonl")
    }

    /// Store the receipt (idempotent) and append a ledger line. Returns the receipt file path.
    pub fn save(&self, r: &Receipt, cwd: &str, argv: &[String]) -> std::io::Result<PathBuf> {
        let dir = self.root.join("receipts");
        std::fs::create_dir_all(&dir)?;
        let hex = r.id_hex();
        let path = self.receipt_path(&hex);
        let bytes = r.canonical_bytes();
        if std::fs::read(&path).ok().as_deref() != Some(bytes.as_slice()) {
            let tmp = dir.join(format!(".{hex}.{}.tmp", std::process::id()));
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(&tmp, &path)?;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64);
        let mut line = Json::obj(vec![
            (
                "argv",
                Json::Arr(argv.iter().map(|a| Json::Str(a.clone())).collect()),
            ),
            ("cwd", Json::str(cwd)),
            ("id", Json::Str(r.id())),
            ("ts", Json::Int(ts)),
        ])
        .to_canonical();
        line.push('\n');
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.ledger_path())?;
        f.write_all(line.as_bytes())?;
        Ok(path)
    }

    /// Full 64-digit hex for `calc:<hex>`, bare hex, or a unique prefix of at least 12 digits.
    pub fn resolve(&self, id: &str) -> Result<String, String> {
        let h = bare_hex(id)?;
        if h.len() > 64 {
            return Err(format!("`{id}` is longer than a receipt id"));
        }
        if h.len() == 64 {
            return if self.receipt_path(h).is_file() {
                Ok(h.to_string())
            } else {
                Err(format!("no stored receipt calc:{h}"))
            };
        }
        if h.len() < MIN_PREFIX {
            return Err(format!(
                "`{id}` is too short: give at least {MIN_PREFIX} hex digits"
            ));
        }
        let mut found = Vec::new();
        if let Ok(rd) = std::fs::read_dir(self.root.join("receipts")) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(stem) = name.strip_suffix(".json") {
                    if stem.len() == 64 && stem.starts_with(h) {
                        found.push(stem.to_string());
                    }
                }
            }
        }
        match found.len() {
            0 => Err(format!("no stored receipt matches `{id}`")),
            1 => Ok(found.remove(0)),
            n => Err(format!("`{id}` is ambiguous: {n} stored receipts match")),
        }
    }

    /// The stored bytes of a receipt.
    pub fn load(&self, id: &str) -> Result<Vec<u8>, String> {
        let hex = self.resolve(id)?;
        std::fs::read(self.receipt_path(&hex)).map_err(|e| format!("cannot read calc:{hex}: {e}"))
    }

    /// Copy a receipt to `<dir>/calc-receipts/<hex>.json` (byte-identical) for committing next to evidence.
    pub fn export(&self, id: &str, dir: &Path) -> Result<PathBuf, String> {
        let hex = self.resolve(id)?;
        let bytes = self.load(&hex)?;
        let out_dir = dir.join("calc-receipts");
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("cannot create {}: {e}", out_dir.display()))?;
        let out = out_dir.join(format!("{hex}.json"));
        std::fs::write(&out, bytes).map_err(|e| format!("cannot write {}: {e}", out.display()))?;
        Ok(out)
    }
}
