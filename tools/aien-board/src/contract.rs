//! A consumer contract: what one repository uses of another, and the check that proves it still works.
//!
//! Lives in the CONSUMER repository at `contracts/<producer>.contract` and is read from the consumer's
//! main through git objects (like `.aien/always`, trust anchor A1: a producer change cannot widen or
//! narrow a consumer's contract). Line format, `#` starts a comment:
//!
//! ```text
//! producer omega                       # the repository this contract is about (siblings map name)
//! pin omega.lock                       # the consumer's pin file naming the producer commit it ships with
//! watch src/runtime/rxc_host_abi.h     # a producer path the consumer depends on; a trailing / is a directory
//! watch-file contracts/omega.watch     # more watch lines, one path per line (generated from the last
//!                                      #   build's dependency files by the check itself)
//! check scripts/contract-omega.sh      # run from the consumer checkout: check PRODUCER_DIR PRODUCER_SHA OUT_DIR
//! timeout 30m
//! ```
//!
//! `touched` is the whole interface rule: a producer change needs this consumer's check exactly when
//! one of its changed paths is under a watched path. Nothing else is guessed.
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    pub consumer: String,
    pub producer: String,
    pub pin: String,
    pub watch: Vec<String>,
    pub watch_file: Option<String>,
    pub check: String,
    pub timeout_secs: u64,
}

pub fn parse_duration_secs(s: &str) -> Option<u64> {
    let (n, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit())?);
    let n: u64 = n.parse().ok()?;
    match unit {
        "s" => Some(n),
        "m" => Some(n * 60),
        "h" => Some(n * 3600),
        _ => None,
    }
}

fn clean_path(p: &str) -> Result<String, String> {
    if p.is_empty() || p.starts_with('/') || p.split('/').any(|c| c == ".." || c == ".") {
        return Err(format!("path {p:?} must be repository-relative without . or .."));
    }
    Ok(p.to_string())
}

pub fn parse(consumer: &str, text: &str) -> Result<Contract, String> {
    let mut c = Contract {
        consumer: consumer.to_string(),
        producer: String::new(),
        pin: String::new(),
        watch: Vec::new(),
        watch_file: None,
        check: String::new(),
        timeout_secs: 1800,
    };
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, val) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let val = val.trim();
        let at = |e: String| format!("line {}: {e}", n + 1);
        match key {
            "producer" => c.producer = val.to_string(),
            "pin" => c.pin = clean_path(val).map_err(at)?,
            "watch" => c.watch.push(clean_path(val).map_err(at)?),
            "watch-file" => c.watch_file = Some(clean_path(val).map_err(at)?),
            "check" => c.check = clean_path(val).map_err(at)?,
            "timeout" => {
                c.timeout_secs =
                    parse_duration_secs(val).ok_or_else(|| at(format!("bad timeout {val:?}")))?
            }
            other => return Err(at(format!("unknown key {other:?}"))),
        }
    }
    if c.producer.is_empty() || c.check.is_empty() || (c.watch.is_empty() && c.watch_file.is_none()) {
        return Err("a contract needs producer, check and at least one watch or watch-file".into());
    }
    Ok(c)
}

/// One path per line from a watch file (comments and blanks skipped).
pub fn parse_watch_file(text: &str) -> Result<Vec<String>, String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(clean_path)
        .collect()
}

/// Every contract in `contracts/` of `consumer` at `rev`, through git objects.
pub fn load_all(clone: &Path, consumer: &str, rev: &str) -> Result<Vec<Contract>, String> {
    let rows = aien_test::plan::tree_rows(clone, rev)?;
    let mut out = Vec::new();
    for path in rows.keys().filter(|p| p.starts_with("contracts/") && p.ends_with(".contract")) {
        let bytes = aien_test::plan::show(clone, rev, path).ok_or_else(|| format!("cannot read {path}"))?;
        let mut c = parse(consumer, &String::from_utf8_lossy(&bytes)).map_err(|e| format!("{path}: {e}"))?;
        if let Some(wf) = c.watch_file.clone() {
            let w = aien_test::plan::show(clone, rev, &wf).ok_or_else(|| format!("{path}: watch-file {wf} missing"))?;
            c.watch.extend(parse_watch_file(&String::from_utf8_lossy(&w)).map_err(|e| format!("{wf}: {e}"))?);
        }
        c.watch.sort();
        c.watch.dedup();
        out.push(c);
    }
    Ok(out)
}

/// The producer paths of `changed` that this contract watches.
pub fn touched(c: &Contract, changed: &[String]) -> Vec<String> {
    changed
        .iter()
        .filter(|p| c.watch.iter().any(|w| aien_test::plan::covers(w, p)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMEGA: &str = "producer omega\npin omega.lock\nwatch src/runtime/rxc_host_abi.h\nwatch mk/\n\
                         check scripts/contract-omega.sh # the compose contract\ntimeout 20m\n";

    #[test]
    fn a_contract_parses_and_touched_is_path_coverage_only() {
        let c = parse("aien-sovereign-core", OMEGA).unwrap();
        assert_eq!(c.producer, "omega");
        assert_eq!(c.timeout_secs, 1200);
        let changed = vec![
            "src/runtime/rxc_host_abi.h".to_string(),
            "mk/rx_compose_lib.mk".to_string(),
            "src/runtime/rxc_host_abi.c".to_string(),
            "docs/x.md".to_string(),
        ];
        assert_eq!(touched(&c, &changed), ["src/runtime/rxc_host_abi.h", "mk/rx_compose_lib.mk"]);
        assert!(touched(&c, &["docs/x.md".to_string()]).is_empty());
    }

    #[test]
    fn a_contract_refuses_escaping_paths_unknown_keys_and_missing_parts() {
        assert!(parse("x", "producer p\nwatch ../etc\ncheck c.sh\n").is_err());
        assert!(parse("x", "producer p\nwatch /etc\ncheck c.sh\n").is_err());
        assert!(parse("x", "producer p\nwatch a\ncheck c.sh\nbogus 1\n").is_err());
        assert!(parse("x", "producer p\ncheck c.sh\n").is_err());
        assert!(parse("x", "watch a\ncheck c.sh\n").is_err());
        assert!(parse("x", "producer p\nwatch a\ncheck c.sh\ntimeout 5x\n").is_err());
    }
}
