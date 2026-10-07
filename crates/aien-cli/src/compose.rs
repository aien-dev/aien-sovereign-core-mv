//! `aien compose ...`: the NEXT-PHASE-1 bounded task, one operator step per
//! subcommand, against the running daemon's composition home (omega
//! COMPOSITION-2, `RunComposeTask` / `ComposeNote` / `ComposeRecall` /
//! `RecoverComposeHome`).
//!
//! The daemon owns the Cortex journal; this side owns the effects. Every
//! workspace path goes through `SafetyEngine::validate_path` with the
//! authorized workspace as the only root, and every effect (inspect,
//! authorize, write_file) leaves a sovereign-core effect receipt
//! (`record_effect_receipt`) that the Cortex record then cites by sha256.
//!
//! Each subcommand prints one JSON object on stdout (`"ok": true|false`) and
//! exits 1 on failure.
//!
//!   remember  --text T                                   S1 constraint record
//!   inspect   --workspace W                              S2 read-only listing + receipt
//!   propose   --goal G --workspace W                     S3 one RunComposeTask
//!   authorize --report S3.json --workspace W --approver NAME [--constraint ID]
//!                                                        S4 the one approval
//!   execute   --report S3.json --workspace W --authorization ID
//!                                                        S5 the write, confined
//!   explain   --report S3.json --cite ID,.. --receipts P,..
//!                                                        S6 evidence-citing explanation
//!   recall    [--ids ID,..] [--prefix N]                 S8 constraints + effects
//!   shutdown                                             S7 stop the daemon (control Shutdown)
//!   recover                                              repair a refused home
//!   effects                                              the effect ledger (NEXT-PHASE-2)
//!   reconcile [--intent ID --declare done|not_done --approver NAME]
//!                                                        settle unsettled effects
//!   stop      --approver NAME                            durable operator stop
//!   resume    --approver NAME                            end the stop (old grants stay stale)
//!   revoke    --authorization ID --approver NAME         revoke an unspent grant
//!   desk-key  [--create 1]                               the approval desk key (#249):
//!                                                        prints its id and path, never the key;
//!                                                        --create 1 makes one if none exists
//!
//! S5 `execute` brackets the write with a durable intent and an ack (see
//! `aien_runtime::effects`); it prints `"state"` and exits 3 when the effect
//! is UNRESOLVED.
use crate::safety::SafetyEngine;
use crate::tools::record_effect_receipt;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::control::{
    ComposeRecallReport, ComposeTaskReport, ControlCommand, ControlResponse, ReconcileDeclare,
};
use aien_runtime::effects;
use aien_runtime::spine::parse_file_proposal;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const INSPECT_MAX_ENTRIES: usize = 4096;

fn sha256_hex(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("expected --flag, got {:?}", args[i]))?;
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("--{k} needs a value"))?;
        m.insert(k.to_string(), v.clone());
        i += 2;
    }
    Ok(m)
}

fn need<'a>(m: &'a HashMap<String, String>, k: &str) -> Result<&'a str, String> {
    m.get(k)
        .map(|s| s.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("--{k} is required"))
}

fn ids(s: &str) -> Result<Vec<u64>, String> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim()
                .parse::<u64>()
                .map_err(|e| format!("bad id {x:?}: {e}"))
        })
        .collect()
}

async fn send(cmd: ControlCommand) -> Result<ControlResponse, String> {
    match AienRuntimeClient::default_client()
        .send_command(cmd)
        .await?
    {
        ControlResponse::Error(e) => Err(e),
        r => Ok(r),
    }
}

/// The authorized workspace as the only root the safety engine accepts.
fn confine(workspace: &str) -> Result<(SafetyEngine, PathBuf), String> {
    let mut engine = SafetyEngine::new();
    engine.set_workspaces(vec![PathBuf::from(workspace)]);
    let ws = engine
        .validate_path(workspace)
        .map_err(|e| format!("CONFINEMENT DENIAL: {e}"))?;
    if !ws.is_dir() {
        return Err(format!("workspace {} is not a directory", ws.display()));
    }
    Ok((engine, ws))
}

/// One effect receipt, read back: (path, sha256 of the file, its JSON).
fn receipt(tool: &str, args: &Value, result: &Value, success: bool) -> Result<Value, String> {
    let path = record_effect_receipt(tool, args, result, success)
        .ok_or_else(|| format!("effect receipt for {tool} could not be written"))?;
    let bytes =
        std::fs::read(&path).map_err(|e| format!("read receipt {}: {e}", path.display()))?;
    let body: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("receipt {}: {e}", path.display()))?;
    Ok(json!({
        "path": path.display().to_string(),
        "sha256": sha256_hex(&bytes),
        "tool": tool,
        "success": success,
        "arguments_digest": body["arguments_digest"],
        "result_digest": body["result_digest"],
    }))
}

async fn note(kind: &str, text: &str, links: Vec<u64>) -> Result<Value, String> {
    match send(ControlCommand::ComposeNote {
        kind: kind.into(),
        text: text.into(),
        links,
    })
    .await?
    {
        ControlResponse::ComposeNoted(n) => serde_json::to_value(n).map_err(|e| e.to_string()),
        other => Err(format!("unexpected response {other:?}")),
    }
}

async fn recall(ids: Vec<u64>, prefix: Option<u64>) -> Result<ComposeRecallReport, String> {
    match send(ControlCommand::ComposeRecall { ids, prefix }).await? {
        ControlResponse::ComposeRecalled(r) => Ok(*r),
        other => Err(format!("unexpected response {other:?}")),
    }
}

fn read_report(path: &str) -> Result<ComposeTaskReport, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
    // `aien compose propose` prints {"ok":true,"report":{...}}; accept both.
    let r = v.get("report").cloned().unwrap_or(v);
    serde_json::from_value(r).map_err(|e| format!("{path}: not a compose report: {e}"))
}

/// The committed proposal of an S3 report, re-parsed and re-hashed here.
fn committed_proposal(r: &ComposeTaskReport) -> Result<(String, String, String), String> {
    if !r.committed {
        return Err("the composition did not commit a proposal".into());
    }
    let text = r.proposal.as_deref().ok_or("report holds no proposal")?;
    let sha = sha256_hex(text.as_bytes());
    if r.proposal_sha256.as_deref() != Some(sha.as_str()) {
        return Err("proposal text does not match its recorded sha256".into());
    }
    let p = parse_file_proposal(text).ok_or("proposal is not one file change")?;
    let csha = sha256_hex(p.content.as_bytes());
    if r.proposal_content_sha256.as_deref() != Some(csha.as_str())
        || r.proposal_path.as_deref() != Some(p.path.as_str())
    {
        return Err("parsed proposal differs from the report".into());
    }
    Ok((p.path, p.content, sha))
}

fn tree(ws: &Path) -> Result<(Vec<Value>, String), String> {
    let mut out = Vec::new();
    let mut stack = vec![ws.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = std::fs::read_dir(&d).map_err(|e| format!("read {}: {e}", d.display()))?;
        for e in rd {
            let e = e.map_err(|e| e.to_string())?;
            let p = e.path();
            let ft = e.file_type().map_err(|e| e.to_string())?;
            let rel = p
                .strip_prefix(ws)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();
            if ft.is_dir() {
                stack.push(p);
                out.push(json!({"path": rel + "/", "kind": "dir"}));
            } else if ft.is_file() {
                let b = std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
                out.push(json!({"path": rel, "kind": "file", "bytes": b.len(), "sha256": sha256_hex(&b)}));
            } else {
                out.push(json!({"path": rel, "kind": "other"}));
            }
            if out.len() > INSPECT_MAX_ENTRIES {
                return Err(format!(
                    "workspace holds more than {INSPECT_MAX_ENTRIES} entries"
                ));
            }
        }
    }
    out.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let mut h = Sha256::new();
    for e in &out {
        h.update(e["path"].as_str().unwrap_or("").as_bytes());
        h.update([0]);
        h.update(e["sha256"].as_str().unwrap_or("-").as_bytes());
        h.update(b"\n");
    }
    Ok((out, hex::encode(h.finalize())))
}

async fn step(sub: &str, m: &HashMap<String, String>) -> Result<Value, String> {
    match sub {
        "remember" => {
            let text = need(m, "text")?;
            let n = note("constraint", text, vec![]).await?;
            Ok(json!({"step": "S1", "constraint": n}))
        }
        "inspect" => {
            let (_, ws) = confine(need(m, "workspace")?)?;
            let (entries, tree_sha) = tree(&ws)?;
            let args = json!({"workspace": ws.display().to_string()});
            let result = json!({"entries": entries.len(), "tree_sha256": tree_sha});
            let rc = receipt("inspect", &args, &result, true)?;
            let text = json!({"tool": "inspect", "receipt_sha256": rc["sha256"],
                "arguments_digest": rc["arguments_digest"], "result_digest": rc["result_digest"],
                "workspace": ws.display().to_string(), "tree_sha256": tree_sha});
            let n = note("effect", &text.to_string(), vec![]).await?;
            Ok(json!({"step": "S2", "receipt": rc, "effect": n, "tree_sha256": tree_sha,
                "entries": entries}))
        }
        "propose" => {
            let (_, ws) = confine(need(m, "workspace")?)?;
            match send(ControlCommand::RunComposeTask {
                goal: need(m, "goal")?.to_string(),
                workspace: ws.display().to_string(),
            })
            .await?
            {
                ControlResponse::ComposeTaskResult(r) => Ok(json!({"step": "S3", "report": r})),
                other => Err(format!("unexpected response {other:?}")),
            }
        }
        "authorize" => {
            let r = read_report(need(m, "report")?)?;
            let (engine, ws) = confine(need(m, "workspace")?)?;
            let approver = need(m, "approver")?;
            let (path, content, psha) = committed_proposal(&r)?;
            let target = engine
                .validate_path(ws.join(&path).to_str().ok_or("path is not UTF-8")?)
                .map_err(|e| format!("CONFINEMENT DENIAL: {e}"))?;
            // sovereign-core #249: the grant names the canonical workspace
            // (symlinks resolved, whatever form --workspace took) and the
            // target must be exactly <workspace>/<path>, as the daemon checks.
            let wsc = std::fs::canonicalize(&ws)
                .map_err(|e| format!("workspace {}: {e}", ws.display()))?;
            if target != wsc.join(&path) {
                return Err(format!(
                    "CONFINEMENT DENIAL: target {} is not {}",
                    target.display(),
                    wsc.join(&path).display()
                ));
            }
            let csha = sha256_hex(content.as_bytes());
            // NEXT-PHASE-2: the grant names the target and the state it was
            // granted against; a changed target makes it stale.
            let prior = effects::file_sha256(&target)?;
            let tgt = target.display().to_string();
            let args = json!({"proposal_sha256": psha, "path": path, "content_sha256": csha,
                "approver": approver, "cx_promotion": r.cx_promotion, "target": tgt,
                "prior_sha256": prior});
            let result = json!({"approved": true, "target": target.display().to_string()});
            let rc = receipt("authorize", &args, &result, true)?;
            let constraint = m.get("constraint").map(|s| ids(s)).transpose()?;
            let mut links = vec![r.cx_promotion, r.cx_evidence];
            links.extend(constraint.unwrap_or_default().into_iter().take(2));
            let text = json!({"proposal_sha256": psha, "path": path, "content_sha256": csha,
                "approver": approver, "receipt_sha256": rc["sha256"], "target": tgt,
                "prior_sha256": prior, "workspace": wsc.display().to_string()});
            let n = note("authorization", &text.to_string(), links).await?;
            Ok(json!({"step": "S4", "receipt": rc, "authorization": n, "approvals": 1}))
        }
        "execute" => {
            let r = read_report(need(m, "report")?)?;
            let (engine, ws) = confine(need(m, "workspace")?)?;
            let auth: u64 = need(m, "authorization")?
                .parse()
                .map_err(|e| format!("--authorization: {e}"))?;
            let (path, content, psha) = committed_proposal(&r)?;
            let csha = sha256_hex(content.as_bytes());
            let target = engine
                .validate_path(ws.join(&path).to_str().ok_or("path is not UTF-8")?)
                .map_err(|e| format!("CONFINEMENT DENIAL: {e}"))?;
            // NEXT-PHASE-2 (ACCEPTANCE-v2 2.1-2.3): the daemon checks the grant
            // (stop, revoke, stale, spent) and records the intent durably
            // before anything touches the world. No intent, no write.
            fault_hold("before_intent");
            let (pid, start) = effects::self_executor();
            let intent = match send(ControlCommand::ComposeEffectIntent {
                authorization: auth,
                proposal_sha256: psha.clone(),
                path: path.clone(),
                target: target.display().to_string(),
                content_sha256: csha.clone(),
                executor_pid: pid,
                executor_start: start,
            })
            .await?
            {
                ControlResponse::ComposeNoted(n) => n,
                other => return Err(format!("unexpected response {other:?}")),
            };
            fault_hold("after_intent");
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            let written = std::fs::write(&target, content.as_bytes())
                .map_err(|e| format!("write {}: {e}", target.display()))
                .and_then(|_| std::fs::read(&target).map_err(|e| e.to_string()));
            let (disk_sha, bytes) = match &written {
                Ok(b) => (sha256_hex(b), b.len()),
                Err(_) => (String::new(), 0),
            };
            fault_hold("after_write");
            let reported = json!({"disk_sha256": disk_sha, "bytes": bytes,
                "error": written.as_ref().err()});
            // The ack's state is the daemon's reading of the world. If the
            // ack cannot be recorded the effect is UNRESOLVED until reconcile.
            let ack = send(ControlCommand::ComposeEffectAck {
                intent: intent.id,
                reported: reported.clone(),
            })
            .await;
            let (state, ack_id, ack_err) = match &ack {
                Ok(ControlResponse::ComposeNoted(n)) => {
                    let rec = recall(vec![n.id], None).await.ok();
                    let st = rec
                        .and_then(|r| r.cited.into_iter().next())
                        .and_then(|c| c.text)
                        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                        .and_then(|v| v["state"].as_str().map(str::to_string))
                        .unwrap_or_else(|| "UNRESOLVED".into());
                    (st, Some(n.id), None)
                }
                Ok(other) => ("UNRESOLVED".to_string(), None, Some(format!("{other:?}"))),
                Err(e) => ("UNRESOLVED".to_string(), None, Some(e.clone())),
            };
            let ok = state == "DONE";
            let args = json!({"path": target.display().to_string(), "content_sha256": csha,
                "authorization": auth, "intent": intent.id});
            let result = json!({"bytes": bytes, "disk_sha256": disk_sha, "error": written.err(),
                "state": state, "ack": ack_id, "ack_error": ack_err});
            let rc = receipt("write_file", &args, &result, ok)?;
            let out = json!({"step": "S5", "receipt": rc, "intent": intent,
                "effect": {"id": ack_id}, "state": state, "path": target.display().to_string(),
                "content_sha256": csha, "disk_sha256": disk_sha, "bytes": bytes,
                "ack_error": ack_err});
            // handle_compose_command turns a state other than DONE into
            // "ok": false (exit 3 for UNRESOLVED, 1 otherwise).
            Ok(out)
        }
        "explain" => {
            let r = read_report(need(m, "report")?)?;
            let (path, _, _) = committed_proposal(&r)?;
            let mut cite = ids(need(m, "cite")?)?;
            let mut all = vec![r.cx_goal];
            all.extend(&r.cx_candidates);
            all.extend([r.cx_evidence, r.cx_promotion]);
            all.append(&mut cite);
            let rec = recall(all.clone(), None).await?;
            if !rec.missing.is_empty() || rec.cited.iter().any(|c| !c.verified) {
                return Err(format!("cited records missing or unverified: {:?}", rec.missing));
            }
            let mut receipts = Vec::new();
            for p in need(m, "receipts")?.split(',').filter(|p| !p.is_empty()) {
                let b = std::fs::read(p).map_err(|e| format!("receipt {p}: {e}"))?;
                let v: Value = serde_json::from_slice(&b).map_err(|e| format!("{p}: {e}"))?;
                receipts.push(json!({"path": p, "sha256": sha256_hex(&b), "tool": v["tool"],
                    "success": v["success"]}));
            }
            let short = |d: &str| d.chars().take(16).collect::<String>();
            let mut lines = vec![format!(
                "Result: {path} in the authorized workspace was written once, as proposed by {} \
                 and approved by the operator. Evidence (Cortex journal {}/cortex.cx, machine {}):",
                r.proposer,
                rec.compose_dir,
                short(&rec.machine_id)
            )];
            for c in rec.cited.iter() {
                let what = match (c.note.as_deref(), c.id) {
                    (Some(n), _) => n.to_string(),
                    (None, id) if id == r.cx_goal => "goal".into(),
                    (None, id) if id == r.cx_evidence => {
                        format!("AEGIS verdict (pass mask {:#x})", r.aegis_pass_mask)
                    }
                    (None, id) if id == r.cx_promotion => "promotion of the winner".into(),
                    (None, _) => "candidate".into(),
                };
                let text = c
                    .text
                    .as_deref()
                    .map(|t| format!(": {}", t.chars().take(160).collect::<String>()))
                    .unwrap_or_default();
                lines.push(format!("- cortex.cx#{} {what} (digest {}){text}", c.id, short(&c.digest)));
            }
            for rc in &receipts {
                lines.push(format!(
                    "- receipt {} {} success={} (sha256 {})",
                    rc["tool"].as_str().unwrap_or("?"),
                    rc["path"].as_str().unwrap_or("?"),
                    rc["success"],
                    short(rc["sha256"].as_str().unwrap_or(""))
                ));
            }
            let text = lines.join("\n");
            Ok(json!({"step": "S6", "text": text, "cited": rec.cited, "receipts": receipts,
                "machine_id": rec.machine_id}))
        }
        "recall" => {
            let want = m.get("ids").map(|s| ids(s)).transpose()?.unwrap_or_default();
            let prefix = m
                .get("prefix")
                .map(|s| s.parse::<u64>().map_err(|e| format!("--prefix: {e}")))
                .transpose()?;
            let rec = recall(want, prefix).await?;
            let of = |k: &str| -> Vec<Value> {
                rec.host
                    .iter()
                    .filter(|h| h.note.as_deref() == Some(k))
                    .map(|h| json!({"id": h.id, "digest": h.digest, "verified": h.verified,
                        "text": h.text}))
                    .collect()
            };
            Ok(json!({"step": "S8", "machine_id": rec.machine_id,
                "constraints": of("constraint"), "authorizations": of("authorization"),
                "effects": of("effect"), "repairs": of("repair_tail"), "recall": rec}))
        }
        "shutdown" => {
            AienRuntimeClient::default_client().shutdown().await?;
            Ok(json!({"step": "S7", "shutdown": true}))
        }
        // NEXT-PHASE-2 operator commands (ACCEPTANCE-v2 2.4-2.6).
        "effects" => {
            let rec = recall(vec![], None).await?;
            let ledger = effects::Ledger::from_records(&rec.host).map_err(|r| r.to_string())?;
            Ok(json!({"machine_id": rec.machine_id, "ledger": ledger.view()}))
        }
        "reconcile" => {
            let declare = match m.get("declare") {
                None => None,
                Some(d) => Some(ReconcileDeclare {
                    intent: need(m, "intent")?
                        .parse()
                        .map_err(|e| format!("--intent: {e}"))?,
                    state: d.clone(),
                    approver: need(m, "approver")?.to_string(),
                }),
            };
            match send(ControlCommand::ComposeReconcile { declare }).await? {
                ControlResponse::ComposeReconciled(r) => Ok(json!({"reconcile": r})),
                other => Err(format!("unexpected response {other:?}")),
            }
        }
        "stop" | "resume" | "revoke" => {
            let authorization = if sub == "revoke" {
                Some(
                    need(m, "authorization")?
                        .parse::<u64>()
                        .map_err(|e| format!("--authorization: {e}"))?,
                )
            } else {
                None
            };
            match send(ControlCommand::ComposeControl {
                action: sub.to_string(),
                approver: need(m, "approver")?.to_string(),
                authorization,
            })
            .await?
            {
                ControlResponse::ComposeControlled(r) => Ok(json!({"control": r})),
                other => Err(format!("unexpected response {other:?}")),
            }
        }
        "desk-key" => {
            use aien_runtime::approved_auth::{desk_key_path, DeskKey};
            let dir = aien_runtime::spine::compose_dir_from_env()?;
            let path = desk_key_path(&dir);
            let key = match (DeskKey::load(&path), m.contains_key("create")) {
                (Ok(k), _) => k,
                (Err(_), true) if !path.exists() => DeskKey::create(&path)?,
                (Err(e), _) => return Err(e),
            };
            Ok(json!({"desk_key_id": key.id(), "path": path.display().to_string()}))
        }
        "recover" => match send(ControlCommand::RecoverComposeHome).await? {
            ControlResponse::ComposeRecovered(r) => Ok(json!({"repair": r, "opens": r.opens})),
            other => Err(format!("unexpected response {other:?}")),
        },
        other => Err(format!(
            "unknown compose step {other:?} (remember, inspect, propose, authorize, execute, explain, recall, shutdown, recover, effects, reconcile, stop, resume, revoke, desk-key)"
        )),
    }
}

pub async fn handle_compose_command(args: &[String]) {
    let Some(sub) = args.first() else {
        println!(
            "{}",
            json!({"ok": false, "error": "usage: aien compose <step> [--flag value ...]"})
        );
        std::process::exit(1);
    };
    let out = match flags(&args[1..]) {
        Ok(m) => step(sub, &m).await,
        Err(e) => Err(e),
    };
    match out {
        Ok(mut v) => {
            // An effect step reports its state; only DONE is success.
            let state = v.get("state").and_then(Value::as_str).map(str::to_string);
            let ok = state.as_deref().is_none_or(|s| s == "DONE");
            v["ok"] = json!(ok);
            println!("{v}");
            if !ok {
                std::process::exit(if state.as_deref() == Some("UNRESOLVED") {
                    3
                } else {
                    1
                });
            }
        }
        Err(e) => {
            println!("{}", json!({"ok": false, "step": sub, "error": e}));
            std::process::exit(1);
        }
    }
}

/// Test builds only (cargo feature `fault-hold`, ACCEPTANCE-v2 2.9): stop at
/// the named point when `AIEN_FAULT_HOLD` names it, until the harness deletes
/// `AIEN_FAULT_HOLD_FILE`. The default build compiles this to nothing.
#[cfg(feature = "fault-hold")]
fn fault_hold(point: &str) {
    if std::env::var("AIEN_FAULT_HOLD").as_deref() != Ok(point) {
        return;
    }
    let Ok(file) = std::env::var("AIEN_FAULT_HOLD_FILE") else {
        eprintln!("AIEN_FAULT_HOLD={point} without AIEN_FAULT_HOLD_FILE");
        std::process::exit(86);
    };
    if let Err(e) = std::fs::write(&file, format!("{} {point}\n", std::process::id())) {
        eprintln!("fault hold {point}: write {file}: {e}");
        std::process::exit(86);
    }
    let t0 = std::time::Instant::now();
    while Path::new(&file).exists() {
        if t0.elapsed() > std::time::Duration::from_secs(300) {
            eprintln!("fault hold {point}: not released in 300 s");
            std::process::exit(86);
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(not(feature = "fault-hold"))]
fn fault_hold(_point: &str) {}
