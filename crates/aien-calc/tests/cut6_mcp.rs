//! Cut 6 tests: the MCP server, driven by a real rmcp client over an in-process pipe.
#![cfg(feature = "mcp")]

use aien_calc::mcp::{schema_digest, CalcServer};
use aien_calc::store::Store;
use aien_calc::Json;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Pinned digest of the advertised tool list (names, descriptions, input schemas). Changing a
/// schema changes this on purpose: update it together with the change.
const PINNED_SCHEMA_DIGEST: &str =
    "16cc960a2c0cbc8de835f3b4cf471442505381db5717c3de5a2e29200fb4a124";

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("aien-calc-mcp-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

async fn connect(server: CalcServer) -> RunningService<RoleClient, ()> {
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    ().serve(client_io).await.expect("client handshake")
}

async fn call(
    c: &RunningService<RoleClient, ()>,
    tool: &'static str,
    args: Value,
) -> CallToolResult {
    let params = CallToolRequestParams::new(tool).with_arguments(args.as_object().unwrap().clone());
    c.call_tool(params).await.expect("call_tool")
}

fn structured(r: &CallToolResult) -> &Value {
    r.structured_content.as_ref().expect("structured content")
}

fn text(r: &CallToolResult) -> String {
    r.content
        .iter()
        .find_map(|b| b.as_text().map(|t| t.text.clone()))
        .unwrap_or_default()
}

fn digest_of(tools: &[rmcp::model::Tool]) -> String {
    // Independent of the library: rebuild the canonical listing from what the client received.
    let mut listing: Vec<Json> = tools
        .iter()
        .map(|t| {
            let schema = Json::parse(&serde_json::to_string(&*t.input_schema).unwrap()).unwrap();
            Json::obj(vec![
                ("description", Json::str(t.description.as_deref().unwrap())),
                ("name", Json::str(&t.name)),
                ("schema", schema),
            ])
        })
        .collect();
    let order = ["eval", "sweep", "verify"];
    listing.sort_by_key(|j| {
        order
            .iter()
            .position(|n| Some(&Json::str(n)) == j.get("name"))
            .unwrap()
    });
    blake3::hash(Json::Arr(listing).to_canonical().as_bytes())
        .to_hex()
        .to_string()
}

#[tokio::test]
async fn lists_exactly_the_three_tools_with_object_schemas() {
    let c = connect(CalcServer::new(None)).await;
    let tools = c.list_all_tools().await.unwrap();
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    names.sort();
    assert_eq!(names, ["eval", "sweep", "verify"]);
    for t in &tools {
        assert_eq!(
            t.input_schema.get("type"),
            Some(&json!("object")),
            "{}",
            t.name
        );
        assert!(
            t.description.as_deref().is_some_and(|d| d.len() > 20),
            "{} needs a description",
            t.name
        );
    }
    let required = |n: &str| -> Vec<String> {
        let t = tools.iter().find(|t| t.name == n).unwrap();
        let mut r: Vec<String> = t.input_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().into())
            .collect();
        r.sort();
        r
    };
    assert_eq!(required("eval"), ["expr"]);
    assert_eq!(required("sweep"), ["expect", "format", "got"]);
    assert_eq!(required("verify"), ["id"]);
}

#[tokio::test]
async fn schema_digest_is_stable_across_servers_and_pinned() {
    let a = connect(CalcServer::new(None)).await;
    let b = connect(CalcServer::new(None)).await;
    let da = digest_of(&a.list_all_tools().await.unwrap());
    let db = digest_of(&b.list_all_tools().await.unwrap());
    assert_eq!(da, db);
    assert_eq!(
        da,
        schema_digest(),
        "the library's digest must describe what the server advertises"
    );
    assert_eq!(
        schema_digest(),
        PINNED_SCHEMA_DIGEST,
        "tool schemas changed: update the pin deliberately"
    );
}

#[tokio::test]
async fn eval_returns_the_receipt_as_structured_content_and_a_human_text_block() {
    let dir = tmp("eval");
    let c = connect(CalcServer::new(Some(Store::new(&dir)))).await;
    let r = call(&c, "eval", json!({"expr": "1/3"})).await;
    assert_ne!(r.is_error, Some(true));
    let direct = aien_calc::eval("1/3", 20).unwrap();
    let s = structured(&r);
    assert_eq!(s["id"], json!(direct.receipt().id()));
    assert_eq!(s["exactness"], json!("exact"));
    assert_eq!(s["refused"], json!(false));
    assert_eq!(s["receipt"]["result"]["value"], json!("1/3"));
    let t = text(&r);
    assert!(t.contains("value   1/3"), "{t}");
    assert_eq!(
        t.lines().last().unwrap(),
        format!("receipt {}", direct.receipt().id())
    );
    // the call was stored, so verify by id works afterwards
    let v = call(&c, "verify", json!({"id": direct.receipt().id()})).await;
    assert_eq!(
        structured(&v)["matches"],
        json!(true),
        "{:?}",
        structured(&v)
    );
    assert_eq!(
        text(&v).lines().next().unwrap(),
        format!("MATCH {}", direct.receipt().id())
    );
}

#[tokio::test]
async fn precision_argument_changes_the_decimal_and_the_receipt() {
    let c = connect(CalcServer::new(None)).await;
    let r = call(&c, "eval", json!({"expr": "1/3", "precision": 5})).await;
    assert_eq!(
        structured(&r)["receipt"]["result"]["decimal"],
        json!("0.33333")
    );
    let r20 = call(&c, "eval", json!({"expr": "1/3"})).await;
    assert_ne!(structured(&r)["id"], structured(&r20)["id"]);
}

#[tokio::test]
async fn a_refusal_is_an_answer_with_a_receipt_not_a_tool_error() {
    let c = connect(CalcServer::new(None)).await;
    let r = call(&c, "eval", json!({"expr": "1/0"})).await;
    assert_ne!(r.is_error, Some(true));
    let s = structured(&r);
    assert_eq!(s["refused"], json!(true));
    assert_eq!(s["exactness"], json!("refused"));
    assert!(s["id"].as_str().unwrap().starts_with("calc:"));
}

#[tokio::test]
async fn malformed_requests_are_tool_errors_without_a_receipt() {
    let c = connect(CalcServer::new(None)).await;
    for (tool, args) in [
        ("eval", json!({"expr": "1 +"})),
        ("eval", json!({})),
        ("eval", json!({"expr": 5})),
        ("eval", json!({"expr": "1", "precision": 0})),
        ("eval", json!({"expr": "1", "precision": "x"})),
        (
            "sweep",
            json!({"format": "f32", "expect": "$x", "got": "$x"}),
        ),
        ("verify", json!({"id": "calc:ffffffffffff"})),
        ("verify", json!({"id": "../../etc/passwd"})),
        ("nonesuch", json!({})),
    ] {
        let r = call(&c, tool, args.clone()).await;
        assert_eq!(r.is_error, Some(true), "{tool} {args}");
        assert!(
            r.structured_content
                .as_ref()
                .is_none_or(|s| s.get("id").is_none()),
            "{tool} {args}"
        );
        assert!(!text(&r).is_empty());
    }
}

#[tokio::test]
async fn sweep_reproduces_the_spec_golden_and_the_negative_control() {
    let c = connect(CalcServer::new(None)).await;
    let w = "(($b & 0x80) << 8) | (($b & 0x7f) << 4)";
    let good = call(
        &c,
        "sweep",
        json!({"format": "e4m3", "expect": "$x", "got": format!("val(bf16, {w}) * 2^120"), "where": "class(e4m3, $x) != nan"}),
    )
    .await;
    let s = structured(&good);
    assert_eq!(s["receipt"]["result"]["mismatch_count"], json!(0));
    assert_eq!(s["receipt"]["result"]["considered"], json!(254));
    let bad = call(
        &c,
        "sweep",
        json!({"format": "e4m3", "expect": "$x", "got": format!("val(bf16, {w})"), "where": "class(e4m3, $x) != nan"}),
    )
    .await;
    assert_eq!(
        structured(&bad)["receipt"]["result"]["mismatch_count"],
        json!(252)
    );
    assert_ne!(structured(&good)["id"], structured(&bad)["id"]);
}

#[tokio::test]
async fn verify_reports_a_tampered_stored_receipt() {
    let dir = tmp("tamper");
    let store = Store::new(&dir);
    let c = connect(CalcServer::new(Some(Store::new(&dir)))).await;
    let r = call(&c, "eval", json!({"expr": "1/4"})).await;
    let id = structured(&r)["id"].as_str().unwrap().to_string();
    let path = store.receipt_path(id.strip_prefix("calc:").unwrap());
    let forged = std::fs::read_to_string(&path)
        .unwrap()
        .replace("\"decimal_exact\":true", "\"decimal_exact\":false");
    std::fs::write(&path, forged).unwrap();
    let v = call(&c, "verify", json!({"id": id})).await;
    assert_ne!(v.is_error, Some(true));
    assert_eq!(structured(&v)["matches"], json!(false));
    assert!(structured(&v)["diffs"].as_array().unwrap().len() >= 2);
    assert!(text(&v).starts_with("MISMATCH"));
}

/// The real binary over stdio, as Claude Code drives it: a tools/list result that carries
/// ttlMs must also carry cacheScope, or Claude Code refuses the whole tool list.
#[test]
fn stdio_tools_list_carries_cache_scope_with_ttl() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    let mut child = Command::new(env!("CARGO_BIN_EXE_aien-calc"))
        .arg("--mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for line in [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ] {
        writeln!(stdin, "{line}").unwrap();
    }
    let mut reply = None;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if v["id"] == 2 {
            reply = Some(v);
            break;
        }
    }
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    let result = &reply.expect("no tools/list reply")["result"];
    assert!(result.get("ttlMs").is_some(), "ttlMs missing: {result}");
    assert_eq!(
        result["cacheScope"], "public",
        "cacheScope missing: {result}"
    );
}
