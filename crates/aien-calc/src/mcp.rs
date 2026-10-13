//! `aien-calc --mcp`: the three tools (`eval`, `sweep`, `verify`) over MCP.
//!
//! Results follow the shape `aien-mcp` reads: `structured_content` carries the machine answer
//! (including the full receipt) and one text block carries the human lines. A well-formed question
//! that exact arithmetic declines is an answer with a receipt (`refused: true`), not a tool error;
//! malformed requests are tool errors and issue no receipt.

use crate::json::Json;
use crate::receipt::{eval as run_eval, Receipt};
use crate::store::Store;
use crate::sweep::sweep as run_sweep;
use crate::verify::verify_bytes;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    JsonObject, ListToolsResult, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::{ServerHandler, ServiceExt};
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// One advertised tool.
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: Value,
}

/// The advertised tools, in a fixed order.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "eval",
            description: "Evaluate one expression exactly (rational arithmetic, number formats, units) and return a receipt id. \
Use it instead of mental math for any figure that goes into a decision or document. Examples: `2^-126`, `98 KB -> B`, \
`1 / (6.1 GB/tok / 273 GB/s) -> tok/s`, `bits(e4m3, 17/16)`. Cite the returned id (calc:<hex>) next to the number.",
            schema: json!({
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "calc-lang v1 expression"},
                    "precision": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "significant digits for the decimal rendering (default 20)"}
                },
                "required": ["expr"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "sweep",
            description: "Compare two expressions for every code of a small number format (at most 16 bits: e4m3, e5m2, f16, bf16, int8, int4, uint8, uint4). \
`$b` is the code and `$x` its exact value. Returns exact counts, the mismatching codes and a receipt id; use it to prove a conversion or encoding is right for all codes.",
            schema: json!({
                "type": "object",
                "properties": {
                    "format": {"type": "string", "description": "e4m3, e5m2, f16, bf16, int8, int4, uint8 or uint4"},
                    "expect": {"type": "string", "description": "expected value, in terms of $b and $x"},
                    "got": {"type": "string", "description": "value actually produced by the construction under test"},
                    "flag": {"type": "string", "description": "optional true/false expression; codes where it is true are listed"},
                    "where": {"type": "string", "description": "optional true/false filter; only codes where it is true are compared"}
                },
                "required": ["format", "expect", "got"],
                "additionalProperties": false
            }),
        },
        ToolSpec {
            name: "verify",
            description: "Recompute a stored receipt from its own recorded inputs and report MATCH or MISMATCH with the differing fields. \
Use it before relying on a calc:<hex> id someone else cited.",
            schema: json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "receipt id, calc:<hex> (12 or more hex digits)"}
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        },
    ]
}

/// BLAKE3 over the canonical serialisation of the advertised tools; changes when any name,
/// description or schema changes.
pub fn schema_digest() -> String {
    let listing: Vec<Json> = tool_specs()
        .iter()
        .map(|t| {
            Json::obj(vec![
                ("description", Json::str(t.description)),
                ("name", Json::str(t.name)),
                ("schema", from_value(&t.schema)),
            ])
        })
        .collect();
    blake3::hash(Json::Arr(listing).to_canonical().as_bytes())
        .to_hex()
        .to_string()
}

fn from_value(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => Json::Int(n.as_i64().unwrap_or(0)),
        Value::String(s) => Json::Str(s.clone()),
        Value::Array(a) => Json::Arr(a.iter().map(from_value).collect()),
        Value::Object(m) => Json::Obj(m.iter().map(|(k, v)| (k.clone(), from_value(v))).collect()),
    }
}

fn to_value(j: &Json) -> Value {
    match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Int(n) => Value::from(*n),
        Json::Str(s) => Value::String(s.clone()),
        Json::Arr(a) => Value::Array(a.iter().map(to_value).collect()),
        Json::Obj(p) => Value::Object(p.iter().map(|(k, v)| (k.clone(), to_value(v))).collect()),
    }
}

/// Server state: where receipts are saved (`None`: nowhere; verify then has nothing to read).
#[derive(Clone)]
pub struct CalcServer {
    store: Option<Arc<Store>>,
}

impl Default for CalcServer {
    /// Store from `AIEN_CALC_HOME` / `~/.aien/calc`.
    fn default() -> Self {
        CalcServer::new(Store::from_env().ok())
    }
}

type ToolResult = Result<(Value, String, bool), String>;

fn str_arg<'a>(args: &'a JsonObject, key: &str) -> Result<&'a str, String> {
    match args.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(_) => Err(format!("`{key}` must be a string")),
        None => Err(format!("`{key}` is required")),
    }
}

fn opt_str_arg<'a>(args: &'a JsonObject, key: &str) -> Result<Option<&'a str>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(format!("`{key}` must be a string")),
    }
}

impl CalcServer {
    pub fn new(store: Option<Store>) -> Self {
        CalcServer {
            store: store.map(Arc::new),
        }
    }

    fn save(
        &self,
        r: &Receipt,
        tool: &str,
        args: &JsonObject,
        structured: &mut Map<String, Value>,
    ) {
        let Some(store) = &self.store else { return };
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let argv = vec![
            "aien-calc".to_string(),
            "--mcp".to_string(),
            tool.to_string(),
            Value::Object(args.clone()).to_string(),
        ];
        if let Err(e) = store.save(r, &cwd, &argv) {
            structured.insert(
                "store_warning".into(),
                Value::String(format!("receipt not stored: {e}")),
            );
        }
    }

    fn answer(
        &self,
        tool: &str,
        args: &JsonObject,
        r: &Receipt,
        text: &str,
        refused: bool,
    ) -> ToolResult {
        let mut s = Map::new();
        s.insert("id".into(), Value::String(r.id()));
        s.insert(
            "exactness".into(),
            to_value(r.body().get("exactness").unwrap_or(&Json::Null)),
        );
        s.insert("refused".into(), Value::Bool(refused));
        s.insert("receipt".into(), to_value(r.body()));
        self.save(r, tool, args, &mut s);
        Ok((Value::Object(s), text.to_string(), false))
    }

    fn dispatch(&self, tool: &str, args: &JsonObject) -> ToolResult {
        match tool {
            "eval" => {
                let expr = str_arg(args, "expr")?;
                let precision = match args.get("precision") {
                    None | Some(Value::Null) => 20,
                    Some(v) => v
                        .as_u64()
                        .filter(|p| *p <= u32::MAX as u64)
                        .ok_or("`precision` must be an integer from 1 to 1000")?
                        as u32,
                };
                let out = run_eval(expr, precision).map_err(|e| e.to_string())?;
                self.answer(
                    "eval",
                    args,
                    out.receipt(),
                    out.text(),
                    out.refusal().is_some(),
                )
            }
            "sweep" => {
                let out = run_sweep(
                    str_arg(args, "format")?,
                    str_arg(args, "expect")?,
                    str_arg(args, "got")?,
                    opt_str_arg(args, "flag")?,
                    opt_str_arg(args, "where")?,
                )
                .map_err(|e| e.to_string())?;
                self.answer("sweep", args, out.receipt(), out.text(), false)
            }
            "verify" => {
                let id = str_arg(args, "id")?;
                let store = self
                    .store
                    .as_ref()
                    .ok_or("no receipt store is configured")?;
                let hex = store.resolve(id)?;
                let bytes = store.load(&hex)?;
                let v = verify_bytes(&bytes, Some(&hex));
                let text = if v.matches {
                    format!("MATCH {}", v.id)
                } else {
                    format!("MISMATCH {}\n  {}", v.id, v.diffs.join("\n  "))
                };
                let s = json!({"matches": v.matches, "id": v.id, "diffs": v.diffs});
                Ok((s, text, false))
            }
            other => Err(format!("unknown tool `{other}`")),
        }
    }
}

impl ServerHandler for CalcServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("aien-calc", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Exact arithmetic with receipts. Call `eval` for any number you would otherwise compute in your head, \
and cite the returned calc:<hex> id beside the figure.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let listed = tool_specs()
            .into_iter()
            .map(|t| {
                let schema = t.schema.as_object().cloned().unwrap_or_default();
                Tool::new(t.name, t.description, Arc::new(schema))
            })
            .collect();
        Ok(ListToolsResult::with_all_items(listed).with_ttl_ms(0))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, rmcp::ErrorData> {
        let args = request.arguments.unwrap_or_default();
        Ok(match self.dispatch(request.name.as_ref(), &args) {
            Ok((structured, text, _)) => {
                let mut r = CallToolResult::structured(structured);
                r.content = vec![ContentBlock::text(text)];
                r.into()
            }
            Err(reason) => CallToolResult::error(vec![ContentBlock::text(reason)]).into(),
        })
    }
}

/// Serve on stdin/stdout until the client disconnects.
pub fn serve_stdio() -> std::io::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let running = CalcServer::default()
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        running
            .waiting()
            .await
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    })
}
