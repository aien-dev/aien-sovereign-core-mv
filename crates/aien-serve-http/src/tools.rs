//! Tool calling in the model's own chat format, rendered exactly as its chat template does it (the
//! `tools` system block, assistant `tool_calls`, and `tool` messages folded into one user turn of
//! `<tool_response>` blocks), and the reply parsed back into OpenAI `tool_calls`. Two formats:
//! Qwen3.6/3.8 qwen3_coder XML (`<function=..><parameter=..>`) and the Hermes JSON of Qwen3-2507
//! (`{"name": .., "arguments": ..}`); [`ToolFormat::from_model_dir`] picks one from the template.
//! Drake 2026-10-21: "make it support tool use ... build it in".

use std::path::Path;

use serde_json::{Map, Value};

/// The tool-calling layout of a model's chat template.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolFormat {
    /// Qwen3.6-35B-A3B / Qwen3.8-27B (`qwen3_coder`): XML function and parameter tags; the
    /// tools block comes before the system text.
    #[default]
    Qwen3CoderXml,
    /// Qwen3-2507 Instruct (e.g. Qwen3-30B-A3B-Instruct-2507): Hermes JSON inside `<tool_call>`;
    /// the system text comes before the tools block.
    HermesJson,
}

/// The sentence only the Hermes tools block carries.
const HERMES_MARKER: &str = "You may call one or more functions to assist with the user query.";
const HERMES_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const HERMES_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

impl ToolFormat {
    /// Hermes JSON when the template carries the Hermes tools instruction, qwen3_coder XML otherwise.
    pub fn detect(template_text: &str) -> Self {
        if template_text.contains(HERMES_MARKER) {
            Self::HermesJson
        } else {
            Self::Qwen3CoderXml
        }
    }

    /// Reads the template from the sources `ChatTokenizer::from_model_dir` renders with, in its
    /// order: `tokenizer_config.json` `chat_template`, else `chat_template.jinja`. No template:
    /// qwen3_coder XML.
    pub fn from_model_dir(dir: &Path) -> Result<Self, String> {
        let read =
            |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
        let config = dir.join("tokenizer_config.json");
        if config.exists() {
            let v: Value = serde_json::from_str(&read(&config)?)
                .map_err(|e| format!("{}: {e}", config.display()))?;
            if let Some(t) = v.get("chat_template").and_then(Value::as_str) {
                return Ok(Self::detect(t));
            }
        }
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            return Ok(Self::detect(&read(&jinja)?));
        }
        Ok(Self::Qwen3CoderXml)
    }

    pub fn system_with_tools(self, tools: &[Value], system: Option<&str>) -> String {
        match self {
            Self::Qwen3CoderXml => system_with_tools(tools, system),
            Self::HermesJson => hermes_system_with_tools(tools, system),
        }
    }

    pub fn assistant_with_calls(self, content: &str, calls: &[Value]) -> String {
        match self {
            Self::Qwen3CoderXml => assistant_with_calls(content, calls),
            Self::HermesJson => hermes_assistant_with_calls(content, calls),
        }
    }
}

/// The chat turns of a request in this format: `msgs` are `(role, text, tool_calls)` with roles
/// already normalized to system/user/assistant/tool. Runs of `tool` messages fold into one user
/// turn; with tools, the first system turn carries the tools block (one is added when absent).
/// Both templates fold tool results the same way.
pub fn fold_turns(
    format: ToolFormat,
    tool_defs: &[Value],
    msgs: Vec<(&str, String, Option<&[Value]>)>,
) -> Vec<(String, String)> {
    let mut turns: Vec<(String, String)> = Vec::with_capacity(msgs.len() + 1);
    let mut pending_tool: Vec<String> = Vec::new();
    for (role, text, calls) in msgs {
        if role == "tool" {
            pending_tool.push(text);
            continue;
        }
        if !pending_tool.is_empty() {
            turns.push(("user".into(), tool_responses(&pending_tool)));
            pending_tool.clear();
        }
        let text = match (role, calls) {
            ("assistant", Some(calls)) if !calls.is_empty() => {
                format.assistant_with_calls(&text, calls)
            }
            _ => text,
        };
        turns.push((role.to_string(), text));
    }
    if !pending_tool.is_empty() {
        turns.push(("user".into(), tool_responses(&pending_tool)));
    }
    if !tool_defs.is_empty() {
        match turns.first_mut() {
            Some((r, c)) if r == "system" => {
                *c = format.system_with_tools(tool_defs, Some(c.as_str()));
            }
            _ => turns.insert(
                0,
                ("system".into(), format.system_with_tools(tool_defs, None)),
            ),
        }
    }
    turns
}

const TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

pub const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Python `json.dumps`-style text (", " and ": " separators), which is what jinja's `tojson` emits
/// for the template. Keys come out sorted (serde_json without preserve_order); the model reads either order.
pub fn tojson(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let parts: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), tojson(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(tojson).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

/// qwen3_coder: the system turn's content when tools are present: the tools block, then the
/// request's own system text after a blank line (the template trims it and drops it when empty).
pub fn system_with_tools(tools: &[Value], system: Option<&str>) -> String {
    let mut s = String::from(TOOLS_HEAD);
    for t in tools {
        s.push('\n');
        s.push_str(&tojson(t));
    }
    s.push_str(TOOLS_TAIL);
    if let Some(sys) = system.map(str::trim).filter(|x| !x.is_empty()) {
        s.push_str("\n\n");
        s.push_str(sys);
    }
    s
}

/// qwen3_coder: an assistant turn's content with its `tool_calls` appended in the template's layout.
pub fn assistant_with_calls(content: &str, calls: &[Value]) -> String {
    let mut s = content.to_string();
    for (i, call) in calls.iter().enumerate() {
        let f = call.get("function").unwrap_or(call);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        if i == 0 {
            if content.trim().is_empty() {
                s.push_str("<tool_call>\n<function=");
            } else {
                s.push_str("\n\n<tool_call>\n<function=");
            }
        } else {
            s.push_str("\n<tool_call>\n<function=");
        }
        s.push_str(name);
        s.push_str(">\n");
        // OpenAI sends arguments as a JSON string; the template iterates a mapping.
        let args = match f.get("arguments") {
            Some(Value::String(t)) => serde_json::from_str::<Value>(t).unwrap_or(Value::Null),
            Some(v) => v.clone(),
            None => Value::Null,
        };
        if let Value::Object(m) = args {
            for (k, v) in m {
                s.push_str("<parameter=");
                s.push_str(&k);
                s.push_str(">\n");
                match v {
                    Value::String(t) => s.push_str(&t),
                    other => s.push_str(&tojson(&other)),
                }
                s.push_str("\n</parameter>\n");
            }
        }
        s.push_str("</function>\n</tool_call>");
    }
    s
}

/// Hermes: the system turn's content when tools are present: the request's system text verbatim
/// and a blank line (the template neither trims nor drops it), then the tools block, one
/// `tojson` line per tool.
pub fn hermes_system_with_tools(tools: &[Value], system: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s.push_str(sys);
        s.push_str("\n\n");
    }
    s.push_str(HERMES_HEAD);
    for t in tools {
        s.push('\n');
        s.push_str(&tojson(t));
    }
    s.push_str(HERMES_TAIL);
    s
}

/// Hermes: an assistant turn's content with each call as `<tool_call>\n{"name": "N", "arguments":
/// A}\n</tool_call>`, a newline before every call but a first one with empty content. String
/// arguments (the OpenAI form) go in verbatim, others through `tojson`, as the template does.
pub fn hermes_assistant_with_calls(content: &str, calls: &[Value]) -> String {
    let mut s = content.to_string();
    for (i, call) in calls.iter().enumerate() {
        if i > 0 || !content.is_empty() {
            s.push('\n');
        }
        let f = call.get("function").unwrap_or(call);
        s.push_str("<tool_call>\n{\"name\": \"");
        s.push_str(f.get("name").and_then(Value::as_str).unwrap_or(""));
        s.push_str("\", \"arguments\": ");
        match f.get("arguments") {
            Some(Value::String(t)) => s.push_str(t),
            Some(v) => s.push_str(&tojson(v)),
            None => s.push_str("null"),
        }
        s.push_str("}\n</tool_call>");
    }
    s
}

/// A run of consecutive `tool` messages as one user turn's content.
pub fn tool_responses(contents: &[String]) -> String {
    let mut s = String::new();
    for (i, c) in contents.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str("<tool_response>\n");
        s.push_str(c);
        s.push_str("\n</tool_response>");
    }
    s
}

/// Splits a reply into its text before the first `<tool_call>` and the parsed calls
/// (OpenAI shape: `{id, type: "function", function: {name, arguments: <json string>}}`).
pub fn parse_reply(text: &str, id_base: &str) -> (String, Vec<Value>) {
    let Some(first) = text.find(OPEN) else {
        return (text.to_string(), Vec::new());
    };
    let content = text[..first].trim_end().to_string();
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(p) = rest.find(OPEN) {
        let body_start = p + OPEN.len();
        let (body, next) = match rest[body_start..].find(CLOSE) {
            Some(e) => (
                &rest[body_start..body_start + e],
                &rest[body_start + e + CLOSE.len()..],
            ),
            None => (&rest[body_start..], ""),
        };
        if let Some((name, args)) = parse_call(body) {
            calls.push(serde_json::json!({
                "id": format!("call_{id_base}_{}", calls.len()),
                "type": "function",
                "function": {"name": name, "arguments": Value::Object(args).to_string()},
            }));
        }
        rest = next;
    }
    (content, calls)
}

fn parse_call(body: &str) -> Option<(String, Map<String, Value>)> {
    let b = body.trim();
    // XML form: <function=NAME> <parameter=K>\nV\n</parameter> ... </function>
    if let Some(fs) = b.find("<function=") {
        let after = &b[fs + "<function=".len()..];
        let ne = after.find('>')?;
        let name = after[..ne].trim().to_string();
        let mut inner = &after[ne + 1..];
        if let Some(fe) = inner.rfind("</function>") {
            inner = &inner[..fe];
        }
        let mut args = Map::new();
        let mut r = inner;
        while let Some(ps) = r.find("<parameter=") {
            let a = &r[ps + "<parameter=".len()..];
            let Some(ke) = a.find('>') else { break };
            let key = a[..ke].trim().to_string();
            let v_all = &a[ke + 1..];
            let (raw, next) = match v_all.find("</parameter>") {
                Some(e) => (&v_all[..e], &v_all[e + "</parameter>".len()..]),
                None => match v_all.find("<parameter=") {
                    Some(e) => (&v_all[..e], &v_all[e..]),
                    None => (v_all, ""),
                },
            };
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            args.insert(key, value_of(raw));
            r = next;
        }
        return Some((name, args));
    }
    // JSON form: {"name": ..., "arguments": {...}}
    let v: Value = serde_json::from_str(b).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        _ => Map::new(),
    };
    Some((name, args))
}

/// A parameter value: JSON when it parses as a number, bool, null, object or array; else the text.
fn value_of(raw: &str) -> Value {
    let t = raw.trim();
    let looks_json = t.starts_with('{')
        || t.starts_with('[')
        || t == "true"
        || t == "false"
        || t == "null"
        || t.parse::<f64>().is_ok();
    if looks_json {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            return v;
        }
    }
    Value::String(raw.to_string())
}

/// Streaming guard: how many bytes at the end of `buf` could be the start of `<tool_call>`
/// (held back so a split tag is never streamed as content).
pub fn partial_open_len(buf: &str) -> usize {
    let max = OPEN.len().min(buf.len());
    for n in (1..=max).rev() {
        if buf.is_char_boundary(buf.len() - n) && OPEN.starts_with(&buf[buf.len() - n..]) {
            return n;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_xml_call_with_text_before() {
        let r = "Let me check.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";
        let (c, calls) = parse_reply(r, "x");
        assert_eq!(c, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "read_file");
        let a: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["path"], "/tmp/a.txt");
        assert_eq!(a["limit"], 20);
    }

    #[test]
    fn parses_two_calls_and_multiline_value() {
        let r = "<tool_call>\n<function=a>\n<parameter=s>\nline1\nline2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>";
        let (c, calls) = parse_reply(r, "x");
        assert_eq!(c, "");
        assert_eq!(calls.len(), 2);
        let a: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["s"], "line1\nline2");
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn parses_json_form() {
        let (_, calls) = parse_reply(
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>",
            "x",
        );
        assert_eq!(calls[0]["function"]["name"], "f");
    }

    #[test]
    fn plain_reply_has_no_calls() {
        let (c, calls) = parse_reply("Hello there.", "x");
        assert_eq!(c, "Hello there.");
        assert!(calls.is_empty());
    }

    #[test]
    fn renders_round_trip_layout() {
        let call = serde_json::json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"p\": \"v\", \"n\": 2}"}});
        let s = assistant_with_calls("", &[call]);
        // serde_json maps are key-sorted here (no preserve_order): n before p.
        assert_eq!(s, "<tool_call>\n<function=f>\n<parameter=n>\n2\n</parameter>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>");
        let (_, calls) = parse_reply(&s, "y");
        let a: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["p"], "v");
        assert_eq!(a["n"], 2);
    }

    #[test]
    fn tool_responses_fold() {
        assert_eq!(
            tool_responses(&["a".into(), "b".into()]),
            "<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response>"
        );
    }

    #[test]
    fn tojson_matches_python_separators() {
        let v: Value = serde_json::from_str(
            "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"p\":[1,2]}}",
        )
        .unwrap();
        assert_eq!(
            tojson(&v),
            "{\"function\": {\"name\": \"f\", \"p\": [1, 2]}, \"type\": \"function\"}"
        );
    }

    #[test]
    fn partial_open_hold_back() {
        assert_eq!(partial_open_len("hello <tool"), 5);
        assert_eq!(partial_open_len("hello"), 0);
        assert_eq!(partial_open_len("x<"), 1);
    }

    /// The Qwen3-4B-Instruct-2507 template pin; byte-identical to the `chat_template` of
    /// Qwen3-30B-A3B-Instruct-2507-FP8's tokenizer_config.json (checked 2026-10-26).
    const QWEN3_2507_TEMPLATE: &str = include_str!(
        "../../aien-inference-abi/fixtures/chat-templates/qwen3-4b-instruct-2507/chat_template.jinja"
    );
    const QWEN36_TEMPLATE: &str = include_str!(
        "../../aien-inference-abi/fixtures/chat-templates/qwen3.6-35b-a3b/chat_template.jinja"
    );
    const QWEN38_TEMPLATE: &str = include_str!(
        "../../aien-inference-abi/fixtures/chat-templates/qwen3.8-27b/chat_template.jinja"
    );
    /// 2 tools, a system message, an assistant turn with text and two calls (string and object
    /// arguments), two tool results, an assistant turn with no text and one call, one tool result.
    const HERMES_REQUEST: &str = include_str!("tools/hermes_request.json");
    /// `tools/hermes_request.json` rendered with add_generation_prompt=True by the real template
    /// (python3 jinja2 3.1.2, ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
    /// loopcontrols), `tojson` = transformers' `json.dumps(x, ensure_ascii=False)`):
    ///   env.from_string(json.load(open("<30B dir>/tokenizer_config.json"))["chat_template"])
    ///      .render(messages=req["messages"], tools=req["tools"], add_generation_prompt=True)
    /// The request's keys are sorted at every level, so key order is the same in both renderers.
    const HERMES_EXPECTED: &str = include_str!("tools/hermes_expected.txt");

    /// The full prompt for the fixture request, through the same fold as `server::prepare` and
    /// the engine's own ChatML renderer for the 2507 template.
    fn render_fixture(format: ToolFormat) -> String {
        let req: Value = serde_json::from_str(HERMES_REQUEST).unwrap();
        let msgs = req["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["role"].as_str().unwrap(),
                    m["content"].as_str().unwrap_or("").to_string(),
                    m.get("tool_calls")
                        .and_then(Value::as_array)
                        .map(Vec::as_slice),
                )
            })
            .collect();
        let turns = fold_turns(format, req["tools"].as_array().unwrap(), msgs);
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let template =
            aien_inference_abi::tokenizer::ChatTemplate::detect(QWEN3_2507_TEMPLATE).unwrap();
        template.try_render(&refs).unwrap()
    }

    #[test]
    fn hermes_render_matches_the_2507_template_byte_for_byte() {
        assert_eq!(render_fixture(ToolFormat::HermesJson), HERMES_EXPECTED);
    }

    /// Negative control: the qwen3_coder renderer does not produce the Hermes prompt.
    #[test]
    fn xml_render_does_not_match_the_hermes_expected() {
        let xml = render_fixture(ToolFormat::Qwen3CoderXml);
        assert_ne!(xml, HERMES_EXPECTED);
        assert!(xml.contains("<function=read_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>"));
    }

    #[test]
    fn detects_format_from_template_text() {
        assert_eq!(
            ToolFormat::detect(QWEN3_2507_TEMPLATE),
            ToolFormat::HermesJson
        );
        assert_eq!(
            ToolFormat::detect(QWEN36_TEMPLATE),
            ToolFormat::Qwen3CoderXml
        );
        assert_eq!(
            ToolFormat::detect(QWEN38_TEMPLATE),
            ToolFormat::Qwen3CoderXml
        );
    }

    #[test]
    fn detects_format_from_model_dir() {
        let dir = std::env::temp_dir().join(format!("aien-tools-fmt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            ToolFormat::from_model_dir(&dir).unwrap(),
            ToolFormat::Qwen3CoderXml
        );
        std::fs::write(dir.join("chat_template.jinja"), QWEN3_2507_TEMPLATE).unwrap();
        assert_eq!(
            ToolFormat::from_model_dir(&dir).unwrap(),
            ToolFormat::HermesJson
        );
        // tokenizer_config.json's template wins, as in ChatTokenizer::from_model_dir.
        let cfg = serde_json::json!({"chat_template": QWEN36_TEMPLATE});
        std::fs::write(dir.join("tokenizer_config.json"), cfg.to_string()).unwrap();
        assert_eq!(
            ToolFormat::from_model_dir(&dir).unwrap(),
            ToolFormat::Qwen3CoderXml
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hermes_reply_parses_back() {
        let s = hermes_assistant_with_calls(
            "",
            &[serde_json::json!({"function": {"name": "f", "arguments": {"x": 1}}})],
        );
        assert_eq!(
            s,
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>"
        );
        let (c, calls) = parse_reply(&s, "h");
        assert_eq!(c, "");
        assert_eq!(calls[0]["function"]["name"], "f");
        assert_eq!(calls[0]["function"]["arguments"], "{\"x\":1}");
    }
}
