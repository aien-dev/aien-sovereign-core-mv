//! Tool calling for the Qwen3.6 (qwen3_coder XML) chat format, rendered exactly as the model's own
//! chat template does it (model dir chat_template.jinja: the `tools` system block, assistant
//! `tool_calls`, and `tool` messages folded into one user turn of `<tool_response>` blocks), and the
//! reply parsed back into OpenAI `tool_calls`. Drake 2026-10-21: "make it support tool use ... build it in".

use serde_json::{Map, Value};

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

/// The system turn's content when tools are present: the tools block, then the request's own system
/// text after a blank line (the template trims it and drops it when empty).
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

/// An assistant turn's content with its `tool_calls` appended in the template's layout.
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

/// A reply split into plain text and parsed calls. `dropped` holds one reason per `<tool_call>`
/// block that could not be parsed; the raw text of such a block stays in `content` so nothing is lost.
#[derive(Debug, Default)]
pub struct Parsed {
    pub content: String,
    pub calls: Vec<Value>,
    pub dropped: Vec<String>,
}

/// Splits a reply into its text before the first `<tool_call>` and the parsed calls
/// (OpenAI shape: `{id, type: "function", function: {name, arguments: <json string>}}`).
/// `tools` are the request's tool definitions: parameter types come from their schemas (see `coerce`).
///
/// Closing tags inside a value are data, not structure (a `write_file` of an XML document, a reply
/// that quotes another call). The rules, applied on the structure of the reply:
/// - a parameter ends at the first `</parameter>` that is followed (after whitespace) by another
///   `<parameter=` or by the end of the function body, so a `</parameter>` followed by ordinary text
///   stays in the value;
/// - a function ends at a `</function>` followed (after whitespace) by `</tool_call>` or the end of the
///   reply, whose body is a clean run of parameters; candidates are tried in order, and the first one
///   whose remaining text (up to the next `<tool_call>`) holds no structural tag wins, because the tail
///   of an enclosing value would. With no such candidate the first one that parses is used.
///
/// A quoted call laid out exactly like a real one (closing tags on their own lines, in structural order,
/// inside a value) is the one case the format cannot tell apart; the split by these rules applies.
pub fn parse_reply(text: &str, id_base: &str, tools: &[Value]) -> Parsed {
    let Some(first) = text.find(OPEN) else {
        return Parsed {
            content: text.to_string(),
            ..Parsed::default()
        };
    };
    let mut out = Parsed {
        content: text[..first].trim_end().to_string(),
        ..Parsed::default()
    };
    let mut rest = &text[first..];
    while let Some(p) = rest.find(OPEN) {
        let after_open = &rest[p + OPEN.len()..];
        match parse_call(after_open, tools) {
            Ok((name, args, used)) => {
                out.calls.push(serde_json::json!({
                    "id": format!("call_{id_base}_{}", out.calls.len()),
                    "type": "function",
                    "function": {"name": name, "arguments": Value::Object(args).to_string()},
                }));
                rest = &after_open[used..];
            }
            Err(reason) => {
                // Keep the raw block: up to and including the next `</tool_call>`, else the rest.
                let end = after_open
                    .find(CLOSE)
                    .map_or(after_open.len(), |e| e + CLOSE.len());
                if !out.content.is_empty() {
                    out.content.push_str("\n\n");
                }
                out.content.push_str(&rest[p..p + OPEN.len() + end]);
                out.dropped.push(reason);
                rest = &after_open[end..];
            }
        }
    }
    out
}

/// Parses one call from the text after `<tool_call>`. Ok = (name, arguments, bytes consumed incl. the
/// closing `</tool_call>`); Err = why the block was dropped.
fn parse_call(s: &str, tools: &[Value]) -> Result<(String, Map<String, Value>, usize), String> {
    let lead = s.len() - s.trim_start().len();
    let b = &s[lead..];
    // XML form: <function=NAME> <parameter=K>\nV\n</parameter> ... </function>
    if let Some(after) = b.strip_prefix("<function=") {
        let ne = after.find('>').ok_or("unterminated <function=NAME> tag")?;
        let name = after[..ne].trim().to_string();
        if name.is_empty() || name.contains(['<', '\n']) {
            return Err("empty or malformed function name".into());
        }
        let body_start = lead + "<function=".len() + ne + 1;
        // Candidates: a `</function>` followed by `</tool_call>` (or the end of the reply) whose body is
        // a clean run of parameters. A candidate inside a value is told apart by what follows it: up to
        // the next `<tool_call>` the rest must hold no structural tag (the tail of the enclosing value
        // would). The first such candidate wins; if none is that clean, the first parsing one is used.
        let mut ends = 0;
        let mut lenient = None;
        for (e, _) in s[body_start..].match_indices("</function>") {
            ends += 1;
            let after_fn = body_start + e + "</function>".len();
            let tail = s[after_fn..].trim_start();
            let used = if let Some(t) = tail.strip_prefix(CLOSE) {
                s.len() - t.len()
            } else if tail.is_empty() {
                s.len()
            } else {
                continue;
            };
            let Some(params) = parse_params(&s[body_start..body_start + e]) else {
                continue;
            };
            let rest = &s[used..];
            let rest = &rest[..rest.find(OPEN).unwrap_or(rest.len())];
            if !has_structural_tag(rest) {
                return Ok(build_call(name, params, tools, used));
            }
            lenient.get_or_insert((params, used));
        }
        if let Some((params, used)) = lenient {
            return Ok(build_call(name, params, tools, used));
        }
        return Err(if ends == 0 {
            format!("function {name}: no closing </function>")
        } else {
            format!("function {name}: parameters are not well formed before any </function></tool_call>")
        });
    }
    // JSON form: {"name": ..., "arguments": {...}}; the body may itself contain "</tool_call>".
    if b.starts_with('{') {
        for (e, _) in s.match_indices(CLOSE) {
            if let Ok(v) = serde_json::from_str::<Value>(s[..e].trim()) {
                let name = v
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("JSON call without a name")?
                    .to_string();
                let args = match v.get("arguments").or_else(|| v.get("parameters")) {
                    Some(Value::Object(m)) => m.clone(),
                    Some(Value::String(t)) => match serde_json::from_str::<Value>(t) {
                        Ok(Value::Object(m)) => m,
                        _ => Map::new(),
                    },
                    _ => Map::new(),
                };
                return Ok((name, args, e + CLOSE.len()));
            }
        }
        return Err("JSON call body does not parse".into());
    }
    Err("unknown call shape (neither <function=NAME> nor a JSON object)".into())
}

fn has_structural_tag(s: &str) -> bool {
    ["</parameter>", "<parameter=", "</function>", CLOSE]
        .iter()
        .any(|t| s.contains(t))
}

fn build_call(
    name: String,
    params: Vec<(String, &str)>,
    tools: &[Value],
    used: usize,
) -> (String, Map<String, Value>, usize) {
    let mut args = Map::new();
    for (k, raw) in params {
        let v = coerce(raw, param_types(tools, &name, &k).as_deref());
        args.insert(k, v);
    }
    (name, args, used)
}

/// A function body as `(name, raw value)` pairs, or None when it is not a clean run of parameters.
fn parse_params(body: &str) -> Option<Vec<(String, &str)>> {
    let mut out = Vec::new();
    let mut r = body.trim_start();
    while !r.is_empty() {
        let a = r.strip_prefix("<parameter=")?;
        let ke = a.find('>')?;
        let key = a[..ke].trim();
        if key.is_empty() || key.contains(['<', '\n']) {
            return None;
        }
        let v_all = &a[ke + 1..];
        let v_all = v_all.strip_prefix('\n').unwrap_or(v_all);
        let mut found = None;
        for (e, _) in v_all.match_indices("</parameter>") {
            let tail = v_all[e + "</parameter>".len()..].trim_start();
            if tail.is_empty() || tail.starts_with("<parameter=") {
                found = Some((e, tail));
                break;
            }
        }
        let (e, tail) = found?;
        let raw = &v_all[..e];
        out.push((key.to_string(), raw.strip_suffix('\n').unwrap_or(raw)));
        r = tail;
    }
    Some(out)
}

/// The declared JSON-schema type names of parameter `key` of tool `name`, if the request carried them.
fn param_types(tools: &[Value], name: &str, key: &str) -> Option<Vec<String>> {
    let t = tools.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name").and_then(Value::as_str) == Some(name)).then_some(f)
    })?;
    match t
        .get("parameters")?
        .get("properties")?
        .get(key)?
        .get("type")?
    {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => Some(
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
        ),
        _ => None,
    }
}

/// A parameter value, typed the way llama.cpp's qwen3-coder parser does it (common/parsers/qwen3-coder.cpp:
/// the tool schema decides). Followed here:
/// - declared `string` only: the text, untouched ("007" and "true" stay strings);
/// - declared types without `string`: the JSON value when it parses and matches one of them, else the text;
/// - declared `string` plus other types: the JSON value if it parses as one of the other types, else the text;
/// - no schema for this parameter (llama.cpp only accepts declared parameters, so it has no rule here):
///   an object or array when the text parses as one, else the text. Numbers, booleans and null stay strings.
fn coerce(raw: &str, types: Option<&[String]>) -> Value {
    let t = raw.trim();
    let parsed = || serde_json::from_str::<Value>(t).ok();
    let v = match types {
        None => parsed().filter(|v| v.is_object() || v.is_array()),
        Some(ty) if ty.len() == 1 && ty[0] == "string" => None,
        Some(ty) => parsed().filter(|v| {
            ty.iter().any(|n| match n.as_str() {
                "number" | "integer" => v.is_number(),
                "boolean" => v.is_boolean(),
                "object" => v.is_object(),
                "array" => v.is_array(),
                "null" => v.is_null(),
                _ => false,
            })
        }),
    };
    v.unwrap_or_else(|| Value::String(raw.to_string()))
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
    fn p(r: &str, id: &str) -> (String, Vec<Value>) {
        let o = parse_reply(r, id, &[]);
        (o.content, o.calls)
    }

    fn schema(name: &str, props: Value) -> Vec<Value> {
        vec![
            serde_json::json!({"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props}}}),
        ]
    }

    fn args_of(c: &Value) -> Value {
        serde_json::from_str(c["function"]["arguments"].as_str().unwrap()).unwrap()
    }

    fn call(name: &str, params: &[(&str, &str)]) -> String {
        let mut s = format!("<tool_call>\n<function={name}>\n");
        for (k, v) in params {
            s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        s.push_str("</function>\n</tool_call>");
        s
    }

    #[test]
    fn value_with_closing_tags_survives() {
        let doc = "<html>\n<parameter=x>y</parameter>\n</function>\n</tool_call>\n</html>";
        let r = call("write_file", &[("content", doc), ("path", "a.html")]);
        let (_, calls) = p(&r, "x");
        assert_eq!(calls.len(), 1);
        let a = args_of(&calls[0]);
        assert_eq!(a["content"], doc);
        assert_eq!(a["path"], "a.html");
    }

    #[test]
    fn value_quoting_a_whole_call_survives() {
        let quoted = call("inner", &[("a", "1")]);
        let r = format!(
            "{}\n{}",
            call("outer", &[("note", &format!("see {quoted} above"))]),
            call("second", &[("b", "2")])
        );
        let o = parse_reply(&r, "x", &[]);
        assert!(o.dropped.is_empty(), "{:?}", o.dropped);
        assert_eq!(o.calls.len(), 2);
        assert_eq!(args_of(&o.calls[0])["note"], format!("see {quoted} above"));
        assert_eq!(o.calls[1]["function"]["name"], "second");
        assert_eq!(args_of(&o.calls[1])["b"], "2");
    }

    #[test]
    fn closing_parameter_tag_inside_text_value_survives() {
        let r = call("f", &[("a", "x</parameter> not the end"), ("b", "y")]);
        let (_, calls) = p(&r, "x");
        let a = args_of(&calls[0]);
        assert_eq!(a["a"], "x</parameter> not the end");
        assert_eq!(a["b"], "y");
    }

    #[test]
    fn string_params_stay_strings_by_schema() {
        let tools = schema(
            "f",
            serde_json::json!({"path": {"type": "string"}, "flag": {"type": "string"}, "text": {"type": "string"}}),
        );
        let r = call(
            "f",
            &[("path", "007"), ("flag", "true"), ("text", "{\"a\": 1}")],
        );
        let a = args_of(&parse_reply(&r, "x", &tools).calls[0]);
        assert_eq!(a["path"], "007");
        assert_eq!(a["flag"], "true");
        assert_eq!(a["text"], "{\"a\": 1}");
    }

    #[test]
    fn typed_params_are_coerced_by_schema() {
        let tools = schema(
            "f",
            serde_json::json!({
                "n": {"type": "integer"}, "x": {"type": "number"}, "b": {"type": "boolean"},
                "o": {"type": "object"}, "l": {"type": "array"}, "m": {"type": ["string", "number"]},
                "bad": {"type": "integer"}, "z": {"type": ["string", "null"]}
            }),
        );
        let r = call(
            "f",
            &[
                ("n", "20"),
                ("x", "1.5"),
                ("b", "false"),
                ("o", "{\"k\": [1]}"),
                ("l", "[1, 2]"),
                ("m", "7"),
                ("bad", "twenty"),
                ("z", "null"),
            ],
        );
        let a = args_of(&parse_reply(&r, "x", &tools).calls[0]);
        assert_eq!(a["n"], 20);
        assert_eq!(a["x"], 1.5);
        assert_eq!(a["b"], false);
        assert_eq!(a["o"], serde_json::json!({"k": [1]}));
        assert_eq!(a["l"], serde_json::json!([1, 2]));
        assert_eq!(a["m"], 7);
        assert_eq!(a["bad"], "twenty");
        assert_eq!(a["z"], Value::Null);
    }

    #[test]
    fn without_schema_only_objects_and_arrays_are_coerced() {
        let r = call(
            "f",
            &[
                ("p", "007"),
                ("t", "true"),
                ("n", "20"),
                ("o", "{\"a\": 1}"),
                ("l", "[1]"),
                ("s", "{not json"),
            ],
        );
        let a = args_of(&p(&r, "x").1[0]);
        assert_eq!(a["p"], "007");
        assert_eq!(a["t"], "true");
        assert_eq!(a["n"], "20");
        assert_eq!(a["o"], serde_json::json!({"a": 1}));
        assert_eq!(a["l"], serde_json::json!([1]));
        assert_eq!(a["s"], "{not json");
    }

    #[test]
    fn legacy_function_definitions_also_give_types() {
        let tools = vec![
            serde_json::json!({"name": "f", "parameters": {"properties": {"n": {"type": "integer"}}}}),
        ];
        let a = args_of(&parse_reply(&call("f", &[("n", "3")]), "x", &tools).calls[0]);
        assert_eq!(a["n"], 3);
    }

    #[test]
    fn malformed_call_text_is_kept_and_reported() {
        let r = "ok\n<tool_call>\n<function=f>\n<parameter=p>\n1\n</parameter>\n";
        let o = parse_reply(r, "x", &[]);
        assert!(o.calls.is_empty());
        assert_eq!(o.dropped.len(), 1);
        assert!(
            o.dropped[0].contains("no closing </function>"),
            "{:?}",
            o.dropped
        );
        assert!(
            o.content.starts_with("ok\n\n<tool_call>"),
            "content: {}",
            o.content
        );
        assert!(o.content.contains("<function=f>"), "content: {}", o.content);
    }

    #[test]
    fn unknown_shape_and_good_call_after_a_bad_one() {
        let r = format!(
            "<tool_call>\nhello there\n</tool_call>\n{}",
            call("g", &[("a", "1")])
        );
        let o = parse_reply(&r, "x", &[]);
        assert_eq!(o.calls.len(), 1);
        assert_eq!(o.calls[0]["function"]["name"], "g");
        assert_eq!(o.dropped.len(), 1);
        assert!(o.dropped[0].contains("unknown call shape"));
        assert!(o.content.contains("hello there"));
        assert!(!o.content.contains("<function=g>"));
    }

    #[test]
    fn unclosed_parameter_is_dropped_not_guessed() {
        let r = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>\n</tool_call>";
        let o = parse_reply(r, "x", &[]);
        assert!(o.calls.is_empty());
        assert!(o.dropped[0].contains("not well formed"), "{:?}", o.dropped);
        assert_eq!(o.content, r);
    }

    #[test]
    fn call_closed_by_end_of_reply_is_accepted() {
        let r = "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>";
        let o = parse_reply(r, "x", &[]);
        assert_eq!(o.calls.len(), 1);
        assert!(o.dropped.is_empty());
    }

    #[test]
    fn json_form_with_closing_tag_inside_a_string() {
        let o = parse_reply("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": \"a</tool_call>b\"}}\n</tool_call>", "x", &[]);
        assert_eq!(args_of(&o.calls[0])["s"], "a</tool_call>b");
    }

    #[test]
    fn parses_xml_call_with_text_before() {
        let tools = schema(
            "read_file",
            serde_json::json!({"limit": {"type": "integer"}}),
        );
        let r = "Let me check.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";
        let o = parse_reply(r, "x", &tools);
        let (c, calls) = (o.content, o.calls);
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
        let (c, calls) = p(r, "x");
        assert_eq!(c, "");
        assert_eq!(calls.len(), 2);
        let a: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["s"], "line1\nline2");
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn parses_json_form() {
        let (_, calls) = p(
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>",
            "x",
        );
        assert_eq!(calls[0]["function"]["name"], "f");
    }

    #[test]
    fn plain_reply_has_no_calls() {
        let (c, calls) = p("Hello there.", "x");
        assert_eq!(c, "Hello there.");
        assert!(calls.is_empty());
    }

    #[test]
    fn renders_round_trip_layout() {
        let call = serde_json::json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"p\": \"v\", \"n\": 2}"}});
        let s = assistant_with_calls("", &[call]);
        // serde_json maps are key-sorted here (no preserve_order): n before p.
        assert_eq!(s, "<tool_call>\n<function=f>\n<parameter=n>\n2\n</parameter>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>");
        let tools = schema("f", serde_json::json!({"n": {"type": "integer"}}));
        let calls = parse_reply(&s, "y", &tools).calls;
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
}
