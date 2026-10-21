//! The server's half of the 35B door's shared prompt cache (sc L375, coordinator 12:10Z).
//!
//! - **Stable-first tool order.** Hermes sends the same ~11k tokens of tool definitions with
//!   every chat. The door restores the longest token prefix it has already read, so the server
//!   renders the tools in a remembered order: tools whose definition is unchanged keep their
//!   remembered place, a changed or new tool goes to the end. One changed tool then costs only
//!   the tools after its old place once, and itself (plus what follows the block) after that.
//!   The order (names and definition digests only) lives in `tool-order.json` in
//!   `AIEN_Q36_PREFIX_DIR` when set, so it survives restarts with the pool it matches.
//! - **Tool boundary marks.** The token positions right after each tool line, handed to the door
//!   ([`crate::engine::Engine::generate_marked`]) as checkpoint hints. A mark is kept only when
//!   the encoded text up to it is exactly a prefix of the prompt's tokens; marks are cached per
//!   rendered head, so a repeated tools block is encoded once.
//!
//! On only for the batch door (`omega-qwen36-batch`) with `AIEN_SERVE_TOOL_ORDER` not `0`.
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::codec::Codec;

/// Rendered tools blocks whose marks are kept.
const MARK_CACHE: usize = 8;

#[derive(Default)]
struct Inner {
    /// Remembered order: (tool name, definition digest).
    order: Vec<(String, String)>,
    /// Marks per digest of the rendered text up to the end of the tools block.
    marks: HashMap<[u8; 32], (u64, Vec<usize>)>,
    clock: u64,
}

pub struct ToolCache {
    on: bool,
    file: Option<PathBuf>,
    inner: Mutex<Inner>,
}

fn tool_name(t: &Value) -> Option<&str> {
    t.get("function")
        .and_then(|f| f.get("name"))
        .or_else(|| t.get("name"))
        .and_then(Value::as_str)
}

fn digest(t: &Value) -> String {
    Sha256::digest(crate::tools::tojson(t).as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl ToolCache {
    /// For an engine of `kind`: on for the batch door unless `AIEN_SERVE_TOOL_ORDER=0`; the
    /// remembered order is read from `AIEN_Q36_PREFIX_DIR/tool-order.json` when there.
    pub fn for_engine(kind: &str) -> Self {
        let on = kind == "omega-qwen36-batch"
            && std::env::var("AIEN_SERVE_TOOL_ORDER").as_deref() != Ok("0");
        let file = std::env::var_os("AIEN_Q36_PREFIX_DIR")
            .filter(|_| on)
            .map(|d| PathBuf::from(d).join("tool-order.json"));
        let order = file
            .as_ref()
            .and_then(|f| std::fs::read(f).ok())
            .and_then(|b| serde_json::from_slice::<Vec<(String, String)>>(&b).ok())
            .unwrap_or_default();
        Self {
            on,
            file,
            inner: Mutex::new(Inner {
                order,
                ..Inner::default()
            }),
        }
    }

    /// Off: tools render in request order and no marks are made.
    pub fn off() -> Self {
        Self {
            on: false,
            file: None,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// The request's tools in the stable-first order (see the module doc); the remembered order
    /// is updated (and saved) when the request changed it. Unchanged when off, or when a tool has
    /// no name or two tools share one.
    pub fn arrange(&self, tools: Vec<Value>) -> Vec<Value> {
        if !self.on || tools.len() < 2 {
            return tools;
        }
        let keys: Option<Vec<(String, String)>> = tools
            .iter()
            .map(|t| tool_name(t).map(|n| (n.to_string(), digest(t))))
            .collect();
        let Some(keys) = keys else { return tools };
        let mut names: Vec<&str> = keys.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return tools;
        }
        let Ok(mut g) = self.inner.lock() else {
            return tools;
        };
        let place = |k: &(String, String)| g.order.iter().position(|o| o == k);
        let mut kept: Vec<(usize, usize)> = Vec::new(); // (remembered place, request index)
        let mut moved: Vec<usize> = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            match place(k) {
                Some(p) => kept.push((p, i)),
                None => moved.push(i),
            }
        }
        kept.sort_unstable();
        let idx: Vec<usize> = kept.iter().map(|&(_, i)| i).chain(moved.iter().copied()).collect();
        if !moved.is_empty() {
            let changed: Vec<&str> = moved.iter().map(|&i| keys[i].0.as_str()).collect();
            g.order.retain(|(n, _)| !changed.contains(&n.as_str()));
            g.order.extend(moved.iter().map(|&i| keys[i].clone()));
            if let Some(f) = &self.file {
                if let Err(e) = save(f, &g.order) {
                    eprintln!(
                        "aien-serve-http: tool order: saving {} failed: {e}",
                        f.display()
                    );
                }
            }
            eprintln!(
                "aien-serve-http: tool order: {} of {} tools new or changed, moved to the end: {}",
                moved.len(),
                tools.len(),
                changed.join(", ")
            );
        }
        drop(g);
        let mut slots: Vec<Option<Value>> = tools.into_iter().map(Some).collect();
        idx.into_iter().filter_map(|i| slots[i].take()).collect()
    }

    /// Token positions of `tokens` (the encoded `text`) right after each tool line of the tools
    /// block (`tools` as rendered, in order); only positions where the encoded text up to the
    /// boundary is exactly a prefix of `tokens`.
    pub fn marks(&self, codec: &dyn Codec, text: &str, tools: &[Value], tokens: &[u32]) -> Vec<usize> {
        if !self.on || tools.is_empty() {
            return Vec::new();
        }
        // Boundaries: just after the "\n" before each tool line and before "</tools>".
        let Some(mut at) = text.find("<tools>").map(|k| k + "<tools>".len()) else {
            return Vec::new();
        };
        let mut cuts = Vec::with_capacity(tools.len() + 1);
        for t in tools {
            let line = format!("\n{}", crate::tools::tojson(t));
            match text[at..].find(&line) {
                Some(k) if k == 0 => {
                    cuts.push(at + 1);
                    at += line.len();
                }
                _ => return Vec::new(),
            }
        }
        if !text[at..].starts_with("\n</tools>") {
            return Vec::new();
        }
        cuts.push(at + 1);
        let key: [u8; 32] = Sha256::digest(text[..at].as_bytes()).into();
        let Ok(mut g) = self.inner.lock() else {
            return Vec::new();
        };
        g.clock += 1;
        let c = g.clock;
        if let Some((used, m)) = g.marks.get_mut(&key) {
            *used = c;
            return m.clone();
        }
        drop(g);
        let marks: Vec<usize> = cuts
            .iter()
            .filter_map(|&b| {
                let t = codec.encode(&text[..b]).ok()?;
                tokens.starts_with(&t).then_some(t.len())
            })
            .collect();
        if let Ok(mut g) = self.inner.lock() {
            if g.marks.len() >= MARK_CACHE {
                if let Some(old) = g.marks.iter().min_by_key(|(_, (u, _))| *u).map(|(k, _)| *k) {
                    g.marks.remove(&old);
                }
            }
            g.marks.insert(key, (c, marks.clone()));
        }
        eprintln!(
            "aien-serve-http: tool marks: {} of {} boundaries exact",
            marks.len(),
            cuts.len()
        );
        marks
    }
}

fn save(f: &std::path::Path, order: &[(String, String)]) -> std::io::Result<()> {
    let tmp = f.with_extension("json.tmp");
    let mut o = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)?;
    o.write_all(&serde_json::to_vec(order).map_err(std::io::Error::other)?)?;
    o.sync_all()?;
    std::fs::rename(tmp, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn on() -> ToolCache {
        ToolCache {
            on: true,
            file: None,
            inner: Mutex::new(Inner::default()),
        }
    }

    fn t(name: &str, d: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "description": d}})
    }

    fn names(v: &[Value]) -> Vec<&str> {
        v.iter().map(|x| tool_name(x).unwrap()).collect()
    }

    #[test]
    fn a_changed_tool_moves_to_the_end_and_the_rest_keep_their_place() {
        let c = on();
        let first = c.arrange(vec![t("a", "1"), t("b", "1"), t("c", "1")]);
        assert_eq!(names(&first), ["a", "b", "c"]);
        // Another client order: the remembered one wins.
        let again = c.arrange(vec![t("c", "1"), t("a", "1"), t("b", "1")]);
        assert_eq!(names(&again), ["a", "b", "c"]);
        // b changes: it goes last, and stays last while unchanged.
        let changed = c.arrange(vec![t("a", "1"), t("b", "2"), t("c", "1")]);
        assert_eq!(names(&changed), ["a", "c", "b"]);
        let next = c.arrange(vec![t("a", "1"), t("b", "2"), t("c", "1"), t("d", "1")]);
        assert_eq!(names(&next), ["a", "c", "b", "d"]);
        // A tool missing from one request keeps its place for the next.
        let without = c.arrange(vec![t("b", "2"), t("a", "1")]);
        assert_eq!(names(&without), ["a", "b"]);
        let back = c.arrange(vec![t("d", "1"), t("c", "1"), t("a", "1"), t("b", "2")]);
        assert_eq!(names(&back), ["a", "c", "b", "d"]);
    }

    #[test]
    fn off_or_duplicate_names_keep_the_request_order() {
        let v = vec![t("b", "1"), t("a", "1")];
        assert_eq!(names(&ToolCache::off().arrange(v.clone())), ["b", "a"]);
        let c = on();
        c.arrange(vec![t("a", "1"), t("b", "1")]);
        let dup = vec![t("b", "1"), t("a", "1"), t("a", "2")];
        assert_eq!(names(&c.arrange(dup)), ["b", "a", "a"]);
    }

    /// A codec whose tokens are the text's characters.
    struct Chars;
    impl Codec for Chars {
        fn render_chat(&self, _: &[(&str, &str)]) -> Result<String, crate::codec::CodecError> {
            Ok(String::new())
        }
        fn stop_token_ids(&self) -> Vec<u32> {
            Vec::new()
        }
        fn encode(&self, text: &str) -> Result<Vec<u32>, crate::codec::CodecError> {
            Ok(text.chars().map(u32::from).collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String, crate::codec::CodecError> {
            Ok(ids.iter().filter_map(|&i| char::from_u32(i)).collect())
        }
    }

    #[test]
    fn marks_sit_right_after_each_tool_line() {
        let tools = vec![t("a", "1"), t("b", "22")];
        let sys = crate::tools::system_with_tools(&tools, Some("be brief"));
        let text = format!("<|im_start|>system\n{sys}<|im_end|>\n");
        let tokens = Chars.encode(&text).unwrap();
        let m = on().marks(&Chars, &text, &tools, &tokens);
        assert_eq!(m.len(), 3);
        let open = text.find("<tools>").unwrap() + "<tools>\n".len();
        assert_eq!(m[0], open);
        assert!(text[..m[1]].ends_with("}\n") && text[m[1]..].starts_with('{'));
        assert!(text[m[2]..].starts_with("</tools>"));
    }
}
