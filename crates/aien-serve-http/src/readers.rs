//! Eight readers (sc L371, Drake 2026-10-21: "if it's reading, it spawns eight agents at once and
//! reads eight different chunks at once so it can be summarized, followed up on, and connected").
//!
//! A request to the readers model id whose long input (the last user message and the tool results
//! after it) passes a threshold is answered in two phases. Map: the input is split into `n` parts
//! on paragraph and line boundaries, equal in tokens with a small overlap, and `n` sub-requests
//! (system prompt + the user's request + part i, thinking off, greedy) each write notes on their
//! part; a batching engine seats them in the same beat ([`crate::engine::Engine::generate_group`]).
//! Connect: the original request, with every long segment replaced by the labelled notes, writes
//! the answer. Each reader sees only its part, so this is a different reading method from the
//! plain id, which stays exact; short inputs take the plain path unchanged.
//!
//! Everything here is pure (JSON in, JSON out, a token counter passed in): the server renders the
//! sub-requests through its usual `prepare`, so tool calling and every request check apply to them.
use serde_json::{json, Value};

/// The readers' settings (`AIEN_READERS_*` in `main.rs`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReadersConfig {
    /// The second id in `/v1/models`.
    pub model_id: String,
    /// Parts (and readers) per long input.
    pub readers: usize,
    /// Long-input tokens from which the readers take over.
    pub threshold_tokens: usize,
    /// Cap on each reader's notes.
    pub note_tokens: usize,
    /// Tokens each part repeats from the end of the part before it.
    pub overlap_tokens: usize,
}

impl ReadersConfig {
    pub fn new(base_model_id: &str) -> Self {
        Self {
            model_id: format!("{base_model_id}-readers"),
            readers: 8,
            threshold_tokens: 3000,
            note_tokens: 400,
            overlap_tokens: 64,
        }
    }

    /// The readers from the environment: `AIEN_READERS` (`1` on, `0` off; default `default_on`,
    /// which `main` sets for a batching engine), `AIEN_READERS_N` (parts, default 8),
    /// `AIEN_READERS_THRESHOLD` (tokens, default 3000), `AIEN_READERS_NOTE_TOKENS` (default 400),
    /// `AIEN_READERS_OVERLAP` (tokens, default 64).
    pub fn from_env(base_model_id: &str, default_on: bool) -> Result<Option<Self>, String> {
        let on = match std::env::var("AIEN_READERS").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            Ok(v) => return Err(format!("AIEN_READERS={v}: 0 or 1")),
            Err(_) => default_on,
        };
        if !on {
            return Ok(None);
        }
        let mut c = Self::new(base_model_id);
        for (var, field, min) in [
            ("AIEN_READERS_N", &mut c.readers, 1),
            ("AIEN_READERS_THRESHOLD", &mut c.threshold_tokens, 1),
            ("AIEN_READERS_NOTE_TOKENS", &mut c.note_tokens, 16),
            ("AIEN_READERS_OVERLAP", &mut c.overlap_tokens, 0),
        ] {
            if let Ok(v) = std::env::var(var) {
                *field = v
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n >= min)
                    .ok_or_else(|| format!("{var}={v}: a whole number >= {min}"))?;
            }
        }
        Ok(Some(c))
    }
}

/// One part of the long input.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    /// The text the reader gets (the overlap included).
    pub text: String,
    /// Tokens of `text`.
    pub tokens: usize,
    /// Lines of the long input this part covers, 1-based, inclusive, the overlap excluded.
    pub first_line: usize,
    pub last_line: usize,
}

/// What the readers do with one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// The user's request as every reader and the connect phase see it.
    pub question: String,
    /// `question` is the beginning and end of a long user message, not the whole of it.
    pub question_is_excerpt: bool,
    /// Messages (indices into `messages`) whose text the parts were cut from.
    pub long_msgs: Vec<usize>,
    pub total_tokens: usize,
    pub total_lines: usize,
    pub parts: Vec<Part>,
}

/// The text of a message's `content`: a string, or the text parts of an array.
pub fn content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if p.get("type").and_then(Value::as_str) == Some("text") {
                    out.push_str(p.get("text").and_then(Value::as_str)?);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn role(m: &Value) -> String {
    m.get("role")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// A segment counts as long input from this many tokens (an eighth of the threshold, at least 200):
/// a short question next to a long tool result stays the question.
fn long_segment_tokens(cfg: &ReadersConfig) -> usize {
    (cfg.threshold_tokens / 8).max(200)
}

/// The readers' plan for `messages`, or `None` when the long input is under the threshold (the
/// request takes the plain path).
pub fn plan(
    messages: &[Value],
    cfg: &ReadersConfig,
    count: &dyn Fn(&str) -> usize,
) -> Option<Plan> {
    let n = cfg.readers.max(1);
    let u = messages.iter().rposition(|m| role(m) == "user")?;
    let mut segs: Vec<(usize, String, usize)> = Vec::new();
    for (i, m) in messages.iter().enumerate().skip(u) {
        let r = role(m);
        if i == u || r == "tool" || r == "function" {
            let text = content_text(m.get("content").unwrap_or(&Value::Null))?;
            let t = count(&text);
            segs.push((i, text, t));
        }
    }
    let min_seg = long_segment_tokens(cfg);
    let long: Vec<&(usize, String, usize)> = segs.iter().filter(|s| s.2 >= min_seg).collect();
    let total: usize = long.iter().map(|s| s.2).sum();
    if long.is_empty() || total < cfg.threshold_tokens {
        return None;
    }
    let user_text = &segs[0].1;
    let user_long = segs[0].2 >= min_seg;
    let (question, question_is_excerpt) = if user_long {
        (excerpt(user_text, 600), true)
    } else {
        (user_text.trim().to_string(), false)
    };
    // The long input: its segments in order, each tool result under a label when there are several.
    let several = long.len() > 1;
    let mut doc = String::new();
    let mut tool_no = 0;
    for s in &segs {
        let is_tool = s.0 != u;
        if is_tool {
            tool_no += 1;
        }
        if s.2 < min_seg {
            continue;
        }
        if several {
            if !doc.is_empty() && !doc.ends_with('\n') {
                doc.push('\n');
            }
            if is_tool {
                doc.push_str(&format!("[tool result {tool_no}]\n"));
            } else {
                doc.push_str("[user message]\n");
            }
        }
        doc.push_str(&s.1);
    }
    let pieces = pieces(&doc, total.div_ceil(n * 4).max(16), count);
    let total_lines = pieces.last().map_or(0, |p| p.line);
    let parts = split(&pieces, n, cfg.overlap_tokens);
    Some(Plan {
        question,
        question_is_excerpt,
        long_msgs: long.iter().map(|s| s.0).collect(),
        total_tokens: pieces.iter().map(|p| p.tokens).sum(),
        total_lines,
        parts,
    })
}

/// The first and last `chars` characters of a long text, cut at line ends when one is near.
fn excerpt(text: &str, chars: usize) -> String {
    let t = text.trim();
    let len = t.chars().count();
    if len <= chars * 2 + 40 {
        return t.to_string();
    }
    let head_end = t.char_indices().nth(chars).map_or(t.len(), |(b, _)| b);
    let head_end = t[..head_end]
        .rfind('\n')
        .filter(|&b| b > head_end / 2)
        .unwrap_or(head_end);
    let tail_start = t.char_indices().nth(len - chars).map_or(0, |(b, _)| b);
    let tail_start = t[tail_start..]
        .find('\n')
        .map(|b| tail_start + b + 1)
        .filter(|&b| b < t.len() - (t.len() - tail_start) / 2)
        .unwrap_or(tail_start);
    format!(
        "{}\n[...]\n{}",
        t[..head_end].trim_end(),
        t[tail_start..].trim_start()
    )
}

/// A cut-able piece of the long input: a line (with its newline), or part of a very long line.
#[derive(Debug, Clone, PartialEq)]
struct Piece {
    text: String,
    tokens: usize,
    /// 1-based line number.
    line: usize,
    /// The line is empty (a paragraph boundary follows it).
    blank: bool,
}

/// The input as lines; a line over `max_tokens` is cut at spaces (or anywhere, when it has none)
/// into pieces of about `max_tokens`.
fn pieces(doc: &str, max_tokens: usize, count: &dyn Fn(&str) -> usize) -> Vec<Piece> {
    let mut out = Vec::new();
    for (k, line) in doc.split_inclusive('\n').enumerate() {
        let tokens = count(line);
        let blank = line.trim().is_empty();
        if tokens <= max_tokens {
            out.push(Piece {
                text: line.to_string(),
                tokens,
                line: k + 1,
                blank,
            });
            continue;
        }
        let cuts = tokens.div_ceil(max_tokens);
        let chars: Vec<(usize, char)> = line.char_indices().collect();
        let mut start = 0usize;
        for c in 1..=cuts {
            let end = if c == cuts {
                line.len()
            } else {
                let goal = chars[(chars.len() * c / cuts).min(chars.len() - 1)].0;
                // The nearest space before the goal, within a tenth of a piece.
                let floor = goal.saturating_sub(line.len() / cuts / 10).max(start + 1);
                line[..goal]
                    .rfind(char::is_whitespace)
                    .filter(|&b| b >= floor)
                    .map(|b| b + 1)
                    .unwrap_or(goal)
            };
            if end <= start {
                continue;
            }
            let text = &line[start..end];
            out.push(Piece {
                text: text.to_string(),
                tokens: count(text),
                line: k + 1,
                blank: false,
            });
            start = end;
        }
    }
    out
}

/// Cuts the pieces into `n` parts of about equal tokens, preferring paragraph boundaries within a
/// tenth of a part of each target; each part after the first repeats up to `overlap` tokens of
/// the one before.
fn split(pieces: &[Piece], n: usize, overlap: usize) -> Vec<Part> {
    let total: usize = pieces.iter().map(|p| p.tokens).sum();
    let mut cum = Vec::with_capacity(pieces.len());
    let mut s = 0usize;
    for p in pieces {
        s += p.tokens;
        cum.push(s);
    }
    // cuts[k] = index of the first piece of part k.
    let mut cuts = vec![0usize];
    for k in 1..n {
        let target = total * k / n;
        let slack = (total / n / 10).max(1);
        let lo = *cuts.last().expect("starts with 0") + 1;
        if lo >= pieces.len() {
            break;
        }
        // Cut before piece j (j in lo..len): tokens before it = cum[j - 1].
        let dist = |j: usize| cum[j - 1].abs_diff(target);
        let best_any = (lo..pieces.len())
            .min_by_key(|&j| dist(j))
            .expect("non-empty");
        let best_par = (lo..pieces.len())
            .filter(|&j| pieces[j - 1].blank && dist(j) <= slack)
            .min_by_key(|&j| dist(j));
        cuts.push(best_par.unwrap_or(best_any));
    }
    cuts.push(pieces.len());
    let mut parts = Vec::with_capacity(n);
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= b {
            continue;
        }
        let mut from = a;
        let mut extra = 0usize;
        while from > 0 && extra + pieces[from - 1].tokens <= overlap {
            from -= 1;
            extra += pieces[from].tokens;
        }
        let text: String = pieces[from..b].iter().map(|p| p.text.as_str()).collect();
        parts.push(Part {
            text,
            tokens: pieces[from..b].iter().map(|p| p.tokens).sum(),
            first_line: pieces[a].line,
            last_line: pieces[b - 1].line,
        });
    }
    parts
}

fn lines_label(p: &Part) -> String {
    if p.first_line == p.last_line {
        format!("line {}", p.first_line)
    } else {
        format!("lines {}-{}", p.first_line, p.last_line)
    }
}

/// The system prompt of the request (its system and developer messages), if any.
fn system_text(messages: &[Value]) -> Option<String> {
    let s: Vec<String> = messages
        .iter()
        .filter(|m| matches!(role(m).as_str(), "system" | "developer"))
        .filter_map(|m| content_text(m.get("content").unwrap_or(&Value::Null)))
        .filter(|t| !t.trim().is_empty())
        .collect();
    (!s.is_empty()).then(|| s.join("\n\n"))
}

/// Reader `i`'s sub-request: the system prompt, the user's request and part `i`; notes only.
pub fn map_body(
    messages: &[Value],
    plan: &Plan,
    i: usize,
    base_model_id: &str,
    cfg: &ReadersConfig,
) -> Value {
    let n = plan.parts.len();
    let p = &plan.parts[i];
    let ask = if plan.question_is_excerpt {
        "The user's message (it holds the input below; its beginning and end are shown)"
    } else {
        "The user's request"
    };
    let user = format!(
        "You are reader {k} of {n}. A long input ({total} tokens, {lines} lines) was split into {n} parts; you read part {k} only. The other readers have the other parts, and a final writer will connect every reader's notes to answer the user.\n\n\
         {ask}:\n<<<\n{q}\n>>>\n\n\
         Part {k} of {n} ({range}):\n<<<\n{text}\n>>>\n\n\
         Write notes on this part only, under these headings:\n\
         Summary: 2 to 5 sentences.\n\
         Key facts: bullet points with exact quotes, numbers, names and dates as written.\n\
         For the request: what in this part answers or bears on the user's request (\"nothing\" if none).\n\
         Open questions: what this part leaves unclear, or points to elsewhere, worth following up.\n\
         Links: names, terms, ids and references that may connect to other parts.\n\
         Be concise. Write nothing that is not in the part.",
        k = i + 1,
        total = plan.total_tokens,
        lines = plan.total_lines,
        q = plan.question,
        range = lines_label(p),
        text = p.text.trim_end(),
    );
    let mut msgs = Vec::new();
    if let Some(s) = system_text(messages) {
        msgs.push(json!({"role": "system", "content": s}));
    }
    msgs.push(json!({"role": "user", "content": user}));
    json!({
        "model": base_model_id,
        "messages": msgs,
        "max_tokens": cfg.note_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false},
    })
}

/// The labelled notes block the connect phase reads.
pub fn notes_block(plan: &Plan, notes: &[String]) -> String {
    let n = plan.parts.len();
    let mut s = format!(
        "The long input ({} tokens, {} lines) was read in {n} parts by {n} readers. Their notes, in order:\n",
        plan.total_tokens, plan.total_lines
    );
    for (k, (p, note)) in plan.parts.iter().zip(notes).enumerate() {
        s.push_str(&format!(
            "\n## Part {} of {n} ({})\n{}\n",
            k + 1,
            lines_label(p),
            note.trim()
        ));
    }
    s.push_str(
        "\nAnswer the user's request from these notes. Connect facts across the parts, say which part a fact comes from when it matters, and say so when the notes do not cover something.",
    );
    s
}

/// The connect request: the original request with every long segment replaced by a pointer, the
/// last one carrying the notes (and, when it was the user message, the user's request first).
pub fn connect_body(raw: &Value, plan: &Plan, notes: &[String], base_model_id: &str) -> Value {
    let mut body = raw.clone();
    body["model"] = Value::String(base_model_id.into());
    if let Some(obj) = body.as_object_mut() {
        obj.remove("aien_show_notes");
    }
    let last = *plan.long_msgs.last().expect("a plan has long input");
    if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for &i in &plan.long_msgs {
            let is_user = role(&msgs[i]) == "user";
            let mut text = if is_user {
                format!("{}\n\n", plan.question)
            } else {
                String::from("(This tool result was long; it was read in parts with the rest of the long input.)\n\n")
            };
            if i == last {
                text.push_str(&notes_block(plan, notes));
            } else {
                text.truncate(text.trim_end().len());
            }
            msgs[i]["content"] = Value::String(text);
        }
    }
    body
}

/// The collapsed preface `aien_show_notes` puts before the answer.
pub fn notes_preface(plan: &Plan, notes: &[String]) -> String {
    let n = plan.parts.len();
    let mut s = format!(
        "<details>\n<summary>Notes from {n} readers ({} tokens read in {n} parts)</summary>\n",
        plan.total_tokens
    );
    for (k, (p, note)) in plan.parts.iter().zip(notes).enumerate() {
        s.push_str(&format!(
            "\n**Part {} of {n} ({}, {} tokens)**\n\n{}\n",
            k + 1,
            lines_label(p),
            p.tokens,
            note.trim()
        ));
    }
    s.push_str("\n</details>\n\n");
    s
}

/// A reader's notes from its reply text: a stray thinking block dropped, trimmed.
pub fn clean_notes(reply: &str) -> String {
    let t = match reply.find("</think>") {
        Some(e) => &reply[e + "</think>".len()..],
        None => reply,
    };
    t.trim().to_string()
}

#[cfg(test)]
mod tests;
