//! Host tests of the readers' split and request building (a word counter stands in for the tokenizer).
use super::*;

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

fn cfg() -> ReadersConfig {
    ReadersConfig::new("m")
}

/// `paras` paragraphs of `lines` lines of `w` words each, every word naming its line.
fn doc(paras: usize, lines: usize, w: usize) -> String {
    let mut s = String::new();
    let mut n = 0;
    for p in 0..paras {
        if p > 0 {
            s.push('\n');
        }
        for _ in 0..lines {
            n += 1;
            let line: Vec<String> = (0..w).map(|k| format!("l{n}w{k}")).collect();
            s.push_str(&line.join(" "));
            s.push('\n');
        }
    }
    s
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

#[test]
fn short_input_takes_the_plain_path() {
    let m = vec![user(&doc(3, 10, 10))]; // 300 words
    assert!(plan(&m, &cfg(), &words).is_none());
}

#[test]
fn long_input_is_eight_equal_parts_covering_every_line_with_overlap() {
    let text = format!("Summarize this design.\n\n{}", doc(40, 6, 20)); // 4800 words + 3
    let m = vec![
        json!({"role": "system", "content": "be brief"}),
        user(&text),
    ];
    let p = plan(&m, &cfg(), &words).expect("over the threshold");
    assert_eq!(p.parts.len(), 8);
    assert!(p.question_is_excerpt);
    assert!(p.question.starts_with("Summarize this design."));
    assert!(p.question.contains("[...]"));
    assert_eq!(p.long_msgs, vec![1]);
    // Lines: each part starts where the one before ended, the last ends at the last line.
    assert_eq!(p.parts[0].first_line, 1);
    for w in p.parts.windows(2) {
        assert_eq!(
            w[1].first_line,
            w[0].last_line + 1,
            "{:?}",
            (w[0].last_line, w[1].first_line)
        );
    }
    assert_eq!(p.parts[7].last_line, p.total_lines);
    // Equal within a sixth of a part (a part is ~600 words, paragraphs are 120).
    let each = p.total_tokens / 8;
    for (k, part) in p.parts.iter().enumerate() {
        let own = part
            .tokens
            .saturating_sub(if k > 0 { cfg().overlap_tokens } else { 0 });
        assert!(
            own.abs_diff(each) <= each / 4,
            "part {k}: {} vs {each}",
            part.tokens
        );
    }
    // Overlap: a part after the first starts with the end of the part before it.
    for w in p.parts.windows(2) {
        let head: String = w[1].text.lines().next().unwrap().to_string();
        assert!(w[0].text.contains(&head), "overlap line {head:?}");
        assert!(w[1].tokens > 0);
    }
}

#[test]
fn cuts_prefer_paragraph_boundaries() {
    let m = vec![user(&doc(16, 10, 30))]; // 16 paragraphs of 300 words: 2 per part
    let p = plan(&m, &cfg(), &words).unwrap();
    assert_eq!(p.parts.len(), 8);
    // Each paragraph is 10 lines + a blank line: a part after the first starts on a paragraph's first line.
    for part in &p.parts[1..] {
        assert_eq!(
            (part.first_line - 1) % 11,
            0,
            "part starts at line {}",
            part.first_line
        );
    }
}

#[test]
fn one_huge_line_is_still_cut_into_eight() {
    let line: Vec<String> = (0..5000).map(|k| format!("w{k}")).collect();
    let m = vec![user(&line.join(" "))];
    let p = plan(&m, &cfg(), &words).unwrap();
    assert_eq!(p.parts.len(), 8);
    let joined: usize = p.parts.iter().map(|x| x.tokens).sum();
    assert!(joined >= 5000);
    assert!(
        p.parts.iter().all(|x| x.tokens >= 400),
        "{:?}",
        p.parts.iter().map(|x| x.tokens).collect::<Vec<_>>()
    );
}

#[test]
fn short_question_and_long_tool_result_keep_the_question_and_the_tools() {
    let raw = json!({
        "model": "m-readers",
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "aien_show_notes": true,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What does the log say about the crash?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": doc(20, 5, 40)},
        ]
    });
    let msgs = raw["messages"].as_array().unwrap();
    let p = plan(msgs, &cfg(), &words).unwrap();
    assert!(!p.question_is_excerpt);
    assert_eq!(p.question, "What does the log say about the crash?");
    assert_eq!(p.long_msgs, vec![3]);
    let notes: Vec<String> = (1..=8).map(|k| format!("note {k}")).collect();
    let c = connect_body(&raw, &p, &notes, "m");
    assert_eq!(c["model"], "m");
    assert!(c.get("aien_show_notes").is_none());
    assert_eq!(c["tools"], raw["tools"]);
    assert_eq!(c["messages"][1], raw["messages"][1]);
    assert_eq!(c["messages"][2], raw["messages"][2]);
    let tool = c["messages"][3]["content"].as_str().unwrap();
    assert_eq!(c["messages"][3]["tool_call_id"], "c1");
    for k in 1..=8 {
        assert!(tool.contains(&format!("## Part {k} of 8")), "part {k}");
        assert!(tool.contains(&format!("note {k}")));
    }
    // The notes are in order.
    assert!(tool.find("note 1").unwrap() < tool.find("note 8").unwrap());
}

#[test]
fn long_user_message_connect_starts_with_the_request() {
    let text = format!("{}\nWhat is the main risk?", doc(30, 5, 30));
    let raw = json!({"model": "m-readers", "stream": true, "messages": [user(&text)]});
    let p = plan(raw["messages"].as_array().unwrap(), &cfg(), &words).unwrap();
    assert!(p.question.ends_with("What is the main risk?"));
    let notes = vec!["n".to_string(); 8];
    let c = connect_body(&raw, &p, &notes, "m");
    let u = c["messages"][0]["content"].as_str().unwrap();
    assert!(u.starts_with(&p.question));
    assert!(u.contains("read in 8 parts by 8 readers"));
    assert_eq!(c["stream"], true);
    assert!(
        words(u) < 600,
        "the connect prompt carries notes, not the input"
    );
}

#[test]
fn map_body_carries_system_question_and_its_part_only() {
    let text = format!("Find the numbers.\n\n{}", doc(20, 6, 30));
    let m = vec![
        json!({"role": "developer", "content": "house rules"}),
        user(&text),
    ];
    let p = plan(&m, &cfg(), &words).unwrap();
    let b = map_body(&m, &p, 2, "m", &cfg());
    assert_eq!(b["model"], "m");
    assert_eq!(b["max_tokens"], 400);
    assert_eq!(b["temperature"], 0.0);
    assert_eq!(b["chat_template_kwargs"]["enable_thinking"], false);
    assert!(b.get("tools").is_none() && b.get("stream").is_none());
    assert_eq!(b["messages"][0]["content"], "house rules");
    let u = b["messages"][1]["content"].as_str().unwrap();
    assert!(u.contains("You are reader 3 of 8"));
    assert!(u.contains("Find the numbers."));
    assert!(u.contains(p.parts[2].text.trim_end()));
    let other = p.parts[5].text.lines().nth(3).unwrap();
    assert!(!u.contains(other), "another part's line leaked in");
}

#[test]
fn notes_are_cleaned_and_the_preface_is_collapsed() {
    assert_eq!(
        clean_notes("<think>\nhm\n</think>\n\n Summary: x \n"),
        "Summary: x"
    );
    assert_eq!(clean_notes(" plain "), "plain");
    let m = vec![user(&doc(30, 5, 30))];
    let p = plan(&m, &cfg(), &words).unwrap();
    let notes: Vec<String> = (1..=8).map(|k| format!("n{k}")).collect();
    let s = notes_preface(&p, &notes);
    assert!(s.starts_with("<details>\n<summary>Notes from 8 readers"));
    assert!(s.ends_with("</details>\n\n"));
    assert!(s.contains("**Part 8 of 8"));
}
