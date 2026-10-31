//! Qwen3.6-35B-A3B chat template (L312: the motor is the 35B server). Pinned text: Hugging Face
//! `Qwen/Qwen3.6-35B-A3B-FP8` revision 95a723d0 (fixtures/chat-templates/qwen3.6-35b-a3b).
//!
//! Checks of the rendered bytes:
//! 1. JINJA ORACLE: `render_oracle.json`, the template rendered by jinja2 3.1.2 (scratch script
//!    outside the repository) for 8 conversations x 6 option sets (thinking undefined, on, off;
//!    preserve_thinking on, off; a reasoning_effort the template ignores), plus the template's own
//!    refusals.
//! 2. GOLDEN PROMPT: with the checkpoint present (`AIEN_QWEN36_DIR`, ignored test), the
//!    thinking-off rendering of the golden question tokenizes to the host reference's
//!    `prompt_ids` (`evidence/q36-host-20261020/gen-chat-max-24.json`), the ids the HTTP gate replays.
use aien_inference_abi::tokenizer::{ChatOptions, ChatTemplate, ChatTokenizer};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3.6-35b-a3b"
);
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../evidence/q36-host-20261020/gen-chat-max-24.json"
);
const QUESTION: &str = "Write a C function that returns the larger of two ints.";

fn t() -> ChatTemplate {
    let text = std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap();
    ChatTemplate::detect(&text).expect("Qwen3.6 template accepted")
}

fn oracle() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/render_oracle.json")).unwrap())
        .unwrap()
}

fn opts(v: &serde_json::Value) -> ChatOptions {
    ChatOptions {
        enable_thinking: v.get("enable_thinking").and_then(|x| x.as_bool()),
        reasoning_effort: v
            .get("reasoning_effort")
            .and_then(|x| x.as_str())
            .map(String::from),
        preserve_thinking: v.get("preserve_thinking").and_then(|x| x.as_bool()),
    }
}

fn turns_of(case: &serde_json::Value) -> Vec<(String, String)> {
    case["turns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into()))
        .collect()
}

#[test]
fn pinned_template_is_selected() {
    assert_eq!(t(), ChatTemplate::Qwen36);
    // The tokenizer_config.json copy of the same template lacks the final newline; it still selects.
    let text = std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap();
    assert_eq!(
        ChatTemplate::detect(text.trim_end()).unwrap(),
        ChatTemplate::Qwen36
    );
}

#[test]
fn rendering_matches_the_jinja_reference_rendering() {
    let oracle = oracle();
    let tpl = t();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for (i, case) in cases.iter().enumerate() {
        let turns = turns_of(case);
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let o = opts(&case["options"]);
        assert_eq!(
            tpl.try_render_with(&refs, &o).unwrap(),
            case["text"].as_str().unwrap(),
            "case {i} options {o:?}"
        );
        // The default is the template's default (thinking on, earlier assistant turns plain).
        if o == ChatOptions::default() {
            assert_eq!(
                tpl.try_render(&refs).unwrap(),
                case["text"].as_str().unwrap()
            );
        }
    }
}

#[test]
fn refusals_match_the_template_exceptions() {
    let oracle = oracle();
    let tpl = t();
    let refusals = oracle["refusals"].as_array().unwrap();
    assert_eq!(refusals.len(), 5);
    for case in refusals {
        let turns = turns_of(case);
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let want = case["error"].as_str().expect("the template raised");
        let err = tpl
            .try_render_with(&refs, &opts(&case["options"]))
            .expect_err(&format!("{}: must be refused", case["name"]));
        assert!(
            err.to_string().contains(want),
            "{}: {err} should carry {want:?}",
            case["name"]
        );
    }
}

#[test]
fn options_change_the_rendering_and_the_ignored_one_does_not() {
    let tpl = t();
    let turns = [("user", "Hi")];
    let on = tpl
        .try_render_with(&turns, &ChatOptions::default())
        .unwrap();
    let off = tpl
        .try_render_with(
            &turns,
            &ChatOptions {
                enable_thinking: Some(false),
                ..ChatOptions::default()
            },
        )
        .unwrap();
    assert_ne!(on, off);
    assert!(on.ends_with("<|im_start|>assistant\n<think>\n"));
    assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // The 35B template has no reasoning-effort switch: any value renders the default text.
    let effort = ChatOptions {
        reasoning_effort: Some("max".into()),
        ..ChatOptions::default()
    };
    assert_eq!(tpl.try_render_with(&turns, &effort).unwrap(), on);
    // preserve_thinking only touches assistant turns before the last user query.
    let multi = [("user", "Q1"), ("assistant", "A1"), ("user", "Q2")];
    let plain = tpl
        .try_render_with(&multi, &ChatOptions::default())
        .unwrap();
    let kept = tpl
        .try_render_with(
            &multi,
            &ChatOptions {
                preserve_thinking: Some(true),
                ..ChatOptions::default()
            },
        )
        .unwrap();
    assert!(plain.contains("<|im_start|>assistant\nA1<|im_end|>\n"));
    assert!(kept.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>\n"));
    assert_eq!(
        tpl.try_render_with(
            &multi,
            &ChatOptions {
                preserve_thinking: Some(false),
                ..ChatOptions::default()
            }
        )
        .unwrap(),
        plain
    );
    assert!(tpl
        .try_render_with(&[("user", "a"), ("tool", "out")], &ChatOptions::default())
        .is_err());
}

#[test]
#[ignore = "needs AIEN_QWEN36_DIR (the checkpoint) and evidence/q36-host-20261020/gen-chat-max-24.json"]
fn golden_prompt_tokenizes_to_the_reference_ids() {
    let dir = std::env::var("AIEN_QWEN36_DIR").expect("AIEN_QWEN36_DIR");
    let g: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(GOLDEN).unwrap()).unwrap();
    let want: Vec<u32> = g["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(want.len(), 24);
    let tk = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).unwrap();
    assert_eq!(tk.template(), ChatTemplate::Qwen36);
    let off = ChatOptions {
        enable_thinking: Some(false),
        ..ChatOptions::default()
    };
    let text = tk
        .try_format_chat_with(&[("user", QUESTION)], &off)
        .unwrap();
    assert_eq!(tk.encode_with_special(&text, true).unwrap(), want);
    // Negative control: the thinking-on rendering of the same question must not match.
    let on = tk
        .try_format_chat_with(&[("user", QUESTION)], &ChatOptions::default())
        .unwrap();
    assert_ne!(tk.encode_with_special(&on, true).unwrap(), want);
    assert_eq!(tk.stop_token_ids(), &[248046, 248044]);
    // The limit comes from config.json (text_config), not the 2048 default: the server's 128k
    // context needs a long prompt to encode.
    let long = "word ".repeat(30_000);
    assert!(tk.encode_with_special(&long, true).unwrap().len() > 2048);
}

/// Qwen3.5-9B (Hugging Face `Qwen/Qwen3.5-9B` revision c2022362,
/// fixtures/chat-templates/qwen3.5-9b): the 35B template minus `preserve_thinking`. Its
/// `render_oracle.json` is the same 48 conversations x options and 5 refusals rendered by jinja2
/// 3.1.2 from the 9B text; it differs from the 35B oracle only in the 3 cases that set
/// `preserve_thinking: true` on a conversation with an assistant turn before the last user query.
const DIR35: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3.5-9b"
);

fn oracle35() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{DIR35}/render_oracle.json")).unwrap())
        .unwrap()
}

#[test]
fn qwen35_pinned_template_is_selected_and_distinct() {
    let text = std::fs::read_to_string(format!("{DIR35}/chat_template.jinja")).unwrap();
    assert_eq!(ChatTemplate::detect(&text).unwrap(), ChatTemplate::Qwen35);
    assert_eq!(
        ChatTemplate::detect(text.trim_end()).unwrap(),
        ChatTemplate::Qwen35
    );
    // One byte off the pinned text is refused, not approximated.
    let edited = text.replacen("last_query_index", "last_query_indeks", 1);
    assert!(ChatTemplate::detect(&edited).is_err());
}

#[test]
fn qwen35_rendering_and_refusals_match_the_jinja_reference() {
    let q36 = oracle();
    let oracle = oracle35();
    let text = std::fs::read_to_string(format!("{DIR35}/chat_template.jinja")).unwrap();
    let tpl = ChatTemplate::detect(&text).unwrap();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    let mut differ_from_35b = 0;
    for (i, case) in cases.iter().enumerate() {
        let turns = turns_of(case);
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let o = opts(&case["options"]);
        let got = tpl.try_render_with(&refs, &o).unwrap();
        assert_eq!(
            got,
            case["text"].as_str().unwrap(),
            "case {i} options {o:?}"
        );
        if got != q36["cases"][i]["text"].as_str().unwrap() {
            assert_eq!(o.preserve_thinking, Some(true), "case {i}");
            differ_from_35b += 1;
        }
    }
    // The oracle exercises the one switch the 9B template lacks.
    assert_eq!(differ_from_35b, 3);
    let refusals = oracle["refusals"].as_array().unwrap();
    assert_eq!(refusals.len(), 5);
    for case in refusals {
        let turns = turns_of(case);
        let refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let want = case["error"].as_str().expect("the template raised");
        let err = tpl
            .try_render_with(&refs, &opts(&case["options"]))
            .expect_err(&format!("{}: must be refused", case["name"]));
        assert!(
            err.to_string().contains(want),
            "{}: {err} should carry {want:?}",
            case["name"]
        );
    }
}
