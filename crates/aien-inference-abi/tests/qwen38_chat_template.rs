//! Qwen3.8-27B chat template (sc#642 serving). Pinned text: Hugging Face `Qwen/Qwen3.8-27B-FP8`
//! revision 017b9c7a (fixtures/chat-templates/qwen3.8-27b).
//!
//! Checks of the rendered bytes:
//! 1. JINJA ORACLE: `render_oracle.json`, the template rendered by jinja2 3.1.2 (scratch script
//!    outside the repository) for 8 conversations x 6 option sets, thinking on, off and default.
//! 2. GOLDEN PROMPT: with the checkpoint present (`AIEN_QWEN38_DIR`, ignored test), the
//!    thinking-off rendering of the golden question tokenizes to the host reference's
//!    `prompt_ids` (`~/handoffs/qwen38-ref/gen-chat-max-24.json`), the ids the chip gate replays.
use aien_inference_abi::tokenizer::{ChatOptions, ChatTemplate, ChatTokenizer};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/chat-templates/qwen3.8-27b"
);

fn t() -> ChatTemplate {
    let text = std::fs::read_to_string(format!("{DIR}/chat_template.jinja")).unwrap();
    ChatTemplate::detect(&text).expect("Qwen3.8 template accepted")
}

fn opts(v: &serde_json::Value) -> ChatOptions {
    ChatOptions {
        enable_thinking: v.get("enable_thinking").and_then(|x| x.as_bool()),
        reasoning_effort: v
            .get("reasoning_effort")
            .and_then(|x| x.as_str())
            .map(String::from),
    }
}

#[test]
fn pinned_template_is_selected() {
    assert_eq!(t(), ChatTemplate::Qwen38);
}

#[test]
fn rendering_matches_the_jinja_reference_rendering() {
    let oracle: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/render_oracle.json")).unwrap(),
    )
    .unwrap();
    let tpl = t();
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for (i, case) in cases.iter().enumerate() {
        let turns: Vec<(String, String)> = case["turns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into()))
            .collect();
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
        // The default is the template's default (thinking on, effort xhigh).
        if o == ChatOptions::default() {
            assert_eq!(
                tpl.try_render(&refs).unwrap(),
                case["text"].as_str().unwrap()
            );
        }
    }
}

#[test]
fn options_change_the_rendering_and_bad_ones_are_refused() {
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
                reasoning_effort: None,
            },
        )
        .unwrap();
    assert_ne!(on, off);
    assert!(on.ends_with("<|im_start|>assistant\n<think>\n"));
    assert!(off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let bad = ChatOptions {
        enable_thinking: None,
        reasoning_effort: Some("max".into()),
    };
    assert!(tpl.try_render_with(&turns, &bad).is_err());
    assert!(tpl
        .try_render_with(
            &[("user", "a"), ("system", "late")],
            &ChatOptions::default()
        )
        .is_err());
    assert!(tpl
        .try_render_with(&[("assistant", "no user")], &ChatOptions::default())
        .is_err());
}

#[test]
#[ignore = "needs AIEN_QWEN38_DIR (the checkpoint) and ~/handoffs/qwen38-ref/gen-chat-max-24.json"]
fn golden_prompt_tokenizes_to_the_reference_ids() {
    let dir = std::env::var("AIEN_QWEN38_DIR").expect("AIEN_QWEN38_DIR");
    let home = std::env::var("HOME").unwrap();
    let g: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{home}/handoffs/qwen38-ref/gen-chat-max-24.json"))
            .unwrap(),
    )
    .unwrap();
    let want: Vec<u32> = g["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let tk = ChatTokenizer::from_model_dir(std::path::Path::new(&dir), None).unwrap();
    assert_eq!(tk.template(), ChatTemplate::Qwen38);
    let off = ChatOptions {
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    let text = tk
        .try_format_chat_with(
            &[(
                "user",
                "Write a C function that returns the larger of two ints.",
            )],
            &off,
        )
        .unwrap();
    assert_eq!(tk.encode_with_special(&text, true).unwrap(), want);
    // Negative control: the thinking-on rendering of the same question must not match.
    let on = tk
        .try_format_chat_with(
            &[(
                "user",
                "Write a C function that returns the larger of two ints.",
            )],
            &ChatOptions::default(),
        )
        .unwrap();
    assert_ne!(tk.encode_with_special(&on, true).unwrap(), want);
    assert_eq!(tk.stop_token_ids(), &[248046, 248044]);
}
