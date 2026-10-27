//! A base (non-chat) model directory has no chat template: it loads with an explicit
//! "no chat template" state, plain encode/decode work, and chat rendering is refused.
//! An existing but unrecognized template is still refused at load.

use aien_inference_abi::tokenizer::{ChatTemplate, ChatTokenizer};
use std::path::{Path, PathBuf};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/openwaldo-byte");

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aien-plain-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["tokenizer.json", "config.json", "generation_config.json"] {
        std::fs::copy(Path::new(DIR).join(f), dir.join(f)).unwrap();
    }
    dir
}

#[test]
fn model_dir_without_template_loads_as_plain() {
    let tok = ChatTokenizer::from_model_dir(Path::new(DIR), None).expect("plain model loads");
    assert_eq!(tok.template(), ChatTemplate::None);
    assert!(tok.stop_token_ids().contains(&2));
    let ids = tok.encode("hi").expect("plain encode");
    assert!(!ids.is_empty());
    assert!(tok
        .decode_opts(&ids, true)
        .expect("plain decode")
        .contains("hi"));
}

#[test]
fn chat_render_on_plain_model_is_refused() {
    let tok = ChatTokenizer::from_model_dir(Path::new(DIR), None).unwrap();
    let err = tok
        .try_format_chat(&[("user", "hello")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("no chat template"), "{err}");
    let err = ChatTemplate::None
        .try_render(&[("user", "hello")])
        .unwrap_err()
        .to_string();
    assert!(err.contains("no chat template"), "{err}");
}

#[test]
fn unrecognized_template_still_refused_at_load() {
    let dir = scratch("unrec");
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template": "{{ messages }}"}"#,
    )
    .unwrap();
    let err = ChatTokenizer::from_model_dir(&dir, None)
        .err()
        .expect("refused")
        .to_string();
    assert!(err.contains("unsupported chat template"), "{err}");
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
    assert!(ChatTokenizer::from_model_dir(&dir, None).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plain prompt format: the text is passed to the tokenizer as-is, nothing is wrapped
/// around it. Whatever the model's own tokenizer.json adds (BOS via its post-processor)
/// is added, exactly as `from_file` + `encode` does; the OpenWALDO byte tokenizer has
/// none, so "hi" is its two bytes (h=104, i=105, ids = byte + 3).
#[test]
fn plain_prompt_is_text_encoded_as_is() {
    let tok = ChatTokenizer::from_model_dir(Path::new(DIR), None).unwrap();
    let direct = ChatTokenizer::from_file(format!("{DIR}/tokenizer.json")).unwrap();
    assert_eq!(tok.encode("hi").unwrap(), direct.encode("hi").unwrap());
    assert_eq!(tok.encode("hi").unwrap(), vec![107, 108]);
}

/// A `chat_template` key that is present but not a string (Hugging Face's list form
/// `[{name, template}, ...]`, `null`, or an object) is an unrecognized template, not
/// "no template": it is refused at load and never falls back to the plain state.
#[test]
fn non_string_template_key_is_refused_not_plain() {
    let dir = scratch("nonstring");
    for form in [
        r#"{"chat_template": [{"name": "default", "template": "{{ messages }}"}]}"#,
        r#"{"chat_template": null}"#,
        r#"{"chat_template": {"template": "{{ messages }}"}}"#,
    ] {
        std::fs::write(dir.join("tokenizer_config.json"), form).unwrap();
        let err = ChatTokenizer::from_model_dir(&dir, None)
            .err()
            .unwrap_or_else(|| panic!("loaded as plain: {form}"))
            .to_string();
        assert!(
            err.contains("unrecognized chat template form"),
            "{form}: {err}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn render_refusal_is_its_own_variant_not_a_load_failure() {
    let tok = ChatTokenizer::from_model_dir(Path::new(DIR), None).unwrap();
    let err = tok.try_format_chat(&[("user", "hello")]).unwrap_err();
    assert!(
        matches!(err, aien_inference_abi::TokenizerError::NoChatTemplate(_)),
        "{err:?}"
    );
    let text = err.to_string();
    assert!(text.contains("no chat template"), "{text}");
    assert!(!text.contains("Failed to load tokenizer"), "{text}");
}

/// Without `generation_config.json` the stop set is the config's `eos_token_id`, also read from
/// `text_config` (the Qwen3.5-9B layout), preceded by the tokenizer's declared `eos_token`.
/// With a generation config, its list is used as is and the tokenizer's token is not added.
#[test]
fn stop_set_without_generation_config_reads_text_config_and_tokenizer_eos() {
    let dir = scratch("eos-fallback");
    std::fs::remove_file(dir.join("generation_config.json")).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"qwen3_5","text_config":{"eos_token_id":2,"max_position_embeddings":16}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"eos_token":"<pad>"}"#,
    )
    .unwrap();
    let tok = ChatTokenizer::from_model_dir(&dir, None).unwrap();
    assert_eq!(tok.stop_token_ids(), &[0, 2]);
    // The declared token already in the list is not repeated.
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"eos_token":"<eos>"}"#,
    )
    .unwrap();
    let tok = ChatTokenizer::from_model_dir(&dir, None).unwrap();
    assert_eq!(tok.stop_token_ids(), &[2]);
    // A generation config decides alone.
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"eos_token":"<pad>"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":2}"#).unwrap();
    let tok = ChatTokenizer::from_model_dir(&dir, None).unwrap();
    assert_eq!(tok.stop_token_ids(), &[2]);
    // No eos anywhere is still refused.
    std::fs::remove_file(dir.join("generation_config.json")).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
    assert!(ChatTokenizer::from_model_dir(&dir, None).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}
