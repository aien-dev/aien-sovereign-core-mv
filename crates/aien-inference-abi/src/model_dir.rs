// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Model description from a Hugging Face model directory.
//!
//! `config.json` (and `generation_config.json` for the stop set) of a `LlamaForCausalLM`
//! checkpoint becomes a [`ModelConfig`]; [`crate::checkpoint::llama_catalog`] turns that into
//! the tensor catalog the strict loader checks. Anything this engine does not implement
//! (another architecture, biases, another activation, another rope type) is refused with
//! an error naming the field, never approximated.

use aien_abi_core::{Llama3RopeScaling, ModelConfig};
use serde::Deserialize;
use std::path::Path;

/// The Llama architecture this engine runs (also SmolLM2 and Llama 3.x).
pub const LLAMA_ARCHITECTURE: &str = "LlamaForCausalLM";

/// The Qwen3 dense architecture: Llama plus per-head RMSNorm of Q and K before RoPE,
/// `head_dim` set explicitly in `config.json`.
pub const QWEN3_ARCHITECTURE: &str = "Qwen3ForCausalLM";

/// Every `config.json` key a `Qwen3ForCausalLM` checkpoint may carry. The ones that change
/// the maths (`sliding_window`, `use_sliding_window`, `layer_types`, `rope_scaling`,
/// `attention_bias`, `hidden_act`) are checked by value below; any key outside this list is
/// refused by name, never ignored.
const QWEN3_KNOWN_KEYS: &[&str] = &[
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "dtype",
    "eos_token_id",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "max_window_layers",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "sliding_window",
    "tie_word_embeddings",
    "torch_dtype",
    "transformers_version",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
];

/// Refusals specific to `Qwen3ForCausalLM`, each naming the field.
fn check_qwen3_fields(raw: &serde_json::Value) -> Result<(), String> {
    let obj = raw.as_object().ok_or("config.json is not an object")?;
    if let Some(key) = obj.keys().find(|k| !QWEN3_KNOWN_KEYS.contains(&k.as_str())) {
        return Err(format!("unsupported Qwen3 config field {key:?}"));
    }
    match obj.get("model_type").and_then(|v| v.as_str()) {
        Some("qwen3") => {}
        other => {
            return Err(format!(
                "unsupported model_type {other:?} for Qwen3ForCausalLM (expected \"qwen3\")"
            ))
        }
    }
    if !obj.get("head_dim").is_some_and(|v| v.is_u64()) {
        return Err(
            "Qwen3 config.json must set head_dim (it is not hidden_size / num_attention_heads)"
                .to_string(),
        );
    }
    if obj
        .get("use_sliding_window")
        .is_some_and(|v| v != &serde_json::Value::Bool(false))
    {
        return Err("unsupported use_sliding_window (supported: false)".to_string());
    }
    if obj.get("sliding_window").is_some_and(|v| !v.is_null()) {
        return Err("unsupported sliding_window (supported: null)".to_string());
    }
    if let Some(types) = obj.get("layer_types") {
        let all_full = types
            .as_array()
            .is_some_and(|a| a.iter().all(|t| t.as_str() == Some("full_attention")));
        if !all_full {
            return Err(
                "unsupported layer_types (supported: every layer full_attention)".to_string(),
            );
        }
    }
    Ok(())
}

/// KV page size used for every model loaded from a directory.
const BLOCK_SIZE: usize = 16;

#[derive(Debug, Deserialize)]
struct HfLlamaConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    tie_word_embeddings: bool,
    max_position_embeddings: usize,
    hidden_act: Option<String>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
}

fn default_rope_theta() -> f32 {
    10000.0
}

/// Reads `rope_scaling`: absent, null or `rope_type: "default"` is plain rope; `"llama3"`
/// needs all four llama3 fields. Any other type is refused.
fn parse_rope_scaling(
    value: Option<&serde_json::Value>,
) -> Result<Option<Llama3RopeScaling>, String> {
    let Some(obj) = value.and_then(|v| v.as_object()) else {
        return match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(other) => Err(format!("rope_scaling must be an object, got {other}")),
        };
    };
    let rope_type = obj
        .get("rope_type")
        .or_else(|| obj.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    match rope_type {
        "default" => Ok(None),
        "llama3" => {
            let num = |key: &str| {
                obj.get(key)
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("rope_scaling (llama3) has no numeric {key}"))
            };
            let scaling = Llama3RopeScaling {
                factor: num("factor")?,
                low_freq_factor: num("low_freq_factor")?,
                high_freq_factor: num("high_freq_factor")?,
                original_max_position_embeddings: num("original_max_position_embeddings")? as usize,
            };
            if scaling.factor <= 0.0
                || scaling.high_freq_factor <= scaling.low_freq_factor
                || scaling.original_max_position_embeddings == 0
            {
                return Err(format!("rope_scaling (llama3) is degenerate: {scaling:?}"));
            }
            Ok(Some(scaling))
        }
        other => Err(format!(
            "unsupported rope_scaling rope_type {other:?} (supported: default, llama3)"
        )),
    }
}

/// Builds the [`ModelConfig`] of a `LlamaForCausalLM` or `Qwen3ForCausalLM` checkpoint from the text of its
/// `config.json` and, when present, its `generation_config.json` (stop set: its
/// `eos_token_id`, else the one in `config.json`).
pub fn model_config_from_hf_json(
    model_id: &str,
    config_json: &str,
    generation_config_json: Option<&str>,
) -> Result<ModelConfig, String> {
    let raw: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
    let hf: HfLlamaConfig =
        serde_json::from_value(raw.clone()).map_err(|e| format!("config.json: {e}"))?;
    let qk_norm = match hf.architectures.as_slice() {
        [a] if a == LLAMA_ARCHITECTURE => false,
        [a] if a == QWEN3_ARCHITECTURE => {
            check_qwen3_fields(&raw)?;
            true
        }
        other => {
            return Err(format!(
                "unsupported architectures {other:?}: this engine runs [\"{LLAMA_ARCHITECTURE}\"] or [\"{QWEN3_ARCHITECTURE}\"] only"
            ))
        }
    };
    if let Some(act) = hf.hidden_act.as_deref().filter(|a| *a != "silu") {
        return Err(format!("unsupported hidden_act {act:?} (supported: silu)"));
    }
    if hf.attention_bias || hf.mlp_bias {
        return Err("unsupported attention_bias/mlp_bias = true (no bias tensors)".to_string());
    }
    let num_heads = hf.num_attention_heads;
    if num_heads == 0 || (!qk_norm && !hf.hidden_size.is_multiple_of(num_heads)) {
        return Err(format!(
            "hidden_size {} is not a multiple of num_attention_heads {num_heads}",
            hf.hidden_size
        ));
    }
    let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / num_heads.max(1));
    if !qk_norm && head_dim * num_heads != hf.hidden_size {
        return Err(format!(
            "unsupported head_dim {head_dim}: num_attention_heads * head_dim must equal hidden_size {}",
            hf.hidden_size
        ));
    }
    let rope_scaling = parse_rope_scaling(hf.rope_scaling.as_ref())?;
    let generation = generation_config_json
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()
        .map_err(|e| format!("generation_config.json: {e}"))?;
    let eos_token_ids = generation
        .as_ref()
        .and_then(crate::tokenizer::eos_token_ids_from_json)
        .or_else(|| crate::tokenizer::eos_token_ids_from_json(&raw))
        .ok_or("no eos_token_id in generation_config.json or config.json")?;
    let config = ModelConfig {
        model_id: model_id.to_string(),
        max_sequence_length: hf.max_position_embeddings,
        block_size: BLOCK_SIZE,
        num_layers: hf.num_hidden_layers,
        num_heads,
        head_dim,
        num_kv_heads: hf.num_key_value_heads.unwrap_or(num_heads),
        hidden_dim: hf.hidden_size,
        intermediate_dim: hf.intermediate_size,
        vocab_size: hf.vocab_size,
        rms_norm_eps: hf.rms_norm_eps,
        rope_theta: hf.rope_theta,
        rope_scaling,
        tie_word_embeddings: hf.tie_word_embeddings,
        eos_token_ids,
        qk_norm,
    };
    config
        .attention_geometry()
        .map_err(|e| format!("attention geometry refused: {e:?}"))?;
    Ok(config)
}

/// A readable model id for a directory: `org/name` for a Hugging Face cache snapshot
/// (`.../models--org--name/snapshots/<hash>`), else the directory name.
pub fn model_id_for_dir(dir: &Path) -> String {
    let name = |p: &Path| p.file_name().and_then(|n| n.to_str()).map(str::to_string);
    let cache_repo = dir
        .parent()
        .filter(|p| name(p).as_deref() == Some("snapshots"))
        .and_then(|p| p.parent())
        .and_then(name)
        .and_then(|repo| {
            repo.strip_prefix("models--")
                .map(|r| r.replacen("--", "/", 1))
        });
    cache_repo
        .or_else(|| name(dir))
        .unwrap_or_else(|| "checkpoint".to_string())
}

/// Reads `<dir>/config.json` and `<dir>/generation_config.json` into a [`ModelConfig`].
pub fn load_model_config(dir: &Path) -> Result<ModelConfig, String> {
    let config_path = dir.join("config.json");
    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let generation_path = dir.join("generation_config.json");
    let generation_json = match std::fs::read_to_string(&generation_path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("cannot read {}: {e}", generation_path.display())),
    };
    model_config_from_hf_json(
        &model_id_for_dir(dir),
        &config_json,
        generation_json.as_deref(),
    )
    .map_err(|e| format!("{}: {e}", config_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{llama_catalog, tinyllama_catalog};

    /// TinyLlama-1.1B-Chat-v1.0 config.json, verbatim.
    const TINYLLAMA_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"], "attention_bias": false, "bos_token_id": 1,
  "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02,
  "intermediate_size": 5632, "max_position_embeddings": 2048, "model_type": "llama",
  "num_attention_heads": 32, "num_hidden_layers": 22, "num_key_value_heads": 4,
  "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000.0,
  "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.35.0",
  "use_cache": true, "vocab_size": 32000
}"#;

    /// unsloth/Llama-3.2-1B-Instruct config.json and generation_config.json, verbatim.
    const LLAMA32_1B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "bos_token_id": 128000, "eos_token_id": 128009, "head_dim": 64, "hidden_act": "silu",
  "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 8192,
  "max_position_embeddings": 131072, "mlp_bias": false, "model_type": "llama",
  "num_attention_heads": 32, "num_hidden_layers": 16, "num_key_value_heads": 8,
  "pad_token_id": 128004, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
  "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192, "rope_type": "llama3"},
  "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
  "transformers_version": "4.52.0.dev0", "unsloth_fixed": true, "use_cache": true,
  "vocab_size": 128256
}"#;
    const LLAMA32_1B_GENERATION: &str = r#"{"bos_token_id": 128000, "do_sample": true,
  "eos_token_id": [128001, 128008, 128009], "max_length": 131072, "pad_token_id": 128004,
  "temperature": 0.6, "top_p": 0.9, "transformers_version": "4.52.0.dev0"}"#;

    #[test]
    fn tinyllama_config_json_reproduces_the_hard_wired_config_and_catalog() {
        let from_json = model_config_from_hf_json(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            TINYLLAMA_CONFIG,
            Some(
                r#"{"bos_token_id": 1, "eos_token_id": 2, "max_length": 2048, "pad_token_id": 0}"#,
            ),
        )
        .unwrap();
        let wired = ModelConfig::tinyllama_1_1b();
        // Every architecture field equal; the stop set comes from the file (EOS 2) where the
        // hard-wired config keeps the legacy backend set.
        assert_eq!(
            ModelConfig {
                eos_token_ids: Vec::new(),
                qk_norm: false,
                ..from_json.clone()
            },
            wired
        );
        assert_eq!(from_json.eos_token_ids, vec![2]);
        assert_eq!(llama_catalog(&from_json), tinyllama_catalog());
        assert_eq!(llama_catalog(&wired), tinyllama_catalog());
        assert_eq!(from_json.rope(), aien_abi_core::RopeParams::plain(10000.0));
    }

    #[test]
    fn llama32_1b_config_has_tied_head_llama3_rope_and_three_stop_ids() {
        let c =
            model_config_from_hf_json("l", LLAMA32_1B_CONFIG, Some(LLAMA32_1B_GENERATION)).unwrap();
        assert_eq!(
            (c.num_layers, c.num_heads, c.num_kv_heads, c.head_dim),
            (16, 32, 8, 64)
        );
        assert_eq!(
            (c.hidden_dim, c.intermediate_dim, c.vocab_size),
            (2048, 8192, 128256)
        );
        assert!(c.tie_word_embeddings);
        assert_eq!(c.eos_token_ids, vec![128001, 128008, 128009]);
        assert_eq!(
            c.rope_scaling,
            Some(Llama3RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            })
        );
        let catalog = llama_catalog(&c);
        assert_eq!(catalog.len(), 2 + 9 * 16);
        assert!(catalog.iter().all(|(n, _)| n != "lm_head.weight"));
        let shape = |name: &str| catalog.iter().find(|(n, _)| n == name).unwrap().1.clone();
        assert_eq!(shape("model.embed_tokens.weight"), vec![128256, 2048]);
        assert_eq!(
            shape("model.layers.15.self_attn.k_proj.weight"),
            vec![512, 2048]
        );
        assert_eq!(
            shape("model.layers.0.mlp.down_proj.weight"),
            vec![2048, 8192]
        );
        // Without generation_config.json the config.json eos is used.
        let c2 = model_config_from_hf_json("l", LLAMA32_1B_CONFIG, None).unwrap();
        assert_eq!(c2.eos_token_ids, vec![128009]);
    }

    /// Expected inverse frequencies for Llama-3.2-1B (theta 500000, head_dim 64, llama3
    /// factor 32, low 1, high 4, original 8192), computed independently in f64 outside the
    /// repository from the transformers `_compute_llama3_parameters` formula.
    const LLAMA32_1B_INV_FREQ: [f64; 32] = [
        1.0,
        0.6636012376960885,
        0.44036660267178046,
        0.2922278225730151,
        0.19392274474868576,
        0.12868737343265052,
        0.08539710028576561,
        0.056669621445291044,
        0.03760603093086393,
        0.024955408670558694,
        0.016560440080994446,
        0.010989528534539826,
        0.0072926647372171085,
        0.004839421345719893,
        0.0032114459947525913,
        0.001290547928209264,
        0.00042955679655936815,
        9.70828780262767e-05,
        1.9461638184831125e-05,
        1.291476718704739e-05,
        8.570255489881478e-06,
        5.687232150457046e-06,
        3.7740542941082826e-06,
        2.5044671007024935e-06,
        1.661967467795309e-06,
        1.102883668639601e-06,
        7.318749675440419e-07,
        4.856731343010108e-07,
        3.2229329303788936e-07,
        2.138742281610915e-07,
        1.4192720251899592e-07,
        9.418306725434909e-08,
    ];

    #[test]
    fn llama3_rope_inv_freq_matches_independent_values() {
        let c =
            model_config_from_hf_json("l", LLAMA32_1B_CONFIG, Some(LLAMA32_1B_GENERATION)).unwrap();
        let got = c.rope().inv_freqs(64);
        assert_eq!(got.len(), 32);
        for (i, (g, want)) in got.iter().zip(LLAMA32_1B_INV_FREQ).enumerate() {
            assert!(
                ((g - want) / want).abs() < 1e-12,
                "inv_freq[{i}] = {g}, expected {want}"
            );
        }
        // The three bands: high frequencies kept, low divided by 32, the middle smoothed.
        let plain = aien_abi_core::RopeParams::plain(500000.0).inv_freqs(64);
        assert_eq!(got[14], plain[14]);
        assert_eq!(got[31], plain[31] / 32.0);
        assert!(got[16] < plain[16] && got[16] > plain[16] / 32.0);
        assert_eq!(got[18], plain[18] / 32.0);
    }

    #[test]
    fn unsupported_models_are_refused_by_name() {
        let qwen = LLAMA32_1B_CONFIG.replace("LlamaForCausalLM", "Qwen2ForCausalLM");
        let err = model_config_from_hf_json("q", &qwen, None).unwrap_err();
        assert!(err.contains("Qwen2ForCausalLM"), "{err}");

        let yarn = LLAMA32_1B_CONFIG.replace("\"llama3\"", "\"yarn\"");
        let err = model_config_from_hf_json("y", &yarn, None).unwrap_err();
        assert!(err.contains("yarn"), "{err}");

        let bias =
            LLAMA32_1B_CONFIG.replace("\"attention_bias\": false", "\"attention_bias\": true");
        assert!(model_config_from_hf_json("b", &bias, None).is_err());

        let gelu = LLAMA32_1B_CONFIG.replace("\"silu\"", "\"gelu\"");
        assert!(model_config_from_hf_json("g", &gelu, None)
            .unwrap_err()
            .contains("gelu"));

        let default_rope = LLAMA32_1B_CONFIG.replace("\"llama3\"", "\"default\"");
        let c = model_config_from_hf_json("d", &default_rope, None).unwrap();
        assert_eq!(c.rope_scaling, None);
    }

    /// HuggingFaceTB/SmolLM2-1.7B-Instruct config.json at revision 31b70e2e869a, verbatim
    /// (sha256 994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea), and its
    /// generation_config.json (sha256 87b916ed...a013).
    const SMOLLM2_17B_CONFIG: &str = r#"{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "num_key_value_heads": 32,
  "pad_token_id": 2,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 130000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "transformers.js_config": {
    "dtype": "q4",
    "kv_cache_dtype": {
      "q4f16": "float16",
      "fp16": "float16"
    },
    "use_external_data_format": {
      "model.onnx": true,
      "model_fp16.onnx": true
    }
  },
  "use_cache": true,
  "vocab_size": 49152
}
"#;
    const SMOLLM2_17B_GENERATION: &str = r#"{
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 2,
  "transformers_version": "4.42.3"
}
"#;

    #[test]
    fn smollm2_config_is_full_mha_tied_plain_rope() {
        let c = model_config_from_hf_json("s", SMOLLM2_17B_CONFIG, Some(SMOLLM2_17B_GENERATION))
            .unwrap();
        // 32 KV heads = 32 query heads: no GQA sharing (group size 1).
        assert_eq!(
            (c.num_layers, c.num_heads, c.num_kv_heads, c.head_dim),
            (24, 32, 32, 64)
        );
        assert_eq!(
            (c.hidden_dim, c.intermediate_dim, c.vocab_size),
            (2048, 8192, 49152)
        );
        assert_eq!(c.max_sequence_length, 8192);
        assert_eq!(c.rms_norm_eps, 1e-5);
        assert!(c.tie_word_embeddings);
        assert_eq!(c.eos_token_ids, vec![2]);
        // "rope_theta": 130000 is a JSON integer; rope_scaling null is plain rope.
        assert_eq!(c.rope_scaling, None);
        assert_eq!(c.rope(), aien_abi_core::RopeParams::plain(130000.0));
        let catalog = llama_catalog(&c);
        // 218 tensors in the published model.safetensors: embed + norm + 9 per layer.
        assert_eq!(catalog.len(), 2 + 9 * 24);
        assert!(catalog.iter().all(|(n, _)| n != "lm_head.weight"));
        let shape = |name: &str| catalog.iter().find(|(n, _)| n == name).unwrap().1.clone();
        assert_eq!(shape("model.embed_tokens.weight"), vec![49152, 2048]);
        assert_eq!(
            shape("model.layers.23.self_attn.k_proj.weight"),
            vec![2048, 2048]
        );
        assert_eq!(
            shape("model.layers.0.self_attn.v_proj.weight"),
            vec![2048, 2048]
        );
        assert_eq!(
            shape("model.layers.0.mlp.down_proj.weight"),
            vec![2048, 8192]
        );
    }

    #[test]
    fn hf_cache_snapshot_dirs_get_the_repo_id() {
        let dir = Path::new("/c/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8a");
        assert_eq!(model_id_for_dir(dir), "unsloth/Llama-3.2-1B-Instruct");
        assert_eq!(
            model_id_for_dir(Path::new("/m/TinyLlama-1.1B-Chat-v1.0")),
            "TinyLlama-1.1B-Chat-v1.0"
        );
    }
}
