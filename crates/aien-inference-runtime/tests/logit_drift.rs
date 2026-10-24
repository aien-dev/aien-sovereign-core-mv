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

//! Teacher-forced logit drift between the CPU reference and the Omega backend, printed
//! per step (diagnostic), plus host tests proving the drift judge used by the strict gate
//! rejects what it must.
//!   AIEN_E2E_CHECKPOINT=~/models/TinyLlama-1.1B-Chat-v1.0 \
//!   cargo test -p aien-inference-runtime --release --test logit_drift -- --ignored --nocapture
mod drift;

use aien_inference_runtime::model::EmbeddedModel;
use drift::{compare, judge, logits_per_step, top2, StepCmp, MAX_ABS_DLOGIT};
use std::path::{Path, PathBuf};

fn step(ref_tok: usize, ref_margin: f32, cand_tok: usize, d: f32) -> StepCmp {
    StepCmp {
        step: 0,
        ref_tok,
        ref_margin,
        cand_tok,
        max_abs_dlogit: d,
    }
}

#[test]
fn judge_accepts_agreement_and_explained_near_tie() {
    assert_eq!(judge(&[step(5, 0.3, 5, 0.04)]), Ok(0));
    // the measured cut 3d case: margin 0.0005, difference 0.0314
    assert_eq!(judge(&[step(9068, 0.0005, 3370, 0.0314)]), Ok(1));
}

#[test]
fn judge_rejects_unexplained_flip_and_large_drift() {
    // negative controls: a flip the measured difference cannot explain, and drift over the bound
    assert!(judge(&[step(9068, 1.0, 3370, 0.03)]).is_err());
    assert!(judge(&[step(5, 0.3, 5, MAX_ABS_DLOGIT * 2.0)]).is_err());
    assert!(judge(&[step(5, 0.3, 5, f32::NAN)]).is_err());
}

#[test]
fn compare_finds_flip_margin_and_difference() {
    let r = vec![vec![0.0, 2.0, 1.9]];
    let c = vec![vec![0.0, 1.9, 2.0]];
    let s = &compare(&r, &c)[0];
    assert_eq!((s.ref_tok, s.cand_tok), (1, 2));
    assert!((s.ref_margin - 0.1).abs() < 1e-6 && (s.max_abs_dlogit - 0.1).abs() < 1e-6);
}

#[test]
#[ignore = "needs the real checkpoint and the GB10 with a native Omega build; heavy queue only"]
fn teacher_forced_logit_drift() {
    let ckpt = PathBuf::from(std::env::var("AIEN_E2E_CHECKPOINT").expect("AIEN_E2E_CHECKPOINT"));
    let dir = if ckpt.is_dir() {
        ckpt.clone()
    } else {
        ckpt.parent().map(Path::to_path_buf).unwrap_or_default()
    };
    let (model_path, tokenizer_path) = (dir.join("model.safetensors"), dir.join("tokenizer.json"));
    let prompt = "The DGX Spark is a small computer with a large";
    let steps = 32usize;

    let mut cpu = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, false, true)
        .expect("reference load");
    let ptoks = cpu.tokenizer.encode(prompt).expect("encode");
    let (ref_logits, ref_toks) = logits_per_step(&mut cpu.transformer, &ptoks, steps, &[]);
    let tok = cpu.tokenizer;
    drop(cpu.transformer);

    let mut om = EmbeddedModel::load_checkpoint(&model_path, &tokenizer_path, true, true)
        .expect("omega load");
    println!(
        "DRIFT omega backend: {}",
        om.transformer.tensor_backend.name()
    );
    let (om_logits, _) = logits_per_step(&mut om.transformer, &ptoks, steps, &ref_toks);
    println!("DRIFT {}", om.transformer.tensor_backend.op_report().line());

    let cmp = compare(&ref_logits, &om_logits);
    for (s, c) in cmp.iter().enumerate() {
        let (_, _, r2, _) = top2(&ref_logits[s]);
        println!(
            "DRIFT step={s:2} ref_tok={:5} {:?} ref_margin={:.4} omega_tok={:5} max_abs_dlogit={:.4} ref2={r2} {:?}{}",
            c.ref_tok,
            tok.decode(&[c.ref_tok as u32]).unwrap_or_default(),
            c.ref_margin,
            c.cand_tok,
            c.max_abs_dlogit,
            tok.decode(&[r2 as u32]).unwrap_or_default(),
            if c.ref_tok != c.cand_tok { "  <-- FLIP" } else { "" }
        );
    }
    let worst = cmp.iter().map(|c| c.max_abs_dlogit).fold(0.0f32, f32::max);
    println!(
        "DRIFT summary steps={steps} worst_max_abs_dlogit={worst:.4} judge={:?}",
        judge(&cmp)
    );
}

#[test]
fn compare_keeps_nan_so_the_judge_fails_it() {
    let s = compare(&[vec![1.0, 0.0]], &[vec![f32::NAN, 0.0]]);
    assert!(s[0].max_abs_dlogit.is_nan());
    assert!(judge(&s).is_err());
}
