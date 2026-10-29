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

//! Host integration: the real Qwen vocabulary against the real oracle on the public dev programs.
//! Each program is walked through the constraint one step at a time; at every masked step the
//! token Qwen's own tokenizer would write next must be allowed, unless it crosses two word
//! boundaries (the one refusal the mapping makes on purpose), in which case the longest allowed
//! token that still writes the program is taken. Needs `OMEGESE_BIN`, `QWEN_TOKENIZER` (the
//! model's tokenizer.json) and `OMEGESE_DEV_DIRS` (colon-separated folders searched for
//! `*.omegese`).

mod common;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aien_inference_abi::generation::{Constraint, StepRule};
use aien_omegese_constraint::word::is_sep;
use aien_omegese_constraint::{
    Answer, Framing, ItemOracle, OmegeseConstraint, Shape, Signature, SubprocessOracle, TokenTable,
};
use common::*;

fn programs() -> Vec<PathBuf> {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        let mut es: Vec<_> = std::fs::read_dir(d)
            .unwrap_or_else(|e| panic!("{}: {e}", d.display()))
            .map(|e| e.unwrap().path())
            .collect();
        es.sort();
        for p in es {
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "omegese") {
                out.push(p);
            }
        }
    }
    let dirs = std::env::var("OMEGESE_DEV_DIRS").expect("set OMEGESE_DEV_DIRS");
    let mut out = Vec::new();
    for d in dirs.split(':').filter(|d| !d.is_empty()) {
        walk(Path::new(d), &mut out);
    }
    assert!(!out.is_empty(), "no *.omegese under OMEGESE_DEV_DIRS");
    out
}

#[derive(Default)]
struct Tally {
    masked: u64,
    canonical_ok: u64,
    refused_multi: u64,
    refused_other: Vec<String>,
    checked: u64,
    unsound: Vec<String>,
    forced: u64,
    calls: u64,
    masked_out_multi: u64,
    times: Vec<Duration>,
}

/// Walks `text` (a whole program) through a fresh constraint; returns what it saw.
fn walk(table: &Arc<TokenTable>, bin: &Path, stop: u32, text: &str, header: bool, t: &mut Tally) {
    let framing = Framing {
        open_fence: Some("```\n".into()),
        header: header.then(|| Signature::from_header(text.lines().next().unwrap()).unwrap()),
        mask_body: true,
        close_fence: "```".into(),
        stop_token: stop,
    };
    let oracle = SubprocessOracle::new(bin).unwrap();
    let mut check = SubprocessOracle::new(bin).unwrap();
    let mut k = OmegeseConstraint::new(table.clone(), oracle, framing).unwrap();
    let mut gen: Vec<u32> = Vec::new();
    loop {
        let rule = k
            .rule(&[], &gen)
            .unwrap_or_else(|e| panic!("{e}\nafter {:?}", String::from_utf8_lossy(k.program())));
        let prog = k.program();
        assert!(
            text.as_bytes().starts_with(prog),
            "the program left the text"
        );
        let rest = &text[prog.len()..];
        match rule {
            StepRule::Force(id) => {
                t.forced += 1;
                gen.push(id);
                if id == stop {
                    break;
                }
            }
            StepRule::Allow(set) => {
                t.masked += 1;
                // Soundness: every allowed token that ends in a separator (the blank and bracket
                // tokens among them) and about 8 others spread over the set must leave a prefix
                // the oracle does not refuse.
                let mut keys = HashSet::new();
                let spread = (set.len() / 8).max(1);
                for (i, &id) in set.iter().enumerate() {
                    let Some(x) = table.text(id) else { continue };
                    if x.contains(&b'`')
                        || !(x.last().is_some_and(|&b| is_sep(b)) || i % spread == 0)
                    {
                        continue;
                    }
                    let mut p = prog.to_vec();
                    p.extend_from_slice(x);
                    if !keys.insert(normalize(&p)) {
                        continue;
                    }
                    t.checked += 1;
                    if let Answer::Refused(r) = check.next(&p).unwrap() {
                        t.unsound.push(format!(
                            "{:?} after {:?}: {r}",
                            String::from_utf8_lossy(x),
                            String::from_utf8_lossy(prog)
                        ));
                    }
                }
                if rest.is_empty() {
                    let close = set
                        .iter()
                        .copied()
                        .find(|&id| table.text(id).is_some_and(|x| x.contains(&b'`')))
                        .expect("eof offers the closing fence");
                    gen.push(close);
                    continue;
                }
                let canon = table.encode(rest).unwrap()[0];
                if set.binary_search(&canon).is_ok() {
                    t.canonical_ok += 1;
                    gen.push(canon);
                    continue;
                }
                let ct = String::from_utf8_lossy(table.text(canon).unwrap()).to_string();
                if table.shape(canon) == Shape::Multi {
                    t.refused_multi += 1;
                } else {
                    t.refused_other
                        .push(format!("{ct:?} after {:?}", String::from_utf8_lossy(prog)));
                }
                let fallback = set
                    .iter()
                    .copied()
                    .filter(|&id| {
                        table
                            .text(id)
                            .is_some_and(|x| rest.as_bytes().starts_with(x))
                    })
                    .max_by_key(|&id| table.text(id).unwrap().len())
                    .unwrap_or_else(|| panic!("no allowed token writes {rest:?}"));
                gen.push(fallback);
            }
            StepRule::Finish => break,
            StepRule::Any => panic!("Any inside a masked body"),
        }
    }
    assert_eq!(k.program(), text.as_bytes());
    let reply = table.decode(&gen[..gen.len() - 1]).unwrap();
    assert_eq!(reply, format!("```\n{text}```").into_bytes());
    let st = k.stats();
    t.calls += st.oracle_calls;
    t.masked_out_multi += st.multi_boundary_refused;
    t.times.extend(st.step_times.iter().copied());
}

#[test]
#[ignore = "needs OMEGESE_BIN, QWEN_TOKENIZER and OMEGESE_DEV_DIRS"]
fn qwen_vocab_writes_every_dev_program_through_the_real_oracle() {
    let bin = omegese_bin();
    let tok = std::env::var("QWEN_TOKENIZER").expect("set QWEN_TOKENIZER to tokenizer.json");
    let table = Arc::new(TokenTable::from_tokenizer_json(Path::new(&tok)).unwrap());
    let stop = table.special_id("<|im_end|>").expect("<|im_end|>");
    println!("vocabulary: {} tokens, stop {stop}", table.len());
    // The vocabulary by shape, over tokens made only of bytes Omegese text can hold.
    let mut census = [0u64; 3];
    for id in 0..table.len() as u32 {
        let Some(x) = table.text(id) else { continue };
        if !x
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || b"_-= \t\n()".contains(&b))
        {
            continue;
        }
        census[match table.shape(id) {
            Shape::Word => 0,
            Shape::OneBoundary { .. } => 1,
            _ => 2,
        }] += 1;
    }
    println!(
        "omegese-byte tokens: no boundary {}, one boundary {}, two or more boundaries {}",
        census[0], census[1], census[2]
    );
    assert!(
        census[2] > 0,
        "the vocabulary has tokens the mapping refuses"
    );
    let mut all = Tally::default();
    for p in programs() {
        let mut text = std::fs::read_to_string(&p).unwrap();
        if !text.ends_with('\n') {
            text.push('\n');
        }
        emit(&bin, text.as_bytes()).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let header_ok = Signature::from_header(text.lines().next().unwrap()).is_ok()
            && text.starts_with("fn ")
            && text.matches("\nfn ").count() == 0;
        for header in [false, true] {
            if header && !header_ok {
                continue;
            }
            let mut t = Tally::default();
            walk(&table, &bin, stop, &text, header, &mut t);
            println!(
                "{} header={header}: masked {} canonical-ok {} two-boundary-refused {} other-refused {} forced {} oracle-calls {}",
                p.display(),
                t.masked,
                t.canonical_ok,
                t.refused_multi,
                t.refused_other.len(),
                t.forced,
                t.calls
            );
            all.masked += t.masked;
            all.canonical_ok += t.canonical_ok;
            all.refused_multi += t.refused_multi;
            all.refused_other.extend(t.refused_other);
            all.checked += t.checked;
            all.unsound.extend(t.unsound);
            all.forced += t.forced;
            all.calls += t.calls;
            all.masked_out_multi += t.masked_out_multi;
            all.times.extend(t.times);
        }
    }
    all.times.sort();
    let n = all.times.len();
    let us = |d: Duration| d.as_micros();
    let mean = all.times.iter().sum::<Duration>() / n as u32;
    println!(
        "TOTAL masked {} canonical-ok {} two-boundary-refused {} other-refused {} forced {} oracle-calls {} ({:.2} per masked step); two-boundary tokens masked out (token x step) {}",
        all.masked,
        all.canonical_ok,
        all.refused_multi,
        all.refused_other.len(),
        all.forced,
        all.calls,
        all.calls as f64 / all.masked as f64
,
        all.masked_out_multi
    );
    println!(
        "STEP LATENCY (oracle calls + vocabulary scan) over {n} masked steps: mean {} us, median {} us, p95 {} us, max {} us",
        us(mean),
        us(all.times[n / 2]),
        us(all.times[n * 95 / 100]),
        us(all.times[n - 1])
    );
    println!(
        "SOUNDNESS {} allowed continuations checked against the oracle, {} refused",
        all.checked,
        all.unsound.len()
    );
    assert!(
        all.unsound.is_empty(),
        "allowed tokens the oracle refuses: {:#?}",
        all.unsound
    );
    assert!(
        all.refused_other.is_empty(),
        "tokens refused although they cross at most one boundary: {:#?}",
        all.refused_other
    );
}
