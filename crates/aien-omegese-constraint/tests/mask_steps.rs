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

//! The mask step by step, without a model: a driver writes a known program through the
//! constraint, and at every masked step every allowed token is checked against an exact oracle
//! (soundness), not only the token the driver takes.

mod common;

use std::sync::Arc;

use aien_inference_abi::generation::{Constraint, StepRule};
use aien_omegese_constraint::{Answer, Framing, Item, ItemOracle, OmegeseConstraint, TokenTable};
use common::*;

const TARGET: &str = "fn f a uint b uint to uint\nret add a b\nend\n";

fn framing(fence: bool, header: bool) -> Framing {
    Framing {
        open_fence: fence.then(|| "```\n".into()),
        header: header.then(sig_f),
        mask_body: true,
        close_fence: "```".into(),
        stop_token: STOP,
    }
}

fn offers_eof(check: &mut ExactText, p: &[u8]) -> bool {
    matches!(check.next(p).unwrap(), Answer::Items(i) if i.contains(&Item::Eof))
}

/// Writes `text` through `k`, the longest allowed token that continues it at each step (the
/// closing fence or the stop token at its end). At every `Allow` step every allowed token is
/// checked: a body token must leave a prefix `check` does not refuse, and a token that ends the
/// program (a fence marker, or the stop token) needs eof offered. Returns the tokens (stop
/// included) and how many allowed tokens were checked.
fn drive<O: ItemOracle>(
    table: &TokenTable,
    k: &mut OmegeseConstraint<O>,
    text: &str,
    check: &mut ExactText,
) -> (Vec<u32>, usize) {
    let mut gen: Vec<u32> = Vec::new();
    let mut checked = 0;
    loop {
        let rule = k.rule(&[], &gen).unwrap_or_else(|e| panic!("{e}"));
        let prog = k.program().to_vec();
        assert!(
            text.as_bytes().starts_with(&prog),
            "the program left the text"
        );
        let rest = &text.as_bytes()[prog.len()..];
        match rule {
            StepRule::Force(id) => {
                gen.push(id);
                if id == STOP {
                    break;
                }
            }
            StepRule::Allow(set) => {
                for &id in &set {
                    checked += 1;
                    match table.text(id) {
                        None => assert!(offers_eof(check, &prog), "stop without eof"),
                        Some(t) if t.contains(&b'`') => {
                            assert!(offers_eof(check, &prog), "fence {t:?} without eof")
                        }
                        Some(t) => {
                            let mut p = prog.clone();
                            p.extend_from_slice(t);
                            assert!(
                                matches!(check.next(&p).unwrap(), Answer::Items(_)),
                                "token {:?} allowed after {:?} leads nowhere",
                                String::from_utf8_lossy(t),
                                String::from_utf8_lossy(&prog)
                            );
                        }
                    }
                }
                let pick = if rest.is_empty() {
                    set.iter()
                        .copied()
                        .find(|&id| table.text(id).is_none_or(|t| t.contains(&b'`')))
                        .expect("eof offers the closing")
                } else {
                    set.iter()
                        .copied()
                        .filter(|&id| table.text(id).is_some_and(|t| rest.starts_with(t)))
                        .max_by_key(|&id| table.text(id).unwrap().len())
                        .unwrap_or_else(|| panic!("no allowed token writes {rest:?}"))
                };
                gen.push(pick);
                if pick == STOP {
                    break;
                }
            }
            StepRule::Finish => break,
            StepRule::Any => panic!("Any inside a masked body"),
        }
    }
    (gen, checked)
}

/// Every token the mask allows keeps the program a prefix of the target, in every framing, for a
/// target that ends on a newline and one that does not (eof right after `end`).
#[test]
fn every_allowed_token_keeps_the_target_reachable() {
    let table = Arc::new(synthetic_table());
    let mut total = 0;
    for target in [TARGET, TARGET.trim_end()] {
        for (fence, header) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut k = OmegeseConstraint::new(
                table.clone(),
                ExactText::new(target),
                framing(fence, header),
            )
            .unwrap();
            let mut check = ExactText::new(target);
            let (gen, checked) = drive(&table, &mut k, target, &mut check);
            total += checked;
            assert_eq!(k.program(), target.as_bytes());
            assert_eq!(*gen.last().unwrap(), STOP);
            let reply = table.decode(&gen[..gen.len() - 1]).unwrap();
            let want = if fence {
                let nl = if target.ends_with('\n') { "" } else { "\n" };
                format!("```\n{target}{nl}```")
            } else {
                target.to_string()
            };
            assert_eq!(String::from_utf8(reply).unwrap(), want, "{fence} {header}");
            // Without a fence, after `end\n` the stop token is one choice among others (more blank
            // space may follow): the constraint cannot tell a consumed stop from a cut by length
            // after another token, so it claims the reply finished only when the stop was forced
            // (after a bare `end` nothing else may follow).
            let forced_stop = !target.ends_with('\n');
            assert_eq!(k.finished(), fence || forced_stop, "{fence} {header}");
        }
    }
    assert!(total > 100, "{total} tokens checked");
}

/// With no token that both starts the closing text and marks it, eof forces the closing: a
/// newline (the program ends on `end`), the fence one backtick at a time, then the stop token.
#[test]
fn the_closing_is_forced_when_no_token_marks_it() {
    let mut texts = synthetic_texts();
    for t in texts.iter_mut() {
        if t.as_deref() == Some(b"```") || t.as_deref() == Some(b"\n```") {
            *t = Some(b"zz".to_vec());
        }
    }
    let table = Arc::new(TokenTable::from_texts(texts));
    let target = TARGET.trim_end();
    let mut k =
        OmegeseConstraint::new(table.clone(), ExactText::new(target), framing(true, true)).unwrap();
    let mut check = ExactText::new(target);
    let (gen, _) = drive(&table, &mut k, target, &mut check);
    let tail: Vec<u32> = ["\n", "`", "`", "`"]
        .iter()
        .map(|s| table.encode(s).unwrap()[0])
        .chain([STOP])
        .collect();
    assert_eq!(&gen[gen.len() - 5..], &tail[..]);
    let reply = table.decode(&gen[..gen.len() - 1]).unwrap();
    assert_eq!(reply, format!("```\n{target}\n```").into_bytes());
}

/// A header the oracle refuses is an error when the constraint is built, before anything is
/// written; with the mask off the header is not asked about.
#[test]
fn a_header_the_oracle_refuses_fails_at_construction() {
    let table = Arc::new(synthetic_table());
    let other = "fn g a uint to uint\nret a\nend\n";
    let e = OmegeseConstraint::new(table.clone(), ExactText::new(other), framing(true, true))
        .err()
        .expect("the header is refused");
    assert!(e.contains("refuses the header"), "{e}");
    let mut unmasked = framing(true, true);
    unmasked.mask_body = false;
    let k = OmegeseConstraint::new(table, ExactText::new(other), unmasked).unwrap();
    assert_eq!(k.oracle().calls, 0);
}
