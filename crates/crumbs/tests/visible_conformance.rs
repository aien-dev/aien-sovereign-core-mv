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

//! CRB1 shared conformance: the Rust reference reader against the
//! language-neutral corpus in aien-protocols (specs/crumb-visible).
//!
//! Every vector listed in `expected.txt` is decoded with
//! `VisibleCrumb::from_bytes`; the outcome is printed in the contract's line
//! format and compared byte for byte. Refusals are compared by the contract's
//! numbered code, mapped here at the edge so no Rust type crosses the contract.
//!
//! Corpus location: `$CRB1_VECTORS`, else the sibling checkout
//! `../aien-protocols/specs/crumb-visible/vectors` next to this repository.
//! When the corpus is absent the test reports NOT_RUN, unless
//! `CRB1_REQUIRE=1` (set by the dedicated CI job), in which case it fails.

use crumbs::visible::{VisibleCrumb, VisibleError};
use std::path::PathBuf;

/// Contract refusal codes (CRUMB_READER_CONTRACT.md, section "Refusal codes").
fn contract_code(e: VisibleError) -> i32 {
    match e {
        VisibleError::Magic => -1,
        VisibleError::Version => -2,
        VisibleError::Shape => -3,
        VisibleError::Lane => -4,
        VisibleError::Length => -5,
        VisibleError::NonCanonical => -6,
    }
}

/// Contract outcome line (without the leading file name).
fn outcome(bytes: &[u8]) -> String {
    let c = match VisibleCrumb::from_bytes(bytes) {
        Ok(c) => c,
        Err(e) => return format!("refuse {}", contract_code(e)),
    };
    let mut line = format!(
        "ok enc={} in={}x{} out={}x{} n={}",
        c.encoding as u8,
        c.in_arity,
        c.in_lane_bytes,
        c.out_arity,
        c.out_lane_bytes,
        c.examples.len()
    );
    if let Some(b) = c.budget {
        line.push_str(&format!(
            " budget={},{},{},{}",
            b.max_candidates, b.max_depth, b.max_oracle_queries, b.max_program_ops
        ));
    }
    for (i, o) in c.values() {
        line.push_str(" |");
        for v in i {
            line.push_str(&format!(" {v}"));
        }
        line.push_str(" ->");
        for v in o {
            line.push_str(&format!(" {v}"));
        }
    }
    line
}

fn corpus_dir() -> PathBuf {
    match std::env::var_os("CRB1_VECTORS") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../aien-protocols/specs/crumb-visible/vectors"),
    }
}

#[test]
fn crumb_visible_v1_conformance() {
    let dir = corpus_dir();
    let manifest = dir.join("expected.txt");
    let Ok(text) = std::fs::read_to_string(&manifest) else {
        let required = std::env::var("CRB1_REQUIRE").is_ok_and(|v| v == "1");
        println!(
            "CRUMB_VISIBLE_V1_CONFORMANCE impl=rust NOT_RUN (no corpus at {})",
            manifest.display()
        );
        assert!(!required, "CRB1_REQUIRE=1 but the corpus is missing");
        return;
    };
    let (mut pass, mut fail) = (0usize, 0usize);
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, want) = line
            .split_once(' ')
            .expect("manifest line: <file> <outcome>");
        let bytes = std::fs::read(dir.join(name)).expect("vector file listed in manifest");
        // Determinism: two decodes of the same bytes give the same line.
        let got = outcome(&bytes);
        assert_eq!(got, outcome(&bytes), "non-deterministic outcome for {name}");
        if got == want {
            pass += 1;
        } else {
            fail += 1;
            println!("MISMATCH {name}\n  want: {want}\n  got:  {got}");
        }
    }
    println!("CRUMB_VISIBLE_V1_CONFORMANCE impl=rust pass={pass} fail={fail}");
    assert!(pass > 0, "empty corpus");
    assert_eq!(fail, 0);
}
