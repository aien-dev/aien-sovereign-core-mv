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

//! TRN1 shared corpus through the Rust reader (spec section 9).
//!
//! TRN1_VECTORS points at aien-protocols `specs/execution-transcript/vectors`
//! (default: the sibling checkout `../aien-protocols` next to this repo).
//! With TRN1_REQUIRE=1 a missing corpus is a failure; otherwise the test
//! prints NOT_RUN and passes, so a plain local `cargo test` does not need the
//! sibling checkout.

use aien_replay::{compare_line, verdict_line, verify, Verdict};
use std::path::{Path, PathBuf};

fn vectors_dir() -> PathBuf {
    std::env::var_os("TRN1_VECTORS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../aien-protocols/specs/execution-transcript/vectors")
        })
}

fn load(dir: &Path, name: &str) -> Verdict {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
    verify(&bytes)
}

fn lines(dir: &Path, manifest: &str) -> Vec<String> {
    std::fs::read_to_string(dir.join(manifest))
        .unwrap_or_else(|e| panic!("read {manifest}: {e}"))
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect()
}

#[test]
fn trn1_shared_corpus() {
    let dir = vectors_dir();
    if !dir.join("expected.txt").exists() {
        assert!(
            std::env::var("TRN1_REQUIRE").as_deref() != Ok("1"),
            "TRN1_REQUIRE=1 but no corpus at {}",
            dir.display()
        );
        println!(
            "TRN1_CONFORMANCE impl=rust NOT_RUN (no corpus at {})",
            dir.display()
        );
        return;
    }

    let (mut pass, mut fail) = (0u32, 0u32);
    for line in lines(&dir, "expected.txt") {
        let (file, want) = line.split_once(' ').expect("<file> <verdict>");
        let got = verdict_line(&load(&dir, file));
        if got == want {
            pass += 1;
        } else {
            fail += 1;
            println!("MISMATCH {file}\n  expected: {want}\n  got:      {got}");
        }
    }
    for line in lines(&dir, "compare.txt") {
        let mut parts = line.splitn(3, ' ');
        let (e, a, want) = (
            parts.next().expect("expected file"),
            parts.next().expect("actual file"),
            parts.next().expect("compare line"),
        );
        let got = compare_line(&load(&dir, e), &load(&dir, a));
        if got == want {
            pass += 1;
        } else {
            fail += 1;
            println!("MISMATCH {e} {a}\n  expected: {want}\n  got:      {got}");
        }
    }
    println!("TRN1_CONFORMANCE impl=rust pass={pass} fail={fail}");
    assert!(pass > 0 && fail == 0, "pass={pass} fail={fail}");
}
