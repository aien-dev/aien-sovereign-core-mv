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

//! `verify-closure --root <repo> [--store <receipt dir>] [--write-lock]`
//!
//! Exit 0 PASS. Exit 1 internal error. Exit 2 usage error. Otherwise the exit
//! code of the first finding (sorted by code, then component), 10 to 18.

use aien_closure::{verify, Options};
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!("usage: verify-closure --root <repo> [--store <receipt dir>] [--write-lock]");
    ExitCode::from(2)
}

fn default_store() -> PathBuf {
    std::env::var_os("AIEN_PROOF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local/state/aien-proof")
        })
}

fn main() -> ExitCode {
    let mut root = None;
    let mut store = None;
    let mut write_lock = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--root" => root = args.next().map(PathBuf::from),
            "--store" => store = args.next().map(PathBuf::from),
            "--write-lock" => write_lock = true,
            _ => return usage(),
        }
    }
    let Some(root) = root else {
        return usage();
    };
    let opts = Options {
        root,
        store: store.unwrap_or_else(default_store),
        write_lock,
    };
    match verify(&opts) {
        Err(e) => {
            println!(
                "SUMMARY status=ERROR findings=0 components=0 detail={}",
                e.replace('\n', " ")
            );
            ExitCode::from(1)
        }
        Ok(out) => {
            for (name, id) in &out.passed {
                println!("PASS component={name} receipt={id}");
            }
            for f in &out.findings {
                println!("{}", f.line());
            }
            let code = out.findings.first().map_or(0, |f| f.code.exit_code());
            println!(
                "SUMMARY status={} findings={} components={}",
                if out.ok() { "PASS" } else { "FAIL" },
                out.findings.len(),
                out.components
            );
            ExitCode::from(code as u8)
        }
    }
}
