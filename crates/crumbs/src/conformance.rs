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

//! Crumb v1 conformance vectors for other implementations (e.g. Omega's reader).

use crate::gen::{self, registry};
use std::path::Path;

/// Write one visible crumb per family (seed 7) plus an expected-values file.
pub fn write_vectors(dir: &Path) -> i32 {
    if std::fs::create_dir_all(dir).is_err() {
        return 1;
    }
    let mut index = String::new();
    for f in &registry().families {
        let g = gen::generate(f, 7);
        let name = format!("v{:03}.crb", f.id);
        if std::fs::write(dir.join(&name), g.visible.to_bytes()).is_err() {
            return 1;
        }
        let mut line = format!(
            "{name} ok enc={} in={}x{} out={}x{} n={}",
            g.visible.encoding as u8,
            g.visible.in_arity,
            g.visible.in_lane_bytes,
            g.visible.out_arity,
            g.visible.out_lane_bytes,
            g.visible.examples.len()
        );
        for (i, o) in g.visible.values() {
            line.push_str(" |");
            for v in i {
                line.push_str(&format!(" {v}"));
            }
            line.push_str(" ->");
            for v in o {
                line.push_str(&format!(" {v}"));
            }
        }
        index.push_str(&line);
        index.push('\n');
    }
    if std::fs::write(dir.join("expected.txt"), index).is_err() {
        return 1;
    }
    0
}
