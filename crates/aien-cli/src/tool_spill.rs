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

//! sovereign-core #666: oversized tool results spill to files.
//!
//! Mechanism after Hermes Agent (Nous Research, MIT, commit 5480918),
//! re-implemented; no code copied.
//!
//! A tool can return far more text than the model can use (a 1 MB command
//! output, a huge search). Such a result is written whole to a file in the
//! session's spill folder, and the conversation gets a short preview (the first
//! 40 percent and the last 60 percent of a fixed size, with one marker line)
//! plus the file's path, so the model can page the rest instead of running the
//! tool again. Three budgets decide what spills:
//!
//! - **per result:** any single result larger than this spills;
//! - **per turn:** if the results of one model reply together are still
//!   larger, the largest remaining ones spill first until they fit;
//! - **window-scaled:** both budgets are a share of the model's context window
//!   (15 and 30 percent at 4 characters a token), with floors and caps.
//!
//! Reading a file (`view_file`) never spills: a spilled read would be read
//! back and spilled again in a loop. The pure part ([`SpillBudget`],
//! [`choose`], [`preview`], [`render_full`], [`stub`]) does no I/O;
//! [`write_spill`] is the only edge.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const DEFAULT_PER_RESULT_CHARS: usize = 100_000;
pub const DEFAULT_PER_TURN_CHARS: usize = 200_000;
pub const PREVIEW_CHARS: usize = 1_500;
const FLOOR_PER_RESULT_CHARS: usize = 8_000;
const FLOOR_PER_TURN_CHARS: usize = 16_000;
/// Room the stub takes besides its preview (path, counts, field names).
const STUB_OVERHEAD_CHARS: usize = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillBudget {
    pub per_result: usize,
    pub per_turn: usize,
    pub preview: usize,
}

impl SpillBudget {
    /// Budgets for a model window of `window_tokens`.
    pub fn for_window(window_tokens: usize) -> Self {
        let chars = window_tokens.saturating_mul(4);
        Self {
            per_result: (chars * 15 / 100).clamp(FLOOR_PER_RESULT_CHARS, DEFAULT_PER_RESULT_CHARS),
            per_turn: (chars * 30 / 100).clamp(FLOOR_PER_TURN_CHARS, DEFAULT_PER_TURN_CHARS),
            preview: PREVIEW_CHARS,
        }
    }

    fn stub_chars(&self) -> usize {
        self.preview + STUB_OVERHEAD_CHARS
    }
}

/// Tools whose results never spill.
pub fn exempt(tool: &str) -> bool {
    matches!(tool, "view_file")
}

/// Which results spill. `sizes[i]` is result `i`'s size in characters as it
/// would enter the conversation; `exempt[i]` marks results that never spill;
/// `turn_room` is what is left of the per-turn budget. Pure.
pub fn choose(
    sizes: &[usize],
    exempt: &[bool],
    budget: &SpillBudget,
    turn_room: usize,
) -> Vec<bool> {
    let shrinks = |i: usize| !exempt[i] && sizes[i] > budget.stub_chars();
    let mut spill: Vec<bool> = (0..sizes.len())
        .map(|i| shrinks(i) && sizes[i] > budget.per_result)
        .collect();
    let kept = |spill: &[bool]| -> usize {
        (0..sizes.len())
            .map(|i| {
                if spill[i] {
                    budget.stub_chars()
                } else {
                    sizes[i]
                }
            })
            .sum()
    };
    let mut order: Vec<usize> = (0..sizes.len()).filter(|&i| shrinks(i)).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    for i in order {
        if kept(&spill) <= turn_room {
            break;
        }
        spill[i] = true;
    }
    spill
}

/// The first 40 percent and the last 60 percent of `keep` characters, with
/// one marker line between them. Text that already fits is returned as is.
pub fn preview(full: &str, keep: usize) -> String {
    let total = full.chars().count();
    if total <= keep {
        return full.to_string();
    }
    let head_n = keep * 40 / 100;
    let tail_n = keep - head_n;
    let head: String = full.chars().take(head_n).collect();
    let tail: String = full.chars().skip(total - tail_n).collect();
    format!(
        "{head}\n... [TOOL OUTPUT TRUNCATED - {} chars omitted out of {total} total] ...\n{tail}",
        total - head_n - tail_n
    )
}

/// The whole result as a file a person or `view_file` can page: string
/// fields with their real line breaks, one heading each; other fields as JSON.
pub fn render_full(tool: &str, call_id: &str, result: &Value) -> String {
    let mut out =
        format!("# Full result of {tool} ({call_id}). The conversation holds only a preview.\n");
    match result {
        Value::Object(map) => {
            for (k, v) in map {
                match v {
                    Value::String(s) => {
                        out.push_str(&format!("## {k}\n{s}\n"));
                    }
                    other => out.push_str(&format!("## {k}: {other}\n")),
                }
            }
        }
        other => {
            out.push_str(&serde_json::to_string_pretty(other).unwrap_or_default());
            out.push('\n');
        }
    }
    out
}

/// What the conversation gets instead of a spilled result. A top-level
/// `status` is carried over so readers of the status still see it.
pub fn stub(result: &Value, full_chars: usize, path: Option<&Path>, budget: &SpillBudget) -> Value {
    let text = serde_json::to_string(result).unwrap_or_default();
    let mut v = json!({
        "spilled": path.is_some(),
        "original_chars": full_chars,
        "preview": preview(&text, budget.preview),
    });
    match path {
        Some(p) => {
            v["path"] = json!(p.display().to_string());
            v["read_with"] = json!("view_file with start_line and end_line, a page at a time; do not run the tool again");
        }
        None => {
            v["note"] =
                json!("the full result could not be saved to a file; only this preview is left");
        }
    }
    if let Some(status) = result.get("status") {
        v["status"] = status.clone();
    }
    v
}

/// Writes `text` into `dir` under a name taken from its digest, so the same
/// output twice is one file. Whole or not at all; owner-only permissions.
pub fn write_spill(dir: &Path, text: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    let digest = hex::encode(Sha256::digest(text.as_bytes()));
    let path = dir.join(format!("{}.txt", &digest[..16]));
    if path.is_file() {
        return Ok(path);
    }
    let tmp = dir.join(format!("{}.txt.tmp", &digest[..16]));
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        use std::io::Write;
        let mut f = opts.open(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_scale_with_the_window_inside_floors_and_caps() {
        let small = SpillBudget::for_window(8_000);
        assert_eq!((small.per_result, small.per_turn), (8_000, 16_000));
        let mid = SpillBudget::for_window(24_000);
        assert_eq!((mid.per_result, mid.per_turn), (14_400, 28_800));
        let big = SpillBudget::for_window(1_000_000);
        assert_eq!((big.per_result, big.per_turn), (100_000, 200_000));
    }

    #[test]
    fn a_result_over_the_per_result_budget_spills() {
        let b = SpillBudget::for_window(24_000);
        assert_eq!(
            choose(&[20_000, 100], &[false, false], &b, b.per_turn),
            vec![true, false]
        );
    }

    #[test]
    fn the_turn_budget_spills_the_largest_first() {
        let b = SpillBudget {
            per_result: 50_000,
            per_turn: 15_000,
            preview: 1_500,
        };
        // 10k + 9k + 2k = 21k > 15k: spilling the 10k alone brings it to ~12.9k.
        assert_eq!(
            choose(
                &[9_000, 10_000, 2_000],
                &[false, false, false],
                &b,
                b.per_turn
            ),
            vec![false, true, false]
        );
    }

    #[test]
    fn file_reads_never_spill() {
        let b = SpillBudget::for_window(24_000);
        assert_eq!(choose(&[1_000_000], &[true], &b, b.per_turn), vec![false]);
        assert!(exempt("view_file") && !exempt("run_command"));
    }

    #[test]
    fn preview_keeps_40_percent_head_and_60_percent_tail() {
        let text: String = (0..1000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let p = preview(&text, 100);
        let (head, rest) = p.split_once('\n').unwrap();
        let (marker, tail) = rest.split_once('\n').unwrap();
        assert_eq!(head, &text[..40]);
        assert_eq!(tail, &text[940..]);
        assert!(
            marker.contains("900 chars omitted out of 1000 total"),
            "{marker}"
        );
        assert_eq!(preview("short", 100), "short");
        // Multibyte text is cut on character boundaries.
        assert!(preview(&"\u{1F30A}".repeat(500), 10)
            .starts_with("\u{1F30A}\u{1F30A}\u{1F30A}\u{1F30A}\n"));
    }

    #[test]
    fn the_same_output_twice_is_one_file() {
        let d = tempfile::tempdir().unwrap();
        let a = write_spill(d.path(), "same").unwrap();
        let b = write_spill(d.path(), "same").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "same");
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 1);
    }

    #[test]
    fn the_full_file_keeps_real_line_breaks() {
        let r = json!({"stdout": "one\ntwo\n", "exit_code": 0});
        let f = render_full("run_command", "call-3", &r);
        assert!(f.contains("## stdout\none\ntwo\n"), "{f}");
        assert!(f.contains("## exit_code: 0"), "{f}");
    }
}
