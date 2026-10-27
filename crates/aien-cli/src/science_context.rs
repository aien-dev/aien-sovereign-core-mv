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

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const DEFAULT_SCIENCE_TOKENS: usize = 700;
const MAX_SCIENCE_RESULTS: usize = 4;

#[derive(Debug, Clone)]
struct ScienceNote {
    id: String,
    title: String,
    content: String,
}

#[derive(Debug, Clone, Default)]
struct ScienceIndex {
    revision: String,
    notes: Vec<ScienceNote>,
}

static SCIENCE_INDEX: OnceLock<ScienceIndex> = OnceLock::new();

fn default_repo_path() -> PathBuf {
    crate::platform::PlatformContext::detect()
        .home_dir
        .join("workspace/atlas-forgejo/data/git/repositories/drake/cortex.git")
}

fn read_science_source(repo: &Path) -> Option<(String, String)> {
    if repo.is_file() {
        return fs::read_to_string(repo)
            .ok()
            .map(|content| ("working-tree".to_string(), content));
    }
    if !repo.is_dir() {
        return None;
    }
    let source = Command::new("git")
        .arg(format!("--git-dir={}", repo.display()))
        .args(["show", "HEAD:src/lib/brain-seed.ts"])
        .output()
        .ok()?;
    if !source.status.success() {
        return None;
    }
    let revision = Command::new("git")
        .arg(format!("--git-dir={}", repo.display()))
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    Some((
        revision,
        String::from_utf8_lossy(&source.stdout).into_owned(),
    ))
}

fn parse_notes(source: &str) -> Vec<ScienceNote> {
    let mut notes = Vec::new();
    let mut id = String::new();
    let mut title = String::new();
    let mut string_args = Vec::new();
    let mut in_note = false;
    let mut in_content = false;
    let mut content = String::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if !in_note && trimmed == "note(" {
            in_note = true;
            string_args.clear();
            continue;
        }
        if !in_note {
            continue;
        }
        if !in_content {
            if trimmed.starts_with('`') {
                in_content = true;
                content.clear();
                content.push_str(trimmed.trim_start_matches('`'));
                content.push('\n');
                continue;
            }
            if let Some(value) = trimmed
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix("\","))
            {
                string_args.push(value.to_string());
                if string_args.len() == 1 {
                    id = value.to_string();
                } else if string_args.len() == 2 {
                    title = value.to_string();
                }
            }
            continue;
        }

        if trimmed == "`," {
            if !id.is_empty() && !title.is_empty() && !content.trim().is_empty() {
                notes.push(ScienceNote {
                    id: std::mem::take(&mut id),
                    title: std::mem::take(&mut title),
                    content: content.trim().to_string(),
                });
            }
            in_note = false;
            in_content = false;
            content.clear();
        } else {
            content.push_str(line);
            content.push('\n');
        }
    }
    notes
}

fn index() -> &'static ScienceIndex {
    SCIENCE_INDEX.get_or_init(|| {
        let configured = std::env::var("AIEN_CORTEX_SCIENCE_REPO")
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_repo_path());
        let Some((revision, source)) = read_science_source(&configured) else {
            return ScienceIndex::default();
        };
        ScienceIndex {
            revision,
            notes: parse_notes(&source),
        }
    })
}

fn query_terms(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
        "about", "after", "again", "could", "from", "have", "into", "more", "some", "that",
        "theory", "their", "there", "these", "they", "this", "what", "when", "where", "which",
        "with", "would", "your",
    ];
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '-')
        .map(|term| term.to_lowercase())
        .filter(|term| term.len() >= 3 && !STOP_WORDS.contains(&term.as_str()))
        .collect()
}

fn best_paragraph(note: &ScienceNote, terms: &[String]) -> String {
    note.content
        .split("\n\n")
        .max_by_key(|paragraph| {
            let lower = paragraph.to_lowercase();
            terms
                .iter()
                .map(|term| lower.matches(term).count())
                .sum::<usize>()
        })
        .unwrap_or(&note.content)
        .trim()
        .to_string()
}

fn truncate_words(text: &str, limit: usize) -> String {
    let mut words = text.split_whitespace();
    let selected: Vec<&str> = words.by_ref().take(limit).collect();
    let suffix = if words.next().is_some() { " ..." } else { "" };
    format!("{}{}", selected.join(" "), suffix)
}

pub fn search_foundation_science(query: &str, limit: usize) -> Vec<Value> {
    let terms = query_terms(query);
    if terms.is_empty() {
        return Vec::new();
    }
    let index = index();
    let mut ranked: Vec<(usize, &ScienceNote)> = index
        .notes
        .iter()
        .filter_map(|note| {
            let title = note.title.to_lowercase();
            let content = note.content.to_lowercase();
            let score = terms
                .iter()
                .map(|term| title.matches(term).count() * 8 + content.matches(term).count())
                .sum::<usize>();
            (score > 0).then_some((score, note))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.title.cmp(&b.1.title)));

    let result_limit = limit.clamp(1, MAX_SCIENCE_RESULTS);
    let words_per_result = DEFAULT_SCIENCE_TOKENS / result_limit;
    ranked
        .into_iter()
        .take(result_limit)
        .map(|(score, note)| {
            json!({
                "source": format!("cortex-founding-science:{}@{}", note.id, index.revision),
                "title": note.title,
                "score": score,
                "content": truncate_words(&best_paragraph(note, &terms), words_per_result)
            })
        })
        .collect()
}

pub fn render_foundation_science(query: &str, limit: usize) -> Option<String> {
    let results = search_foundation_science(query, limit);
    if results.is_empty() {
        return None;
    }
    let entries = results
        .iter()
        .map(|item| {
            format!(
                "[{}] {}\n{}",
                item["source"].as_str().unwrap_or("cortex-founding-science"),
                item["title"].as_str().unwrap_or("Untitled"),
                item["content"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    Some(format!(
        "<CORTEX_FOUNDING_SCIENCE>\nRetrieved neuroscience context from the original Cortex repository. Cite the source IDs when used.\n{}\n</CORTEX_FOUNDING_SCIENCE>",
        entries
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seed_notes() {
        let source = r#"
note(
  "ca1",
  "CA1",
  "Structures",
  "structure",
  "detail",
  null,
  ["hippocampus"],
  `# CA1

Comparator circuit for prediction error.
`,
),
"#;
        let notes = parse_notes(source);
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, "ca1");
        assert!(notes[0].content.contains("prediction error"));
    }

    #[test]
    fn query_terms_drop_common_words() {
        assert_eq!(
            query_terms("what is hippocampal prediction"),
            ["hippocampal", "prediction"]
        );
    }
}
