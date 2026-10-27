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

use colored::*;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub path: PathBuf,
    pub content: String,
}

pub fn discover_rules(cwd: &Path) -> Vec<Rule> {
    let mut rules = Vec::new();
    let mut current = Some(cwd.to_path_buf());
    let mut seen_paths = HashSet::new();

    while let Some(dir) = current {
        // 1. Check AGENTS.md
        let agents_md = dir.join("AGENTS.md");
        if agents_md.is_file() {
            if let Ok(canon) = agents_md.canonicalize() {
                if seen_paths.insert(canon.clone()) {
                    if let Ok(content) = fs::read_to_string(&canon) {
                        rules.push(Rule {
                            name: format!("AGENTS.md ({})", dir.display()),
                            path: canon,
                            content,
                        });
                    }
                }
            }
        }

        // 2. Check GEMINI.md
        let gemini_md = dir.join("GEMINI.md");
        if gemini_md.is_file() {
            if let Ok(canon) = gemini_md.canonicalize() {
                if seen_paths.insert(canon.clone()) {
                    if let Ok(content) = fs::read_to_string(&canon) {
                        rules.push(Rule {
                            name: format!("GEMINI.md ({})", dir.display()),
                            path: canon,
                            content,
                        });
                    }
                }
            }
        }

        // 3. Check .agents/rules/*.md
        let rules_dir = dir.join(".agents/rules");
        if rules_dir.is_dir() {
            if let Ok(entries) = fs::read_dir(&rules_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
                        if let Ok(canon) = path.canonicalize() {
                            if seen_paths.insert(canon.clone()) {
                                if let Ok(content) = fs::read_to_string(&canon) {
                                    let fname = canon
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| "rule".to_string());
                                    rules.push(Rule {
                                        name: format!("{} ({})", fname, dir.display()),
                                        path: canon,
                                        content,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Stop ascending if we reached git repository root or root filesystem
        if dir.join(".git").exists() || dir.parent().is_none() {
            break;
        }
        current = dir.parent().map(|p| p.to_path_buf());
    }

    rules
}

pub fn format_rules_for_prompt(rules: &[Rule]) -> String {
    if rules.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "
ACTIVE REPOSITORY & WORKSPACE RULES:
",
    );
    for r in rules {
        out.push_str(&format!(
            "
[Rule: {}]
",
            r.name
        ));
        out.push_str(&r.content);
        out.push_str(
            "
[/Rule]
",
        );
    }
    out
}

pub fn format_rules_tui(cwd: &Path) -> String {
    let rules = discover_rules(cwd);
    let mut out = String::new();
    out.push_str(&format!(
        "{}
",
        "=== Active Antigravity Repository Rules ===".cyan().bold()
    ));
    out.push_str(&format!(
        "Working Directory: {}

",
        cwd.display().to_string().dimmed()
    ));

    if rules.is_empty() {
        out.push_str(&format!(
            "{}
",
            "No active rules discovered in current working tree.".dimmed()
        ));
        return out;
    }

    for (i, r) in rules.iter().enumerate() {
        out.push_str(&format!(
            "{}. {} ({})
",
            i + 1,
            r.name.green().bold(),
            r.path.display().to_string().dimmed()
        ));
    }
    out
}
