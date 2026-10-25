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
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const SKILL_METADATA_BYTE_LIMIT: u64 = 16 * 1024;
const PREVIEW_BYTE_LIMIT: u64 = 32 * 1024;
const SEARCH_BYTE_LIMIT: u64 = 256 * 1024;
const DEFAULT_PREVIEW_TOKENS: usize = 96;
const MAX_PREVIEW_TOKENS: usize = 256;
const DEFAULT_SEARCH_SNIPPET_TOKENS: usize = 80;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub path: String,
    pub has_scripts: bool,
}

fn extract_frontmatter_field(content: &str, field: &str) -> Option<String> {
    let mut in_frontmatter = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "---" {
            if in_frontmatter {
                break;
            } else {
                in_frontmatter = true;
                continue;
            }
        }
        if in_frontmatter {
            if let Some(rest) = trimmed.strip_prefix(&format!("{}:", field)) {
                return Some(
                    rest.trim()
                        .trim_matches('"')
                        .trim_matches('\x27')
                        .to_string(),
                );
            }
        }
    }
    None
}

fn read_prefix(path: &Path, byte_limit: u64) -> String {
    read_capped(path, byte_limit)
        .map(|(text, _)| text)
        .unwrap_or_default()
}

fn read_capped(path: &Path, byte_limit: u64) -> Result<(String, bool), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let mut bytes = Vec::new();
    file.take(byte_limit)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    Ok((
        String::from_utf8_lossy(&bytes).into_owned(),
        len > byte_limit,
    ))
}

fn body_without_frontmatter(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("---") else {
        return content;
    };
    let Some(end) = rest.find("\n---") else {
        return content;
    };
    rest[end + 4..].trim_start_matches(['\r', '\n'])
}

fn truncate_tokens(text: &str, token_limit: usize) -> (String, bool) {
    let limit = token_limit.max(1);
    let mut words = text.split_whitespace();
    let selected: Vec<&str> = words.by_ref().take(limit).collect();
    let truncated = words.next().is_some();
    (selected.join(" "), truncated)
}

fn first_body_paragraph(content: &str) -> &str {
    let body = body_without_frontmatter(content);
    body.split("\n\n")
        .map(str::trim)
        .find(|paragraph| {
            !paragraph.is_empty()
                && !paragraph
                    .lines()
                    .all(|line| line.trim_start().starts_with('#'))
        })
        .unwrap_or("")
}

fn scan_dir_for_skills(dir: &Path, skills: &mut Vec<Skill>, seen_names: &mut HashSet<String>) {
    if !dir.is_dir() {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_md = path.join("SKILL.md");
                if skill_md.is_file() {
                    let content = read_prefix(&skill_md, SKILL_METADATA_BYTE_LIMIT);
                    let name = extract_frontmatter_field(&content, "name").unwrap_or_else(|| {
                        path.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string()
                    });
                    let norm_name = name.to_lowercase();
                    if seen_names.insert(norm_name) {
                        let desc = extract_frontmatter_field(&content, "description")
                            .unwrap_or_else(|| "No description provided.".to_string());
                        let has_scripts = path.join("scripts").is_dir();
                        skills.push(Skill {
                            name,
                            description: desc,
                            path: skill_md.display().to_string(),
                            has_scripts,
                        });
                    }
                } else {
                    // Check one level deeper (e.g. vendor-skills/benchmark-model)
                    if let Ok(sub_entries) = fs::read_dir(&path) {
                        for sub_entry in sub_entries.flatten() {
                            let sub_path = sub_entry.path();
                            if sub_path.is_dir() {
                                let sub_skill_md = sub_path.join("SKILL.md");
                                if sub_skill_md.is_file() {
                                    let content =
                                        read_prefix(&sub_skill_md, SKILL_METADATA_BYTE_LIMIT);
                                    let name = extract_frontmatter_field(&content, "name")
                                        .unwrap_or_else(|| {
                                            sub_path
                                                .file_name()
                                                .unwrap_or_default()
                                                .to_string_lossy()
                                                .to_string()
                                        });
                                    let norm_name = name.to_lowercase();
                                    if seen_names.insert(norm_name) {
                                        let desc =
                                            extract_frontmatter_field(&content, "description")
                                                .unwrap_or_else(|| {
                                                    "No description provided.".to_string()
                                                });
                                        let has_scripts = sub_path.join("scripts").is_dir();
                                        skills.push(Skill {
                                            name,
                                            description: desc,
                                            path: sub_skill_md.display().to_string(),
                                            has_scripts,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn discover_skills() -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen_names = HashSet::new();
    let platform = crate::platform::PlatformContext::detect();

    // Priority 1: Current workspace .agents/skills and skills
    let cwd = std::env::current_dir().unwrap_or_else(|_| platform.home_dir.clone());
    scan_dir_for_skills(&cwd.join(".agents/skills"), &mut skills, &mut seen_names);
    scan_dir_for_skills(&cwd.join("skills"), &mut skills, &mut seen_names);

    // Priority 2: Home skills (~/skills)
    let home_skills = platform.home_dir.join("skills");
    scan_dir_for_skills(&home_skills, &mut skills, &mut seen_names);

    // Priority 3: Home .agents/skills (~/.agents/skills)
    let home_agents_skills = platform.home_dir.join(".agents/skills");
    scan_dir_for_skills(&home_agents_skills, &mut skills, &mut seen_names);

    // Priority 4: Gemini global config skills (~/.gemini/config/skills)
    let gemini_skills = platform.home_dir.join(".gemini/config/skills");
    scan_dir_for_skills(&gemini_skills, &mut skills, &mut seen_names);

    skills
}

pub fn format_skills_progressive_summary() -> String {
    let skills = discover_skills();
    if skills.is_empty() {
        return String::new();
    }

    let mut out = String::from("\nAGENT SKILLS (CPU-filtered progressive disclosure):\n");
    out.push_str(&format!(
        "{} skills are indexed locally. Use skill discover for a task, preview one matching skill, search inside that skill for details, and request full only when necessary. Never load multiple full skills into one turn.\n",
        skills.len()
    ));
    out
}

pub fn format_skills_tui() -> String {
    let skills = discover_skills();
    let mut out = String::new();
    out.push_str(&format!(
        "{}\n",
        "=== AIEN Antigravity Skill Registry ===".cyan().bold()
    ));

    if skills.is_empty() {
        out.push_str(&format!(
            "{}\n",
            "No skills currently discovered. Create a folder with SKILL.md in ~/skills/ or .agents/skills/ to register one.".dimmed()
        ));
        return out;
    }

    for (i, s) in skills.iter().enumerate() {
        let scripts_badge = if s.has_scripts {
            "[scripts]".yellow()
        } else {
            "".normal()
        };
        out.push_str(&format!(
            "{}. {} {} ({})\n",
            i + 1,
            s.name.bold().green(),
            scripts_badge,
            s.path.dimmed()
        ));
        out.push_str(&format!("   {}\n\n", s.description.dimmed()));
    }

    out
}

pub fn read_skill_content(name: &str) -> Result<String, String> {
    let skills = discover_skills();
    let norm = name.trim().to_lowercase();

    for s in skills {
        if s.name.to_lowercase() == norm {
            return fs::read_to_string(&s.path).map_err(|e| e.to_string());
        }
    }
    Err(format!(
        "Skill '{}' not found in discovered skill paths",
        name
    ))
}

fn find_skill(name: &str) -> Result<Skill, String> {
    let norm = name.trim().to_lowercase();
    discover_skills()
        .into_iter()
        .find(|skill| skill.name.to_lowercase() == norm)
        .ok_or_else(|| format!("Skill '{}' not found in discovered skill paths", name))
}

fn preview_from_content(content: &str, token_limit: usize) -> (String, usize, bool) {
    let paragraph = first_body_paragraph(content);
    let limit = token_limit.clamp(1, MAX_PREVIEW_TOKENS);
    let (preview, truncated) = truncate_tokens(paragraph, limit);
    (preview, limit, truncated)
}

pub fn preview_skill(name: &str, token_limit: usize) -> Result<Value, String> {
    let skill = find_skill(name)?;
    let (content, scan_truncated) = read_capped(Path::new(&skill.path), PREVIEW_BYTE_LIMIT)?;
    let (preview, limit, truncated) = preview_from_content(&content, token_limit);
    Ok(json!({
        "status": "ok",
        "mode": "preview",
        "name": skill.name,
        "description": skill.description,
        "preview": preview,
        "token_limit": limit,
        "truncated": truncated || scan_truncated,
        "next": "Use skill search for a targeted passage or skill full for explicit full context."
    }))
}

fn search_paragraphs(content: &str, terms: &[String], limit: usize) -> Vec<Value> {
    let mut ranked: Vec<(usize, usize, String)> = body_without_frontmatter(content)
        .split("\n\n")
        .enumerate()
        .filter_map(|(index, paragraph)| {
            let paragraph = paragraph.trim();
            let lower = paragraph.to_lowercase();
            let score = terms
                .iter()
                .map(|term| lower.matches(term).count())
                .sum::<usize>();
            (score > 0).then(|| {
                let (snippet, truncated) =
                    truncate_tokens(paragraph, DEFAULT_SEARCH_SNIPPET_TOKENS);
                let suffix = if truncated { " ..." } else { "" };
                (score, index, format!("{snippet}{suffix}"))
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    ranked
        .into_iter()
        .take(limit.clamp(1, 8))
        .map(|(score, paragraph, snippet)| {
            json!({"score": score, "paragraph": paragraph, "snippet": snippet})
        })
        .collect()
}

fn discover_matching_skills(query: &str, limit: usize) -> Vec<Value> {
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|term| term.to_lowercase())
        .filter(|term| term.len() > 1)
        .collect();
    let mut ranked: Vec<(usize, Skill)> = discover_skills()
        .into_iter()
        .filter_map(|skill| {
            let name = skill.name.to_lowercase();
            let description = skill.description.to_lowercase();
            let score = terms
                .iter()
                .map(|term| {
                    usize::from(name.contains(term)) * 4 + usize::from(description.contains(term))
                })
                .sum::<usize>();
            (score > 0).then_some((score, skill))
        })
        .collect();
    ranked.sort_by(|(score_a, skill_a), (score_b, skill_b)| {
        score_b
            .cmp(score_a)
            .then_with(|| skill_a.name.cmp(&skill_b.name))
    });
    ranked
        .into_iter()
        .take(limit.clamp(1, 10))
        .map(|(score, skill)| {
            let (description, _) = truncate_tokens(&skill.description, 32);
            json!({"name": skill.name, "description": description, "score": score})
        })
        .collect()
}

pub fn search_skill(name: &str, query: &str, limit: usize) -> Result<Value, String> {
    let skill = find_skill(name)?;
    let terms: Vec<String> = query
        .split_whitespace()
        .map(|term| term.to_lowercase())
        .filter(|term| term.len() > 1)
        .collect();
    if terms.is_empty() {
        return Err("Skill search requires a non-empty query".to_string());
    }

    let (content, scan_truncated) = read_capped(Path::new(&skill.path), SEARCH_BYTE_LIMIT)?;
    let matches = search_paragraphs(&content, &terms, limit);
    Ok(json!({
        "status": "ok",
        "mode": "search",
        "name": skill.name,
        "query": query,
        "matches": matches,
        "scan_truncated": scan_truncated
    }))
}

pub fn skills_dispatch_tool(args: &Value) -> Value {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("list");

    match action {
        "list" => {
            let skills = discover_skills();
            json!({
                "status": "ok",
                "total": skills.len(),
                "skills": skills.into_iter().map(|skill| skill.name).collect::<Vec<_>>()
            })
        }
        "discover" | "match" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(5) as usize;
            json!({
                "status": "ok",
                "query": query,
                "skills": discover_matching_skills(query, limit)
            })
        }
        "read" | "load" | "preview" => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("");
            let token_limit = args
                .get("max_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(DEFAULT_PREVIEW_TOKENS as u64) as usize;
            match preview_skill(name, token_limit) {
                Ok(preview) => preview,
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "search" => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("");
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(3) as usize;
            match search_skill(name, query, limit) {
                Ok(results) => results,
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "full" => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("");
            match read_skill_content(name) {
                Ok(content) => json!({
                    "status": "ok",
                    "mode": "full",
                    "name": name,
                    "content": content
                }),
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "optimize" => {
            let name = args
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("atlas-skillopt");
            optimize_skill(name)
        }
        _ => json!({"error": format!("Unknown skill action '{}'", action)}),
    }
}

pub fn optimize_skill(name: &str) -> Value {
    let platform = crate::platform::PlatformContext::detect();
    let script = platform.home_dir.join("atlas-skillopt-stage.sh");
    if !script.exists() {
        return json!({"status": "error", "error": "atlas-skillopt-stage.sh not found"});
    }
    let output = std::process::Command::new("bash").arg(&script).output();
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            json!({
                "status": if out.status.success() { "ok" } else { "failed" },
                "target_skill": name,
                "stdout": stdout,
                "stderr": stderr
            })
        }
        Err(e) => json!({"status": "error", "error": e.to_string()}),
    }
}

pub fn run_optimize_cli(name: &str) {
    println!(
        "{}",
        format!(
            "Starting Microsoft SkillOpt self-improvement loop for {}...",
            name
        )
        .cyan()
        .bold()
    );
    let res = optimize_skill(name);
    println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_uses_first_body_paragraph_and_budget() {
        let content = "---\nname: huge\ndescription: Large skill\n---\n# Heading\n\nOne two three four five.\n\nSecond paragraph.";
        assert_eq!(first_body_paragraph(content), "One two three four five.");
        let (text, truncated) = truncate_tokens("one two three four", 3);
        assert_eq!(text, "one two three");
        assert!(truncated);
    }

    #[test]
    fn frontmatter_is_not_returned_as_body() {
        let content = "---\nname: test\n---\nFirst paragraph.\n\nSecond paragraph.";
        assert_eq!(first_body_paragraph(content), "First paragraph.");
    }

    #[test]
    fn preview_stops_at_the_first_paragraph() {
        let content = "---\nname: huge\ndescription: Large skill\n---\n# Heading\n\nAlpha beta gamma.\n\nLater delta epsilon, which must stay out of the preview.";
        let (preview, limit, truncated) = preview_from_content(content, 96);
        assert_eq!(preview, "Alpha beta gamma.");
        assert_eq!(limit, 96);
        assert!(!truncated);
        assert!(!preview.contains("delta"));
    }

    #[test]
    fn search_returns_one_matching_paragraph() {
        let content = "---\nname: huge\n---\nUnrelated opening.\n\nThe reactor uses a graphite moderator.\n\nAnother note about orchards.";
        let terms = vec!["graphite".to_string(), "moderator".to_string()];
        let matches = search_paragraphs(content, &terms, 3);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["paragraph"], 1);
        assert!(matches[0]["snippet"]
            .as_str()
            .unwrap()
            .contains("graphite moderator"));
    }
}
