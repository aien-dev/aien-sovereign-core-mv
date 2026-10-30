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

use chrono::Utc;
use colored::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalStatus {
    Active,
    Completed,
    Paused,
}

impl std::fmt::Display for GoalStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GoalStatus::Active => write!(f, "Active"),
            GoalStatus::Completed => write!(f, "Completed"),
            GoalStatus::Paused => write!(f, "Paused"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub id: usize,
    pub description: String,
    pub completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: GoalStatus,
    pub milestones: Vec<Milestone>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalManifest {
    pub project_name: String,
    pub goals: Vec<Goal>,
}

pub fn get_goals_path() -> PathBuf {
    let local = Path::new(".goals.json");
    if local.exists() {
        return local.to_path_buf();
    }
    crate::platform::PlatformContext::detect()
        .home_dir
        .join("basecamp/goals.json")
}

pub fn load_goals() -> GoalManifest {
    let path = get_goals_path();
    if path.exists() {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(manifest) = serde_json::from_str::<GoalManifest>(&content) {
                return manifest;
            }
        }
    }
    GoalManifest {
        project_name: "Sovereign Workspace".to_string(),
        goals: Vec::new(),
    }
}

pub fn save_goals(manifest: &GoalManifest) -> Result<PathBuf, String> {
    let path = get_goals_path();
    let json_str = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    fs::write(&path, json_str).map_err(|e| e.to_string())?;
    Ok(path)
}

pub fn add_goal(title: &str, description: &str, milestones_raw: Vec<String>) -> Goal {
    let mut manifest = load_goals();
    let id = format!("goal-{}", Utc::now().timestamp());

    let milestones: Vec<Milestone> = milestones_raw
        .into_iter()
        .enumerate()
        .map(|(idx, d)| Milestone {
            id: idx + 1,
            description: d,
            completed: false,
        })
        .collect();

    let now_str = Utc::now().to_rfc3339();
    let goal = Goal {
        id: id.clone(),
        title: title.to_string(),
        description: description.to_string(),
        status: GoalStatus::Active,
        milestones,
        created_at: now_str.clone(),
        updated_at: now_str,
    };

    manifest.goals.push(goal.clone());
    let _ = save_goals(&manifest);
    goal
}

pub fn complete_goal(target_id: &str) -> Result<String, String> {
    complete_goal_with(target_id, load_goals(), save_goals)
}

fn complete_goal_with(
    target_id: &str, mut manifest: GoalManifest,
    save: impl FnOnce(&GoalManifest) -> Result<PathBuf, String>,
) -> Result<String, String> {
    let mut found = false;
    let mut title = String::new();

    for g in manifest.goals.iter_mut() {
        if g.id == target_id || g.title.to_lowercase().contains(&target_id.to_lowercase()) {
            g.status = GoalStatus::Completed;
            for m in g.milestones.iter_mut() {
                m.completed = true;
            }
            g.updated_at = Utc::now().to_rfc3339();
            title = g.title.clone();
            found = true;
            break;
        }
    }

    if found {
        save(&manifest)?;
        Ok(format!(
            "Goal '{}' ({}) marked as Completed.",
            title, target_id
        ))
    } else {
        Err(format!("Goal '{}' not found in manifest.", target_id))
    }
}

pub fn complete_milestone(target_id: &str, milestone_id: usize) -> Result<String, String> {
    complete_milestone_with(target_id, milestone_id, load_goals(), save_goals)
}

fn complete_milestone_with(
    target_id: &str, milestone_id: usize, mut manifest: GoalManifest,
    save: impl FnOnce(&GoalManifest) -> Result<PathBuf, String>,
) -> Result<String, String> {
    let mut found = false;
    let mut all_completed = false;

    for g in manifest.goals.iter_mut() {
        if g.id == target_id || g.title.to_lowercase().contains(&target_id.to_lowercase()) {
            for m in g.milestones.iter_mut() {
                if m.id == milestone_id {
                    m.completed = true;
                    found = true;
                }
            }
            g.updated_at = Utc::now().to_rfc3339();
            if g.milestones.iter().all(|m| m.completed) {
                g.status = GoalStatus::Completed;
                all_completed = true;
            }
            break;
        }
    }

    if found {
        save(&manifest)?;
        if all_completed {
            Ok(format!(
                "Milestone {} completed. All milestones finished! Goal is now Completed.",
                milestone_id
            ))
        } else {
            Ok(format!("Milestone {} completed.", milestone_id))
        }
    } else {
        Err(format!(
            "Milestone {} for goal '{}' not found.",
            milestone_id, target_id
        ))
    }
}

pub fn format_goals_tui() -> String {
    let manifest = load_goals();
    let mut out = String::new();

    out.push_str(&format!(
        "{}\n",
        format!("=== Project Goals: {} ===", manifest.project_name)
            .cyan()
            .bold()
    ));
    if manifest.goals.is_empty() {
        out.push_str(&format!(
            "{}\n",
            "No active goals in manifest. Use '/goal new <title>' to create one.".dimmed()
        ));
        return out;
    }

    for (i, g) in manifest.goals.iter().enumerate() {
        let status_colored = match g.status {
            GoalStatus::Active => "[ACTIVE]".green().bold(),
            GoalStatus::Completed => "[DONE]".dimmed(),
            GoalStatus::Paused => "[PAUSED]".yellow(),
        };

        out.push_str(&format!(
            "{}. {} {} ({})\n",
            i + 1,
            status_colored,
            g.title.bold(),
            g.id.cyan()
        ));
        if !g.description.is_empty() {
            out.push_str(&format!("   {}\n", g.description.dimmed()));
        }

        for m in &g.milestones {
            let mark = if m.completed {
                "✓".green()
            } else {
                "○".dimmed()
            };
            out.push_str(&format!("     {} {}. {}\n", mark, m.id, m.description));
        }
        out.push('\n');
    }

    out
}

pub fn goals_dispatch_tool(args: &Value) -> Value {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("list");

    match action {
        "milestone_done" | "advance" => {
            let id = args.get("id").and_then(Value::as_str).unwrap_or("");
            let m_id = args
                .get("milestone_id")
                .and_then(Value::as_u64)
                .unwrap_or(1) as usize;
            match complete_milestone(id, m_id) {
                Ok(msg) => json!({"status": "ok", "message": msg}),
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "new" | "create" => {
            let title = args
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled Goal");
            let desc = args
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let milestones: Vec<String> = args
                .get("milestones")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default();

            let g = add_goal(title, desc, milestones);
            json!({
                "status": "ok",
                "goal": g,
                "message": format!("Goal '{}' initialized successfully.", title)
            })
        }
        "done" | "complete" => {
            let id = args.get("id").and_then(Value::as_str).unwrap_or("");
            match complete_goal(id) {
                Ok(msg) => json!({"status": "ok", "message": msg}),
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "list" => {
            let manifest = load_goals();
            json!({
                "status": "ok",
                "project": manifest.project_name,
                "total_goals": manifest.goals.len(),
                "goals": manifest.goals
            })
        }
        _ => json!({"error": format!("Unknown goal action '{}'", action)}),
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    fn manifest() -> GoalManifest {
        GoalManifest {
            project_name: "isolated".into(),
            goals: vec![Goal {
                id:"active".into(), title:"active".into(), description:"test".into(),
                status:GoalStatus::Active, created_at:"test".into(), updated_at:"test".into(),
                milestones:vec![Milestone{id:7, description:"test".into(), completed:false}],
            }],
        }
    }
    #[test]
    fn round3_milestone_persistence_failure_is_error() {
        let scratch = tempfile::tempdir().unwrap();
        let blocked = scratch.path().join("goals.json");
        std::fs::create_dir(&blocked).unwrap();
        let result = complete_milestone_with("active",7,manifest(), |m| {
            std::fs::write(&blocked, serde_json::to_vec(m).unwrap())
                .map(|_| blocked.clone()).map_err(|e| format!("write goals: {e}"))
        });
        assert!(result.unwrap_err().contains("write goals"));
    }
    #[test]
    fn round3_goal_persistence_failure_is_error() {
        let result = complete_goal_with("active",manifest(), |_| Err("storage unavailable".into()));
        assert!(result.unwrap_err().contains("storage unavailable"));
    }
    #[test]
    fn round3_milestone_persistence_success() {
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("goals.json");
        complete_milestone_with("active",7,manifest(), |m| {
            std::fs::write(&path,serde_json::to_vec(m).unwrap())
                .map(|_|path.clone()).map_err(|e|e.to_string())
        }).unwrap();
        let saved: GoalManifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(saved.goals[0].milestones[0].completed);
        assert_eq!(saved.goals[0].status, GoalStatus::Completed);
    }
}
