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

pub(crate) fn goal_index(goals: &[Goal], target: &str) -> Option<usize> {
    goals.iter().position(|goal| goal.id == target).or_else(|| {
        let title = target.to_lowercase();
        goals
            .iter()
            .position(|goal| goal.title.to_lowercase().contains(&title))
    })
}

pub fn complete_goal(target_id: &str) -> Result<String, String> {
    complete_goal_with(target_id, load_goals(), save_goals)
}

fn complete_goal_with(
    target_id: &str,
    mut manifest: GoalManifest,
    save: impl FnOnce(&GoalManifest) -> Result<PathBuf, String>,
) -> Result<String, String> {
    let index = goal_index(&manifest.goals, target_id)
        .ok_or_else(|| format!("Goal '{}' not found in manifest.", target_id))?;
    let goal = &mut manifest.goals[index];
    goal.status = GoalStatus::Completed;
    for milestone in &mut goal.milestones {
        milestone.completed = true;
    }
    goal.updated_at = Utc::now().to_rfc3339();
    save(&manifest)?;
    Ok(format!(
        "Goal '{}' ({}) marked as Completed.",
        manifest.goals[index].title, target_id
    ))
}

pub fn complete_milestone(target_id: &str, milestone_id: usize) -> Result<String, String> {
    complete_milestone_with(target_id, milestone_id, load_goals(), save_goals)
}

fn complete_milestone_with(
    target_id: &str,
    milestone_id: usize,
    mut manifest: GoalManifest,
    save: impl FnOnce(&GoalManifest) -> Result<PathBuf, String>,
) -> Result<String, String> {
    let missing = || {
        format!(
            "Milestone {} for goal '{}' not found.",
            milestone_id, target_id
        )
    };
    let index = goal_index(&manifest.goals, target_id).ok_or_else(missing)?;
    let goal = &mut manifest.goals[index];
    let milestone = goal
        .milestones
        .iter_mut()
        .find(|milestone| milestone.id == milestone_id)
        .ok_or_else(missing)?;
    milestone.completed = true;
    goal.updated_at = Utc::now().to_rfc3339();
    let all_completed = goal.milestones.iter().all(|milestone| milestone.completed);
    if all_completed {
        goal.status = GoalStatus::Completed;
    }
    save(&manifest)?;
    if all_completed {
        Ok(format!(
            "Milestone {} completed. All milestones finished! Goal is now Completed.",
            milestone_id
        ))
    } else {
        Ok(format!("Milestone {} completed.", milestone_id))
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
                id: "active".into(),
                title: "active".into(),
                description: "test".into(),
                status: GoalStatus::Active,
                created_at: "test".into(),
                updated_at: "test".into(),
                milestones: vec![Milestone {
                    id: 7,
                    description: "test".into(),
                    completed: false,
                }],
            }],
        }
    }
    #[test]
    fn round3_milestone_persistence_failure_is_error() {
        let scratch = tempfile::tempdir().unwrap();
        let blocked = scratch.path().join("goals.json");
        std::fs::create_dir(&blocked).unwrap();
        let result = complete_milestone_with("active", 7, manifest(), |m| {
            std::fs::write(&blocked, serde_json::to_vec(m).unwrap())
                .map(|_| blocked.clone())
                .map_err(|e| format!("write goals: {e}"))
        });
        assert!(result.unwrap_err().contains("write goals"));
    }
    #[test]
    fn round3_goal_persistence_failure_is_error() {
        let result =
            complete_goal_with("active", manifest(), |_| Err("storage unavailable".into()));
        assert!(result.unwrap_err().contains("storage unavailable"));
    }
    #[test]
    fn round3_milestone_persistence_success() {
        let scratch = tempfile::tempdir().unwrap();
        let path = scratch.path().join("goals.json");
        complete_milestone_with("active", 7, manifest(), |m| {
            std::fs::write(&path, serde_json::to_vec(m).unwrap())
                .map(|_| path.clone())
                .map_err(|e| e.to_string())
        })
        .unwrap();
        let saved: GoalManifest = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(saved.goals[0].milestones[0].completed);
        assert_eq!(saved.goals[0].status, GoalStatus::Completed);
    }

    fn collision_manifest() -> GoalManifest {
        let mut manifest = manifest();
        let mut goal_a = manifest.goals.remove(0);
        goal_a.id = "goal-A".into();
        goal_a.title = "Support goal-B".into();
        goal_a.milestones[0].id = 1;
        let mut goal_b = goal_a.clone();
        goal_b.id = "goal-B".into();
        goal_b.title = "Actual task".into();
        manifest.goals = vec![goal_a, goal_b];
        manifest
    }

    #[test]
    fn round4_exact_goal_resolution_precedes_title() {
        let manifest = collision_manifest();
        assert_eq!(goal_index(&manifest.goals, "goal-B"), Some(1));
        assert_eq!(goal_index(&manifest.goals, "Actual task"), Some(1));
        assert_eq!(goal_index(&manifest.goals, "ACTUAL"), Some(1));
        assert_eq!(goal_index(&manifest.goals, "Support"), Some(0));
        assert_eq!(goal_index(&manifest.goals, "missing"), None);
    }

    fn persisted_dispatch_collision(test_name: &str, actions: &[&str]) {
        const CHILD_ACTION: &str = "AIEN_TEST_ROUND4_GOAL_DISPATCH_ACTION";
        if let Ok(action) = std::env::var(CHILD_ACTION) {
            assert!(actions.contains(&action.as_str()));
            let before = load_goals();
            // This is the same resolver used by autonomous mode. Resolving
            // the title selects goal-B; its id must stay authoritative when
            // the real tool dispatcher persists the completion.
            let selected = goal_index(&before.goals, "Actual task").unwrap();
            let id = &before.goals[selected].id;
            assert_eq!(id, "goal-B");
            let result = goals_dispatch_tool(&json!({
                "action": action, "id": id, "milestone_id": 1,
            }));
            assert_eq!(result["status"], "ok", "{result}");
            let saved: GoalManifest =
                serde_json::from_slice(&fs::read(".goals.json").unwrap()).unwrap();
            assert!(
                saved.goals[1].milestones[0].completed,
                "{action}: goal-B not persisted"
            );
            assert_eq!(saved.goals[1].status, GoalStatus::Completed);
            assert_eq!(
                serde_json::to_value(&saved.goals[0]).unwrap(),
                serde_json::to_value(&before.goals[0]).unwrap(),
                "{action}: unrelated goal-A changed",
            );
            return;
        }
        for action in actions {
            let scratch = tempfile::tempdir().unwrap();
            fs::write(
                scratch.path().join(".goals.json"),
                serde_json::to_vec(&collision_manifest()).unwrap(),
            )
            .unwrap();
            // Isolate the real path resolver and writer without changing the
            // unit-test process's cwd or using synthetic dispatch success.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .current_dir(scratch.path())
                .env(CHILD_ACTION, action)
                .args(["--exact", test_name, "--nocapture"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{action}: child failed ({:?})\n{}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    #[test]
    fn round4_milestone_dispatch_persists_exact_goal() {
        persisted_dispatch_collision(
            "goals::completion_tests::round4_milestone_dispatch_persists_exact_goal",
            &["milestone_done", "advance"],
        );
    }

    #[test]
    fn round4_goal_dispatch_persists_exact_goal() {
        persisted_dispatch_collision(
            "goals::completion_tests::round4_goal_dispatch_persists_exact_goal",
            &["done", "complete"],
        );
    }
}
