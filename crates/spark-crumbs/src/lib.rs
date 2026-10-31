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

use chrono::{DateTime, Utc};
use colored::*;
use serde::{Deserialize, Serialize};
use std::fs;

use std::path::Path;

pub const CRUMB_FILENAME: &str = ".crumb";
pub const LOCAL_CRUMB_FILENAME: &str = ".crumb.local";

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct LocalCrumbData {
    #[serde(default)]
    pub history: Vec<CrumbHistoryItem>,
    #[serde(default)]
    pub whispers: Vec<CrumbWhisper>,
}

pub fn load_local_crumb(dir: &Path) -> LocalCrumbData {
    let local_path = dir.join(LOCAL_CRUMB_FILENAME);
    if local_path.exists() {
        if let Ok(c) = fs::read_to_string(&local_path) {
            if let Ok(data) = serde_json::from_str::<LocalCrumbData>(&c) {
                return data;
            }
        }
    }
    LocalCrumbData::default()
}

pub fn save_local_crumb(dir: &Path, data: &LocalCrumbData) {
    let local_path = dir.join(LOCAL_CRUMB_FILENAME);
    if let Ok(json_str) = serde_json::to_string_pretty(data) {
        let _ = fs::write(&local_path, json_str);
    }
}

pub const DEFAULT_TTL_SECS: u64 = 1800; // 30 minutes

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AboveInfo {
    pub path: String,
    pub canonical: String,
    pub name: String,
    pub role: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum AboveValue {
    Structured(AboveInfo),
    Path(String),
}

impl AboveValue {
    pub fn to_above_info(&self) -> AboveInfo {
        match self {
            AboveValue::Structured(a) => a.clone(),
            AboveValue::Path(p) => {
                let path = std::path::Path::new(p);
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "above".to_string());
                AboveInfo {
                    path: "..".to_string(),
                    canonical: p.clone(),
                    name,
                    role: "Parent directory context".to_string(),
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BelowItem {
    pub name: String,
    pub item_type: String, // "dir" or "file"
    pub role: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrumbHistoryItem {
    pub crumb_id: String,
    pub timestamp: String,
    pub agent_id: String,
    pub session_id: String,
    pub action: String, // "signin", "view", "edit", "create", "test"
    pub target: String,
    pub intent: String,
    pub vector: String,
}

impl CrumbHistoryItem {
    pub fn elapsed_secs(&self) -> u64 {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(&self.timestamp) {
            let age = Utc::now().signed_duration_since(parsed.with_timezone(&Utc));
            age.num_seconds().max(0) as u64
        } else {
            0
        }
    }

    pub fn is_active(&self) -> bool {
        self.elapsed_secs() < DEFAULT_TTL_SECS
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrumbWhisper {
    #[serde(default)]
    pub whisper_id: String,
    pub from_agent: String,
    pub timestamp: String,
    #[serde(default)]
    pub target_file: Option<String>,
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrumbPurpose {
    pub statement: String,
    #[serde(default)]
    pub created_by: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default = "default_lifecycle")]
    pub lifecycle: String, // "permanent" | "ephemeral"
}

fn default_lifecycle() -> String {
    "permanent".to_string()
}

/// A `.crumb` written by this tool. Unknown fields are rejected so that a file
/// carrying anything this tool would drop on save (for example hand-written
/// `invariants`) is treated as foreign and never rewritten.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DirCrumb {
    pub version: u32,
    pub dir_path: String,
    pub dir_name: String,
    pub description: String,
    #[serde(default)]
    pub purpose: Option<CrumbPurpose>,
    pub above: Option<AboveValue>,
    pub below: Vec<BelowItem>,
    pub history: Vec<CrumbHistoryItem>,
    pub whispers: Vec<CrumbWhisper>,
}

impl DirCrumb {
    pub fn new(dir: &Path, description: Option<&str>, purpose: Option<CrumbPurpose>) -> Self {
        let (canon, dir_name) = dir_identity(dir);

        // Inspect Above (from the resolved path, so `.` still has a parent)
        let canon_path = Path::new(&canon);
        let above = if let Some(parent) = canon_path.parent() {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_default();
            if !home.is_empty() && parent.starts_with(&home) && parent != canon_path {
                let p_canon = parent
                    .canonicalize()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|_| parent.to_string_lossy().to_string());
                let p_name = parent
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "home".to_string());

                Some(AboveInfo {
                    path: "..".to_string(),
                    canonical: p_canon,
                    name: p_name,
                    role: "Parent directory context".to_string(),
                })
            } else {
                None
            }
        } else {
            None
        };

        // Inspect Below
        let mut below = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(".") {
                    continue;
                }
                let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                below.push(BelowItem {
                    name,
                    item_type: if is_dir {
                        "dir".to_string()
                    } else {
                        "file".to_string()
                    },
                    role: String::new(),
                });
            }
        }
        below.sort_by(|a, b| a.name.cmp(&b.name));

        DirCrumb {
            version: 1,
            dir_path: canon,
            dir_name,
            description: description.unwrap_or("Workspace directory").to_string(),
            purpose,
            above: above.map(AboveValue::Structured),
            below,
            history: Vec::new(),
            whispers: Vec::new(),
        }
    }
}

/// Reads `crumb_path`. `Ok(None)`: no file. `Ok(Some)`: a crumb in this tool's
/// `DirCrumb` format. `Err`: a file exists that is not in that format (for
/// example a hand-written schema 1.0.0 crumb) or cannot be read; such a file is
/// never rewritten.
fn read_dir_crumb_file(crumb_path: &Path) -> Result<Option<DirCrumb>, String> {
    match fs::symlink_metadata(crumb_path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {}", crumb_path.display(), e)),
        Ok(_) => {}
    }
    let content =
        fs::read_to_string(crumb_path).map_err(|e| format!("{}: {}", crumb_path.display(), e))?;
    serde_json::from_str::<DirCrumb>(&content)
        .map(Some)
        .map_err(|e| format!("{} is not a DirCrumb: {}", crumb_path.display(), e))
}

/// Read-only view of a directory whose `.crumb` is not in `DirCrumb` format.
/// The purpose comes from the file's `purpose` string when it has one.
fn foreign_crumb_view(dir: &Path) -> DirCrumb {
    let statement = fs::read_to_string(dir.join(CRUMB_FILENAME))
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|v| v.get("purpose").and_then(|p| p.as_str()).map(str::to_string));
    let purpose = statement.map(|statement| CrumbPurpose {
        statement,
        created_by: "hand-written .crumb".to_string(),
        session_id: String::new(),
        created_at: String::new(),
        lifecycle: "permanent".to_string(),
    });
    let mut crumb = DirCrumb::new(
        dir,
        Some("Existing .crumb is not in DirCrumb format; shown read-only and never rewritten"),
        purpose,
    );
    merge_local_crumb(&mut crumb, dir);
    crumb
}

/// Merges ephemeral telemetry from `.crumb.local` into `crumb`.
fn merge_local_crumb(crumb: &mut DirCrumb, dir: &Path) {
    let local_data = load_local_crumb(dir);
    for h in local_data.history {
        if !crumb
            .history
            .iter()
            .any(|existing| existing.crumb_id == h.crumb_id)
        {
            crumb.history.push(h);
        }
    }
    for w in local_data.whispers {
        if !crumb
            .whispers
            .iter()
            .any(|existing| existing.whisper_id == w.whisper_id || existing.message == w.message)
        {
            crumb.whispers.push(w);
        }
    }
}

/// Loads `dir/.crumb`. Seeds and writes a new crumb only when no `.crumb` file
/// exists. A `.crumb` that is not in `DirCrumb` format (a hand-written crumb)
/// is never written; a read-only view is returned instead.
pub fn load_or_init_dir_crumb(dir: &Path, purpose_opt: Option<CrumbPurpose>) -> DirCrumb {
    let crumb_path = dir.join(CRUMB_FILENAME);
    match read_dir_crumb_file(&crumb_path) {
        Err(_) => return foreign_crumb_view(dir),
        Ok(Some(mut crumb)) => {
            // The stored location may be stale (the crumb was moved or copied):
            // the directory it was loaded from is authoritative.
            let (dir_path, dir_name) = dir_identity(dir);
            crumb.dir_path = dir_path;
            crumb.dir_name = dir_name;
            // Refresh below items dynamically
            let mut below = Vec::new();
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(".") {
                        continue;
                    }
                    let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                    below.push(BelowItem {
                        name,
                        item_type: if is_dir {
                            "dir".to_string()
                        } else {
                            "file".to_string()
                        },
                        role: String::new(),
                    });
                }
            }
            below.sort_by(|a, b| a.name.cmp(&b.name));
            crumb.below = below;
            if crumb.purpose.is_none() && purpose_opt.is_some() {
                crumb.purpose = purpose_opt.clone();
                let _ = save_dir_crumb(dir, &crumb);
            }
            merge_local_crumb(&mut crumb, dir);
            return crumb;
        }
        Ok(None) => {}
    }

    // Auto-Sense & Seed if purpose not provided
    let final_purpose = match purpose_opt {
        Some(p) => Some(p),
        None => {
            let sensed_statement = auto_sense_directory_purpose(dir);
            Some(CrumbPurpose {
                statement: sensed_statement,
                created_by: "AIEN (Auto-Sensed)".to_string(),
                session_id: "auto-sense".to_string(),
                created_at: Utc::now().to_rfc3339(),
                lifecycle: "permanent".to_string(),
            })
        }
    };

    let crumb = DirCrumb::new(dir, None, final_purpose);
    let _ = save_dir_crumb(dir, &crumb);
    crumb
}

/// Canonical path and name of `dir`, so `.` and relative paths get the real
/// directory name. Falls back to the path as given when it cannot be resolved.
fn dir_identity(dir: &Path) -> (String, String) {
    let canon = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let name = canon
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "/".to_string());
    (canon.to_string_lossy().to_string(), name)
}

/// Writes the durable part of `crumb` to `dir/.crumb`, never to the crumb's
/// stored `dir_path`, and records `dir` as its location. Refuses, without
/// writing, when an existing `.crumb` is not in `DirCrumb` format.
pub fn save_dir_crumb(dir: &Path, crumb: &DirCrumb) -> Result<(), String> {
    let crumb_path = dir.join(CRUMB_FILENAME);
    if let Err(e) = read_dir_crumb_file(&crumb_path) {
        return Err(format!("refusing to overwrite .crumb: {}", e));
    }

    // Durable .crumb: keep architectural purpose, above/below, clean of ephemeral churn
    let mut durable_crumb = crumb.clone();
    let (dir_path, dir_name) = dir_identity(dir);
    durable_crumb.dir_path = dir_path;
    durable_crumb.dir_name = dir_name;
    durable_crumb.history = Vec::new(); // Ephemeral history lives in .crumb.local
    durable_crumb.whispers = Vec::new(); // Whispers live in .crumb.local

    let json_str = serde_json::to_string_pretty(&durable_crumb).map_err(|e| e.to_string())?;
    fs::write(&crumb_path, json_str).map_err(|e| format!("{}: {}", crumb_path.display(), e))
}

/// Records an action in the directory's `.crumb` file and bubbles awareness to parent.
pub fn record_directory_crumb(
    agent_id: &str,
    session_id: &str,
    target_path: &Path,
    action: &str,
    intent: &str,
    vector: &str,
) {
    let dir = if target_path.is_dir() {
        target_path.to_path_buf()
    } else {
        target_path.parent().unwrap_or(target_path).to_path_buf()
    };

    let target_name = target_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "dir".to_string());

    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "dir".to_string());
    let mut local_data = load_local_crumb(&dir);

    let item = CrumbHistoryItem {
        crumb_id: format!("crumb-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        timestamp: Utc::now().to_rfc3339(),
        agent_id: agent_id.to_string(),
        session_id: session_id.to_string(),
        action: action.to_string(),
        target: target_name.clone(),
        intent: intent.to_string(),
        vector: vector.to_string(),
    };

    local_data.history.push(item);
    if local_data.history.len() > 50 {
        local_data.history.drain(0..local_data.history.len() - 50);
    }
    save_local_crumb(&dir, &local_data);

    // If dir has a parent within project workspace, bubble awareness to parent .crumb.local
    if let Some(parent) = dir.parent() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        if !home.is_empty() && parent.starts_with(&home) && parent != dir {
            let mut parent_local = load_local_crumb(parent);
            let bubble_item = CrumbHistoryItem {
                crumb_id: format!("bubble-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
                timestamp: Utc::now().to_rfc3339(),
                agent_id: agent_id.to_string(),
                session_id: session_id.to_string(),
                action: format!("sub:{}", action),
                target: format!("{}/{}", dir_name, target_name),
                intent: intent.to_string(),
                vector: vector.to_string(),
            };
            parent_local.history.push(bubble_item);
            if parent_local.history.len() > 50 {
                parent_local
                    .history
                    .drain(0..parent_local.history.len() - 50);
            }
            save_local_crumb(parent, &parent_local);
        }
    }
}

/// Leaves a whisper message in a directory .crumb.local file.
pub fn leave_dir_whisper(from_agent: &str, dir: &Path, target_file: Option<&str>, message: &str) {
    let mut local_data = load_local_crumb(dir);
    let whisper = CrumbWhisper {
        whisper_id: format!("whisp-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        from_agent: from_agent.to_string(),
        timestamp: Utc::now().to_rfc3339(),
        target_file: target_file.map(|s| s.to_string()),
        message: message.to_string(),
    };
    local_data.whispers.push(whisper);
    if local_data.whispers.len() > 30 {
        local_data.whispers.drain(0..local_data.whispers.len() - 30);
    }
    save_local_crumb(dir, &local_data);
}

/// Sniffs directory crumb and checks for recent actions or whispers on target file.
pub fn sniff_dir_crumb(target_path: &Path, current_agent: &str) -> Option<String> {
    let dir = if target_path.is_dir() {
        target_path.to_path_buf()
    } else {
        target_path.parent().unwrap_or(target_path).to_path_buf()
    };

    let crumb = load_or_init_dir_crumb(&dir, None);
    let target_name = target_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut notices = Vec::new();

    // Check history for this target or directory
    for item in crumb.history.iter().rev() {
        if (item.target == target_name
            || item.target.ends_with(&target_name)
            || target_path.is_dir())
            && item.agent_id != current_agent
            && item.is_active()
        {
            let elapsed = item.elapsed_secs();
            let el_str = if elapsed < 60 {
                format!("{}s ago", elapsed)
            } else {
                format!("{}m ago", elapsed / 60)
            };
            notices.push(format!(
                "• [{}] {} on '{}' ({}) - Intent: '{}' -> Vector: '{}'",
                item.agent_id, item.action, item.target, el_str, item.intent, item.vector
            ));
        }
    }

    // Check whispers
    for w in crumb.whispers.iter().rev() {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(&w.timestamp) {
            let age = Utc::now().signed_duration_since(parsed.with_timezone(&Utc));
            if age.num_seconds() >= 0 && (age.num_seconds() as u64) < DEFAULT_TTL_SECS * 2 {
                if let Some(tf) = &w.target_file {
                    if tf == &target_name || target_path.is_dir() {
                        notices.push(format!("💬 [Whisper from {}]: {}", w.from_agent, w.message));
                    }
                } else {
                    notices.push(format!("💬 [Whisper from {}]: {}", w.from_agent, w.message));
                }
            }
        }
    }

    if notices.is_empty() {
        return None;
    }

    let mut out = format!("🐾 [DIRECTORY CRUMB: {}/.crumb]\n", dir.display());
    if let Some(above_raw) = &crumb.above {
        let above = above_raw.to_above_info();
        out.push_str(&format!(
            "   ↑ Above: {} ({})\n",
            above.canonical, above.name
        ));
    }
    if let Some(p) = &crumb.purpose {
        out.push_str(&format!("   🎯 Purpose: {}\n", p.statement.cyan()));
    }
    out.push_str("   Recent Peer History in this directory:\n");
    for n in notices.iter().take(4) {
        out.push_str(&format!("   {}\n", n));
    }
    out.push_str("─────────────────────────────────────────────────────────────\n");
    Some(out)
}

/// Pretty-prints a rich TUI inspection card for a directory's crumb.
pub fn format_dir_crumb_tui(dir: &Path) -> String {
    let crumb = load_or_init_dir_crumb(dir, None);

    let mut out = String::new();
    out.push_str(&format!(
        "🧭 DIRECTORY CRUMB: {} ({})\n",
        crumb.dir_name.bold().cyan(),
        crumb.dir_path.dimmed()
    ));
    out.push_str(&format!("   Description: {}\n", crumb.description));
    if let Some(p) = &crumb.purpose {
        out.push_str(&format!(
            "   🎯 PURPOSE:     {} [{} | by {}]\n",
            p.statement.bold().magenta(),
            p.lifecycle.cyan(),
            p.created_by.yellow()
        ));
    }

    if let Some(above_raw) = &crumb.above {
        let above = above_raw.to_above_info();
        out.push_str(&format!(
            "   ↑ ABOVE: {} [{}]\n",
            above.name.bold().yellow(),
            above.canonical.dimmed()
        ));
    } else {
        out.push_str("   ↑ ABOVE: [Root workspace boundary]\n");
    }

    let dirs: Vec<&str> = crumb
        .below
        .iter()
        .filter(|i| i.item_type == "dir")
        .map(|i| i.name.as_str())
        .collect();
    let files: Vec<&str> = crumb
        .below
        .iter()
        .filter(|i| i.item_type == "file")
        .map(|i| i.name.as_str())
        .collect();

    out.push_str(&format!(
        "   ↓ BELOW (Subdirectories): [{}]\n",
        dirs.join(", ").cyan()
    ));
    out.push_str(&format!(
        "   ↓ BELOW (Files):          [{}]\n",
        files.join(", ").white()
    ));

    out.push_str(&format!(
        "\n📜 Agent Chronicle in this Directory (Total: {}):\n",
        crumb.history.len()
    ));
    if crumb.history.is_empty() {
        out.push_str("   (No previous actions recorded in this directory)\n");
    } else {
        for (i, h) in crumb.history.iter().rev().take(5).enumerate() {
            let elapsed = h.elapsed_secs();
            let el_str = if elapsed < 60 {
                format!("{}s ago", elapsed)
            } else {
                format!("{}m ago", elapsed / 60)
            };
            out.push_str(&format!(
                "   {}. [{}] {} '{}' ({})\n      Intent: {}\n      Vector: {}\n",
                i + 1,
                h.agent_id.bold().yellow(),
                h.action,
                h.target.green(),
                el_str,
                h.intent,
                h.vector.dimmed()
            ));
        }
    }

    if !crumb.whispers.is_empty() {
        out.push_str("\n💬 Directory Whispers & Intercommunication:\n");
        for w in crumb.whispers.iter().rev().take(3) {
            out.push_str(&format!(
                "   • From [{}]: \"{}\"\n",
                w.from_agent.bold().cyan(),
                w.message
            ));
        }
    }

    out
}

/// Discovers directory crumbs for nesting ritual, showing immediate directory topography.
pub fn discover_workspace_crumbs_summary(start_dir: &Path) -> String {
    let crumb = load_or_init_dir_crumb(start_dir, None);
    let mut s = format!(
        "[DIRECTORY CRUMB TOPOGRAPHY: {}]\n\
         Current: {} ({})\n",
        crumb.dir_name, crumb.dir_path, crumb.description
    );

    if let Some(above_raw) = &crumb.above {
        let above = above_raw.to_above_info();
        s.push_str(&format!("↑ Above: {} ({})\n", above.name, above.canonical));
    }
    if let Some(p) = &crumb.purpose {
        s.push_str(&format!("🎯 Directory Purpose: {}\n", p.statement));
    }

    let subdirs: Vec<&str> = crumb
        .below
        .iter()
        .filter(|i| i.item_type == "dir")
        .map(|i| i.name.as_str())
        .collect();
    if !subdirs.is_empty() {
        s.push_str(&format!("↓ Below Subdirs: [{}]\n", subdirs.join(", ")));
    }

    if !crumb.history.is_empty() {
        s.push_str("Recent Actions in this Directory:\n");
        for h in crumb.history.iter().rev().take(3) {
            s.push_str(&format!(
                "• [{}] {} '{}' - Intent: '{}' -> Vector: '{}'\n",
                h.agent_id, h.action, h.target, h.intent, h.vector
            ));
        }
    }

    if !crumb.whispers.is_empty() {
        s.push_str("Directory Whispers:\n");
        for w in crumb.whispers.iter().rev().take(2) {
            s.push_str(&format!("• [{}] \"{}\"\n", w.from_agent, w.message));
        }
    }

    s
}

fn auto_sense_directory_purpose(dir: &Path) -> String {
    let readme = dir.join("README.md");
    if readme.exists() {
        if let Ok(c) = fs::read_to_string(&readme) {
            for line in c.lines() {
                let trimmed = line.trim().trim_start_matches('#').trim();
                if !trimmed.is_empty() {
                    return format!("Documentation & implementation for {}", trimmed);
                }
            }
        }
    }
    let cargo = dir.join("Cargo.toml");
    if cargo.exists() {
        if let Ok(c) = fs::read_to_string(&cargo) {
            for line in c.lines() {
                if line.starts_with("name =") {
                    let name = line
                        .replace("name =", "")
                        .replace("\"", "")
                        .trim()
                        .to_string();
                    return format!("Rust crate codebase: {}", name);
                }
            }
        }
    }
    let pkg = dir.join("package.json");
    if pkg.exists() {
        if let Ok(c) = fs::read_to_string(&pkg) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&c) {
                if let Some(name) = val.get("name").and_then(|n| n.as_str()) {
                    return format!("Node/Web package: {}", name);
                }
            }
        }
    }
    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "workspace".to_string());
    format!("Workspace subsystem directory for {}", dir_name)
}

pub fn seed_directory_tree(
    root: &Path,
    recursive: bool,
    agent: &str,
    whisper: Option<&str>,
) -> usize {
    let mut count = 0;
    if !recursive {
        load_or_init_dir_crumb(root, None);
        if let Some(msg) = whisper {
            leave_dir_whisper(agent, root, None, msg);
        }
        return 1;
    }

    for entry in walkdir::WalkDir::new(root)
        .min_depth(0)
        .max_depth(6)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && name != "target" && name != "node_modules"
        })
        .flatten()
    {
        if entry.file_type().is_dir() {
            let path = entry.path();
            load_or_init_dir_crumb(path, None);
            if let Some(msg) = whisper {
                leave_dir_whisper(agent, path, None, msg);
            }
            count += 1;
        }
    }
    count
}

pub fn sniff_crumb(target_path: &Path, current_agent: &str) -> Option<String> {
    sniff_dir_crumb(target_path, current_agent)
}

pub fn add_crumb_whisper(dir: &Path, from_agent: &str, message: &str, target_file: Option<&str>) {
    leave_dir_whisper(from_agent, dir, target_file, message);
}

pub fn record_crumb_action(
    dir: &Path,
    agent_id: &str,
    session_id: &str,
    action: &str,
    target: &str,
    intent: &str,
    vector: &str,
) {
    record_directory_crumb(
        agent_id,
        session_id,
        &dir.join(target),
        action,
        intent,
        vector,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crumb_lifecycle_and_whisper() {
        let temp_dir = std::env::temp_dir().join(format!("test-crumb-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("create temp dir");

        let crumb = load_or_init_dir_crumb(&temp_dir, None);
        assert!(crumb.purpose.is_some());

        record_directory_crumb(
            "Atlas-Prime",
            "test-session-01",
            &temp_dir.join("lib.rs"),
            "edit",
            "Verify crumb persistence",
            "Testing spark-crumbs subsystem",
        );

        leave_dir_whisper(
            "AIEN",
            &temp_dir,
            Some("lib.rs"),
            "Verification lock acquired",
        );

        let sniff = sniff_dir_crumb(&temp_dir.join("lib.rs"), "Peer-Agent");
        assert!(sniff.is_some());
        let sniff_str = sniff.unwrap();
        assert!(sniff_str.contains("Atlas-Prime"));
        assert!(sniff_str.contains("Verification lock"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_seed_directory_tree() {
        let temp_dir = std::env::temp_dir().join(format!("test-seed-{}", uuid::Uuid::new_v4()));
        let sub = temp_dir.join("sub");
        fs::create_dir_all(&sub).expect("create sub dir");

        let count = seed_directory_tree(&temp_dir, true, "AIEN", Some("Seeded tree"));
        assert!(count >= 2);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_format_tui_card() {
        let temp_dir = std::env::temp_dir().join(format!("test-tui-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("create temp dir");

        let card = format_dir_crumb_tui(&temp_dir);
        assert!(card.contains("DIRECTORY CRUMB"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Shape of a hand-written crumb on main (crates/aien-serve-http/.crumb).
    const HAND_WRITTEN_CRUMB: &str = r#"{
  "invariants": [
    "The door refuses to bind a non-loopback address. Check: cargo test -p aien-serve-http --test loopback -- --exact non_loopback_bind_is_refused"
  ],
  "name": "aien-serve-http",
  "purpose": "OpenAI-compatible HTTP door for AIEN's own-kernel model serving (OWN-30B cut 4)",
  "schema_version": "1.0.0"
}
"#;

    #[test]
    fn hand_written_crumb_is_never_rewritten() {
        let temp_dir =
            std::env::temp_dir().join(format!("test-handwritten-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(temp_dir.join("src")).expect("create temp dir");
        fs::write(temp_dir.join("Cargo.toml"), "[package]\nname = \"door\"\n").expect("cargo");
        let crumb_path = temp_dir.join(CRUMB_FILENAME);
        fs::write(&crumb_path, HAND_WRITTEN_CRUMB).expect("write crumb");

        let crumb = load_or_init_dir_crumb(&temp_dir, None);
        let _ = format_dir_crumb_tui(&temp_dir);
        let _ = sniff_dir_crumb(&temp_dir, "Peer-Agent");
        let _ = discover_workspace_crumbs_summary(&temp_dir);
        seed_directory_tree(&temp_dir, false, "AIEN", Some("seeded"));
        seed_directory_tree(&temp_dir, true, "AIEN", None);
        record_crumb_action(&temp_dir, "AIEN", "s", "edit", "Cargo.toml", "i", "v");

        assert_eq!(
            fs::read_to_string(&crumb_path).expect("read crumb"),
            HAND_WRITTEN_CRUMB
        );
        assert_eq!(
            crumb.purpose.map(|p| p.statement).as_deref(),
            Some("OpenAI-compatible HTTP door for AIEN's own-kernel model serving (OWN-30B cut 4)")
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn save_refuses_to_overwrite_foreign_crumb() {
        let temp_dir =
            std::env::temp_dir().join(format!("test-refuse-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).expect("create temp dir");
        let crumb_path = temp_dir.join(CRUMB_FILENAME);

        // A hand-written crumb: load gives a view, save refuses, bytes unchanged.
        fs::write(&crumb_path, HAND_WRITTEN_CRUMB).expect("write crumb");
        let mut crumb = load_or_init_dir_crumb(&temp_dir, None);
        crumb.description = "changed".to_string();
        assert!(save_dir_crumb(&temp_dir, &crumb).is_err());
        assert_eq!(fs::read_to_string(&crumb_path).unwrap(), HAND_WRITTEN_CRUMB);

        // A DirCrumb with an extra hand-written field: save would drop the
        // field, so it is refused too.
        let seeded = DirCrumb::new(&temp_dir, None, None);
        let mut value = serde_json::to_value(&seeded).unwrap();
        value["invariants"] = serde_json::json!(["kept by hand"]);
        let mixed = serde_json::to_string_pretty(&value).unwrap();
        fs::write(&crumb_path, &mixed).expect("write mixed crumb");
        assert!(save_dir_crumb(&temp_dir, &seeded).is_err());
        assert_eq!(fs::read_to_string(&crumb_path).unwrap(), mixed);

        // A crumb this tool wrote is still saved.
        fs::remove_file(&crumb_path).unwrap();
        let fresh = load_or_init_dir_crumb(&temp_dir, None);
        assert!(fresh.purpose.is_some());
        assert!(save_dir_crumb(&temp_dir, &fresh).is_ok());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn save_writes_where_the_crumb_was_loaded_not_to_stored_dir_path() {
        let here = std::env::temp_dir().join(format!("test-here-{}", uuid::Uuid::new_v4()));
        let old = std::env::temp_dir().join(format!("test-old-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&here).expect("create here");
        fs::create_dir_all(&old).expect("create old");

        // A DirCrumb moved (or copied) from `old`: its stored dir_path is stale.
        let mut moved = DirCrumb::new(&old, None, None);
        moved.purpose = None;
        fs::write(
            here.join(CRUMB_FILENAME),
            serde_json::to_string_pretty(&moved).unwrap(),
        )
        .expect("write moved crumb");

        // Loading with a purpose fills the missing purpose and saves.
        let purpose = CrumbPurpose {
            statement: "filled on load".to_string(),
            created_by: "test".to_string(),
            session_id: "test".to_string(),
            created_at: "2026-10-31T00:00:00Z".to_string(),
            lifecycle: "permanent".to_string(),
        };
        let loaded = load_or_init_dir_crumb(&here, Some(purpose));

        assert!(
            !old.join(CRUMB_FILENAME).exists(),
            "save wrote to the stale stored dir_path"
        );
        let saved: DirCrumb =
            serde_json::from_str(&fs::read_to_string(here.join(CRUMB_FILENAME)).unwrap()).unwrap();
        let canon = here.canonicalize().unwrap().to_string_lossy().to_string();
        assert_eq!(saved.dir_path, canon);
        assert_eq!(loaded.dir_path, canon);
        assert_eq!(
            saved.purpose.map(|p| p.statement).as_deref(),
            Some("filled on load")
        );

        let _ = fs::remove_dir_all(&here);
        let _ = fs::remove_dir_all(&old);
    }

    #[test]
    fn dir_name_of_dot_is_the_real_directory_name() {
        let crumb = DirCrumb::new(Path::new("."), None, None);
        let expected = std::env::current_dir()
            .unwrap()
            .canonicalize()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert_eq!(crumb.dir_name, expected);
    }

    #[test]
    fn crumb_for_dot_has_the_parent_entry() {
        let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        let home = std::env::var("HOME").unwrap_or_default();
        let parent = cwd.parent().unwrap();
        assert!(
            !home.is_empty() && parent.starts_with(&home),
            "test needs the crate directory under HOME"
        );
        let crumb = DirCrumb::new(Path::new("."), None, None);
        match crumb.above {
            Some(AboveValue::Structured(above)) => {
                assert_eq!(above.canonical, parent.to_string_lossy());
                assert_eq!(above.name, parent.file_name().unwrap().to_string_lossy());
            }
            other => panic!("no parent entry for \".\": {:?}", other.is_some()),
        }
    }
}
