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
use std::io::{Read, Write};
use std::path::Path;

/// Open a `.crumb` or `.crumb.local` sidecar without following a symlink in
/// its final component. A sidecar planted as a link (say by a cloned repo)
/// must not redirect a crumb read or write outside its directory (DRA-47).
fn open_sidecar(path: &Path, opts: &mut fs::OpenOptions) -> std::io::Result<fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NOFOLLOW);
    }
    opts.open(path)
}

fn read_sidecar(path: &Path) -> std::io::Result<String> {
    let mut text = String::new();
    open_sidecar(path, fs::OpenOptions::new().read(true))?.read_to_string(&mut text)?;
    Ok(text)
}

fn write_sidecar(path: &Path, text: &str) -> std::io::Result<()> {
    open_sidecar(
        path,
        fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true),
    )?
    .write_all(text.as_bytes())
}

/// Create a `.crumb` that does not exist yet. An existing entry, including a
/// dangling symlink that `Path::exists` reports as absent, is left alone
/// rather than written through (DRA-47).
pub(crate) fn write_new_crumb(path: &Path, text: &str) -> std::io::Result<()> {
    open_sidecar(path, fs::OpenOptions::new().write(true).create_new(true))?
        .write_all(text.as_bytes())
}

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
        if let Ok(c) = read_sidecar(&local_path) {
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
        let _ = write_sidecar(&local_path, &json_str);
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DirCrumb {
    pub version: u32,
    pub dir_path: String,
    pub dir_name: String,
    pub description: String,
    #[serde(default)]
    pub purpose: Option<CrumbPurpose>,
    pub above: Option<AboveInfo>,
    pub below: Vec<BelowItem>,
    pub history: Vec<CrumbHistoryItem>,
    pub whispers: Vec<CrumbWhisper>,
}

impl DirCrumb {
    pub fn new(dir: &Path, description: Option<&str>, purpose: Option<CrumbPurpose>) -> Self {
        let canon = dir
            .canonicalize()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| dir.to_string_lossy().to_string());

        let dir_name = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "/".to_string());

        // Inspect Above
        let above = if let Some(parent) = dir.parent() {
            let platform = crate::platform::PlatformContext::detect();
            if parent.starts_with(&platform.home_dir) && parent != dir {
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
            above,
            below,
            history: Vec::new(),
            whispers: Vec::new(),
        }
    }
}

pub fn load_or_init_dir_crumb(dir: &Path, purpose_opt: Option<CrumbPurpose>) -> DirCrumb {
    let crumb_path = dir.join(CRUMB_FILENAME);
    if crumb_path.exists() {
        if let Ok(content) = read_sidecar(&crumb_path) {
            match serde_json::from_str::<DirCrumb>(&content) {
                Err(e) => eprintln!(
                    "Failed to parse DirCrumb at {}: {}",
                    crumb_path.display(),
                    e
                ),
                Ok(mut crumb) => {
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
                        save_dir_crumb(&crumb);
                    }
                    // Merge ephemeral telemetry from .crumb.local
                    let local_data = load_local_crumb(dir);
                    if !local_data.history.is_empty() {
                        for h in local_data.history {
                            if !crumb
                                .history
                                .iter()
                                .any(|existing| existing.crumb_id == h.crumb_id)
                            {
                                crumb.history.push(h);
                            }
                        }
                    }
                    if !local_data.whispers.is_empty() {
                        for w in local_data.whispers {
                            if !crumb.whispers.iter().any(|existing| {
                                existing.whisper_id == w.whisper_id || existing.message == w.message
                            }) {
                                crumb.whispers.push(w);
                            }
                        }
                    }
                    return crumb;
                }
            }
        }
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
    save_dir_crumb(&crumb);
    crumb
}

pub fn save_dir_crumb(crumb: &DirCrumb) {
    let dir = Path::new(&crumb.dir_path);
    let crumb_path = dir.join(CRUMB_FILENAME);

    // Durable .crumb: keep architectural purpose, above/below, clean of ephemeral churn
    let mut durable_crumb = crumb.clone();
    durable_crumb.history = Vec::new(); // Ephemeral history lives in .crumb.local
    durable_crumb.whispers = Vec::new(); // Whispers live in .crumb.local

    if let Ok(json_str) = serde_json::to_string_pretty(&durable_crumb) {
        let _ = write_sidecar(&crumb_path, &json_str);
    }
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

    // Bubble awareness to the parent's .crumb.local only when that parent is
    // itself inside a permitted workspace.
    let platform = crate::platform::PlatformContext::detect();
    if let Some(parent) = bubble_parent(
        &dir,
        &platform.home_dir,
        &crate::fs_bind::permitted_workspaces(),
    ) {
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

/// The parent directory a crumb record bubbles into. It must sit under the
/// home directory and inside a permitted workspace: the parent of a workspace
/// root (`~` for `~/workspace`) is outside the allow list (DRA-47).
fn bubble_parent<'a>(
    dir: &'a Path,
    home: &Path,
    workspaces: &[std::path::PathBuf],
) -> Option<&'a Path> {
    let parent = dir.parent().filter(|p| *p != dir)?;
    if !parent.starts_with(home) {
        return None;
    }
    crate::safety::validate_workspace_path(parent, workspaces).ok()?;
    Some(parent)
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
    if let Some(above) = &crumb.above {
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

    if let Some(above) = &crumb.above {
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

    if let Some(above) = &crumb.above {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// DRA-47: recording in a workspace root must not bubble into the root's
    /// parent, which is under `$HOME` but outside the allow list. One level
    /// down, the parent is the workspace and the bubble stays.
    #[test]
    fn bubble_stays_inside_a_permitted_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().canonicalize().unwrap();
        let ws = home.join("workspace");
        std::fs::create_dir_all(ws.join("sub")).unwrap();
        let workspaces = vec![ws.clone()];
        assert_eq!(bubble_parent(&ws, &home, &workspaces), None);
        assert_eq!(
            bubble_parent(&ws.join("sub"), &home, &workspaces),
            Some(ws.as_path())
        );
        let elsewhere = tempfile::tempdir().unwrap();
        let elsewhere = elsewhere.path().canonicalize().unwrap().join("sub");
        assert_eq!(bubble_parent(&elsewhere, &home, &workspaces), None);
    }

    /// DRA-47: `init_sandbox` writes a fresh worktree's `.crumb` only when
    /// `exists()` is false, which a dangling symlink committed by the repo
    /// also reports. The write must not follow either kind of link.
    #[cfg(unix)]
    #[test]
    fn new_crumb_is_not_written_through_a_symlink() {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir_in("/dev/shm").unwrap();
        let out = outside.path().canonicalize().unwrap();
        let existing = out.join("existing");
        std::fs::write(&existing, "VICTIM").unwrap();
        let missing = out.join("missing");
        let sandbox = tempfile::tempdir().unwrap();
        let dangling = sandbox.path().join("dangling.crumb");
        let linked = sandbox.path().join("linked.crumb");
        symlink(&missing, &dangling).unwrap();
        symlink(&existing, &linked).unwrap();
        assert!(!dangling.exists(), "a dangling link reads as absent");

        let _ = write_new_crumb(&dangling, "SANDBOX_CRUMB");
        let _ = write_new_crumb(&linked, "SANDBOX_CRUMB");
        let mut escaped = Vec::new();
        if missing.symlink_metadata().is_ok() {
            escaped.push(format!("created {}", missing.display()));
        }
        if std::fs::read_to_string(&existing).unwrap() != "VICTIM" {
            escaped.push(format!("rewrote {}", existing.display()));
        }
        assert!(escaped.is_empty(), "{}", escaped.join("; "));

        // Control: a plain absent path is created with the content.
        let fresh = sandbox.path().join(".crumb");
        write_new_crumb(&fresh, "SANDBOX_CRUMB").unwrap();
        assert_eq!(std::fs::read_to_string(&fresh).unwrap(), "SANDBOX_CRUMB");
    }
}
