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
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn get_sandbox_dir() -> PathBuf {
    crate::platform::PlatformContext::detect()
        .home_dir
        .join("workspace/aien-sandbox")
}

pub fn get_repo_dir() -> PathBuf {
    crate::platform::PlatformContext::detect()
        .home_dir
        .join("workspace/aien-sovereign-core")
}

fn cargo_cmd() -> Command {
    let platform = crate::platform::PlatformContext::detect();
    let user_cargo = platform.home_dir.join(".cargo/bin/cargo");
    let cargo_bin = if user_cargo.exists() {
        user_cargo.to_string_lossy().to_string()
    } else {
        "cargo".to_string()
    };
    let mut cmd = Command::new(cargo_bin);
    let cargo_bin_dir = platform.home_dir.join(".cargo/bin");
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", format!("{}:{}", cargo_bin_dir.display(), path));
    } else {
        cmd.env(
            "PATH",
            format!("{}:/usr/local/bin:/usr/bin:/bin", cargo_bin_dir.display()),
        );
    }
    cmd
}

pub fn init_sandbox(branch_opt: Option<&str>) -> Result<Value, String> {
    let sb_dir = get_sandbox_dir();
    let rp_dir = get_repo_dir();
    let sandbox_path = sb_dir.as_path();
    let repo_path = rp_dir.as_path();

    if !repo_path.exists() {
        return Err(format!("Base repository not found at {}", rp_dir.display()));
    }

    let branch = branch_opt.unwrap_or("sandbox-auto");

    if sandbox_path.exists() {
        let _ = clean_sandbox();
    }

    let output = Command::new("git")
        .args([
            "worktree",
            "add",
            "-B",
            branch,
            sb_dir.to_str().unwrap_or(""),
            "main",
        ])
        .current_dir(&rp_dir)
        .output()
        .map_err(|e| format!("Failed to create git worktree: {}", e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git worktree add failed: {}", err));
    }

    let crumb_path = sandbox_path.join(".crumb");
    if !crumb_path.exists() {
        let above = sb_dir
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let crumb_content = format!(
            r#"{{
  "directory": "aien-sandbox",
  "statement": "Isolated Git worktree sandbox for AIEN autonomous edits and testing.",
  "above": "{}",
  "below": []
}}"#,
            above
        );
        let _ = crate::crumbs::write_new_crumb(&crumb_path, &crumb_content);
    }

    Ok(json!({
        "status": "ok",
        "action": "init",
        "sandbox_dir": sb_dir.to_string_lossy(),
        "branch": branch,
        "message": format!("Sandbox initialized cleanly at {} on branch {}", sb_dir.display(), branch)
    }))
}

pub fn status_sandbox() -> Result<Value, String> {
    let sb_dir = get_sandbox_dir();
    let sandbox_path = sb_dir.as_path();
    if !sandbox_path.exists() {
        return Ok(json!({
            "status": "not_initialized",
            "sandbox_dir": sb_dir.to_string_lossy(),
            "exists": false
        }));
    }

    let status_out = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(&sb_dir)
        .output()
        .map_err(|e| e.to_string())?;

    let branch_out = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(&sb_dir)
        .output()
        .map_err(|e| e.to_string())?;

    let branch = String::from_utf8_lossy(&branch_out.stdout)
        .trim()
        .to_string();
    let status_text = String::from_utf8_lossy(&status_out.stdout).to_string();
    let modified_files: Vec<String> = status_text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    Ok(json!({
        "status": "ok",
        "sandbox_dir": sb_dir.to_string_lossy(),
        "branch": branch,
        "is_clean": modified_files.is_empty(),
        "modified_files": modified_files
    }))
}

pub fn test_sandbox() -> Result<Value, String> {
    let sb_dir = get_sandbox_dir();
    let sandbox_path = sb_dir.as_path();
    if !sandbox_path.exists() {
        return Err(format!(
            "Sandbox not found at {}. Run init first.",
            sb_dir.display()
        ));
    }

    let mut cmd = cargo_cmd();
    cmd.args(["check", "--tests"]).current_dir(&sb_dir);
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to run cargo check: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let success = output.status.success();

    Ok(json!({
        "status": if success { "passed" } else { "failed" },
        "action": "check",
        "stdout": stdout,
        "stderr": stderr
    }))
}

pub fn promote_sandbox(commit_msg: &str) -> Result<Value, String> {
    let sb_dir = get_sandbox_dir();
    let rp_dir = get_repo_dir();
    let sandbox_path = sb_dir.as_path();
    if !sandbox_path.exists() {
        return Err(format!("Sandbox not found at {}", sb_dir.display()));
    }

    // 1. Stage and commit in sandbox
    let _ = Command::new("git")
        .args(["add", "-A"])
        .current_dir(&sb_dir)
        .output();

    let commit_out = Command::new("git")
        .args(["commit", "-m", commit_msg])
        .current_dir(&sb_dir)
        .output()
        .map_err(|e| format!("Commit failed: {}", e))?;

    if !commit_out.status.success() {
        let err = String::from_utf8_lossy(&commit_out.stderr);
        return Err(format!("git commit in sandbox failed: {}", err));
    }

    // 2. Fast-forward or merge into main repo
    let branch_out = Command::new("git")
        .args(["branch", "--show-current"])
        .current_dir(&sb_dir)
        .output()
        .map_err(|e| e.to_string())?;
    let branch = String::from_utf8_lossy(&branch_out.stdout)
        .trim()
        .to_string();

    let merge_out = Command::new("git")
        .args(["merge", &branch, "--ff-only"])
        .current_dir(&rp_dir)
        .output()
        .map_err(|e| format!("Merge into base failed: {}", e))?;

    let promoted_ok = merge_out.status.success();

    // 3. Trigger release build and self-update if aien-cli was modified
    let mut install_success = false;
    if promoted_ok {
        let mut build_cmd = cargo_cmd();
        build_cmd
            .args(["build", "--release", "-p", "aien-cli"])
            .current_dir(&rp_dir);
        let _ = build_cmd.output();

        let release_bin = {
            let ws_bin = format!("{}/target/release/aien-cli", rp_dir.display());
            if Path::new(&ws_bin).exists() {
                ws_bin
            } else {
                format!(
                    "{}/crates/aien-cli/target/release/aien-cli",
                    rp_dir.display()
                )
            }
        };
        let dest_path = crate::platform::PlatformContext::detect()
            .home_dir
            .join(".local/bin/aien");
        let dest_bin = dest_path.to_str().unwrap_or("aien");
        let install_status = Command::new("install")
            .args(["-m", "755", &release_bin, dest_bin])
            .status();
        if let Ok(s) = install_status {
            install_success = s.success();
        }
    }

    Ok(json!({
        "status": if promoted_ok { "promoted" } else { "merge_failed" },
        "branch": branch,
        "binary_updated": install_success,
        "merge_output": String::from_utf8_lossy(&merge_out.stdout).to_string(),
        "merge_error": String::from_utf8_lossy(&merge_out.stderr).to_string()
    }))
}

pub fn clean_sandbox() -> Result<String, String> {
    let sb_dir = get_sandbox_dir();
    let rp_dir = get_repo_dir();
    let sandbox_path = sb_dir.as_path();
    if sandbox_path.exists() {
        let _ = Command::new("git")
            .args([
                "worktree",
                "remove",
                "--force",
                sb_dir.to_str().unwrap_or(""),
            ])
            .current_dir(&rp_dir)
            .output();

        if sandbox_path.exists() {
            let _ = fs::remove_dir_all(sandbox_path);
        }
    }

    let _ = Command::new("git")
        .args(["worktree", "prune"])
        .current_dir(&rp_dir)
        .output();

    Ok(format!(
        "Sandbox at {} cleaned successfully",
        sb_dir.display()
    ))
}

pub fn format_sandbox_tui() -> String {
    match status_sandbox() {
        Ok(v) => {
            let status = v.get("status").and_then(Value::as_str).unwrap_or("unknown");
            if status == "not_initialized" {
                format!(
                    "  {} Sandbox: Not initialized (run /sandbox init)
",
                    "○".yellow()
                )
            } else {
                let branch = v.get("branch").and_then(Value::as_str).unwrap_or("none");
                let is_clean = v.get("is_clean").and_then(Value::as_bool).unwrap_or(false);
                let clean_str = if is_clean {
                    "Clean".green()
                } else {
                    "Modified".yellow().bold()
                };
                format!(
                    "  {} Sandbox: Active [branch: {}] • Status: {}
",
                    "●".green(),
                    branch.cyan(),
                    clean_str
                )
            }
        }
        Err(e) => format!(
            "  {} Sandbox: Error ({})
",
            "✗".red(),
            e
        ),
    }
}

pub fn sandbox_dispatch_tool(args: &Value) -> Value {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("status");
    match action {
        "status" => match status_sandbox() {
            Ok(v) => v,
            Err(e) => json!({"status": "error", "error": e}),
        },
        "init" => {
            let branch = args.get("branch").and_then(Value::as_str);
            match init_sandbox(branch) {
                Ok(v) => v,
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "test" | "check" => match test_sandbox() {
            Ok(v) => v,
            Err(e) => json!({"status": "error", "error": e}),
        },
        "promote" => {
            let msg = args
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Autonomous sandbox promotion");
            match promote_sandbox(msg) {
                Ok(v) => v,
                Err(e) => json!({"status": "error", "error": e}),
            }
        }
        "clean" => match clean_sandbox() {
            Ok(msg) => json!({"status": "ok", "message": msg}),
            Err(e) => json!({"status": "error", "error": e}),
        },
        other => json!({"status": "error", "error": format!("Unknown sandbox action '{}'", other)}),
    }
}

pub fn browser_dispatch_tool(action: &str, prompt: Option<&str>) -> Value {
    browser_dispatch_tool_at(action, prompt, None, None)
}

/// Browser mirror with an explicit Cockpit URL and output directory.
/// `None` falls back to the Cockpit default (127.0.0.1:18095) and
/// `~/basecamp/ui-tests`. The AIEN_COCKPIT_URL environment variable
/// overrides the default URL.
pub fn browser_dispatch_tool_at(
    action: &str,
    prompt: Option<&str>,
    url: Option<&str>,
    out_dir: Option<&std::path::Path>,
) -> Value {
    use aien_cli::browser_mirror as bm;
    let env_url = std::env::var("AIEN_COCKPIT_URL").ok();
    let url = url
        .map(str::to_string)
        .or(env_url)
        .unwrap_or_else(|| bm::DEFAULT_URL.to_string());
    let out = out_dir
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(bm::default_out_dir);
    bm::dispatch(action, prompt, &url, &out)
}

pub fn handle_sandbox_command(parts: &[&str]) {
    if parts.is_empty() || parts[0] == "status" {
        match status_sandbox() {
            Ok(val) => {
                println!(
                    "
{}",
                    "=== AIEN Sovereign Sandbox Status ===".cyan().bold()
                );
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            }
            Err(e) => println!("{}", format!("Error: {}", e).red()),
        }
    } else if parts[0] == "init" {
        let branch = if parts.len() > 1 {
            Some(parts[1])
        } else {
            None
        };
        match init_sandbox(branch) {
            Ok(val) => {
                println!(
                    "
{}",
                    "=== AIEN Sandbox Initialized ===".green().bold()
                );
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            }
            Err(e) => println!("{}", format!("Error: {}", e).red()),
        }
    } else if parts[0] == "test" {
        println!("{}", "Running test suite inside sandbox...".yellow());
        match test_sandbox() {
            Ok(val) => {
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            }
            Err(e) => println!("{}", format!("Error: {}", e).red()),
        }
    } else if parts[0] == "promote" {
        let msg = if parts.len() > 1 {
            parts[1..].join(" ")
        } else {
            "Autonomous sandbox promotion".to_string()
        };
        match promote_sandbox(&msg) {
            Ok(val) => {
                println!(
                    "
{}",
                    "=== Sandbox Promoted to Main ===".green().bold()
                );
                println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
            }
            Err(e) => println!("{}", format!("Error: {}", e).red()),
        }
    } else if parts[0] == "clean" {
        match clean_sandbox() {
            Ok(msg) => println!("{}", msg.green()),
            Err(e) => println!("{}", format!("Error: {}", e).red()),
        }
    } else {
        println!("Usage: /sandbox [status | init [branch] | test | promote [commit msg] | clean]");
    }
}

pub fn handle_browser_command(parts: &[&str]) {
    let action = parts.first().copied().unwrap_or("test");
    let rest = if parts.len() > 1 {
        Some(parts[1..].join(" "))
    } else {
        None
    };
    let note = match action {
        "mentor" => format!(
            "Spawning browser mentor session with prompt: '{}'...",
            rest.as_deref()
                .unwrap_or("AIEN, report self-improvement status.")
        ),
        "screenshot" => "Spawning headless Chrome for a Cockpit screenshot...".to_string(),
        "eval" => "Spawning headless Chrome to evaluate JavaScript on the Cockpit...".to_string(),
        _ => "Spawning headless Chrome and running mirror self-test on Cockpit...".to_string(),
    };
    println!("{}", note.yellow());
    let res = browser_dispatch_tool(action, rest.as_deref());
    println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_constants() {
        let platform = crate::platform::PlatformContext::detect();
        assert_eq!(
            get_sandbox_dir(),
            platform.home_dir.join("workspace/aien-sandbox")
        );
        assert_eq!(
            get_repo_dir(),
            platform.home_dir.join("workspace/aien-sovereign-core")
        );
    }

    #[test]
    fn test_sandbox_dispatch_unknown() {
        let args = json!({"action": "invalid_action"});
        let res = sandbox_dispatch_tool(&args);
        assert_eq!(res.get("status").unwrap().as_str().unwrap(), "error");
    }

    #[test]
    fn test_sandbox_status_serialization() {
        let res = status_sandbox();
        assert!(res.is_ok());
    }
}
