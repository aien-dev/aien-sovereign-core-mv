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

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::platform::PlatformContext;

#[derive(Debug, Clone, PartialEq)]
pub enum SafetyDecision {
    Allow,
    AskUser(String),
    Deny(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum PolicyPriority {
    SpecificDeny = 1,
    SpecificAsk = 2,
    SpecificAllow = 3,
    PrefixWildcardDeny = 4,
    PrefixWildcardAsk = 5,
    PrefixWildcardAllow = 6,
    GlobalWildcardDeny = 7,
    GlobalWildcardAsk = 8,
    GlobalWildcardAllow = 9,
}

pub type PredicateFn = Arc<dyn Fn(&Value) -> Result<bool, String> + Send + Sync>;

#[derive(Clone)]
#[allow(dead_code)]
pub struct PolicyRule {
    pub name: String,
    pub tool_pattern: String,
    pub priority: PolicyPriority,
    pub predicate: Option<PredicateFn>,
    pub action: SafetyDecision,
}

pub struct SafetyEngine {
    rules: Vec<PolicyRule>,
    workspaces: Vec<PathBuf>,
}

impl SafetyEngine {
    pub fn new() -> Self {
        let platform = PlatformContext::detect();
        let mut workspaces = platform.default_workspaces;
        workspaces.push(PathBuf::from("/tmp"));

        Self {
            rules: Vec::new(),
            workspaces,
        }
    }

    #[allow(dead_code)]
    pub fn workspaces(&self) -> &[PathBuf] {
        &self.workspaces
    }

    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    #[allow(dead_code)]
    pub fn set_workspaces(&mut self, workspaces: Vec<PathBuf>) {
        self.workspaces = workspaces;
    }

    pub fn validate_path(&self, path: &str) -> Result<PathBuf, String> {
        validate_workspace_path(path, &self.workspaces)
    }

    pub fn evaluate(&self, tool: &str, args: &Value) -> SafetyDecision {
        // Foundational Primitive: Path Confinement for all filesystem tools
        match tool {
            "view_file" | "write_to_file" | "replace_file_content" | "list_dir" | "create_dir" => {
                if let Some(path_str) = args.get("path").and_then(Value::as_str) {
                    if let Err(err) = self.validate_path(path_str) {
                        return SafetyDecision::Deny(format!("CONFINEMENT DENIAL: {}", err));
                    }
                }
            }
            "run_command" => {
                if let Some(cwd_str) = args.get("cwd").and_then(Value::as_str) {
                    if let Err(err) = self.validate_path(cwd_str) {
                        return SafetyDecision::Deny(format!(
                            "CONFINEMENT DENIAL: Invalid command working directory: {}",
                            err
                        ));
                    }
                }
            }
            _ => {}
        }

        for rule in &self.rules {
            if !self.matches_tool_pattern(&rule.tool_pattern, tool) {
                continue;
            }

            if let Some(pred) = &rule.predicate {
                match pred(args) {
                    Ok(true) => return rule.action.clone(),
                    Ok(false) => continue,
                    Err(err) => {
                        // Fail-closed guarantee from Antigravity SDK:
                        // Predicate evaluation failure treats the rule as matched for Deny/Ask
                        return match &rule.action {
                            SafetyDecision::Deny(msg) => SafetyDecision::Deny(format!(
                                "{}: [Predicate Evaluation Error: {}]",
                                msg, err
                            )),
                            SafetyDecision::AskUser(msg) => SafetyDecision::AskUser(format!(
                                "{}: [Predicate Evaluation Error: {}]",
                                msg, err
                            )),
                            SafetyDecision::Allow => SafetyDecision::Deny(format!(
                                "Predicate failed closed on allow rule: {}",
                                err
                            )),
                        };
                    }
                }
            } else {
                return rule.action.clone();
            }
        }

        // Default conservative fallback
        SafetyDecision::AskUser(format!("No matching safety rule for tool '{}'", tool))
    }

    fn matches_tool_pattern(&self, pattern: &str, tool: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        if let Some(prefix) = pattern.strip_suffix("/*") {
            return tool.starts_with(prefix);
        }
        pattern == tool
    }

    pub fn default_sovereign_engine() -> Self {
        let mut engine = Self::new();

        // 1. SPECIFIC DENY: Block writing plaintext secrets to disk (.env / keys)
        engine.add_rule(PolicyRule {
            name: "deny_env_secret_writes".to_string(),
            tool_pattern: "write_to_file".to_string(),
            priority: PolicyPriority::SpecificDeny,
            predicate: Some(Arc::new(|args| {
                let path = args.get("path").and_then(Value::as_str).unwrap_or("");
                Ok(is_secret_leak_path(path))
            })),
            action: SafetyDecision::Deny("REJECTED BY SOVEREIGN VAULT POLICY: Writing plaintext secrets into .env or credential files is forbidden. Use 'atlas-vault add <KEY>'.".to_string()),
        });

        engine.add_rule(PolicyRule {
            name: "deny_env_secret_replaces".to_string(),
            tool_pattern: "replace_file_content".to_string(),
            priority: PolicyPriority::SpecificDeny,
            predicate: Some(Arc::new(|args| {
                let path = args.get("path").and_then(Value::as_str).unwrap_or("");
                Ok(is_secret_leak_path(path))
            })),
            action: SafetyDecision::Deny("REJECTED BY SOVEREIGN VAULT POLICY: Writing plaintext secrets into .env or credential files is forbidden. Use 'atlas-vault add <KEY>'.".to_string()),
        });

        // 2. SPECIFIC DENY: Block writing unmasked API keys or private keys into any file
        engine.add_rule(PolicyRule {
            name: "deny_unmasked_secret_content".to_string(),
            tool_pattern: "write_to_file".to_string(),
            priority: PolicyPriority::SpecificDeny,
            predicate: Some(Arc::new(|args| {
                let content = args.get("content").and_then(Value::as_str).unwrap_or("");
                Ok(contains_unmasked_secret(content).is_some())
            })),
            action: SafetyDecision::Deny("REJECTED BY SOVEREIGN VAULT POLICY: Plaintext API key or credential pattern detected. Use atlas-vault.".to_string()),
        });

        // 3. SPECIFIC ASK: Destructive commands require operator confirmation
        engine.add_rule(PolicyRule {
            name: "ask_destructive_commands".to_string(),
            tool_pattern: "run_command".to_string(),
            priority: PolicyPriority::SpecificAsk,
            predicate: Some(Arc::new(|args| {
                let cmd = args.get("command").and_then(Value::as_str).unwrap_or("");
                Ok(is_high_risk_command(cmd))
            })),
            action: SafetyDecision::AskUser(
                "Potentially destructive command detected.".to_string(),
            ),
        });

        // 4. SPECIFIC ASK: Writing to OS system paths (/etc, /boot, /root, /usr)
        engine.add_rule(PolicyRule {
            name: "ask_system_path_writes".to_string(),
            tool_pattern: "write_to_file".to_string(),
            priority: PolicyPriority::SpecificAsk,
            predicate: Some(Arc::new(|args| {
                let path = args.get("path").and_then(Value::as_str).unwrap_or("");
                Ok(is_high_risk_path(path))
            })),
            action: SafetyDecision::AskUser(
                "Writing to system operating system path requires operator confirmation."
                    .to_string(),
            ),
        });

        // 5. SPECIFIC ALLOW: Safe standard tool execution
        let safe_tools = [
            "view_file",
            "list_dir",
            "grep_search",
            "crumb",
            "vault",
            "goal",
            "skill",
            "cortex",
            "walkthrough",
            "hive",
            "sandbox",
            "browser",
            "invoke_subagent",
            "subagent",
            "subagents",
            "adapter",
            "model_adapter",
            "model_adapters",
            "socratic",
            "socratic_inquiry",
        ];
        for st in safe_tools {
            engine.add_rule(PolicyRule {
                name: format!("allow_{}", st),
                tool_pattern: st.to_string(),
                priority: PolicyPriority::SpecificAllow,
                predicate: None,
                action: SafetyDecision::Allow,
            });
        }

        // 6. GLOBAL WILDCARD ALLOW: Permissive default for verified standard commands
        engine.add_rule(PolicyRule {
            name: "allow_standard_commands".to_string(),
            tool_pattern: "run_command".to_string(),
            priority: PolicyPriority::GlobalWildcardAllow,
            predicate: None,
            action: SafetyDecision::Allow,
        });

        engine.add_rule(PolicyRule {
            name: "allow_standard_writes".to_string(),
            tool_pattern: "write_to_file".to_string(),
            priority: PolicyPriority::GlobalWildcardAllow,
            predicate: None,
            action: SafetyDecision::Allow,
        });

        engine.add_rule(PolicyRule {
            name: "allow_standard_replaces".to_string(),
            tool_pattern: "replace_file_content".to_string(),
            priority: PolicyPriority::GlobalWildcardAllow,
            predicate: None,
            action: SafetyDecision::Allow,
        });

        engine.add_rule(PolicyRule {
            name: "allow_standard_create_dir".to_string(),
            tool_pattern: "create_dir".to_string(),
            priority: PolicyPriority::GlobalWildcardAllow,
            predicate: None,
            action: SafetyDecision::Allow,
        });

        engine
    }
}

pub fn validate_workspace_path<P: AsRef<Path>>(
    path: P,
    workspaces: &[PathBuf],
) -> Result<PathBuf, String> {
    let raw = path.as_ref();
    let absolute = if raw.is_relative() {
        match std::env::current_dir() {
            Ok(cwd) => cwd.join(raw),
            Err(e) => {
                return Err(format!(
                    "Failed to determine current working directory: {}",
                    e
                ))
            }
        }
    } else {
        raw.to_path_buf()
    };

    // Keep the original component order. Collapsing `..` before resolving
    // symlinks can validate a different target from the path the OS opens.
    // This is static path validation, not a descriptor-bound race guarantee.
    let mut existing_ancestor = absolute.as_path();
    let mut subparts = Vec::new();
    loop {
        match std::fs::symlink_metadata(existing_ancestor) {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Only plain missing names can safely be reconstructed. A
                // parent traversal through a missing component has no resolved
                // filesystem meaning; ask the caller to name the target directly.
                match existing_ancestor.components().next_back() {
                    Some(std::path::Component::Normal(name)) => subparts.push(name),
                    _ => {
                        return Err(format!(
                            "Cannot resolve path '{}': use a direct path below an existing workspace directory",
                            raw.display()
                        ));
                    }
                }
                existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
                    format!("Cannot resolve parent of '{}'", raw.display())
                })?;
            }
            Err(e) => {
                return Err(format!(
                    "Failed to inspect path '{}': {}",
                    existing_ancestor.display(), e
                ));
            }
        }
    }

    // symlink_metadata notices a dangling link rather than treating it as
    // an absent name. canonicalize then refuses its unresolved target.
    let canon_ancestor = existing_ancestor.canonicalize().map_err(|e| {
        format!(
            "Failed to canonicalize ancestor '{}': {}",
            existing_ancestor.display(),
            e
        )
    })?;

    let mut canonical = canon_ancestor;
    for part in subparts.into_iter().rev() {
        canonical.push(part);
    }

    // Verify that the canonical path is enclosed within at least one permitted workspace root
    let permitted = workspaces.iter().any(|ws| {
        let ws_canon = ws.canonicalize().unwrap_or_else(|_| ws.clone());
        canonical.starts_with(&ws_canon)
    });

    if !permitted {
        return Err(format!(
            "Path '{}' resolves to '{}' which is outside permitted workspace roots",
            raw.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

pub fn is_high_risk_command(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let dangerous_patterns = [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf /home",
        "mkfs",
        "dd if=",
        "fdisk",
        "> /dev/sd",
        "> /dev/nvme",
        "shutdown",
        "reboot",
        "drop database",
        "drop table",
        "truncate table",
        "kill -9 1",
    ];

    for pattern in dangerous_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }
    false
}

pub fn is_high_risk_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.starts_with("/etc")
        || lower.starts_with("/boot")
        || lower.starts_with("/root")
        || lower.starts_with("/usr")
}

pub fn is_secret_leak_path(path: &str) -> bool {
    let p = Path::new(path);
    let file_name = p
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("")
        .to_lowercase();

    if file_name == ".env" || file_name.starts_with(".env.") || file_name.ends_with(".env") {
        return true;
    }
    if file_name == "secrets.json" || file_name == "credentials.json" || file_name == "id_rsa" {
        return true;
    }
    false
}

pub fn contains_unmasked_secret(content: &str) -> Option<&'static str> {
    let lower = content.to_lowercase();
    if lower.contains("api_key=") || lower.contains("secret_key=") || lower.contains("auth_token=")
    {
        return Some("Plaintext key assignment pattern detected (e.g. API_KEY=...)");
    }
    if content.contains("sk-proj-") || content.contains("ghp_") || content.contains("AIzaSy") {
        return Some("Live API key signature detected (sk-, ghp-, AIzaSy)");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_secret_leak_paths() {
        assert!(is_secret_leak_path(".env"));
        assert!(is_secret_leak_path("/path/to/.env"));
        assert!(is_secret_leak_path(".env.local"));
        assert!(is_secret_leak_path("production.env"));
        assert!(is_secret_leak_path("secrets.json"));
        assert!(!is_secret_leak_path("environment.rs"));
    }

    #[test]
    fn test_unmasked_secret_patterns() {
        assert!(contains_unmasked_secret("API_KEY=\"abcdefghijklmnop\"").is_some());
        assert!(contains_unmasked_secret("export sk-proj-1234567890abcdef").is_some());
        assert!(contains_unmasked_secret("ghp_1234567890abcdef").is_some());
        assert!(contains_unmasked_secret("let normal_code = true;").is_none());
    }

    #[test]
    fn test_path_confinement_validation() {
        let tmp = std::env::temp_dir();
        let allowed = vec![tmp.clone()];

        // Allowed path inside temp_dir
        let ok_file = tmp.join("test_file.txt");
        assert!(validate_workspace_path(&ok_file, &allowed).is_ok());

        // Escape attempt via parent traversal on existing path
        let escape_path = tmp.join("../etc/shadow");
        assert!(validate_workspace_path(&escape_path, &allowed).is_err());

        // Escape attempt via parent traversal on non-existent path
        let nonexistent_escape = tmp.join("does_not_exist/../../etc/passwd");
        assert!(validate_workspace_path(&nonexistent_escape, &allowed).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn path_confinement_respects_parent_after_symlink() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace");
        let outside = root.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir_all(outside.join("child")).unwrap();
        std::fs::write(outside.join("sentinel"), "synthetic").unwrap();
        symlink(outside.join("child"), workspace.join("link")).unwrap();
        let allowed = vec![workspace.clone()];
        assert!(validate_workspace_path(workspace.join("link/../sentinel"), &allowed).is_err());
        assert!(validate_workspace_path(workspace.join("link/../newdir"), &allowed).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn path_confinement_refuses_dangling_symlink_creation() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let workspace = root.path().join("workspace");
        let outside = root.path().join("outside");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(&outside).unwrap();
        symlink(outside.join("newfile"), workspace.join("newfile")).unwrap();
        assert!(validate_workspace_path(workspace.join("newfile"), &[workspace]).is_err());
        assert!(!outside.join("newfile").exists());
    }

    #[test]
    fn path_confinement_allows_normal_missing_suffix() {
        let workspace = tempfile::tempdir().unwrap();
        let path = workspace.path().join("newdir/newfile");
        let expected = workspace.path().canonicalize().unwrap().join("newdir/newfile");
        assert_eq!(
            validate_workspace_path(&path, &[workspace.path().to_path_buf()]).unwrap(),
            expected
        );
    }

    #[cfg(unix)]
    #[test]
    fn path_confinement_preserves_safe_symlinks_and_existing_parent_traversal() {
        use std::os::unix::fs::symlink;
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path();
        std::fs::create_dir(root.join("child")).unwrap();
        std::fs::write(root.join("sentinel"), "synthetic").unwrap();
        symlink(root.join("child"), root.join("link")).unwrap();
        let expected = root.canonicalize().unwrap().join("sentinel");
        assert_eq!(
            validate_workspace_path(root.join("link/../sentinel"), &[root.to_path_buf()]).unwrap(),
            expected
        );
        assert!(validate_workspace_path(root.join("link/newfile"), &[root.to_path_buf()]).is_ok());
    }

    #[test]
    fn path_confinement_refuses_unresolved_parent_components() {
        let workspace = tempfile::tempdir().unwrap();
        assert!(validate_workspace_path(
            workspace.path().join("absent/../newfile"),
            &[workspace.path().to_path_buf()]
        ).is_err());
    }

    #[test]
    fn test_policy_precedence_and_fail_closed() {
        let engine = SafetyEngine::default_sovereign_engine();

        // 1. Specific Deny beats everything: writing to .env
        let env_args = json!({"path": "/tmp/.env", "content": "KEY=123"});
        assert!(matches!(
            engine.evaluate("write_to_file", &env_args),
            SafetyDecision::Deny(_)
        ));

        // 2. Specific Ask: high-risk command
        let rm_args = json!({"command": "rm -rf /", "cwd": "/tmp"});
        assert!(matches!(
            engine.evaluate("run_command", &rm_args),
            SafetyDecision::AskUser(_)
        ));

        // 3. Specific Allow: standard read within allowed path
        let view_args = json!({"path": "/tmp/test.rs"});
        assert_eq!(
            engine.evaluate("view_file", &view_args),
            SafetyDecision::Allow
        );
    }
}
