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

// spark-aegis Deep Scanner
// Scans files, directories, and git diffs for secrets, slop, and dangerous commands

use crate::byte_scan;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationCategory {
    PlaintextSecret,
    AntiSlop,
    DangerousCommand,
    ProhibitedFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub category: ViolationCategory,
    pub severity: ViolationSeverity,
    pub rule: String,
    pub file_path: PathBuf,
    pub line_number: Option<usize>,
    pub column: Option<usize>,
    pub matched_text: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    pub target: String,
    pub scanned_files_count: usize,
    pub violations: Vec<Violation>,
    pub passed: bool,
    pub blake3_hash: String,
    pub elapsed_ms: f64,
}

struct SecretRule {
    name: &'static str,
    regex: LazyLock<Regex>,
    severity: ViolationSeverity,
    remediation: &'static str,
}

struct SlopRule {
    name: &'static str,
    regex: LazyLock<Regex>,
    remediation: &'static str,
}

struct DangerousCommandRule {
    name: &'static str,
    regex: LazyLock<Regex>,
    severity: ViolationSeverity,
    remediation: &'static str,
}

static SECRET_RULES: LazyLock<Vec<SecretRule>> = LazyLock::new(|| {
    vec![
        SecretRule {
            name: "AWS Access Key ID",
            regex: LazyLock::new(|| Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap()),
            severity: ViolationSeverity::Critical,
            remediation: "Store AWS credentials in the hardware TPM key vault.",
        },
        SecretRule {
            name: "GitHub Personal Access Token",
            regex: LazyLock::new(|| {
                Regex::new(r"\b(?:ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36}\b|\bgithub_pat_[0-9a-zA-Z_]{22,82}\b").unwrap()
            }),
            severity: ViolationSeverity::Critical,
            remediation: "Store GitHub token in the hardware TPM vault (atlas-vault add).",
        },
        SecretRule {
            name: "Slack Token",
            regex: LazyLock::new(|| Regex::new(r"\bxox[baprs]-[0-9a-zA-Z]{10,48}\b").unwrap()),
            severity: ViolationSeverity::Critical,
            remediation: "Store Slack token in the hardware TPM vault.",
        },
        SecretRule {
            name: "Private Key Header",
            regex: LazyLock::new(|| {
                Regex::new(
                    r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----",
                )
                .unwrap()
            }),
            severity: ViolationSeverity::Critical,
            remediation: "Remove private key from filesystem. Use TPM-bound keys.",
        },
        SecretRule {
            name: "Generic Hardcoded Secret Assignment",
            regex: LazyLock::new(|| {
                Regex::new(r#"(?i)\b(?:api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token)\s*[:=]\s*['"][0-9a-zA-Z_\-]{16,}['"]"#).unwrap()
            }),
            severity: ViolationSeverity::High,
            remediation: "Resolve secret dynamically in memory via atlas-vault get.",
        },
    ]
});

static SLOP_RULES: LazyLock<Vec<SlopRule>> = LazyLock::new(|| {
    vec![
        SlopRule {
            name: "Forbidden Em Dash",
            regex: LazyLock::new(|| Regex::new("\u{2014}").unwrap()),
            remediation: "Replace em dash with comma, colon, parentheses, or period.",
        },
        SlopRule {
            name: "Forbidden En Dash",
            regex: LazyLock::new(|| Regex::new("\u{2013}").unwrap()),
            remediation: "Replace en dash with hyphen or rewrite clause.",
        },
        SlopRule {
            name: "Banned AI Buzzword",
            regex: LazyLock::new(|| {
                Regex::new(r"(?i)\b(delve[a-z]*|tapestr[a-z]*|testament[a-z]*|beacon[a-z]*|crucial[a-z]*|pivotal[a-z]*|elevat[a-z]+|game-changer|unleash[a-z]*|harness[a-z]*|seamless[a-z]*)\b").unwrap()
            }),
            remediation: "Remove banned AI cliché. State technical facts directly.",
        },
        SlopRule {
            name: "Banned Transitional Fluff",
            regex: LazyLock::new(|| {
                Regex::new(r"(?i)\b(furthermore|moreover|in conclusion|at its core)\b").unwrap()
            }),
            remediation: "Omit transitional filler. Lead immediately with evidence or code.",
        },
        SlopRule {
            name: "Banned Antithesis Trope",
            regex: LazyLock::new(|| {
                Regex::new(r"(?i)\b(?:not only\b[^\n.!?]{1,60}\bbut also|it's not\b[^\n.!?]{1,60}\bit's)\b").unwrap()
            }),
            remediation: "Omit rhetorical antithesis trope. State facts directly.",
        },
    ]
});

static DANGEROUS_COMMAND_RULES: LazyLock<Vec<DangerousCommandRule>> = LazyLock::new(|| {
    vec![
        DangerousCommandRule {
            name: "Destructive Root Filesystem Removal",
            regex: LazyLock::new(|| {
                Regex::new(r#"rm\s+-(?:[a-zA-Z]*r[a-zA-Z]*f|[a-zA-Z]*f[a-zA-Z]*r)\s+(?:/|/\*|~|~/\*)(?:\s|$)"#).unwrap()
            }),
            severity: ViolationSeverity::Critical,
            remediation: "Command removes root or home directory. Abort immediately.",
        },
        DangerousCommandRule {
            name: "Unsandboxed Pipe to Shell",
            regex: LazyLock::new(|| {
                Regex::new(r#"(?:curl|wget)\s+[^|\n]+?\|\s*(?:ba)?sh\b"#).unwrap()
            }),
            severity: ViolationSeverity::High,
            remediation: "Download file, inspect Blake3 checksum, and execute in isolation.",
        },
        DangerousCommandRule {
            name: "Fork Bomb Pattern",
            regex: LazyLock::new(|| Regex::new(r#":\(\)\s*\{\s*:\|:&\s*\};:"#).unwrap()),
            severity: ViolationSeverity::Critical,
            remediation: "Malicious fork bomb detected. Isolate process immediately.",
        },
        DangerousCommandRule {
            name: "Unrestricted Global Permissions",
            regex: LazyLock::new(|| {
                Regex::new(r#"chmod\s+(?:-R\s+)?777\s+(?:/|~)(?:\s|$)"#).unwrap()
            }),
            severity: ViolationSeverity::Critical,
            remediation: "Do not apply 777 permissions to root or home directory.",
        },
        DangerousCommandRule {
            name: "Raw Sudo Command in Script",
            regex: LazyLock::new(|| {
                Regex::new(r#"(?m)^\s*sudo\s+(?:rm|chmod|chown|dd|mkfs)\b"#).unwrap()
            }),
            severity: ViolationSeverity::High,
            remediation: "Explicit privilege escalation requires audited sandbox confinement.",
        },
    ]
});

pub fn is_prohibited_filename(path: &Path) -> Option<(&'static str, &'static str)> {
    if let Some(file_name) = path.file_name().and_then(|f| f.to_str()) {
        if file_name == ".env" || file_name.starts_with(".env.") {
            return Some((
                "Plaintext .env Environment File",
                "Zero Disk Secrets invariant violated. Remove .env and use hardware TPM vault.",
            ));
        }
        if file_name.ends_with(".pem") || file_name.ends_with(".key") {
            return Some((
                "Unencrypted Private Key File",
                "Zero Disk Secrets invariant violated. Store keys in TPM vault.",
            ));
        }
        if file_name == "id_rsa" || file_name == "id_ed25519" || file_name == "id_ecdsa" {
            return Some((
                "Raw SSH Private Key in Workspace",
                "Move SSH keys to ~/.ssh or hardware security token.",
            ));
        }
    }
    None
}

pub fn scan_content(path: &Path, content: &str) -> Vec<Violation> {
    let mut violations = Vec::new();

    // Check prohibited filename
    if let Some((rule, remediation)) = is_prohibited_filename(path) {
        violations.push(Violation {
            category: ViolationCategory::ProhibitedFile,
            severity: ViolationSeverity::Critical,
            rule: rule.to_string(),
            file_path: path.to_path_buf(),
            line_number: None,
            column: None,
            matched_text: path.display().to_string(),
            remediation: remediation.to_string(),
        });
    }

    // Fast check for em and en dashes: skip the per-line scan when no E2 lead byte exists
    let bytes = content.as_bytes();
    // Em dash UTF-8: E2 80 94 (226, 128, 148)
    // En dash UTF-8: E2 80 93 (226, 128, 147)
    let has_utf8_prefix = byte_scan::scan_byte(bytes, 0xE2).is_some();

    for (line_idx, line) in content.lines().enumerate() {
        let line_num = line_idx + 1;

        // 1. Plaintext Secrets
        for rule in SECRET_RULES.iter() {
            for mat in rule.regex.find_iter(line) {
                violations.push(Violation {
                    category: ViolationCategory::PlaintextSecret,
                    severity: rule.severity,
                    rule: rule.name.to_string(),
                    file_path: path.to_path_buf(),
                    line_number: Some(line_num),
                    column: Some(mat.start() + 1),
                    matched_text: "[REDACTED_BY_ATLAS_VAULT]".to_string(),
                    remediation: rule.remediation.to_string(),
                });
            }
        }

        // 2. Anti-Slop Violations
        if has_utf8_prefix || line.contains('\u{2014}') || line.contains('\u{2013}') {
            for rule in &SLOP_RULES[0..2] {
                for mat in rule.regex.find_iter(line) {
                    violations.push(Violation {
                        category: ViolationCategory::AntiSlop,
                        severity: ViolationSeverity::Medium,
                        rule: rule.name.to_string(),
                        file_path: path.to_path_buf(),
                        line_number: Some(line_num),
                        column: Some(mat.start() + 1),
                        matched_text: mat.as_str().to_string(),
                        remediation: rule.remediation.to_string(),
                    });
                }
            }
        }

        for rule in &SLOP_RULES[2..] {
            for mat in rule.regex.find_iter(line) {
                violations.push(Violation {
                    category: ViolationCategory::AntiSlop,
                    severity: ViolationSeverity::Medium,
                    rule: rule.name.to_string(),
                    file_path: path.to_path_buf(),
                    line_number: Some(line_num),
                    column: Some(mat.start() + 1),
                    matched_text: mat.as_str().to_string(),
                    remediation: rule.remediation.to_string(),
                });
            }
        }

        // 3. Dangerous Commands
        for rule in DANGEROUS_COMMAND_RULES.iter() {
            for mat in rule.regex.find_iter(line) {
                violations.push(Violation {
                    category: ViolationCategory::DangerousCommand,
                    severity: rule.severity,
                    rule: rule.name.to_string(),
                    file_path: path.to_path_buf(),
                    line_number: Some(line_num),
                    column: Some(mat.start() + 1),
                    matched_text: mat.as_str().to_string(),
                    remediation: rule.remediation.to_string(),
                });
            }
        }
    }

    violations
}

pub fn scan_path(target_path: &Path) -> Result<ScanReport> {
    let start = Instant::now();
    let mut files_scanned = 0;
    let mut violations = Vec::new();
    let mut hasher = blake3::Hasher::new();

    if target_path.is_file() {
        files_scanned += 1;
        let content = fs::read_to_string(target_path).unwrap_or_default();
        hasher.update(content.as_bytes());
        violations.extend(scan_content(target_path, &content));
    } else if target_path.is_dir() {
        let mut dirs = vec![target_path.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            let entries = match fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let p = entry.path();
                let file_name = match p.file_name().and_then(|f| f.to_str()) {
                    Some(s) => s,
                    None => continue,
                };

                // Skip version control and build directories
                if file_name == ".git" || file_name == "target" || file_name == "node_modules" {
                    continue;
                }

                if p.is_dir() {
                    dirs.push(p);
                } else if p.is_file() {
                    files_scanned += 1;
                    if let Some((rule, remediation)) = is_prohibited_filename(&p) {
                        violations.push(Violation {
                            category: ViolationCategory::ProhibitedFile,
                            severity: ViolationSeverity::Critical,
                            rule: rule.to_string(),
                            file_path: p.clone(),
                            line_number: None,
                            column: None,
                            matched_text: p.display().to_string(),
                            remediation: remediation.to_string(),
                        });
                    }
                    if let Ok(content) = fs::read_to_string(&p) {
                        hasher.update(content.as_bytes());
                        violations.extend(scan_content(&p, &content));
                    }
                }
            }
        }
    } else {
        anyhow::bail!("Target path {:?} does not exist", target_path);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let digest = hasher.finalize().to_hex().to_string();
    let passed = violations.is_empty();

    Ok(ScanReport {
        target: target_path.display().to_string(),
        scanned_files_count: files_scanned,
        violations,
        passed,
        blake3_hash: digest,
        elapsed_ms: elapsed,
    })
}

pub fn scan_diff(diff_content: &str) -> Result<ScanReport> {
    let start = Instant::now();
    let mut violations = Vec::new();
    let mut current_file = PathBuf::from("diff");
    let mut line_counter: usize = 0;

    for line in diff_content.lines() {
        if let Some(path_str) = line.strip_prefix("+++ b/") {
            current_file = PathBuf::from(path_str);
            if let Some((rule, remediation)) = is_prohibited_filename(&current_file) {
                violations.push(Violation {
                    category: ViolationCategory::ProhibitedFile,
                    severity: ViolationSeverity::Critical,
                    rule: rule.to_string(),
                    file_path: current_file.clone(),
                    line_number: None,
                    column: None,
                    matched_text: current_file.display().to_string(),
                    remediation: remediation.to_string(),
                });
            }
            continue;
        }

        if line.starts_with('+') && !line.starts_with("+++") {
            line_counter += 1;
            let added_text = &line[1..];
            let line_violations = scan_content(&current_file, added_text);
            for mut v in line_violations {
                v.line_number = Some(line_counter);
                violations.push(v);
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let digest = blake3::hash(diff_content.as_bytes()).to_hex().to_string();
    let passed = violations.is_empty();

    Ok(ScanReport {
        target: "git-diff".to_string(),
        scanned_files_count: 1,
        violations,
        passed,
        blake3_hash: digest,
        elapsed_ms: elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_detection() {
        let fake_aws = "export AWS_KEY=AKIAIOSFODNN7EXAMPLE\n";
        let v = scan_content(Path::new("test.sh"), fake_aws);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].category, ViolationCategory::PlaintextSecret);
        assert_eq!(v[0].matched_text, "[REDACTED_BY_ATLAS_VAULT]");
    }

    #[test]
    fn test_slop_detection_em_dash() {
        let bad_line = "This is a feature \u{2014} it elevates throughput.\n";
        let v = scan_content(Path::new("doc.md"), bad_line);
        assert!(v.iter().any(|item| item.rule == "Forbidden Em Dash"));
        assert!(v.iter().any(|item| item.rule == "Banned AI Buzzword"));
    }

    #[test]
    fn test_dangerous_command_detection() {
        let dangerous = "rm -rf /\n";
        let v = scan_content(Path::new("script.sh"), dangerous);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].category, ViolationCategory::DangerousCommand);
    }

    #[test]
    fn test_prohibited_env_file() {
        let path = Path::new("/workspace/.env");
        let v = scan_content(path, "FOO=BAR\n");
        assert!(v
            .iter()
            .any(|item| item.category == ViolationCategory::ProhibitedFile));
    }
}
