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

// spark-aegis PR Triage and Autonomous Review Engine
// Verifies git diffs against sovereign invariants and certifies PR safety

use crate::scanner::{self, Violation, ViolationCategory};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictStatus {
    PASS,
    REJECT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrVerdict {
    pub status: VerdictStatus,
    pub pr_id: String,
    pub blake3_proof: String,
    pub zero_disk_secrets_certified: bool,
    pub unslop_certified: bool,
    pub safe_execution_certified: bool,
    pub violations: Vec<Violation>,
    pub summary: String,
    pub timestamp: DateTime<Utc>,
}

impl PrVerdict {
    pub fn is_pass(&self) -> bool {
        self.status == VerdictStatus::PASS
    }
}

pub fn review_pr_diff(diff_content: &str, pr_identifier: &str) -> Result<PrVerdict> {
    let report = scanner::scan_diff(diff_content)?;
    let ts = Utc::now();

    let mut secrets_clean = true;
    let mut slop_clean = true;
    let mut safe_clean = true;

    for v in &report.violations {
        match v.category {
            ViolationCategory::PlaintextSecret | ViolationCategory::ProhibitedFile => {
                secrets_clean = false;
            }
            ViolationCategory::AntiSlop => {
                slop_clean = false;
            }
            ViolationCategory::DangerousCommand => {
                safe_clean = false;
            }
        }
    }

    let passed = secrets_clean && slop_clean && safe_clean;
    let status = if passed {
        VerdictStatus::PASS
    } else {
        VerdictStatus::REJECT
    };

    let summary = if passed {
        format!(
            "PR {} certified safe. Zero disk secrets, unslop compliant, safe execution verified.",
            pr_identifier
        )
    } else {
        format!(
            "PR {} REJECTED with {} sovereign violation(s). Remediation required before merge.",
            pr_identifier,
            report.violations.len()
        )
    };

    Ok(PrVerdict {
        status,
        pr_id: pr_identifier.to_string(),
        blake3_proof: report.blake3_hash,
        zero_disk_secrets_certified: secrets_clean,
        unslop_certified: slop_clean,
        safe_execution_certified: safe_clean,
        violations: report.violations,
        summary,
        timestamp: ts,
    })
}

pub fn review_pr_file<P: AsRef<Path>>(file_path: P) -> Result<PrVerdict> {
    let path = file_path.as_ref();
    let content = fs::read_to_string(path)?;
    let identifier = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pr-diff");
    review_pr_diff(&content, identifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_pr_diff_passes() {
        let clean_diff = r#"
diff --git a/crates/spark-test/src/lib.rs b/crates/spark-test/src/lib.rs
index 1111111..2222222 100644
--- a/crates/spark-test/src/lib.rs
+++ b/crates/spark-test/src/lib.rs
@@ -1,3 +1,4 @@
 pub fn compute(x: i32) -> i32 {
+    x * 2
 }
"#;
        let verdict = review_pr_diff(clean_diff, "PR-101").unwrap();
        assert_eq!(verdict.status, VerdictStatus::PASS);
        assert!(verdict.zero_disk_secrets_certified);
        assert!(verdict.unslop_certified);
        assert!(verdict.safe_execution_certified);
        assert!(verdict.violations.is_empty());
    }

    #[test]
    fn test_slop_pr_diff_rejects() {
        let bad_diff =
            "+++ b/README.md\n+This tool elevates developer productivity \u{2014} guaranteed.\n";
        let verdict = review_pr_diff(bad_diff, "PR-102").unwrap();
        assert_eq!(verdict.status, VerdictStatus::REJECT);
        assert!(!verdict.unslop_certified);
        assert_eq!(verdict.violations.len(), 2); // "elevates" and em dash
    }

    #[test]
    fn test_secret_pr_diff_rejects() {
        let secret_diff = "+++ b/config.rs\n+const GITHUB_TOKEN: &str = \"ghp_123456789012345678901234567890123456\";\n";
        let verdict = review_pr_diff(secret_diff, "PR-103").unwrap();
        assert_eq!(verdict.status, VerdictStatus::REJECT);
        assert!(!verdict.zero_disk_secrets_certified);
    }
}
