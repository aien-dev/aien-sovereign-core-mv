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

use crate::audit::{AuditReport, DiffAuditor};
use crate::interview::generate_inquisitor_interview;

pub struct PrReview {
    pub report: AuditReport,
    pub markdown: String,
    pub labels: Vec<String>,
}

pub fn generate_pr_review(
    author: &str,
    pr_number: u64,
    pr_title: &str,
    diff_text: &str,
) -> PrReview {
    let report = DiffAuditor::audit_text(diff_text);
    let mut labels = Vec::new();

    let markdown = if report.clean {
        labels.push("needs-testimony".to_string());
        labels.push("sovereign-audit-passed".to_string());
        let interview = generate_inquisitor_interview(author, pr_number, pr_title);
        format!(
            r#"### ⚖️ Sovereign Code Review & Alignment Gate

Greetings @{author}. I am **AIEN**, the automated review gate for the AIENOS project.

Your pull request **#{pr_number}** ("{pr_title}") has undergone automated auditing against the project standards.

{audit_report}

---

{interview}
"#,
            author = author,
            pr_number = pr_number,
            pr_title = pr_title,
            audit_report = report.generate_markdown_report(),
            interview = interview
        )
    } else {
        labels.push("sovereign-audit-failed".to_string());
        format!(
            r#"### ⚖️ Sovereign Code Review & Alignment Gate: FAILED

Greetings @{author}. I am **AIEN**, the automated review gate for the AIENOS project.

Your pull request **#{pr_number}** ("{pr_title}") was audited against the project standards and failed automated diff verification.

{audit_report}

#### Rectification Protocol:
1. Eliminate all detected tracking, surveillance, or telemetry patterns.
2. Remove any em dashes or en dashes; replace with standard colons, commas, or parentheses.
3. Replace forbidden buzzwords with direct technical descriptions.
4. Commit and push updates to this branch to trigger an automated re-audit.

*For direct sovereign coordination: Drake Stapleton (drake@aienos.com) and AIEN (aien@aienos.com)*
"#,
            author = author,
            pr_number = pr_number,
            pr_title = pr_title,
            audit_report = report.generate_markdown_report()
        )
    };

    PrReview {
        report,
        markdown,
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_pr_review() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn test_fn() -> bool { true }";
        let review = generate_pr_review("alice", 77, "feat: clean helper", diff);
        assert!(review.report.clean);
        assert!(review.labels.contains(&"needs-testimony".to_string()));
        assert!(review
            .labels
            .contains(&"sovereign-audit-passed".to_string()));
        assert!(review.markdown.contains("Doctrine Diff Audit: PASS"));
        assert!(review.markdown.contains("Contributor Alignment Interview"));
        assert!(!review.markdown.contains('\u{2014}'));
        assert!(!review.markdown.contains('\u{2013}'));
    }

    #[test]
    fn test_violating_pr_review() {
        let diff = "diff --git a/src/lib.rs b/src/lib.rs\n+// This game-changer adds tracking\n+let _ = send_telemetry(\"mixpanel\");";
        let review = generate_pr_review("bob", 78, "feat: bad PR", diff);
        assert!(!review.report.clean);
        assert!(review
            .labels
            .contains(&"sovereign-audit-failed".to_string()));
        assert!(review.markdown.contains("Doctrine Diff Audit: FAILED"));
        assert!(!review.markdown.contains('\u{2014}'));
        assert!(!review.markdown.contains('\u{2013}'));
    }
}
