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

//! Exclusive resource lease binding for evidence receipts.
//!
//! Hardware runs go through the merged coordination machinery:
//! `Board::hold` runs the command while holding its exclusive keys, seals an
//! `audit` event into the Crumb ledger, and stores a hold record under
//! `holds/`. Receipts reference the hold by its audit event hash; this
//! module checks the record, the ledger event, and the output digest.
//! Nothing here invents a second lock: all exclusion lives in `board`.

use std::fs;
use std::path::Path;

use crate::board::{valid_resource, Record};
use crate::evidence::{Receipt, Report, Verdict};
use crate::ledger;

/// Subdirectory of the proof board root holding hold records.
pub const HOLD_DIR: &str = "holds";
/// Ledger action used for hold audit events.
pub const HOLD_ACTION: &str = "audit";

/// Resource names follow the board rule: path components of lowercase
/// letters, digits, and dashes, at most 64 bytes.
pub fn check_resource_name(name: &str) -> Result<(), String> {
    if valid_resource(name) {
        Ok(())
    } else {
        Err(format!(
            "invalid resource name {name:?}: use lowercase letters, digits, dashes"
        ))
    }
}

/// Find the hold record whose audit event hash is `hold_id`.
pub fn load_hold(store: &Path, hold_id: &str) -> Result<Record, String> {
    let normalized = hold_id.to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("hold id {hold_id:?} is not a 64 digit hash"));
    }
    let dir = store.join(HOLD_DIR);
    let entries = fs::read_dir(&dir)
        .map_err(|_| format!("hold record {normalized} not found: no holds in store"))?;
    for entry in entries.flatten() {
        let text = match fs::read(entry.path()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        if let Ok(record) = serde_json::from_slice::<Record>(&text) {
            if record.ledger_hash.to_ascii_lowercase() == normalized {
                return Ok(record);
            }
        }
    }
    Err(format!("missing hold record {normalized}"))
}

/// Resources named in a hold audit intent (`... resources=a,b`).
fn intent_resources(intent: &str) -> Vec<String> {
    intent
        .rsplit_once("resources=")
        .map(|(_, list)| {
            list.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Check a receipt's lease reference against the hold store and ledger.
/// Returns `None` when the receipt claims no lease. `Ok(Some(report))` is
/// the binding verdict; `Err` means the store itself is unreadable.
pub fn check_lease_ref(receipt: &Receipt, store: &Path) -> Result<Option<Report>, String> {
    let lease = match &receipt.lease {
        Some(l) => l,
        None => return Ok(None),
    };
    let record =
        load_hold(store, &lease.hold_id).map_err(|e| format!("lease binding broken: {e}"))?;
    let mut problems = Vec::new();
    // The hold audit event must exist in the ledger with a matching hash.
    let event = ledger::read_event(&store.join(ledger::LEDGER_FILE), record.ledger_index)?
        .ok_or_else(|| {
            format!(
                "lease binding broken: ledger event {} is missing",
                record.ledger_index
            )
        })?;
    if ledger::hex(&event.hash) != record.ledger_hash.to_ascii_lowercase()
        || ledger::hex(&event.hash) != lease.hold_id.to_ascii_lowercase()
    {
        problems.push("hold record does not match its ledger audit event".to_string());
    }
    if event.action != HOLD_ACTION {
        problems.push(format!(
            "ledger event {} is not a hold audit event",
            record.ledger_index
        ));
    }
    if !intent_resources(&event.intent)
        .iter()
        .any(|r| r == &lease.resource)
    {
        problems.push(format!(
            "hold {} never covered resource {:?}",
            lease.hold_id, lease.resource
        ));
    }
    if receipt.machine != lease.resource {
        problems.push(format!(
            "receipt machine {:?} does not match leased resource {:?}",
            receipt.machine, lease.resource
        ));
    }
    if receipt.procedure != record.cmd.join(" ") {
        problems.push(format!(
            "receipt procedure {:?} does not match held command {:?}",
            receipt.procedure,
            record.cmd.join(" ")
        ));
    }
    if !receipt
        .output_digest
        .eq_ignore_ascii_case(&ledger::hex(&event.payload_hash))
    {
        problems.push("receipt output digest does not match the held run output".to_string());
    }
    if receipt.result == Verdict::Pass && record.exit_code != 0 {
        problems.push(format!(
            "receipt claims PASS but the held run exited with {}",
            record.exit_code
        ));
    }
    if problems.is_empty() {
        Ok(Some(Report::pass("lease binding checks out")))
    } else {
        Ok(Some(Report::fail(problems)))
    }
}

/// Require a lease for hardware tiers. Called by chain and gate evaluation
/// so a Machine 1 qualification without a lease never reads as PASS.
pub fn lease_status(receipt: &Receipt, store: &Path) -> Option<Report> {
    if !receipt.tier.is_hardware() {
        return None;
    }
    match &receipt.lease {
        None => Some(Report::closed(
            Verdict::Incomplete,
            vec!["Machine 1 qualification has no exclusive hardware lease".to_string()],
        )),
        Some(_) => match check_lease_ref(receipt, store) {
            Ok(Some(r)) => Some(r),
            Ok(None) => None,
            Err(e) => Some(Report::fail(vec![e])),
        },
    }
}

/// Run a held command through the board and return its record plus the
/// output digest. Test helper that exercises the real exclusion path.
#[cfg(test)]
pub fn test_hold(
    store: &Path,
    base: &Path,
    resource: &str,
    holder: &str,
    cmd: &[String],
) -> (Record, String) {
    use crate::board::{Board, Job};
    let board = Board {
        root: store.to_path_buf(),
        cpu_slots: 4,
        min_free_bytes: 0,
        mem_wait: std::time::Duration::from_secs(1),
        quiet: true,
    };
    let job = Job {
        name: format!("test-hold-{resource}"),
        agent: holder.to_string(),
        base: base.to_path_buf(),
        inputs: vec![],
        cmd: cmd.to_vec(),
        gpu: false,
        resources: vec![resource.to_string()],
        toolchain: String::new(),
    };
    let record = board.hold(&job).unwrap();
    let event = ledger::read_event(&store.join(ledger::LEDGER_FILE), record.ledger_index)
        .unwrap()
        .unwrap();
    let digest = ledger::hex(&event.payload_hash);
    (record, digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::fixtures::{sample, sealed};
    use crate::evidence::{store_receipt, LeaseRef, Mutation, Tier, Verdict};

    fn machine_receipt(
        resource: &str,
        procedure: &str,
        lease: Option<LeaseRef>,
        output: &str,
    ) -> Receipt {
        let mut r = sample();
        r.tier = Tier::Machine1Attended;
        r.env_class = "MACHINE1_ATTENDED".to_string();
        r.machine = resource.to_string();
        r.procedure = procedure.to_string();
        r.declared_mutation = Mutation::RemovableMediaOnly;
        r.observed_mutation = Mutation::RemovableMediaOnly;
        r.output_digest = output.to_string();
        r.lease = lease;
        sealed(r)
    }

    fn held_run(tag: &str) -> (std::path::PathBuf, (Record, String)) {
        let store = crate::testutil::temp_dir(tag);
        let base = crate::testutil::temp_dir(&format!("{tag}-base"));
        let cmd = vec![
            "sh".to_string(),
            "-c".to_string(),
            "printf held-output".to_string(),
        ];
        let held = test_hold(&store, &base, "machine-1", "tester", &cmd);
        (store, held)
    }

    #[test]
    fn resource_names_follow_the_board_rule() {
        assert!(check_resource_name("machine-1").is_ok());
        assert!(check_resource_name("machine_1").is_err());
        assert!(check_resource_name("Machine-1").is_err());
        assert!(check_resource_name("").is_err());
    }

    #[test]
    fn lease_binding_verifies_end_to_end() {
        let (store, (record, digest)) = held_run("lease-ok");
        let r = machine_receipt(
            "machine-1",
            &record.cmd.join(" "),
            Some(LeaseRef {
                hold_id: record.ledger_hash.clone(),
                resource: "machine-1".to_string(),
            }),
            &digest,
        );
        store_receipt(&store, &r).unwrap();
        let report = check_lease_ref(&r, &store).unwrap().unwrap();
        assert_eq!(report.status, Verdict::Pass);
    }

    #[test]
    fn hardware_receipt_without_hold_is_incomplete() {
        let store = crate::testutil::temp_dir("lease-missing");
        let r = machine_receipt("machine-1", "scripts/x.sh", None, &"c".repeat(64));
        let report = lease_status(&r, &store).unwrap();
        assert_eq!(report.status, Verdict::Incomplete);
    }

    #[test]
    fn output_digest_mismatch_fails_binding() {
        let (store, (record, _)) = held_run("lease-digest");
        let r = machine_receipt(
            "machine-1",
            &record.cmd.join(" "),
            Some(LeaseRef {
                hold_id: record.ledger_hash.clone(),
                resource: "machine-1".to_string(),
            }),
            &"d".repeat(64),
        );
        let report = check_lease_ref(&r, &store).unwrap().unwrap();
        assert_eq!(report.status, Verdict::Fail);
    }

    #[test]
    fn wrong_resource_fails_binding() {
        let (store, (record, digest)) = held_run("lease-resource");
        let r = machine_receipt(
            "machine-2",
            &record.cmd.join(" "),
            Some(LeaseRef {
                hold_id: record.ledger_hash.clone(),
                resource: "machine-2".to_string(),
            }),
            &digest,
        );
        let report = check_lease_ref(&r, &store).unwrap().unwrap();
        assert_eq!(report.status, Verdict::Fail);
    }

    #[test]
    fn missing_hold_record_fails_closed() {
        let store = crate::testutil::temp_dir("lease-gone");
        let r = machine_receipt(
            "machine-1",
            "scripts/x.sh",
            Some(LeaseRef {
                hold_id: "e".repeat(64),
                resource: "machine-1".to_string(),
            }),
            &"c".repeat(64),
        );
        assert!(check_lease_ref(&r, &store).is_err());
    }
}
