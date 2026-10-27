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

//! Crumb ledger cross checks for evidence receipts.
//!
//! The ledger remains the append only operational audit stream. Receipts
//! reference relevant ledger event IDs and digests instead of duplicating
//! the ledger, and verification confirms the chain, the referenced event,
//! and the output hash match.

use std::path::Path;

use crate::evidence::Receipt;
use crate::ledger;

/// Check a receipt's ledger reference. Returns `Ok(None)` when the receipt
/// carries no reference, `Ok(Some(problem))` with a fail closed reason when
/// the reference does not check out.
pub fn check_ledger_ref(receipt: &Receipt, store: &Path) -> Result<Option<String>, String> {
    let reference = match &receipt.ledger {
        Some(r) => r,
        None => return Ok(None),
    };
    let path = store.join(ledger::LEDGER_FILE);
    ledger::verify(&path).map_err(|e| format!("ledger chain broken: {e}"))?;
    let event = ledger::read_event(&path, reference.index)?.ok_or_else(|| {
        format!(
            "receipt references ledger event {} which does not exist",
            reference.index
        )
    })?;
    if ledger::hex(&event.hash) != reference.hash.to_ascii_lowercase() {
        return Ok(Some(format!(
            "receipt ledger hash does not match event {}",
            reference.index
        )));
    }
    if ledger::hex(&event.payload_hash) != receipt.output_digest.to_ascii_lowercase() {
        return Ok(Some(
            "receipt output digest does not match the ledger payload hash".to_string(),
        ));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::fixtures::{sample, sealed};
    use crate::evidence::{LedgerRef, Tier, Verdict};

    #[test]
    fn matching_ledger_ref_passes() {
        let store = crate::testutil::temp_dir("ledgerbind-ok");
        let output = b"seed output";
        let event = ledger::append(&store, "tester", "seed-0b", "pass exit=0", output).unwrap();
        let mut r = sample();
        r.output_digest = ledger::hex(&event.payload_hash);
        r.ledger = Some(LedgerRef {
            index: event.index,
            hash: ledger::hex(&event.hash),
        });
        let r = sealed(r);
        assert_eq!(check_ledger_ref(&r, &store).unwrap(), None);
    }

    #[test]
    fn output_digest_mismatch_fails_closed() {
        let store = crate::testutil::temp_dir("ledgerbind-digest");
        let event = ledger::append(&store, "tester", "seed-0b", "pass exit=0", b"real").unwrap();
        let mut r = sample();
        r.output_digest = "d".repeat(64);
        r.ledger = Some(LedgerRef {
            index: event.index,
            hash: ledger::hex(&event.hash),
        });
        let r = sealed(r);
        assert!(check_ledger_ref(&r, &store).unwrap().is_some());
    }

    #[test]
    fn missing_event_fails_closed() {
        let store = crate::testutil::temp_dir("ledgerbind-missing");
        ledger::append(&store, "tester", "seed-0b", "pass exit=0", b"real").unwrap();
        let mut r = sample();
        r.ledger = Some(LedgerRef {
            index: 7,
            hash: "e".repeat(64),
        });
        let r = sealed(r);
        assert!(check_ledger_ref(&r, &store)
            .unwrap_err()
            .contains("does not exist"));
    }

    #[test]
    fn tier_and_verdict_reexports_used() {
        assert_eq!(Tier::Qemu.as_str(), "QEMU");
        assert_eq!(Verdict::Pass.as_str(), "PASS");
    }
}
