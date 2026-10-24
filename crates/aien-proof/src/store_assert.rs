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

//! Stable Store qualification assertion IDs for evidence receipts.
//!
//! This module records the test that happened; it never defines Store
//! wire format rules. Each constant is the stable `Assertion.id` value
//! written into an `EvidenceReceiptV1`.

/// Records that the observed store layout parsed as the expected format.
pub const STORE_FORMAT_VALID: &str = "store_format_valid";
/// Records that observed object identifiers matched the expected shape.
pub const STORE_OBJECT_IDS_VALID: &str = "store_object_ids_valid";
/// Records that the observed catalog listing matched the expected entries.
pub const STORE_CATALOG_VALID: &str = "store_catalog_valid";
/// Records that the observed commit reference matched the expected value.
pub const STORE_COMMIT_VALID: &str = "store_commit_valid";
/// Records that the observed full history graph traversal matched the expected result.
pub const STORE_FULL_GRAPH_VALID: &str = "store_full_graph_valid";
/// Records that a conflicting write attempt was rejected without mutating state.
pub const STORE_CONFLICT_FAIL_CLOSED: &str = "store_conflict_fail_closed";
/// Records that an inconsistent history input was rejected without mutating state.
pub const STORE_INCONSISTENT_HISTORY_FAIL_CLOSED: &str = "store_inconsistent_history_fail_closed";
/// Records that an I/O error was reported instead of silent fallback content.
pub const STORE_IO_ERROR_NOT_FALLBACK: &str = "store_io_error_not_fallback";
/// Records that a no-space condition left zero partial writes visible.
pub const STORE_NOSPACE_ZERO_WRITES: &str = "store_nospace_zero_writes";
/// Records that after a write attempt only the previous or the new content was visible.
pub const STORE_PREVIOUS_OR_NEW_ONLY: &str = "store_previous_or_new_only";
/// Records that opening the store produced no observable side effects.
pub const STORE_OPEN_SIDE_EFFECT_FREE: &str = "store_open_side_effect_free";
/// Records that the store state observed after a QEMU reboot matched the expected committed state.
pub const STORE_QEMU_REBOOT_RECOVERED: &str = "store_qemu_reboot_recovered";

/// All 12 stable Store assertion IDs in canonical order.
pub fn store_assertion_ids() -> &'static [&'static str] {
    &[
        STORE_FORMAT_VALID,
        STORE_OBJECT_IDS_VALID,
        STORE_CATALOG_VALID,
        STORE_COMMIT_VALID,
        STORE_FULL_GRAPH_VALID,
        STORE_CONFLICT_FAIL_CLOSED,
        STORE_INCONSISTENT_HISTORY_FAIL_CLOSED,
        STORE_IO_ERROR_NOT_FALLBACK,
        STORE_NOSPACE_ZERO_WRITES,
        STORE_PREVIOUS_OR_NEW_ONLY,
        STORE_OPEN_SIDE_EFFECT_FREE,
        STORE_QEMU_REBOOT_RECOVERED,
    ]
}

/// Returns true when `id` is one of the 12 stable Store assertion IDs.
pub fn is_store_assertion(id: &str) -> bool {
    store_assertion_ids().contains(&id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn ids_are_non_empty_unique_and_stable() {
        let ids = store_assertion_ids();
        assert_eq!(ids.len(), 12);
        let mut seen = BTreeSet::new();
        for id in ids {
            assert!(!id.is_empty(), "store assertion id must not be empty");
            assert!(seen.insert(*id), "duplicate store assertion id {id:?}");
            assert!(
                is_store_assertion(id),
                "is_store_assertion must accept {id:?}"
            );
        }
        assert_eq!(seen.len(), 12);
    }

    #[test]
    fn rejects_non_store_ids() {
        assert!(!is_store_assertion("artifact_signature_valid"));
        assert!(!is_store_assertion(""));
    }
}
