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

// spark-aegis Sovereign Defensive Boundary and Containment Engine
// Pure Native Rust Systems Architecture

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DOCTRINE_STATEMENT: &str =
    "The network is the house. Cut the session. Isolate the host. Close the door. Keep the evidence.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainmentActionKind {
    CutSession,
    IsolateHost,
    CloseDoor,
    KeepEvidence,
}

impl ContainmentActionKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CutSession => "CutSession",
            Self::IsolateHost => "IsolateHost",
            Self::CloseDoor => "CloseDoor",
            Self::KeepEvidence => "KeepEvidence",
        }
    }

    pub fn directive(&self) -> &'static str {
        match self {
            Self::CutSession => "Cut the session.",
            Self::IsolateHost => "Isolate the host.",
            Self::CloseDoor => "Close the door.",
            Self::KeepEvidence => "Keep the evidence.",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CutSession => "Terminate active unauthorized interactive or socket sessions immediately.",
            Self::IsolateHost => "Suspend, sever, and isolate compromised host processes from execution.",
            Self::CloseDoor => "Quarantine filesystem targets and revoke all access permissions.",
            Self::KeepEvidence => "Compute Blake3 cryptographic integrity hash and commit durable audit record to Cortex.",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainmentAction {
    CutSession {
        session_id: String,
    },
    IsolateHost {
        process_id: u32,
    },
    CloseDoor {
        target_path: PathBuf,
    },
    KeepEvidence {
        incident_id: String,
        blake3_hash: String,
    },
}

impl ContainmentAction {
    pub fn kind(&self) -> ContainmentActionKind {
        match self {
            Self::CutSession { .. } => ContainmentActionKind::CutSession,
            Self::IsolateHost { .. } => ContainmentActionKind::IsolateHost,
            Self::CloseDoor { .. } => ContainmentActionKind::CloseDoor,
            Self::KeepEvidence { .. } => ContainmentActionKind::KeepEvidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AegisInvariant {
    ZeroDiskSecrets,
    SovereignVoice,
    PureCompiledArchitecture,
    ImmutableAuditTrail,
}

impl AegisInvariant {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ZeroDiskSecrets => "SECURE_TPM_ONLY",
            Self::SovereignVoice => "UNSLOP_COMPLIANCE",
            Self::PureCompiledArchitecture => "PURE_NATIVE_SYSTEMS",
            Self::ImmutableAuditTrail => "BLAKE3_CORTEX_TRAIL",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::ZeroDiskSecrets => {
                "Plaintext secrets and .env files must never touch disk. All credentials must resolve via hardware TPM vault."
            }
            Self::SovereignVoice => {
                "Strict zero em dash and en dash policy. Ban AI buzzwords, transitional filler, and antithesis tropes."
            }
            Self::PureCompiledArchitecture => {
                "Core security and agent runtime engines must compile to native Rust binaries."
            }
            Self::ImmutableAuditTrail => {
                "Every defensive incident and PR audit must commit Blake3 cryptographic proof into Cortex memory."
            }
        }
    }
}

pub fn all_invariants() -> &'static [AegisInvariant] {
    &[
        AegisInvariant::ZeroDiskSecrets,
        AegisInvariant::SovereignVoice,
        AegisInvariant::PureCompiledArchitecture,
        AegisInvariant::ImmutableAuditTrail,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doctrine_statement() {
        assert_eq!(
            DOCTRINE_STATEMENT,
            "The network is the house. Cut the session. Isolate the host. Close the door. Keep the evidence."
        );
    }

    #[test]
    fn test_containment_action_kinds() {
        let action = ContainmentAction::CutSession {
            session_id: "test-session-42".to_string(),
        };
        assert_eq!(action.kind(), ContainmentActionKind::CutSession);
        assert_eq!(action.kind().directive(), "Cut the session.");

        let isolate = ContainmentAction::IsolateHost { process_id: 1234 };
        assert_eq!(isolate.kind(), ContainmentActionKind::IsolateHost);
        assert_eq!(isolate.kind().directive(), "Isolate the host.");

        let door = ContainmentAction::CloseDoor {
            target_path: PathBuf::from("/tmp/badfile"),
        };
        assert_eq!(door.kind(), ContainmentActionKind::CloseDoor);
        assert_eq!(door.kind().directive(), "Close the door.");

        let ev = ContainmentAction::KeepEvidence {
            incident_id: "inc-1".to_string(),
            blake3_hash: "abcd".to_string(),
        };
        assert_eq!(ev.kind(), ContainmentActionKind::KeepEvidence);
        assert_eq!(ev.kind().directive(), "Keep the evidence.");
    }

    #[test]
    fn test_all_invariants_present() {
        let inv = all_invariants();
        assert_eq!(inv.len(), 4);
        assert_eq!(inv[0].code(), "SECURE_TPM_ONLY");
    }
}
