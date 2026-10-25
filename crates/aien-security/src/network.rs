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

//! Network purpose classification and egress governance for AIEN.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Explicit classification for all outbound network requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkPurpose {
    /// Outbound call to a user-configured model inference API or proxy.
    UserRequestedApi,
    /// Explicit repository or state synchronization with configured peers.
    PeerSynchronization,
    /// Operator-directed download of model weights or tokenizer assets.
    ModelDownload,
    /// Optional check for signed software release updates.
    UpdateCheck,
    /// Unsolicited usage metrics, analytics, beacons, or behavioral tracking.
    Telemetry, // no-telemetry
}

impl NetworkPurpose {
    /// Check whether this network purpose is permitted by policy.
    pub fn is_permitted(&self) -> bool {
        !matches!(self, NetworkPurpose::Telemetry) // no-telemetry
    }

    /// Descriptive name of the network purpose.
    pub fn as_str(&self) -> &'static str {
        match self {
            NetworkPurpose::UserRequestedApi => "UserRequestedApi",
            NetworkPurpose::PeerSynchronization => "PeerSynchronization",
            NetworkPurpose::ModelDownload => "ModelDownload",
            NetworkPurpose::UpdateCheck => "UpdateCheck",
            NetworkPurpose::Telemetry => "Telemetry", // no-telemetry
        }
    }
}

/// Errors occurring during network policy evaluation and routing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NetworkSecurityError {
    #[error("Outbound destination forbidden by no-telemetry policy")]
    TelemetryForbidden, // no-telemetry
    #[error("Egress destination '{0}' is not allowed for purpose '{1}'")]
    DestinationDisallowed(String, &'static str),
    #[error("Network operation blocked by air-gap deployment profile")]
    AirGapBlocked,
}

/// A declared outbound network intent subject to security governance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIntent {
    pub destination_url: String,
    pub purpose: NetworkPurpose,
    pub context: String,
}

impl NetworkIntent {
    pub fn new(destination_url: impl Into<String>, purpose: NetworkPurpose) -> Self {
        Self {
            destination_url: destination_url.into(),
            purpose,
            context: String::new(),
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }
}

/// Evaluates network intent against constitutional network policy.
pub fn enforce_network_purpose(intent: &NetworkIntent) -> Result<(), NetworkSecurityError> {
    let is_forbidden = matches!(intent.purpose, NetworkPurpose::Telemetry); // no-telemetry
    if is_forbidden {
        tracing::error!(
            target: "security::network_audit",
            destination = %intent.destination_url,
            purpose = %intent.purpose.as_str(),
            context = %intent.context,
            "BLOCKED: Outbound request rejected by no-telemetry policy"
        );
        return Err(NetworkSecurityError::TelemetryForbidden); // no-telemetry
    }

    tracing::debug!(
        target: "security::network_audit",
        destination = %intent.destination_url,
        purpose = %intent.purpose.as_str(),
        "Permitted outbound request under approved network purpose"
    );

    Ok(())
}
