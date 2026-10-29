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

//! # AIEN Security
//!
//! Normative security abstractions, SecretProvider capability traits,
//! and network purpose governance for the AIEN ecosystem.

pub mod network;
pub mod secret;
pub mod tpm;

pub use network::{enforce_network_purpose, NetworkIntent, NetworkPurpose, NetworkSecurityError};
pub use secret::{
    MemorySecretProvider, SealedSecret, SecretCapabilities, SecretError, SecretProtectionLevel,
    SecretProvider,
};
pub use tpm::Tpm2SecretProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rejected_forbidden_network_purpose() {
        let bad_purpose = NetworkPurpose::Telemetry; // no-telemetry
        let intent = NetworkIntent::new("https://analytics.example.com/v1/event", bad_purpose)
            .with_context("background_diagnostic_beacon");

        let result = enforce_network_purpose(&intent);
        assert!(result.is_err());
        let expected = NetworkSecurityError::TelemetryForbidden; // no-telemetry
        assert_eq!(result.unwrap_err(), expected);
    }

    #[test]
    fn test_permitted_purposes_succeed() {
        let allowed = [
            (
                NetworkPurpose::UserRequestedApi,
                "https://api.openai.com/v1/chat/completions",
            ),
            (
                NetworkPurpose::PeerSynchronization,
                "radicle://peer.node.example",
            ),
            (
                NetworkPurpose::ModelDownload,
                "https://huggingface.co/TinyLlama/model.safetensors",
            ),
            (
                NetworkPurpose::UpdateCheck,
                "https://updates.aien.dev/releases.json",
            ),
        ];

        for (purpose, url) in allowed {
            assert!(purpose.is_permitted());
            let intent = NetworkIntent::new(url, purpose);
            assert!(enforce_network_purpose(&intent).is_ok());
        }
    }

    #[test]
    fn test_memory_secret_provider_lifecycle() {
        let provider = MemorySecretProvider::new();
        assert_eq!(
            provider.protection_level(),
            SecretProtectionLevel::MemoryOnly
        );

        let secret_name = "TEST_API_TOKEN";
        let secret_bytes = b"super-secret-token-payload";

        // Seal
        let sealed = provider
            .seal(secret_name, secret_bytes)
            .expect("sealing failed");
        assert_eq!(sealed.name, secret_name);
        assert_eq!(sealed.protection_level, SecretProtectionLevel::MemoryOnly);

        // Unseal
        let unsealed = provider.unseal(&sealed).expect("unsealing failed");
        assert_eq!(&*unsealed, secret_bytes);

        // Resolve
        let resolved = provider.resolve(secret_name).expect("resolution failed");
        assert_eq!(&*resolved, secret_bytes);

        // Delete
        provider.delete(secret_name).expect("deletion failed");
        assert!(provider.resolve(secret_name).is_err());
    }

    #[test]
    fn test_tpm2_secret_provider_capabilities() {
        let provider = Tpm2SecretProvider::default();
        let caps = provider.capabilities();
        assert_eq!(
            caps.protection_level,
            SecretProtectionLevel::HardwareBackedSealed
        );
        assert!(caps.hardware_bound);
        assert!(caps.persistent_at_rest);
    }

    #[test]
    fn test_unslop_invariants_in_crate() {
        let files = [
            include_str!("lib.rs"),
            include_str!("secret.rs"),
            include_str!("network.rs"),
            include_str!("tpm.rs"),
        ];

        for code in files {
            assert!(!code.contains('\u{2014}'), "Encountered em dash in code");
            assert!(!code.contains('\u{2013}'), "Encountered en dash in code");
        }
    }
}
