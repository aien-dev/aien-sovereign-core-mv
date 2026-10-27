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

//! Secret protection abstractions and provider contracts for AIEN.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;
use zeroize::Zeroizing;

/// Classification tiers for cryptographic secret protection at rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretProtectionLevel {
    /// Private key material generated within hardware silicon that cannot be extracted.
    HardwareBackedNonExportable,
    /// Secrets encrypted at rest using keys bound to platform hardware state or TPM 2.0.
    HardwareBackedSealed,
    /// Secrets protected by operating system credential facilities.
    OsProtected,
    /// Ephemeral in-process memory store, wiped on process exit.
    MemoryOnly,
}

/// Operational capabilities exposed by a SecretProvider implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretCapabilities {
    pub protection_level: SecretProtectionLevel,
    pub can_seal: bool,
    pub can_unseal: bool,
    pub hardware_bound: bool,
    pub persistent_at_rest: bool,
}

/// An encrypted or sealed secret container safe for persistent storage or transit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedSecret {
    pub name: String,
    pub ciphertext: Vec<u8>,
    pub protection_level: SecretProtectionLevel,
    pub metadata: HashMap<String, String>,
}

/// Errors occurring during secret resolution, sealing, or unsealing.
#[derive(Debug, Error)]
pub enum SecretError {
    #[error("Secret '{0}' was not found in provider")]
    NotFound(String),
    #[error("Hardware security provider unavailable: {0}")]
    HardwareUnavailable(String),
    #[error("Cryptographic operation failed: {0}")]
    CryptoError(String),
    #[error("Plaintext secret persistence is forbidden by security policy: {0}")]
    PolicyViolation(String),
    #[error("I/O failure during secret operation: {0}")]
    IoError(String),
}

/// Uniform contract for platform secret providers.
pub trait SecretProvider: Send + Sync {
    /// Return capabilities of this provider.
    fn capabilities(&self) -> SecretCapabilities;

    /// Return protection level guaranteed by this provider.
    fn protection_level(&self) -> SecretProtectionLevel;

    /// Seal plaintext bytes into a protected container.
    fn seal(&self, name: &str, secret: &[u8]) -> Result<SealedSecret, SecretError>;

    /// Unseal a protected container into zeroized process memory.
    fn unseal(&self, secret: &SealedSecret) -> Result<Zeroizing<Vec<u8>>, SecretError>;

    /// Delete a secret entry by name.
    fn delete(&self, name: &str) -> Result<(), SecretError>;

    /// Resolve a secret directly by name into zeroized process memory.
    fn resolve(&self, name: &str) -> Result<Zeroizing<Vec<u8>>, SecretError>;
}

/// In-memory secret provider for tests and ephemeral environments.
#[derive(Default, Clone)]
pub struct MemorySecretProvider {
    store: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl MemorySecretProvider {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl SecretProvider for MemorySecretProvider {
    fn capabilities(&self) -> SecretCapabilities {
        SecretCapabilities {
            protection_level: SecretProtectionLevel::MemoryOnly,
            can_seal: true,
            can_unseal: true,
            hardware_bound: false,
            persistent_at_rest: false,
        }
    }

    fn protection_level(&self) -> SecretProtectionLevel {
        SecretProtectionLevel::MemoryOnly
    }

    fn seal(&self, name: &str, secret: &[u8]) -> Result<SealedSecret, SecretError> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| SecretError::CryptoError(e.to_string()))?;
        guard.insert(name.to_string(), secret.to_vec());
        Ok(SealedSecret {
            name: name.to_string(),
            ciphertext: secret.to_vec(),
            protection_level: SecretProtectionLevel::MemoryOnly,
            metadata: HashMap::new(),
        })
    }

    fn unseal(&self, secret: &SealedSecret) -> Result<Zeroizing<Vec<u8>>, SecretError> {
        let guard = self
            .store
            .read()
            .map_err(|e| SecretError::CryptoError(e.to_string()))?;
        match guard.get(&secret.name) {
            Some(bytes) => Ok(Zeroizing::new(bytes.clone())),
            None => Err(SecretError::NotFound(secret.name.clone())),
        }
    }

    fn delete(&self, name: &str) -> Result<(), SecretError> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| SecretError::CryptoError(e.to_string()))?;
        guard.remove(name);
        Ok(())
    }

    fn resolve(&self, name: &str) -> Result<Zeroizing<Vec<u8>>, SecretError> {
        let guard = self
            .store
            .read()
            .map_err(|e| SecretError::CryptoError(e.to_string()))?;
        match guard.get(name) {
            Some(bytes) => Ok(Zeroizing::new(bytes.clone())),
            None => Err(SecretError::NotFound(name.to_string())),
        }
    }
}
