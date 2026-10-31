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

//! Hardware TPM 2.0 secret provider implementation for AIEN.

use crate::secret::{
    SealedSecret, SecretCapabilities, SecretError, SecretProtectionLevel, SecretProvider,
};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use zeroize::Zeroizing;

/// Hardware TPM 2.0 provider backed by platform sealed storage.
pub struct Tpm2SecretProvider {
    vault_binary: PathBuf,
    tpm_device: PathBuf,
}

impl Tpm2SecretProvider {
    /// Probe system for hardware TPM availability.
    pub fn probe() -> Self {
        let default_device = PathBuf::from("/dev/tpmrm0");
        let default_vault = if let Ok(home) = std::env::var("HOME") {
            let candidate = Path::new(&home).join(".local/bin/atlas-vault");
            if candidate.exists() {
                candidate
            } else {
                PathBuf::from("atlas-vault")
            }
        } else {
            PathBuf::from("atlas-vault")
        };

        Self {
            vault_binary: default_vault,
            tpm_device: default_device,
        }
    }

    /// Provider for an explicit vault binary and TPM device node.
    pub fn with_paths(vault_binary: impl Into<PathBuf>, tpm_device: impl Into<PathBuf>) -> Self {
        Self {
            vault_binary: vault_binary.into(),
            tpm_device: tpm_device.into(),
        }
    }

    /// Check if the hardware TPM device node exists on this system.
    pub fn is_hardware_present(&self) -> bool {
        self.tpm_device.exists()
    }
}

impl Default for Tpm2SecretProvider {
    fn default() -> Self {
        Self::probe()
    }
}

impl SecretProvider for Tpm2SecretProvider {
    fn capabilities(&self) -> SecretCapabilities {
        SecretCapabilities {
            protection_level: SecretProtectionLevel::HardwareBackedSealed,
            can_seal: true,
            can_unseal: true,
            hardware_bound: true,
            persistent_at_rest: true,
        }
    }

    fn protection_level(&self) -> SecretProtectionLevel {
        SecretProtectionLevel::HardwareBackedSealed
    }

    fn seal(&self, name: &str, secret: &[u8]) -> Result<SealedSecret, SecretError> {
        let secret_str = std::str::from_utf8(secret).map_err(|_| {
            SecretError::CryptoError("TPM vault requires UTF-8 compatible secret input".to_string())
        })?;

        // `atlas-vault add <NAME>` reads the value from standard input. The value never goes on
        // the command line, where other local processes could read it.
        let mut child = Command::new(&self.vault_binary)
            .args(["add", name])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SecretError::IoError(format!("Failed to execute vault command: {}", e)))?;
        // Dropping `stdin` closes it, so the vault sees end of input. The child is always waited
        // for, even when the write fails, and its own error is reported first.
        let written = match child.stdin.take() {
            Some(mut stdin) => stdin.write_all(secret_str.as_bytes()),
            None => Err(std::io::Error::other("vault command stdin is not piped")),
        };
        let output = child
            .wait_with_output()
            .map_err(|e| SecretError::IoError(format!("Failed to execute vault command: {}", e)))?;

        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(SecretError::CryptoError(format!(
                "TPM vault add failed: {}",
                err_msg.trim()
            )));
        }
        written.map_err(|e| {
            SecretError::IoError(format!("Failed to write secret to vault command: {}", e))
        })?;

        Ok(SealedSecret {
            name: name.to_string(),
            ciphertext: Vec::new(),
            protection_level: SecretProtectionLevel::HardwareBackedSealed,
            metadata: HashMap::new(),
        })
    }

    fn unseal(&self, secret: &SealedSecret) -> Result<Zeroizing<Vec<u8>>, SecretError> {
        self.resolve(&secret.name)
    }

    fn delete(&self, name: &str) -> Result<(), SecretError> {
        let output = Command::new(&self.vault_binary)
            .args(["remove", name])
            .output()
            .map_err(|e| SecretError::IoError(format!("Failed to execute vault command: {}", e)))?;

        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(SecretError::CryptoError(format!(
                "TPM vault remove failed: {}",
                err_msg.trim()
            )));
        }

        Ok(())
    }

    fn resolve(&self, name: &str) -> Result<Zeroizing<Vec<u8>>, SecretError> {
        let output = Command::new(&self.vault_binary)
            .args(["get", name])
            .output()
            .map_err(|e| SecretError::IoError(format!("Failed to execute vault command: {}", e)))?;

        if !output.status.success() {
            return Err(SecretError::NotFound(name.to_string()));
        }

        let val = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if val.is_empty() {
            return Err(SecretError::NotFound(name.to_string()));
        }

        Ok(Zeroizing::new(val.into_bytes()))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// A stand-in for `atlas-vault` that records its arguments and standard input and keeps one
    /// value in a file, so the adapter can be checked without a TPM.
    fn stub_vault(dir: &Path) -> PathBuf {
        let script = dir.join("vault-stub");
        let body = format!(
            "#!/bin/sh\nd='{d}'\nprintf '%s\\n' \"$*\" >> \"$d/argv\"\ncase \"$1\" in\n  add) cat > \"$d/stdin\"; cp \"$d/stdin\" \"$d/value\" ;;\n  get) [ -f \"$d/value\" ] || exit 1; cat \"$d/value\" ;;\n  remove) [ -f \"$d/value\" ] || exit 1; rm \"$d/value\" ;;\n  *) echo \"unknown command $1\" >&2; exit 2 ;;\nesac\n",
            d = dir.display()
        );
        std::fs::write(&script, body).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        script
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aien-security-vault-{}-{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn seal_uses_stdin_not_argv_remove_round_trips_and_refusal_is_an_error() {
        let dir = scratch_dir("roundtrip");
        let provider = Tpm2SecretProvider::with_paths(stub_vault(&dir), dir.join("no-tpm"));
        let value = b"test-value-7f3a";

        provider.seal("AIEN_TEST_NAME", value).expect("seal");
        assert_eq!(std::fs::read(dir.join("stdin")).unwrap(), value);
        assert_eq!(
            &*provider.resolve("AIEN_TEST_NAME").expect("resolve"),
            value
        );
        provider.delete("AIEN_TEST_NAME").expect("remove");
        assert!(provider.resolve("AIEN_TEST_NAME").is_err());

        let argv = std::fs::read_to_string(dir.join("argv")).unwrap();
        assert_eq!(
            argv,
            "add AIEN_TEST_NAME\nget AIEN_TEST_NAME\nremove AIEN_TEST_NAME\nget AIEN_TEST_NAME\n"
        );
        assert!(!argv.contains("test-value-7f3a"));

        // A vault that refuses the add is reported as an error. One test writes both scripts, so
        // no other test in this module forks while a script file is open for writing.
        let script = dir.join("vault-fail");
        std::fs::write(
            &script,
            "#!/bin/sh\ncat > /dev/null\necho refused >&2\nexit 1\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let provider = Tpm2SecretProvider::with_paths(script, dir.join("no-tpm"));
        let err = provider.seal("AIEN_TEST_NAME", b"v").unwrap_err();
        assert!(matches!(err, SecretError::CryptoError(ref m) if m.contains("refused")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
