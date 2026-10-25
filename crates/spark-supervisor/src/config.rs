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

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ServiceConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default = "default_restart")]
    pub restart: String,
    pub health_url: Option<String>,
}

fn default_restart() -> String {
    "always".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct SupervisorConfig {
    #[serde(default)]
    pub services: HashMap<String, ServiceConfig>,
}

impl SupervisorConfig {
    /// Parses the file and refuses it when any `health_url` is unusable.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        let cfg: SupervisorConfig = toml::from_str(&content)?;
        cfg.validate_health_urls()
            .map_err(|e| format!("supervisor config {} is invalid: {e}", path.display()))?;
        Ok(cfg)
    }

    /// Every `health_url` must be a plain `http://` URL: probes go through
    /// aien-localhttp, which never speaks TLS. An `https://` URL is a config
    /// error, reported with the service name, not a probe that fails forever.
    pub fn validate_health_urls(&self) -> Result<(), aien_localhttp::Error> {
        let mut names: Vec<&String> = self.services.keys().collect();
        names.sort();
        for name in names {
            if let Some(url) = &self.services[name].health_url {
                aien_localhttp::Url::parse(url).map_err(|e| match e {
                    aien_localhttp::Error::HttpsRefused { .. } => {
                        aien_localhttp::Error::HttpsRefused {
                            url: format!("{url} (service {name:?} health_url)"),
                        }
                    }
                    other => other,
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_supervisor_config() {
        let sample = r#"
[services.cortex]
command = "/usr/local/bin/cortex-rs"
args = ["--port", "18080"]
health_url = "http://127.0.0.1:18080/health"

[services.cockpit]
command = "/usr/local/bin/spark-cockpit-rs"
restart = "on-failure"
"#;
        let cfg: SupervisorConfig = toml::from_str(sample).expect("valid toml");
        assert_eq!(cfg.services.len(), 2);
        let cortex = cfg.services.get("cortex").unwrap();
        assert_eq!(cortex.command, "/usr/local/bin/cortex-rs");
        assert_eq!(cortex.args, vec!["--port", "18080"]);
        assert_eq!(cortex.restart, "always");
        assert_eq!(
            cortex.health_url.as_deref(),
            Some("http://127.0.0.1:18080/health")
        );

        let cockpit = cfg.services.get("cockpit").unwrap();
        assert_eq!(cockpit.restart, "on-failure");
        assert!(cockpit.health_url.is_none());
    }

    #[test]
    fn test_empty_config_defaults() {
        let sample = "";
        let cfg: SupervisorConfig = toml::from_str(sample).expect("empty toml is valid");
        assert!(cfg.services.is_empty());
    }

    fn write_config(tag: &str, body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "spark-supervisor-{tag}-{}.toml",
            std::process::id()
        ));
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn load_from_file_accepts_http_health_urls() {
        let path = write_config(
            "http",
            "[services.cortex]\ncommand = \"x\"\nhealth_url = \"http://127.0.0.1:18080/health\"\n",
        );
        let cfg = SupervisorConfig::load_from_file(&path).expect("http is valid");
        std::fs::remove_file(&path).unwrap();
        assert_eq!(cfg.services.len(), 1);
    }

    #[test]
    fn load_from_file_refuses_https_health_url_loudly() {
        let path = write_config(
            "https",
            "[services.web]\ncommand = \"x\"\nhealth_url = \"https://127.0.0.1:8443/health\"\n",
        );
        let err = SupervisorConfig::load_from_file(&path)
            .unwrap_err()
            .to_string();
        std::fs::remove_file(&path).unwrap();
        assert!(err.contains("is invalid"), "{err}");
        assert!(err.contains("https is refused"), "{err}");
        assert!(err.contains("service \"web\" health_url"), "{err}");
    }

    #[test]
    fn validate_health_urls_refuses_other_schemes_and_garbage() {
        for bad in ["ftp://127.0.0.1/health", "127.0.0.1:80/health", "http://"] {
            let cfg: SupervisorConfig = toml::from_str(&format!(
                "[services.s]\ncommand = \"x\"\nhealth_url = \"{bad}\"\n"
            ))
            .unwrap();
            assert!(cfg.validate_health_urls().is_err(), "{bad} must be refused");
        }
    }
}
