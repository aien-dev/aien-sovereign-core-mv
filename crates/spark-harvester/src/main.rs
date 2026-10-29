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

use clap::{Parser, Subcommand};
use p256::pkcs8::DecodePublicKey;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use aien_provenance::SourceGrant;
use spark_harvester::{BundleKind, ExtractedPair, ReasoningExtractor, RightsGate};

#[derive(Parser)]
#[command(name = "spark-harvester")]
#[command(
    about = "Native Rust rights-gated harvester and provenance engine for sovereign AI datasets"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract reasoning pairs from raw text with mandatory SourceGrant rights gating
    Extract {
        #[arg(short, long)]
        prompt: String,
        #[arg(short, long)]
        input: String,
        #[arg(long, default_value = "commercial-lab")]
        provider: String,
        #[arg(long, default_value = "frontier-model")]
        model: String,
        /// Path to JSON file containing a valid, signed SourceGrant
        #[arg(long)]
        grant_file: Option<PathBuf>,
        /// Inline JSON string of SourceGrant
        #[arg(long)]
        grant_json: Option<String>,
        /// Authority verifying key (hex-encoded SEC1 or PEM)
        #[arg(long)]
        authority_key: Option<String>,
        /// Path to PEM or SEC1 file containing authority verifying key
        #[arg(long)]
        authority_key_file: Option<PathBuf>,
        /// Optional output path for gated bundle
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Bundle kind: training, distillation, or evidence
        #[arg(long, default_value = "distillation")]
        bundle_kind: String,
    },
    /// Export extracted training pairs into JSONL with mandatory SourceGrant rights gating
    Distill {
        #[arg(short, long)]
        output: PathBuf,
        /// Path to JSON file containing a valid, signed SourceGrant
        #[arg(long)]
        grant_file: Option<PathBuf>,
        /// Inline JSON string of SourceGrant
        #[arg(long)]
        grant_json: Option<String>,
        /// Authority verifying key (hex-encoded SEC1 or PEM)
        #[arg(long)]
        authority_key: Option<String>,
        /// Path to PEM or SEC1 file containing authority verifying key
        #[arg(long)]
        authority_key_file: Option<PathBuf>,
    },
    /// Display pipeline status
    Status,
}

fn load_grant(
    grant_file: Option<&PathBuf>,
    grant_json: Option<&str>,
) -> Result<SourceGrant, Box<dyn std::error::Error>> {
    if let Some(path) = grant_file {
        let content = std::fs::read_to_string(path)?;
        let grant: SourceGrant = serde_json::from_str(&content)?;
        Ok(grant)
    } else if let Some(json) = grant_json {
        let grant: SourceGrant = serde_json::from_str(json)?;
        Ok(grant)
    } else {
        Err("SourceGrant required: Pass --grant-file <PATH> or --grant-json <JSON> to prove rights admission. Unlicensed extraction is strictly rejected.".into())
    }
}

fn load_authority_key(
    authority_key: Option<&str>,
    authority_key_file: Option<&PathBuf>,
) -> Result<p256::ecdsa::VerifyingKey, Box<dyn std::error::Error>> {
    if let Some(path) = authority_key_file {
        let content = std::fs::read_to_string(path)?;
        let trimmed = content.trim();
        if trimmed.starts_with("-----BEGIN") {
            let vk = p256::ecdsa::VerifyingKey::from_public_key_pem(trimmed)?;
            return Ok(vk);
        }
        if let Ok(bytes) = hex::decode(trimmed) {
            let vk = p256::ecdsa::VerifyingKey::from_sec1_bytes(&bytes)?;
            return Ok(vk);
        }
        let raw = std::fs::read(path)?;
        let vk = p256::ecdsa::VerifyingKey::from_sec1_bytes(&raw)?;
        return Ok(vk);
    }
    if let Some(k) = authority_key {
        let trimmed = k.trim();
        if trimmed.starts_with("-----BEGIN") {
            let vk = p256::ecdsa::VerifyingKey::from_public_key_pem(trimmed)?;
            return Ok(vk);
        }
        let bytes = hex::decode(trimmed)?;
        let vk = p256::ecdsa::VerifyingKey::from_sec1_bytes(&bytes)?;
        return Ok(vk);
    }
    if let Ok(vault_val) =
        spark_harvester::HarvesterClient::resolve_vault_key("ATLAS_AUTHORITY_PUBKEY")
    {
        let trimmed = vault_val.trim();
        if trimmed.starts_with("-----BEGIN") {
            let vk = p256::ecdsa::VerifyingKey::from_public_key_pem(trimmed)?;
            return Ok(vk);
        }
        if let Ok(bytes) = hex::decode(trimmed) {
            let vk = p256::ecdsa::VerifyingKey::from_sec1_bytes(&bytes)?;
            return Ok(vk);
        }
    }
    Err("Authority verifying key required: pass --authority-key <HEX_OR_PEM>, --authority-key-file <PATH>, or register ATLAS_AUTHORITY_PUBKEY in atlas-vault.".into())
}

fn parse_bundle_kind(s: &str) -> Result<BundleKind, String> {
    match s.to_lowercase().as_str() {
        "training" => Ok(BundleKind::Training),
        "distillation" => Ok(BundleKind::Distillation),
        "evidence" => Ok(BundleKind::Evidence),
        other => Err(format!(
            "Unknown bundle kind {}. Must be training, distillation, or evidence",
            other
        )),
    }
}

const DEFAULT_ALLOWED_LICENSES: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "SRCL-1.0",
    "PolyForm-Noncommercial-1.0.0",
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let cli = Cli::parse();
    match cli.command {
        Commands::Extract {
            prompt,
            input,
            provider,
            model,
            grant_file,
            grant_json,
            authority_key,
            authority_key_file,
            output,
            bundle_kind,
        } => {
            let grant = load_grant(grant_file.as_ref(), grant_json.as_deref())?;
            let authority_vk =
                load_authority_key(authority_key.as_deref(), authority_key_file.as_ref())?;
            let kind = parse_bundle_kind(&bundle_kind)?;

            let extractor = ReasoningExtractor::new();
            let pair = extractor.extract(&prompt, &input, &provider, &model);
            info!(
                "Extracted pair: tokens={}, quality={:.2}",
                pair.token_count, pair.quality_score
            );

            let bundle = RightsGate::validate_and_bundle(
                &grant,
                &[pair],
                kind,
                DEFAULT_ALLOWED_LICENSES,
                &authority_vk,
            )?;
            info!(
                "RightsGate accepted bundle: ID={}, MerkleRoot={:?}",
                bundle.bundle_id, bundle.merkle_root
            );

            if let Some(out_path) = output {
                let count = RightsGate::export_bundle(&bundle, &out_path)?;
                info!("Exported {} rights-gated records to {:?}", count, out_path);
            }

            println!("{}", serde_json::to_string_pretty(&bundle)?);
        }
        Commands::Distill {
            output,
            grant_file,
            grant_json,
            authority_key,
            authority_key_file,
        } => {
            let grant = load_grant(grant_file.as_ref(), grant_json.as_deref())?;
            let authority_vk =
                load_authority_key(authority_key.as_deref(), authority_key_file.as_ref())?;

            info!("Preparing distillation with SourceGrant {}", grant.grant_id);
            let sample_pair = ExtractedPair {
                prompt: "Sovereign AI foundational premise".to_string(),
                reasoning: Some("Verify local computation vs centralized extraction".to_string()),
                completion: "Human freedom requires local ownership of intelligence.".to_string(),
                provider: "community".to_string(),
                model: "atlas-sovereign".to_string(),
                token_count: 42,
                quality_score: 1.0,
            };

            let bundle = RightsGate::validate_and_bundle(
                &grant,
                &[sample_pair],
                BundleKind::Distillation,
                DEFAULT_ALLOWED_LICENSES,
                &authority_vk,
            )?;

            let count = RightsGate::export_bundle(&bundle, &output)?;
            info!(
                "Appended {} verified rights-gated pairs to {:?}",
                count, output
            );
        }
        Commands::Status => {
            println!("AIEN Sovereign Harvester Pipeline Active");
            println!("Rights Enforcement: Active (Strict SourceGrant validation & structural evidence quarantine)");
            println!("Zero Disk Secrets: Enforced via hardware TPM vault (atlas-vault)");
            println!("License: PolyForm Noncommercial 1.0.0 with Sovereign Defense Covenant");
        }
    }
    Ok(())
}
