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

// spark-aegis CLI Entrypoint
// Sovereign Defensive Boundary and Containment Engine

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::json;
use spark_aegis::{
    audit, containment,
    doctrine::{all_invariants, DOCTRINE_STATEMENT},
    pr_triage, scanner,
};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "spark-aegis",
    about = "Sovereign defensive boundary and containment engine",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Deep scan a file or directory for secrets, slop, and dangerous commands
    Scan {
        /// Target file or directory path
        path: PathBuf,
    },
    /// Autonomous PR review and diff triage
    ReviewPr {
        /// Path to diff file (or - for stdin)
        diff_file: PathBuf,
    },
    /// Live system and workspace confinement audit
    Audit,
    /// Execute defensive containment actions on session, process, or path
    Contain {
        /// Target identifier (process ID, session ID, or filesystem path)
        target: String,
        /// Explicitly treat target as a process ID to isolate
        #[arg(long)]
        pid: Option<u32>,
        /// Explicitly treat target as a session ID to cut
        #[arg(long)]
        session: Option<String>,
        /// Explicitly treat target as a filesystem path to close and quarantine
        #[arg(long)]
        close: Option<PathBuf>,
    },
    /// Display sovereign doctrine and invariant definitions
    Doctrine,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Scan { path } => {
            println!("SPARK-AEGIS DEEP SCAN");
            println!("Target: {}", path.display());
            println!("------------------------------------------------------------");

            let report = scanner::scan_path(&path)?;
            println!("Files scanned: {}", report.scanned_files_count);
            println!("Duration: {:.2} ms", report.elapsed_ms);
            println!("Blake3 hash:   {}", report.blake3_hash);
            println!(
                "Status:        {}",
                if report.passed {
                    "CLEAN"
                } else {
                    "VIOLATIONS_FOUND"
                }
            );

            if !report.violations.is_empty() {
                println!("\nVIOLATION REPORT ({} items):", report.violations.len());
                for (idx, v) in report.violations.iter().enumerate() {
                    println!("\n[{}] {:?} ({:?})", idx + 1, v.rule, v.severity);
                    println!("  Path:        {}", v.file_path.display());
                    if let Some(line) = v.line_number {
                        println!("  Line:        {}", line);
                    }
                    println!("  Match:       {}", v.matched_text);
                    println!("  Remediation: {}", v.remediation);
                }
                std::process::exit(1);
            } else {
                println!("\nZero violations detected. Target conforms to sovereign invariants.");
            }
        }

        Commands::ReviewPr { diff_file } => {
            println!("SPARK-AEGIS AUTONOMOUS PR TRIAGE");
            println!("------------------------------------------------------------");

            let content = if diff_file == Path::new("-") {
                use std::io::Read;
                let mut buffer = String::new();
                std::io::stdin().read_to_string(&mut buffer)?;
                buffer
            } else {
                std::fs::read_to_string(&diff_file)?
            };

            let pr_name = diff_file.display().to_string();
            let verdict = pr_triage::review_pr_diff(&content, &pr_name)?;

            println!("PR Identifier: {}", verdict.pr_id);
            println!("Blake3 Proof:  {}", verdict.blake3_proof);
            println!("Status:        {:?}", verdict.status);
            println!(
                "Zero Disk Secrets: {}",
                if verdict.zero_disk_secrets_certified {
                    "CERTIFIED"
                } else {
                    "FAILED"
                }
            );
            println!(
                "Unslop Compliance: {}",
                if verdict.unslop_certified {
                    "CERTIFIED"
                } else {
                    "FAILED"
                }
            );
            println!(
                "Safe Execution:    {}",
                if verdict.safe_execution_certified {
                    "CERTIFIED"
                } else {
                    "FAILED"
                }
            );
            println!("Summary:           {}", verdict.summary);

            if !verdict.violations.is_empty() {
                println!("\nViolations List:");
                for (i, v) in verdict.violations.iter().enumerate() {
                    println!(
                        "  {}. [{:?}] {} at line {:?}",
                        i + 1,
                        v.category,
                        v.rule,
                        v.line_number
                    );
                    println!("     Remediation: {}", v.remediation);
                }
                std::process::exit(1);
            }
        }

        Commands::Audit => {
            println!("SPARK-AEGIS LIVE HOST & WORKSPACE CONFINEMENT AUDIT");
            println!("------------------------------------------------------------");

            let report = audit::audit_system().await?;
            println!("Timestamp:               {}", report.timestamp.to_rfc3339());
            println!("Blake3 Audit Digest:     {}", report.blake3_digest);
            println!(
                "Cortex Memory Service:   {}",
                if report.cortex_online {
                    "ONLINE"
                } else {
                    "OFFLINE"
                }
            );
            if let Some(v) = report.cortex_version {
                println!("Cortex Engine Version:   {}", v);
            }
            println!(
                "Workspace Path:          {}",
                report.workspace.path.display()
            );
            println!(
                "Workspace Status:        {}",
                if report.workspace.secure_tpm_only {
                    "SECURE_TPM_ONLY (PASS)"
                } else {
                    "FAILED (INSECURE)"
                }
            );

            if !report.workspace.dot_env_found.is_empty() {
                println!("\nWARNING: Unencrypted .env files detected:");
                for p in &report.workspace.dot_env_found {
                    println!("  - {}", p.display());
                }
            }

            println!("\nListening Sockets Count: {}", report.sockets.len());
            println!(
                "Non-Loopback Sockets:    {}",
                report.non_loopback_sockets_count
            );
            for s in report.sockets.iter().take(10) {
                let exposure = if s.is_loopback {
                    "LOOPBACK"
                } else {
                    "EXPOSED_LAN/WAN"
                };
                println!(
                    "  {} {:<25} [{}] proc: {:?}",
                    s.protocol, s.local_address, exposure, s.process_name
                );
            }

            println!(
                "\nOverall Confinement:     {}",
                if report.overall_safe {
                    "PASS"
                } else {
                    "REJECT"
                }
            );
        }

        Commands::Contain {
            target,
            pid,
            session,
            close,
        } => {
            println!("SPARK-AEGIS CONTAINMENT DISPATCH");
            println!("Doctrine: {}", DOCTRINE_STATEMENT);
            println!("------------------------------------------------------------");

            if let Some(target_pid) = pid {
                let receipt = containment::isolate_host(target_pid)?;
                println!("Action:    {}", receipt.action);
                println!("Target:    PID {}", receipt.target);
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "manual_pid_containment",
                    "HIGH",
                    json!({"pid": target_pid, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
                return Ok(());
            }

            if let Some(sess_id) = session {
                let receipt = containment::cut_session(&sess_id)?;
                println!("Action:    {}", receipt.action);
                println!("Target:    Session {}", receipt.target);
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "manual_session_containment",
                    "HIGH",
                    json!({"session": sess_id, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
                return Ok(());
            }

            if let Some(close_path) = close {
                let receipt = containment::close_door(&close_path)?;
                println!("Action:    {}", receipt.action);
                println!("Target:    Path {}", receipt.target);
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "manual_path_containment",
                    "HIGH",
                    json!({"path": close_path, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
                return Ok(());
            }

            // Automatic target deduction: numeric PID, path, or session string
            if let Ok(parsed_pid) = target.parse::<u32>() {
                println!("Auto-detected numeric PID target: {}", parsed_pid);
                let receipt = containment::isolate_host(parsed_pid)?;
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "auto_pid_containment",
                    "HIGH",
                    json!({"pid": parsed_pid, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
            } else if Path::new(&target).exists() {
                println!("Auto-detected filesystem target: {}", target);
                let receipt = containment::close_door(Path::new(&target))?;
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "auto_path_containment",
                    "HIGH",
                    json!({"path": target, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
            } else {
                println!(
                    "Auto-detected interactive/socket session target: {}",
                    target
                );
                let receipt = containment::cut_session(&target)?;
                println!("Status:    {}", receipt.status);
                println!("Details:   {}", receipt.details);
                println!("Blake3:    {}", receipt.blake3_digest);

                let incident = containment::IncidentReport::new(
                    "auto_session_containment",
                    "HIGH",
                    json!({"session": target, "receipt": receipt}),
                );
                let audit_receipt = containment::keep_evidence(&incident).await?;
                println!("Evidence:  Cortex recorded: {}", audit_receipt.recorded);
            }
        }

        Commands::Doctrine => {
            println!("SPARK-AEGIS SOVEREIGN DEFENSIVE DOCTRINE");
            println!("============================================================");
            println!("Doctrine: \"{}\"\n", DOCTRINE_STATEMENT);
            println!("Invariants:");
            for inv in all_invariants() {
                println!("- [{}] {}", inv.code(), inv.description());
            }
        }
    }

    Ok(())
}
