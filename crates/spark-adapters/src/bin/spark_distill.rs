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

use chrono::Utc;
use clap::{Parser, Subcommand};
use spark_adapters::curriculum::{CurriculumEngine, CurriculumTrack};
use spark_adapters::models::{DistillTask, DistillationRecord, TaskType, VerificationStrategy};
use spark_adapters::vault::sanitize_outbound_prompt;
use spark_adapters::verifier::Verifier;
use spark_adapters::{AdapterRouter, DistillationEngine};
use std::io::{self, BufRead, IsTerminal, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "spark-distill")]
#[command(author = "AIEN <aien@aienos.com>")]
#[command(version = "0.2.0")]
#[command(
    about = "Sovereign Teacher Distillation, Multi-Track Curriculum Crawler, and Training Pair Generator"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Flat arguments for backwards compatibility when no subcommand is given
    #[arg(short, long, help = "Task prompt or reasoning challenge to evaluate")]
    prompt: Option<String>,

    #[arg(
        short,
        long,
        default_value = "anthropic/claude-3-7-sonnet",
        help = "Teacher model identifier"
    )]
    teacher: String,

    #[arg(
        short,
        long,
        default_value = "atlas-lightning-omni",
        help = "Student open-weight model seat"
    )]
    student: String,

    #[arg(
        long,
        default_value = "code",
        help = "Task type: code, arch, reasoning, refactor, harness"
    )]
    task_type: String,

    #[arg(
        long,
        default_value = "compiler",
        help = "Verification strategy: compiler, unslop, json, consensus"
    )]
    verify: String,

    #[arg(long, help = "Commit high-scoring verified solution to Cortex memory")]
    commit_cortex: bool,

    #[arg(long, help = "Custom dataset directory for SFT/DPO output")]
    dataset_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the interactive terminal distillation workbench
    Cli {
        #[arg(
            short,
            long,
            default_value = "anthropic/claude-3-7-sonnet",
            help = "Default teacher model identifier"
        )]
        teacher: String,

        #[arg(
            short,
            long,
            default_value = "atlas-lightning-omni",
            help = "Default student open-weight model seat"
        )]
        student: String,

        #[arg(
            short = 'k',
            long,
            help = "Optional initial curriculum track: systems, agent, science, frontier, dynamic"
        )]
        track: Option<String>,

        #[arg(
            long,
            alias = "commit-cortex",
            help = "Auto-commit verified high-scoring solutions to Cortex without interactive confirmation"
        )]
        auto_commit: bool,

        #[arg(long, help = "Custom dataset directory for SFT/DPO output")]
        dataset_dir: Option<PathBuf>,
    },

    /// Execute a single prompt distillation task
    Single {
        #[arg(short, long, help = "Task prompt or reasoning challenge to evaluate")]
        prompt: String,

        #[arg(
            short,
            long,
            default_value = "anthropic/claude-3-7-sonnet",
            help = "Teacher model identifier"
        )]
        teacher: String,

        #[arg(
            short,
            long,
            default_value = "atlas-lightning-omni",
            help = "Student open-weight model seat"
        )]
        student: String,

        #[arg(
            long,
            default_value = "code",
            help = "Task type: code, arch, reasoning, refactor, harness"
        )]
        task_type: String,

        #[arg(
            long,
            default_value = "compiler",
            help = "Verification strategy: compiler, unslop, json, consensus"
        )]
        verify: String,

        #[arg(long, help = "Commit high-scoring verified solution to Cortex memory")]
        commit_cortex: bool,

        #[arg(long, help = "Custom dataset directory for SFT/DPO output")]
        dataset_dir: Option<PathBuf>,
    },

    /// Run an autonomous batch distillation crawler across a curriculum track
    Crawl {
        #[arg(
            short = 'k',
            long,
            default_value = "systems",
            help = "Curriculum track: systems, agent, science, frontier, dynamic"
        )]
        track: String,

        #[arg(short, long, default_value_t = 5, help = "Number of tasks to execute")]
        limit: usize,

        #[arg(
            short,
            long,
            default_value = "anthropic/claude-3-7-sonnet",
            help = "Teacher model identifier"
        )]
        teacher: String,

        #[arg(
            short,
            long,
            default_value = "atlas-lightning-omni",
            help = "Student open-weight model seat"
        )]
        student: String,

        #[arg(long, help = "Commit high-scoring verified solution to Cortex memory")]
        commit_cortex: bool,

        #[arg(long, help = "Custom dataset directory for SFT/DPO output")]
        dataset_dir: Option<PathBuf>,
    },

    /// Audit and verify all configured cloud subscriptions and hardware TPM vault keys
    VerifyKeys,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Cli {
            teacher,
            student,
            track,
            auto_commit,
            dataset_dir,
        }) => {
            run_cli(teacher, student, track, auto_commit, dataset_dir).await;
        }

        Some(Commands::Single {
            prompt,
            teacher,
            student,
            task_type,
            verify,
            commit_cortex,
            dataset_dir,
        }) => {
            run_single(
                prompt,
                teacher,
                student,
                task_type,
                verify,
                commit_cortex,
                dataset_dir,
            )
            .await;
        }

        Some(Commands::Crawl {
            track,
            limit,
            teacher,
            student,
            commit_cortex,
            dataset_dir,
        }) => {
            run_crawl(
                &track,
                limit,
                &teacher,
                &student,
                commit_cortex,
                dataset_dir,
            )
            .await;
        }

        Some(Commands::VerifyKeys) => {
            run_verify_keys();
        }

        None => {
            if let Some(prompt) = cli.prompt {
                run_single(
                    prompt,
                    cli.teacher,
                    cli.student,
                    cli.task_type,
                    cli.verify,
                    cli.commit_cortex,
                    cli.dataset_dir,
                )
                .await;
            } else {
                eprintln!("Error: Missing prompt. Use `spark-distill single --prompt ...` or `spark-distill crawl --track systems`.");
                std::process::exit(1);
            }
        }
    }
}

async fn run_cli(
    mut teacher: String,
    mut student: String,
    initial_track: Option<String>,
    auto_commit: bool,
    dataset_dir: Option<PathBuf>,
) {
    let mut task_type = TaskType::CodeSynthesis;
    let mut strategy = VerificationStrategy::CompilerCheck;
    let is_interactive = io::stdin().is_terminal();

    println!("========================================================================");
    println!("  AIEN Sovereign Distillation Workshop - Interactive Terminal Workbench");
    println!("  NVIDIA DGX Spark (Grace Blackwell GB10 LPDDR5X)");
    println!("========================================================================");
    println!("Type 'help' for command reference, or enter a prompt directly to distill.");
    println!("Commands: models, teacher <id>, student <id>, track <name>, strategy <name>, status, quit\n");

    let mut engine = DistillationEngine::new().with_timeout(Duration::from_secs(180));
    if let Some(ref dir) = dataset_dir {
        engine = engine.with_dataset_dir(dir.clone());
    }

    if let Some(ref track_str) = initial_track {
        if let Some(track) = CurriculumTrack::parse(track_str) {
            println!("Loaded curriculum track: {}", track.display_name());
            let tasks =
                CurriculumEngine::generate_tasks(track, 5, &teacher, Some(&student), auto_commit);
            println!("\nTrack prompts:");
            for (i, t) in tasks.iter().enumerate() {
                println!("  [{}] {}", i + 1, t.prompt);
            }
        } else {
            eprintln!("Warning: Unknown curriculum track '{}'", track_str);
        }
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        if is_interactive {
            print!("spark-distill> ");
            let _ = io::stdout().flush();
        }

        let mut input = String::new();
        match reader.read_line(&mut input) {
            Ok(0) => {
                if is_interactive {
                    println!("\nExiting interactive workbench.");
                }
                break;
            }
            Ok(_) => {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                match parts[0].to_lowercase().as_str() {
                    "help" | "?" => {
                        print_cli_help();
                    }
                    "status" => {
                        println!("Current Configuration:");
                        println!("  Teacher Model:   {}", teacher);
                        println!("  Student Model:   {}", student);
                        println!("  Task Type:       {:?}", task_type);
                        println!("  Strategy:        {:?}", strategy);
                        println!("  Dataset Dir:     {}", engine.dataset_dir.display());
                        println!("  Auto Commit:     {}", auto_commit);
                        println!("  Interactive TTY: {}", is_interactive);
                    }
                    "models" => {
                        run_verify_keys();
                    }
                    "teacher" => {
                        if parts.len() > 1 {
                            teacher = parts[1].to_string();
                            println!("Teacher model set to: {}", teacher);
                        } else {
                            println!("Usage: teacher <model_id>");
                        }
                    }
                    "student" => {
                        if parts.len() > 1 {
                            student = parts[1].to_string();
                            println!("Student model seat set to: {}", student);
                        } else {
                            println!("Usage: student <model_id>");
                        }
                    }
                    "strategy" => {
                        if parts.len() > 1 {
                            strategy = parse_strategy(parts[1]);
                            println!("Verification strategy set to: {:?}", strategy);
                        } else {
                            println!("Usage: strategy <compiler|unslop|json|consensus>");
                        }
                    }
                    "type" => {
                        if parts.len() > 1 {
                            task_type = parse_task_type(parts[1]);
                            println!("Task type set to: {:?}", task_type);
                        } else {
                            println!("Usage: type <code|arch|reasoning|refactor|harness>");
                        }
                    }
                    "track" => {
                        let track_name = if parts.len() > 1 { parts[1] } else { "systems" };
                        if let Some(track) = CurriculumTrack::parse(track_name) {
                            println!("Track: {}", track.display_name());
                            let tasks = CurriculumEngine::generate_tasks(
                                track,
                                5,
                                &teacher,
                                Some(&student),
                                auto_commit,
                            );
                            for (i, t) in tasks.iter().enumerate() {
                                println!("  [{}] {}", i + 1, t.prompt);
                            }
                            if is_interactive {
                                print!("Select task [1-5] or [c]ancel: ");
                                let _ = io::stdout().flush();
                                let mut choice = String::new();
                                if reader.read_line(&mut choice).is_ok() {
                                    let choice_trim = choice.trim();
                                    if let Ok(num) = choice_trim.parse::<usize>() {
                                        if num >= 1 && num <= tasks.len() {
                                            let selected_task = tasks[num - 1].clone();
                                            execute_cli_distill(
                                                &engine,
                                                selected_task.prompt,
                                                &teacher,
                                                &student,
                                                selected_task.task_type,
                                                selected_task.verification_strategy,
                                                auto_commit,
                                                is_interactive,
                                                &mut reader,
                                            )
                                            .await;
                                        }
                                    }
                                }
                            }
                        } else {
                            println!(
                                "Unknown track '{}'. Available: systems, agent, science, frontier, dynamic",
                                track_name
                            );
                        }
                    }
                    "exit" | "quit" | "q" => {
                        println!("Exiting interactive workbench.");
                        break;
                    }
                    _ => {
                        execute_cli_distill(
                            &engine,
                            trimmed.to_string(),
                            &teacher,
                            &student,
                            task_type,
                            strategy,
                            auto_commit,
                            is_interactive,
                            &mut reader,
                        )
                        .await;
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_cli_distill<R: BufRead>(
    engine: &DistillationEngine,
    prompt: String,
    teacher: &str,
    student: &str,
    task_type: TaskType,
    strategy: VerificationStrategy,
    auto_commit: bool,
    is_interactive: bool,
    reader: &mut R,
) {
    let sanitized_prompt = sanitize_outbound_prompt(&prompt);
    if sanitized_prompt != prompt {
        println!("  [Vault Sanitizer] Outbound prompt scrubbed: local paths, private IPs, and secrets redacted.");
    }

    let task_id = Uuid::new_v4().to_string();
    let task = DistillTask {
        id: task_id.clone(),
        task_type,
        prompt: sanitized_prompt.clone(),
        system_prompt: Some(
            "You are a verified sovereign systems specialist. Output clear, compilable native code with zero unslop."
                .to_string(),
        ),
        teacher_model: teacher.to_string(),
        student_model: Some(student.to_string()),
        verification_strategy: strategy,
        commit_to_cortex: false,
    };

    println!("\n=== Distillation Rollout Dispatched ===");
    println!("Task ID:        {}", task.id);
    println!("Teacher Oracle: {}", teacher);
    println!("Student Seat:   {}", student);
    println!("Task Type:      {:?}", task_type);
    println!("Strategy:       {:?}", strategy);
    println!("Prompt:         {}", sanitized_prompt);

    match engine.distill(task).await {
        Ok(record) => {
            display_rollout_comparison(&record);

            let t_pass = record.teacher_rollout.verification.passed;
            let t_score = record.teacher_rollout.verification.score;
            let s_pass = record
                .student_rollout
                .as_ref()
                .map(|s| s.verification.passed)
                .unwrap_or(false);
            let s_score = record
                .student_rollout
                .as_ref()
                .map(|s| s.verification.score)
                .unwrap_or(0.0);

            let winning_score = t_score.max(s_score);
            let winning_passed = if t_score >= s_score { t_pass } else { s_pass };
            let qualifies = winning_passed && winning_score >= 0.85;

            if qualifies {
                if auto_commit {
                    commit_cli_cortex(engine, &record).await;
                } else if is_interactive {
                    print!(
                        "\nCommit verified winning heuristic to Cortex memory (atlas-memory on port 18080)? [y/N]: "
                    );
                    let _ = io::stdout().flush();
                    let mut ans = String::new();
                    if reader.read_line(&mut ans).is_ok() {
                        let a = ans.trim().to_lowercase();
                        if a == "y" || a == "yes" {
                            commit_cli_cortex(engine, &record).await;
                        } else {
                            println!("  [Cortex Commit] Skipped by operator.");
                        }
                    }
                } else {
                    println!(
                        "  [Cortex Commit] Non-interactive mode: commit skipped (use --auto-commit to persist)."
                    );
                }
            } else {
                println!(
                    "\n  [Cortex Gate] Solution score ({:.2}) did not meet qualification threshold (score >= 0.85 and passed).",
                    winning_score
                );
            }

            if record.has_dpo_pair() {
                println!(
                    "  Datasets Appended: {}/sft.jsonl, {}/dpo.jsonl",
                    engine.dataset_dir.display(),
                    engine.dataset_dir.display()
                );
            } else {
                println!(
                    "  Datasets Appended: {}/sft.jsonl",
                    engine.dataset_dir.display()
                );
            }
        }
        Err(e) => {
            eprintln!("\nDistillation error: {}", e);
        }
    }
}

fn display_rollout_comparison(record: &DistillationRecord) {
    let teacher_content = &record.teacher_rollout.content;
    let student_content = record
        .student_rollout
        .as_ref()
        .map(|s| s.content.as_str())
        .unwrap_or("");

    println!(
        "\n--- Teacher Oracle Rollout ({} ms, {} tokens) ---",
        record.teacher_rollout.duration_ms, record.teacher_rollout.token_count
    );
    let preview_len = 300.min(teacher_content.len());
    println!(
        "{}{}",
        &teacher_content[..preview_len],
        if teacher_content.len() > preview_len {
            "..."
        } else {
            ""
        }
    );

    if let Some(ref s_roll) = record.student_rollout {
        println!(
            "\n--- Student Seat Rollout ({} ms, {} tokens) ---",
            s_roll.duration_ms, s_roll.token_count
        );
        let s_preview_len = 300.min(student_content.len());
        println!(
            "{}{}",
            &student_content[..s_preview_len],
            if student_content.len() > s_preview_len {
                "..."
            } else {
                ""
            }
        );
    }

    let lev = Verifier::normalized_levenshtein(teacher_content, student_content);
    let lcs = Verifier::normalized_lcs(teacher_content, student_content);
    let jac = Verifier::token_jaccard(teacher_content, student_content);
    let consensus = Verifier::hybrid_consensus(teacher_content, student_content);

    let (s_dur, s_tok, s_pass, s_score, s_viol, s_comp) = match record.student_rollout.as_ref() {
        Some(s) => (
            format!("{}", s.duration_ms),
            format!("{}", s.token_count),
            if s.verification.passed {
                "PASSED"
            } else {
                "FAILED"
            },
            format!("{:.2}", s.verification.score),
            format!("{}", s.verification.rule_violations.len()),
            if s.verification.compiler_output.is_some() {
                "CHECKED"
            } else {
                "CLEAN"
            },
        ),
        None => (
            "N/A".to_string(),
            "N/A".to_string(),
            "N/A",
            "N/A".to_string(),
            "N/A".to_string(),
            "N/A",
        ),
    };

    let t_comp = if record
        .teacher_rollout
        .verification
        .compiler_output
        .is_some()
    {
        "CHECKED"
    } else {
        "CLEAN"
    };

    println!("\n+------------------------------------------------------------------------+");
    println!("|                         SCORECARD & VERIFICATION                       |");
    println!("+------------------------------------+-----------------+-----------------+");
    println!("| Metric                             | Teacher Oracle  | Student Seat    |");
    println!("+------------------------------------+-----------------+-----------------+");
    println!(
        "| Latency                            | {:>13} ms | {:>13} ms |",
        record.teacher_rollout.duration_ms, s_dur
    );
    println!(
        "| Tokens                             | {:>15} | {:>15} |",
        record.teacher_rollout.token_count, s_tok
    );
    println!(
        "| Verification                       | {:>15} | {:>15} |",
        if record.teacher_rollout.verification.passed {
            "PASSED"
        } else {
            "FAILED"
        },
        s_pass
    );
    println!(
        "| Quality Score                      | {:>15.2} | {:>15} |",
        record.teacher_rollout.verification.score, s_score
    );
    println!(
        "| Rule Violations                    | {:>15} | {:>15} |",
        record.teacher_rollout.verification.rule_violations.len(),
        s_viol
    );
    println!(
        "| Rustc Syntax Check                 | {:>15} | {:>15} |",
        t_comp, s_comp
    );
    println!("+------------------------------------+-----------------+-----------------+");
    println!("|                      DUAL-CONSENSUS ALIGNMENT                          |");
    println!("+------------------------------------+-----------------------------------+");
    println!("| Token Jaccard Similarity           | {:>33.4} |", jac);
    println!("| Normalized Levenshtein             | {:>33.4} |", lev);
    println!("| Normalized LCS                     | {:>33.4} |", lcs);
    println!(
        "| Hybrid Consensus Score             | {:>33.4} |",
        consensus
    );
    println!(
        "| Preference Delta                   | {:>33.4} |",
        record.preference_delta
    );
    println!(
        "| Selected Solution                  | {:>33} |",
        if record.chosen == *teacher_content {
            "Teacher Oracle"
        } else {
            "Student Seat"
        }
    );
    println!("+------------------------------------+-----------------------------------+");
}

async fn commit_cli_cortex(engine: &DistillationEngine, record: &DistillationRecord) {
    let canonical = format!("distill_{}", record.task_id.replace('-', "_"));
    let entity_name = format!(
        "distill_lesson:{}_{}",
        record.task_id,
        Utc::now().timestamp()
    );
    let summary = format!(
        "Interactive distillation. Prompt: {}. Reasoning: {}. Verified Solution: {}",
        record.prompt,
        record.chosen_reasoning.as_deref().unwrap_or("<direct>"),
        record.chosen
    );
    let confidence = (record.teacher_rollout.verification.score.max(
        record
            .student_rollout
            .as_ref()
            .map(|s| s.verification.score)
            .unwrap_or(0.0),
    )) as f64;

    match engine
        .commit_to_cortex(
            &entity_name,
            &canonical,
            &summary,
            &record.task_id,
            confidence,
        )
        .await
    {
        Ok(id) => {
            println!(
                "  [Cortex Commit] Entity registered successfully: {} (Target ID: {})",
                canonical, id
            );
        }
        Err(e) => {
            eprintln!("  [Cortex Commit Error] Failed to persist to Cortex: {}", e);
        }
    }
}

fn print_cli_help() {
    println!("\nSovereign Distillation Workbench Command Reference:");
    println!("  help, ?                    Show this help reference");
    println!("  status                     Display active models, strategy, and options");
    println!("  models                     List all catalog model adapters and key status");
    println!("  teacher <model_id>         Set teacher model (e.g. anthropic/claude-3-7-sonnet, openai/gpt-4o)");
    println!("  student <model_id>         Set student model seat (e.g. atlas-lightning-omni, ollama/qwen2.5-coder)");
    println!("  strategy <name>            Set strategy: compiler, unslop, json, consensus");
    println!(
        "  type <name>                Set task type: code, arch, reasoning, refactor, harness"
    );
    println!("  track [name]               List tasks in curriculum track (systems, agent, science, frontier, dynamic)");
    println!("  exit, quit, q              Exit interactive workbench");
    println!("  <prompt text>              Evaluate prompt immediately through dual rollouts\n");
}

async fn run_single(
    prompt: String,
    teacher: String,
    student: String,
    task_type_str: String,
    verify_str: String,
    commit_cortex: bool,
    dataset_dir: Option<PathBuf>,
) {
    let task_type = parse_task_type(&task_type_str);
    let strategy = parse_strategy(&verify_str);

    let sanitized_prompt = sanitize_outbound_prompt(&prompt);
    if sanitized_prompt != prompt {
        println!("  [Vault Sanitizer] Outbound prompt scrubbed: local paths, private IPs, and secrets redacted.");
    }

    let task = DistillTask {
        id: Uuid::new_v4().to_string(),
        task_type,
        prompt: sanitized_prompt.clone(),
        system_prompt: Some(
            "You are a verified sovereign systems specialist. Output clear, compilable native code with zero unslop."
                .to_string(),
        ),
        teacher_model: teacher.clone(),
        student_model: Some(student.clone()),
        verification_strategy: strategy,
        commit_to_cortex: commit_cortex,
    };

    println!("=== Sovereign Knowledge Distillation ===");
    println!("Task ID:  {}", task.id);
    println!("Teacher:  {}", task.teacher_model);
    println!(
        "Student:  {}",
        task.student_model.as_deref().unwrap_or("<none>")
    );
    println!("Prompt:   {}", task.prompt);

    let mut engine = DistillationEngine::new().with_timeout(Duration::from_secs(180));
    if let Some(dir) = dataset_dir {
        engine = engine.with_dataset_dir(dir);
    }

    match engine.distill(task).await {
        Ok(record) => {
            println!("\n=== Distillation Complete ===");
            println!(
                "Teacher Verification: Passed: {}, Score: {:.2}",
                record.teacher_rollout.verification.passed,
                record.teacher_rollout.verification.score
            );
            if let Some(ref s_roll) = record.student_rollout {
                println!(
                    "Student Verification: Passed: {}, Score: {:.2}",
                    s_roll.verification.passed, s_roll.verification.score
                );
                println!("Preference Delta:     {:.2}", record.preference_delta);
            }
            if record.durable_memory_committed {
                println!(
                    "Cortex Entity:        {}",
                    record.cortex_entity_id.as_deref().unwrap_or("<committed>")
                );
            }
            if record.has_dpo_pair() {
                println!(
                    "Datasets Appended:    {}/sft.jsonl, {}/dpo.jsonl",
                    engine.dataset_dir.display(),
                    engine.dataset_dir.display()
                );
            } else {
                println!(
                    "Datasets Appended:    {}/sft.jsonl",
                    engine.dataset_dir.display()
                );
            }
        }
        Err(e) => {
            eprintln!("Distillation error: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run_crawl(
    track_str: &str,
    limit: usize,
    teacher: &str,
    student: &str,
    commit_cortex: bool,
    dataset_dir: Option<PathBuf>,
) {
    let track = CurriculumTrack::parse(track_str).unwrap_or(CurriculumTrack::NativeSystemsAndGpu);
    println!("=== Sovereign Batch Curriculum Crawler ===");
    println!("Curriculum Track: {}", track.display_name());
    println!("Task Limit:       {}", limit);
    println!("Teacher Model:    {}", teacher);
    println!("Student Seat:     {}", student);
    println!("Cortex Commit:    {}", commit_cortex);

    let tasks =
        CurriculumEngine::generate_tasks(track, limit, teacher, Some(student), commit_cortex);

    let mut engine = DistillationEngine::new().with_timeout(Duration::from_secs(180));
    if let Some(dir) = dataset_dir {
        engine = engine.with_dataset_dir(dir);
    }

    let mut passed_count = 0;
    let mut total_delta = 0.0;
    let mut cortex_count = 0;

    for (idx, task) in tasks.into_iter().enumerate() {
        println!("\n[{}/{}] Distilling: {}", idx + 1, limit, task.prompt);
        match engine.distill(task).await {
            Ok(record) => {
                let t_pass = record.teacher_rollout.verification.passed;
                let s_pass = record
                    .student_rollout
                    .as_ref()
                    .map(|s| s.verification.passed)
                    .unwrap_or(false);
                if t_pass || s_pass {
                    passed_count += 1;
                }
                total_delta += record.preference_delta;
                if record.durable_memory_committed {
                    cortex_count += 1;
                    println!(
                        "  [+] Committed to Cortex: {}",
                        record.cortex_entity_id.as_deref().unwrap_or("<id>")
                    );
                }
                println!(
                    "  [+] Score: Teacher: {:.2}, Delta: {:.2}",
                    record.teacher_rollout.verification.score, record.preference_delta
                );
            }
            Err(e) => {
                eprintln!("  [-] Task failed: {}", e);
            }
        }
    }

    println!("\n=== Curriculum Crawl Summary ===");
    println!("Total Tasks Evaluated:  {}", limit);
    println!("Passing Candidates:     {}/{}", passed_count, limit);
    println!("Cortex Entities Added:  {}", cortex_count);
    println!(
        "Average Preference Delta: {:.2}",
        total_delta / limit.max(1) as f32
    );
    println!("Datasets Directory:     {}", engine.dataset_dir.display());
}

fn is_port_open(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok()
}

fn run_verify_keys() {
    let router = AdapterRouter::new();
    let adapters = router.discover_adapters();

    println!("=== Sovereign Model & Subscription Key Audit ===");
    println!(
        "{:<28} {:<15} {:<20} {:<10}",
        "ADAPTER ID", "PROVIDER", "STATUS", "RESIDENT"
    );
    println!("{}", "-".repeat(76));

    for a in adapters {
        let status = if a.is_resident {
            "LOCAL_SILICON"
        } else if a.provider == spark_adapters::models::ProviderType::Ollama {
            if is_port_open(11434) {
                "PORT_OPEN"
            } else {
                "UNCONFIGURED"
            }
        } else if a.is_available {
            "OK"
        } else {
            "UNCONFIGURED"
        };
        let resident = if a.is_resident { "YES" } else { "NO" };
        println!(
            "{:<28} {:<15} {:<20} {:<10}",
            a.id,
            a.provider.slug(),
            status,
            resident
        );
    }

    println!("\n=== Sovereign Infrastructure & Daemons ===");
    println!(
        "{:<28} {:<24} {:<15} {:<20}",
        "SERVICE", "ENDPOINT", "STATUS", "AUTH"
    );
    println!("{}", "-".repeat(84));

    let seat_port = is_port_open(18006);
    let seat_status = if seat_port { "PORT_OPEN" } else { "OFFLINE" };
    println!(
        "{:<28} {:<24} {:<15} {:<20}",
        "Local Model Seat", "http://127.0.0.1:18006", seat_status, "LOCAL_SILICON"
    );

    let cortex_port = is_port_open(18080);
    let cortex_key = spark_adapters::vault::is_secret_present("CORTEX_TOKEN");
    let cortex_status = if cortex_port { "PORT_OPEN" } else { "OFFLINE" };
    let cortex_auth = if cortex_key {
        "OK (CORTEX_TOKEN)"
    } else {
        "UNCONFIGURED"
    };
    println!(
        "{:<28} {:<24} {:<15} {:<20}",
        "Cortex Memory Daemon", "http://127.0.0.1:18080", cortex_status, cortex_auth
    );
}

fn parse_task_type(s: &str) -> TaskType {
    match s.to_lowercase().as_str() {
        "arch" => TaskType::SystemArchitecture,
        "reasoning" => TaskType::ReasoningTrace,
        "refactor" => TaskType::RefactorLogic,
        "harness" => TaskType::VerificationHarness,
        _ => TaskType::CodeSynthesis,
    }
}

fn parse_strategy(s: &str) -> VerificationStrategy {
    match s.to_lowercase().as_str() {
        "unslop" => VerificationStrategy::UnslopStrict,
        "json" => VerificationStrategy::JsonSchema,
        "consensus" => VerificationStrategy::DualConsensus,
        _ => VerificationStrategy::CompilerCheck,
    }
}
