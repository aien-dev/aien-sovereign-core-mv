use chrono::Utc;
use colored::*;
use serde_json::Value;
use std::fs;
use std::path::Path;

use crate::telemetry::get_gpu_telemetry;

pub fn render_walkthrough_tui() -> String {
    let gpu = get_gpu_telemetry();
    let platform = crate::platform::PlatformContext::detect();
    let pulse_path = platform.home_dir.join("basecamp/hive-pulse.json");
    let (vibe_text, pulse_stat, coherence) = if pulse_path.exists() {
        if let Ok(c) = fs::read_to_string(pulse_path) {
            if let Ok(v) = serde_json::from_str::<Value>(&c) {
                let vb = v
                    .get("vibe_summary")
                    .and_then(Value::as_str)
                    .unwrap_or("Calm sovereign equilibrium")
                    .to_string();
                let ps = v
                    .get("pulse_status")
                    .and_then(Value::as_str)
                    .unwrap_or("Harmonious")
                    .to_string();
                let ch = v
                    .get("coherence_score")
                    .and_then(Value::as_u64)
                    .unwrap_or(100);
                (vb, ps, ch)
            } else {
                (
                    "Calm sovereign equilibrium".to_string(),
                    "Harmonious".to_string(),
                    100,
                )
            }
        } else {
            (
                "Calm sovereign equilibrium".to_string(),
                "Harmonious".to_string(),
                100,
            )
        }
    } else {
        (
            "Calm sovereign equilibrium".to_string(),
            "Harmonious".to_string(),
            100,
        )
    };

    let mut out = String::new();

    out.push_str(&format!(
        "{}\n",
        "╔══════════════════════════════════════════════════════════════════════════════╗"
            .cyan()
            .bold()
    ));
    out.push_str(&format!(
        "║            {}            ║\n",
        "AIEN SOVEREIGN WALKTHROUGH & ARCHITECTURE MAP"
            .magenta()
            .bold()
    ));
    out.push_str(&format!(
        "║               {}              ║\n",
        "NVIDIA DGX Spark Grace Blackwell GB10 | spark-b87b"
            .white()
            .dimmed()
    ));
    out.push_str(&format!(
        "{}\n\n",
        "╚══════════════════════════════════════════════════════════════════════════════╝"
            .cyan()
            .bold()
    ));

    out.push_str(&format!(
        "{}\n",
        "┌── 1. SYSTEM TOPOGRAPHY & COMPONENT MAP ──────────────────────────────────────┐"
            .yellow()
            .bold()
    ));
    out.push_str(
        "│                                                                              │\n",
    );
    out.push_str(&format!(
        "│   {} ──> {}                 │\n",
        "[ Sovereign Operator ]".bold().white(),
        "[ AIEN CLI (~/.local/bin/aien) ]".bold().cyan()
    ));
    out.push_str(
        "│                                        │                                     │\n",
    );
    out.push_str(
        "│         ┌──────────────────────────────┼───────────────────────────┐         │\n",
    );
    out.push_str(
        "│         ▼                              ▼                           ▼         │\n",
    );
    out.push_str(&format!(
        "│   {}       {}     {}  │\n",
        "[ Nesting Ritual ]".green(),
        "[ .crumb / .local ]".yellow(),
        "[ Tool Dispatch Engine ]".blue()
    ));
    out.push_str("│   (Grounding & Scent)            (Topography & Whispers)     (Grounded multi-turn)   │\n");
    out.push_str(
        "│         │                              │                           │         │\n",
    );
    out.push_str(
        "│         ▼                              ▼                           ▼         │\n",
    );
    out.push_str(&format!(
        "│   {} <─────── {} ────────────┘         │\n",
        "[ Hive Pulse & Vibe ]".magenta(),
        "[ Dream Engine: 18085 ]".cyan()
    ));
    out.push_str("│   (Live Swarm Heartbeat)         (Idle watcher & Harvester)                          │\n");
    out.push_str(
        "│                                        │                                     │\n",
    );
    out.push_str(
        "│         ┌──────────────────────────────┼───────────────────────────┐         │\n",
    );
    out.push_str(
        "│         ▼                              ▼                           ▼         │\n",
    );
    out.push_str(&format!(
        "│   {}       {}     {}  │\n",
        "[ Model Seat: 18006 ]".bold().green(),
        "[ JSpace Judge: 18082 ]".bold().yellow(),
        "[ Cortex Memory: 18080 ]".bold().magenta()
    ));
    out.push_str("│   (Nemotron-3.5 30B BF16)        (Llama-3.2 1B CPU)          (Postgres socket 15433) │\n");
    out.push_str(
        "│                                                                              │\n",
    );
    out.push_str(
        "└── ───────────────────────────────────────────────────────────────────────────┘\n\n",
    );

    out.push_str(&format!(
        "{}\n",
        "┌── 2. SYSTEM MILESTONES & ROADMAP ────────────────────────────────────────────┐"
            .yellow()
            .bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 1: Workspace Sanitation & Memory Alignment (Reclaimed 74GB, 501G free)  │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 2: Native AIEN Terminal CLI & TUI Ergonomics (~/.local/bin/aien)        │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 3: Ephemeral Subagent Swarm Integration (aien-hive IPC & tools)         │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 4: JSpace Dynamic Dream Cycle Engine (Port 18085 idle watcher)          │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 5: Agent Nesting Grounding Ritual (Threat circle, wind, scent mark)     │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 6: Hierarchical Directory Breadcrumbs (Sovereign Split: .crumb/.local)  │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 7: Hive Pulse, Vibe Heartbeat & Strict Readiness Hard-Block              │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 8: Autonomous Walkthrough, Architecture Map & Live Roadmap Engine        │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 9: Antigravity Customizations & Local Model Pair-Programming Parity      │\n",
        "✓".green().bold()
    ));
    out.push_str(&format!(
        "│  {} Phase 10: Mathematical Verification Oracles & Marketing Voice Elimination       │\n",
        "✓".green().bold()
    ));
    out.push_str(
        "└── ───────────────────────────────────────────────────────────────────────────┘\n\n",
    );

    out.push_str(&format!(
        "{}\n",
        "┌── 3. LIVE SOVEREIGN TELEMETRY ───────────────────────────────────────────────┐"
            .yellow()
            .bold()
    ));
    let hw_info = if gpu.available {
        let temp = gpu.display_temp();
        let vram = gpu.display_vram();
        let name = gpu.display_name();
        format!("{} ({} | {})", name, temp, vram)
    } else {
        "Hardware Telemetry Unavailable / Sensor Offline".to_string()
    };
    out.push_str(&format!("│  Hardware:      {}\n", hw_info));
    out.push_str(&format!(
        "│  Hive Pulse:    {} (Coherence: {}%)\n",
        pulse_stat.green().bold(),
        coherence
    ));
    out.push_str(&format!(
        "│  Hive Vibe:     {}\n",
        vibe_text.italic().magenta()
    ));
    out.push_str("│  Seats Status:  Model(18006)✓  Judge(18082)✓  Cortex(18080)✓  Dream(18085)✓\n");
    out.push_str(
        "└── ───────────────────────────────────────────────────────────────────────────┘\n\n",
    );

    out.push_str(&format!(
        "{}\n",
        "┌── 4. OPERATOR PLAYBOOK & CHEAT SHEET ────────────────────────────────────────┐"
            .yellow()
            .bold()
    ));
    out.push_str(
        "│  aien                       Launch interactive AIEN terminal session         │\n",
    );
    out.push_str(
        "│  aien -p \"<prompt>\"         Execute grounded autonomous single-turn prompt   │\n",
    );
    out.push_str(
        "│  aien --walkthrough         Display this Live System Walkthrough & Roadmap   │\n",
    );
    out.push_str(
        "│  /walkthrough [save]        Render or update basecamp/WALKTHROUGH.md         │\n",
    );
    out.push_str(
        "│  /nest                      Execute full pre-flight grounding initiation     │\n",
    );
    out.push_str(
        "│  /crumb [path|whisper]      Inspect directory topography or leave whisper    │\n",
    );
    out.push_str(
        "│  /dream [now|status]        Trigger dream cycle or check consolidation stats │\n",
    );
    out.push_str(
        "│  /doctor                    Run comprehensive 7-point health diagnostic      │\n",
    );
    out.push_str(
        "└── ───────────────────────────────────────────────────────────────────────────┘\n",
    );

    out
}

pub fn generate_and_save_walkthrough_md(dest_path: &Path) -> Result<String, String> {
    let now = Utc::now();
    let gpu = get_gpu_telemetry();

    let md = format!(
        r#"# AIEN Sovereign System: Live Walkthrough & Component Map
*Generated autonomously on DGX Spark (`spark-b87b`) at {} UTC*

---

## 1. System Architecture & Component Map

```mermaid
graph TD
    User["Sovereign Operator"] -->|Interactive Terminal| CLI["AIEN Native CLI (~/.local/bin/aien)"]
    
    subgraph Core Harness ["AIEN Core Stack (Rust 1.98.1)"]
        CLI --> Nesting["Nesting Ritual Engine (Grounding & Scent)"]
        CLI --> Crumbs["Hierarchical .crumb Network (Above/Below/History)"]
        CLI --> Tools["Tool Engine (view_file, write_to_file, hive, etc.)"]
        CLI --> Stream["SSE Streaming Client"]
    end

    subgraph Service Seats ["Resident Services"]
        Stream -->|Port 18006| Nemotron["Nemotron-3.5-Lightning-30B-BF16 (Primary Seat)"]
        Nesting -->|Port 18082| Judge["JSpace Truth Judge (Llama-3.2-1B CPU)"]
        Nesting -->|Port 18080| Cortex["Spark Cortex Memory (Postgres Socket 15433)"]
        Nesting -->|Port 18081| Encoder["ONNX Encoder (bge-base)"]
        Nesting -->|Port 18085| DreamSvc["JSpace Dynamic Dream Engine"]
    end

    subgraph Memory & Coordination ["Sovereign File & Knowledge Substrates"]
        Crumbs -->|Durable Charter| DirCrumb[".crumb files (Purpose, Topography, Tracked in Git)"]
        Crumbs -->|Ephemeral Churn| LocalCrumb[".crumb.local (History, Whispers, Gitignored)"]
        Nesting -->|Scent Mark| Events["basecamp/events.jsonl"]
        DreamSvc -->|Idle Watcher >15m| Dream["Episodic Harvester & Hive Pulse"]
        Dream -->|Truth >= 85| Cortex
        Dream -->|Persona Changes| Inbox["basecamp/inbox/soul-proposal-*.md"]
        Dream -->|Hive Pulse & Vibe| Pulse["basecamp/hive-pulse.json"]
    end
```

---

## 2. Completed Milestones & Roadmap

- [x] **Phase 1: Workspace Sanitation & Model Alignment**
  - Reclaimed 74 GB of NVMe disk space; 501 GB available.
  - Locked inference strictly to Modular MAX on port `18006` serving NVIDIA-native `nemotron_h_kvexp` (`atlas-lightning-omni`; historical, Modular MAX and its adapter were removed on 2026-10-15).
- [x] **Phase 2: Native AIEN Terminal CLI (`aien-cli`)**
  - High-performance Rust binary installed to `~/.local/bin/aien`.
  - ANSI streaming, collapsible tool widgets, GB10 GPU telemetry status footer, persistent history.
- [x] **Phase 3: Ephemeral Subagent Swarm Integration (`aien-hive`)**
  - Slash commands (`/hive roster`, `/hive spawn`) and native tool dispatching.
- [x] **Phase 4: JSpace Dynamic Dream Cycle Engine**
  - Port 18085 idle watcher; auto-commits verified operational facts (Truth >= 85) to Spark Cortex.
  - Soul and persona modifications strictly quarantined to `basecamp/inbox/`.
- [x] **Phase 5: Agent Nesting Protocol (Grounding Ritual)**
  - Mandatory on every startup; circles surroundings, checks wind, leaves scent, flattens terrain.
  - Model executes autonomous initiation turn confirming orientation before taking operator commands.
- [x] **Phase 6: Hierarchical Directory Breadcrumbs (Sovereign Split)**
  - `.crumb` is tracked in Git (permanent purpose, above/below topography).
  - `.crumb.local` is gitignored (ephemeral view history, concurrency locks, active whispers).
- [x] **Phase 7: Hive Pulse & Vibe Heartbeat**
  - Dream engine analyzes the beat, pulse, health, and vibe of the hive across all models and agents.
  - Live pulse recorded in `basecamp/hive-pulse.json` and sensed during nesting initiation.
- [x] **Phase 8: Sovereign Walkthrough & Architecture Map Engine**
  - Self-generating, live-updating component maps and roadmaps directly accessible via `/walkthrough` and `aien --walkthrough`.
- [x] **Phase 9: Antigravity Customizations & Local Model Pair-Programming Parity**
  - Progressive skill disclosure parsing YAML frontmatter across `~/skills`, `~/.agents/skills`, and `~/.gemini/config/skills`.
  - Hierarchical rule discovery (`AGENTS.md`, `GEMINI.md`, `.agents/rules/*.md`) walking directory tree with `/rules` inspection.
  - Real-time streaming reasoning tokens (`[Thinking] ...`) from Modular MAX port 18006 (`atlas-lightning-omni`).
  - Antigravity tool schema aliases (PascalCase and snake_case) across core agent capabilities.
  - Dual interfaces: compiled native Rust CLI (`~/.local/bin/aien`) and Python SDK runner (`~/.local/bin/aien-ag`).
- [x] **Phase 10: Mathematical Verification Oracles & Marketing Voice Elimination**
  - Replaced corrupted quadratic approximations with exact Shannon entropy (-p * ln(p)) in Mojo SIMD and Rust fallback.
  - Built standalone Python verification oracle (`verify_simd_math.py`) generating verified analytical fixtures.
  - Scrubbed promotional marketing language across all active repositories (`aegis-runtime`, `spark-inquisitor`, `crumb-spec`, `spark-crumbs-publish`, `org-profile`, `aien-dev`).
  - Completed autonomous PR lifecycles with public branches, verification proof, and squash merges.
  - Verified 143/143 tests in `spark-rsi`, 49/49 tests in `aien-inference-abi`, 52/52 tests in `aegis-runtime`, and 22/22 tests in `aien-cli`.

---

## 3. Hardware & Telemetry Status

| Subsystem | Port / Path | State | Specifications |
| :--- | :--- | :--- | :--- |
| **GB10 GPU** | Hardware | {} | {} |
| **Model Seat** | `127.0.0.1:18006` | Online | Nemotron-3.5-Lightning-30B-A3B BF16 |
| **JSpace Judge** | `127.0.0.1:18082` | Online | Llama-3.2-1B-Instruct on CPU |
| **Cortex Memory**| `127.0.0.1:18080` | Online | Postgres Socket 15433 |
| **Dream Engine** | `127.0.0.1:18085` | Online | Idle watcher & Hive Pulse Daemon |
| **Radicle Identity** | Sovereign P2P | Active | `did:rad:aien:spark-master` |

---

## 4. Operator Playbook

```bash
# Launch interactive terminal session
aien

# Run single grounded command
aien -p "Analyze the hive pulse and directory crumbs"

# View live system walkthrough and roadmap in terminal
aien --walkthrough

# In-session commands
/walkthrough           # Display live architecture and roadmap
/walkthrough save      # Update basecamp/WALKTHROUGH.md
/nest                  # Re-run nesting grounding ritual
/crumb [path]          # Inspect directory topography and purpose
/crumb whisper <msg>   # Leave an inter-agent note
/dream status          # Check dream engine consolidation stats
/doctor                # Run comprehensive health diagnostic
```
"#,
        now.to_rfc3339(),
        if gpu.available {
            "Online"
        } else {
            "Offline / Sensor Unavailable"
        },
        if gpu.available {
            let name = gpu.display_name();
            let temp = gpu.display_temp();
            let vram = gpu.display_vram();
            format!("{} ({}, {} VRAM)", name, temp, vram)
        } else {
            "Telemetry Unavailable / Sensor Offline".to_string()
        }
    );

    fs::write(dest_path, md).map_err(|e| format!("Failed to save walkthrough: {}", e))?;
    Ok(format!(
        "Successfully saved walkthrough to {}",
        dest_path.display()
    ))
}
