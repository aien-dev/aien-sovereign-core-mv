# DOCTRINE-ROADMAP: The 41-Milestone Sovereign Roadmap (M0-M40)

```text
Document ID:     DOCTRINE-ROADMAP
Classification:  Sovereign Machine Canonical Doctrine (Single Roadmap Source)
Target Substrate: Complete Sovereign Stack (Atlas -> AIENOS -> AIEN / OMEGA / FORGE / AEGIS)
Status:          AUTHORITATIVE / CANONICAL
```

---

## 1. Authority of This Document

This file is the **single authoritative source** for Sovereign Machine milestone numbering, milestone code identifiers, and milestone status.

- Every other doctrine page references this file instead of restating the milestone table or milestone status.
- Milestone status is changed **only** here. A status change must cite the qualifying evidence (issue, PR, or receipt).
- `scripts/check_doctrine.sh` enforces this: it fails if another doctrine page duplicates the roadmap table, carries its own milestone status tags, or cites a milestone code identifier under a milestone number different from the one below.

> **Historical naming rule:** ADR 0014 renames the current machine-realization subsystem to **FORGE**. Existing milestone identifiers such as `PHYSICS_BOOT`, `PHYSICS_EFFECTS`, and `PHYSICS_ACCELERATOR_LINK` remain unchanged because milestone/evidence identity is historical provenance.

Status vocabulary:

| Status | Meaning |
| :--- | :--- |
| `COMPLETE` | Milestone gate passed and ratified. |
| `COMPLETE / QEMU QUALIFIED` | Gate passed under QEMU qualification; native hardware qualification remains separately gated. |
| `REOPENED / IN PROGRESS` | Previously reported complete; reopened for requalification. Not complete. |
| `PLANNED` | Not started. |

Note (2026-10-01, Lane 33): rows M20, M22 and M23 stay `PLANNED` because no exit gate is met, but precursor work has merged for each (M20 draft `aien-dev/omega#136` open; M22 substrate `#140`; M23 corpus `#122`). For these rows `PLANNED` means "no gate met, precursor work merged", not "nothing started"; see their notes in §3.

---

## 2. The Canonical Roadmap Table

| Milestone | Era | Code Identifier | Scope & Objective Invariants | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M0** | Foundational | `DOCTRINE_V1` | Foundational Specification. Ratification of complete architectural corpus. | COMPLETE |
| **M1** | Foundational | `ATLAS_BOOT` | Irreducible bootstrap seed (`atlas.bin`). Dual-seam verified. | COMPLETE / QEMU QUALIFIED |
| **M2** | Foundational | `PHYSICS_BOOT` | Physical machine authority nucleus (`physics.bin`). Current-EL VBAR, frame authority, CAP_ROOT. Requalified 2026-09-26 (`aien-dev/physics#3`, receipt `qualification_receipt.json`). QEMU only; `PHYSICS_BOOT_NATIVE_PASS` pending. | COMPLETE / QEMU QUALIFIED |
| **M3** | Foundational | `PHYSICS_EFFECTS` | Capability ledger, monotonic attenuation, effect broker, signed execution receipts (`EFFECT_INTENT` admission, `EFFECT_RECEIPT` accounting). Qualified 2026-09-26 under `aien-dev/physics#9` (commit `766f8fd6...`, receipt commit `a7dc4ef7...`). QEMU only; native hardware qualification remains separately gated. | COMPLETE / QEMU QUALIFIED |
| **M4** | Omega Core Substrate | `OMEGA_SEMANTICS` | Semantic graph ($G_S$), typed AST, invariant envelopes, substrate-independent content-addressed identity. Qualified 2026-09-26 under `aien-dev/omega` (commit `30bb116...`, receipt commit `a975573...`). | COMPLETE |
| **M5** | Omega Core Substrate | `OMEGA_AARCH64` | Native direct AArch64 machine byte realization generator (no LLVM). Qualified 2026-09-26 under `aien-dev/omega` (commit `2dd4102...`, receipt commit `34dbe93...`). | COMPLETE |
| **M6** | Omega Core Substrate | `OMEGA_SELF_HOST` | Closed-loop reproduction check of a fixed realization emitter (`OMEGA_SELF_HOST`). Qualified 2026-09-26 under `aien-dev/omega` (commit `8033c38...`, receipt `evidence/omega_self_host_qualification_receipt.json`). Correction 2026-09-30: the gates proved a fixed-output self-copy fixed point, not compilation; M6 is not evidence of a general Omega compiler (see §3 M6 note). | COMPLETE |
| **M7** | Omega Core Substrate | `OMEGA_VERIFY` | Mandatory V0-V2 verification engine; V3-V5 proof-carrying code framework. Qualified 2026-09-26 under `aien-dev/omega` (commit `202fa3c...`, receipt `evidence/omega_verify_qualification_receipt.json`). | COMPLETE |
| **M8** | Program Synthesis & Library Learning | `OMEGA_PROGRAM_CORE` | Explicit program representation, synthesis task schema (`SYNTHESIS_TASK`), program cost modeling. Qualified 2026-09-26 under `aien-dev/omega` (commit `d25349e...`, receipt `evidence/omega_program_core_qualification_receipt.json`). | COMPLETE |
| **M9** | Program Synthesis & Library Learning | `OMEGA_SYNTHESIS_V0` | Deterministic typed program synthesis over base primitives. Qualified 2026-09-26 under `aien-dev/omega` (commit `39905a3...`, receipt `evidence/omega_synthesis_v0_qualification_receipt.json`). | COMPLETE |
| **M10** | Program Synthesis & Library Learning | `OMEGA_LIBRARY_V1` | Versioned procedural program library, provenance tracking, and verified component catalog. Qualified 2026-09-26 under `aien-dev/omega` (commit `1808ed8...`, receipt commit `d3ac970...`). | COMPLETE |
| **M11** | Program Synthesis & Library Learning | `OMEGA_LIBRARY_DISCOVERY` | Autonomous abstraction discovery from program corpus with verified reuse. Qualified 2026-09-26 under `aien-dev/omega` (commit `1ee43fb...`, receipt commit `8e0a50a...`). | COMPLETE |
| **M12** | Program Synthesis & Library Learning | `OMEGA_LIVING_MATVEC` | Adaptive realization selection across varying input regimes and cache dynamics. Qualified 2026-09-26 under `aien-dev/omega` (commit `44f645f...`, receipt commit `5a22e60...`). | COMPLETE |
| **M13** | Program Synthesis & Library Learning | `OMEGA_MACHINE_GRAPH` | Formal machine hardware graph ($G_M$) describing execution pipelines and memory hierarchies, as reported by Physics. Qualified 2026-09-26 under `aien-dev/omega` (commit `9413558...`, receipt commit `db066d9...`). | COMPLETE |
| **M14** | Program Synthesis & Library Learning | `OMEGA_REALIZATION_SYNTHESIS` | Automated $G_S \times G_M \to G_R$ synthesis targeting declared hardware capabilities. Qualified 2026-09-26 under `aien-dev/omega` (commit `c0c8102...`, receipt commit `03fbcb9...`). | COMPLETE |
| **M15** | Accelerator Cognition Substrate | `PHYSICS_ACCELERATOR_LINK` | Bounded accelerator authority model grounded in observed DGX Spark device, coherent-memory, SMMUv3, IOMMU, and BAR topology. Native Blackwell submission protocol intentionally deferred to M16. Qualified 2026-09-26 under `aien-dev/physics` and `aien-dev/omega`. | COMPLETE / HARDWARE BOUNDARY QUALIFIED |
| **M16** | Accelerator Cognition Substrate | `BLACKWELL_NATIVE_PATH_KNOWN` | Empirical execution characterization of native Blackwell GB10 submission architecture (MMIO, GPFIFO queues, doorbells, completions). Correctively requalified 2026-09-26 under `aien-dev/physics` without libcuda (implementation `a2c0d7f...`, receipt `evidence/m16-blackwell-native-path-requalification-receipt.json`, canonical `b64753d...`). | COMPLETE / CORRECTIVELY REQUALIFIED |
| **M17** | Accelerator Cognition Substrate | `OMEGA_BLACKWELL_VECTOR` | Verified native Blackwell sm_121 vector compute realization bound to the $G_S$ semantic contract, using a canonical verified machine-code artifact and dynamically synthesized QMD, parameter bindings, and native submission. Qualified on physical DGX Spark GB10 silicon under `aien-dev/omega` (implementation `27971cd`, receipt `43f725e`). | COMPLETE / SILICON QUALIFIED |
| **M18** | Accelerator Cognition Substrate | `OMEGA_BLACKWELL_MATMUL` | Verified native Blackwell tensor matrix multiplication with dynamic sm_121 code generation and tensor core acceleration. Qualified on physical DGX Spark GB10 silicon under `aien-dev/omega` (implementation `9421737`, receipt `87349c0`). | COMPLETE / SILICON QUALIFIED |
| **M19** | Accelerator Cognition Substrate | `OMEGA_ACCELERATOR_RESIDENT` | Persistent Omega execution substrate residing in accelerator-accessible coherent memory. Implementation `beaa1185154051f147a5588e9fa78410a879b9a1`, evidence `5a850796c6e0e843178ade5374034066539d4697`, merge `ae2476d2c16beedff6507cbfc36701bb740203de`, authority substrate `aien-dev/physics` (`b64753d95bacb1114ba48decde48239f0c542e12`). Reopened per errata E-M19-1..10; requalified and closed 2026-09-30 via M19R Gate 14 Combined Foundation Admission (`evidence/GATE14-FOUNDATION/50dd611bfd28e8320443810b35f7425df5eb112777662e5840c227154271f872.json`, `aien-dev/omega#111`) on clean Omega `8024e9a` + Physics `e95e3ed`. | COMPLETE / CORRECTIVELY REQUALIFIED |
| **M20** | Sovereign Training Runtime | `OMEGA_TENSOR` | Tensor semantics, multi-dimensional array types, strides, and memory layouts. | PLANNED |
| **M21** | Sovereign Training Runtime | `OMEGA_AUTODIFF` | Sovereign automatic differentiation generating gradient semantic graphs. | PLANNED |
| **M22** | Sovereign Training Runtime | `OMEGA_OPTIMIZER` | Verified sovereign optimizer realizations (SGD, Adam, AdamW). | PLANNED |
| **M23** | Sovereign Training Runtime | `OMEGA_SEARCH_GUIDE_TRAINING` | Sovereign training pipeline for search guides using Omega-generated traces. | PLANNED |
| **M24** | Sovereign Training Runtime | `AIEN_0` | First sovereign neural search guide ranking subgoals and pruning search branches. | PLANNED |
| **M25** | Sovereign Training Runtime | `AIEN_GUIDED_SYNTHESIS` | Neural-guided synthesis achieving lower search cost under invariant verification parity. | PLANNED |
| **M26** | Sovereign Training Runtime | `AIEN_ABSTRACTION_DISCOVERY` | AIEN proposes candidate abstractions; Omega enforces survival based on utility. | PLANNED |
| **M27** | Physics Zero Discovery | `PHYSICS_ZERO_PROTOCOL` | Contamination firewall specification and sealed evaluation benchmark contracts. | PLANNED |
| **M28** | Physics Zero Discovery | `PHYSICS_ZERO_HIDDEN_WORLDS` | Discrete and continuous sealed benchmark worlds with randomized channels (P0-0..P0-2). | PLANNED |
| **M29** | Physics Zero Discovery | `AIEN_THEORY_DISCOVERY` | Executable `OMEGA_THEORY` synthesis, population competition, and calibrated prediction. | PLANNED |
| **M30** | Physics Zero Discovery | `AIEN_ACTIVE_EXPERIMENTATION` | Active experiment design distinguishing competing theories ($P(R \mid A) \neq P(R \mid B)$). | PLANNED |
| **M31** | Physics Zero Discovery | `AIEN_CONCEPT_FORMATION` | Autonomous invention of reusable latent concepts (`OMEGA_DISCOVERED_CONCEPT`). | PLANNED |
| **M32** | Physics Zero Discovery | `PHYSICS_ZERO_ALIEN_WORLDS` | Scientific discovery in universes governed by unfamiliar laws (P0-5). | PLANNED |
| **M33** | Physics Zero Discovery | `PHYSICS_ZERO_NOVEL_REGIME` | Extrapolative theory prediction in unobserved state regimes. | PLANNED |
| **M34** | Physics Zero Discovery | `PHYSICS_ZERO_NOVEL_PHENOMENON` | Predictive discovery and experimental realization of unobserved phenomena. | PLANNED |
| **M35** | Physics Zero Discovery | `PHYSICS_ZERO_REAL_LAB` | Capability-bounded, safety-governed autonomous discovery in physical laboratories. | PLANNED |
| **M36** | General AIEN | `AIEN_HUMAN_INTERFACE` | Bi-directional perceptual adapter mapping human natural language to Omega semantics. | PLANNED |
These are the implementation stages of ADR 0020, not roadmap milestones; they carry no M-number and do not change the table in §2. EST-4 and later wait for R16 closure.

Status 2026-10-01 (R16 wording corrected 2026-10-05: R16 is a historical PASS, not requalified on CAND-0, see the R16 row): R16 was closed (`aien-dev/omega#112` `3dd5eaa`), so EST-4 and later no longer wait on R16; they wait on a passing EST-3 calibration. Three frozen calibration attempts are recorded as FAIL and kept: v1 `aien-dev/omega#105` `6d1ff1d`, v2 `#118` `8a56ace`, v3 `#129` `d78fd11` (Phase A FAIL, sealed run NOT_RUN, omega `docs/estimation/receipts/est3c-v3/RESULT.md`). v4: INCONCLUSIVE (omega#153 merged 2026-10-01); v5 in progress (omega#170, open). All three attempts VOID, no verdict.

| Stage | Exit token | Status |
| :--- | :--- | :--- |
| EST-0 semantic contract | `ESTIMATION_SEMANTIC_CONTRACT` | IN PROGRESS (standalone module; code merged `aien-dev/omega#105` `6d1ff1d`) |
| EST-1 linear Kalman reference | `LINEAR_KALMAN_REFERENCE` | IN PROGRESS (standalone module; code merged `aien-dev/omega#105` `6d1ff1d`) |
| EST-2 real signal | `ESTIMATION_REAL_SIGNAL` | IN PROGRESS (signal tools merged `aien-dev/omega#105` `6d1ff1d`; exit not claimed) |
| EST-3 calibration | `ESTIMATION_CALIBRATION` | FAIL recorded (v1 `omega#105`, v2 `omega#118`, v3 `omega#129`); v4: INCONCLUSIVE (omega#153 merged 2026-10-01); v5 in progress (omega#170, open) |
| EST-4 World belief state | `WORLD_BELIEF_STATE` | PLANNED (blocked on a passing EST-3) |
| EST-5 Omega cost model | `OMEGA_ESTIMATION_INTEGRATION` | PLANNED (blocked on a passing EST-3) |
| EST-6 TURING evidence | `TURING_ESTIMATION_EVIDENCE` | PLANNED |
| EST-7 J-Space | `JSPACE_BELIEF_INTEGRATION` | PLANNED |
| EST-8 Cortex | `CORTEX_PREDICTIVE_MEMORY` | PLANNED |
| EST-9 innovation curiosity signal | `INNOVATION_CURIOSITY_SIGNAL` | PLANNED |
| EST-10 active information gain | `ACTIVE_INFORMATION_GAIN` | PLANNED |

### M20: `OMEGA_TENSOR`
- Status 2026-10-01: table status stays PLANNED; no exit gate is met. A semantic layer + CPU realization is OPEN as `aien-dev/omega#136` (not qualified). Its prerequisite E1 numerical closure is not closed: scalar contract CPU tier `aien-dev/omega#124` `d3194f6` and reductions `#134` `c54d492` merged; transcendental sequences `#127` (`7a5a13c`, CPU only) and standalone GB10 DIV/SQRT `#141` (`6b14354`) are merged (corrected 2026-10-01, Lane 33; they were already merged or merging when this note was written), as are GB10 main-path WP-C `#147` (`2d8cd68`) and DIV/SQRT `#152` (`07004a8`). E1 is still PARTIAL, 2 of 6 exit requirements met; `#136` remains an open draft (see `CURRENT_EXECUTION_PLAN.md` §2 Lane 33 addendum).
- Status 2026-10-02: the E1 prerequisite is CLOSED on the GB10 chip campaign at omega `fb36109` (merged `40d1ea37`, `aien-dev/omega#225`, Physics `e95e3ed`): consolidated receipt omega `evidence/E1-CLOSURE/e7851c69d34ac777a9436af16d0bdd5d69264c8528c62d562b600f5d89153061.json`, six of six E1 bullets met, exclusions recorded in omega `docs/numeric/E1_GAP_TABLE.md` (natural-base EXP/LOG not on GB10). M20 `OMEGA_TENSOR` itself is unchanged: table status PLANNED, `#136` an open draft, no exit gate met.

### M22 — `OMEGA_OPTIMIZER`
Omega-native semantics for SGD, Adam, and AdamW, with verified CPU reference realizations and optional accelerator-fused realizations.
- Status 2026-10-01: table status stays PLANNED; no exit gate is met. Discrepancy recorded: substrate work has started although the vocabulary row reads "Not started". `aien-dev/omega#140` (`54826a3`) merged a shadow-state optimizer substrate with one atomic generation switch, E5 two-tier provenance and a float32 SGD path; receipt omega `evidence/M22/receipts/eaa0cdeae9f348b19b80377baaeb5e5dea6a6577e9385e7a2a3c15caab24c5c8.json` records substrate tests PASS and "M22 NOT QUALIFIED: optimizer substrate + SGD path only; no tensor/autodiff integration (waits M20/M21)". Adam/AdamW are not implemented.
- Scope note 2026-10-03: M22 means training optimizers that update model parameters (SGD, Adam, AdamW). Adaptive pricing of soft machine resource budgets (DUAL, ADR 0031, `CURRENT_EXECUTION_PLAN.md` Lane 8) is a separate workstream with its own gates; it is not part of M22, and no M22 status changes because of it.

### M23: `OMEGA_SEARCH_GUIDE_TRAINING`
- Status 2026-10-01: table status stays PLANNED; no exit gate is met (training NOT_RUN, waits on M22). Discrepancy recorded: the corpus prerequisite exists. G1 search-trace corpus capture PASS (`aien-dev/omega#122` `ce7821d`; receipt omega `evidence/M23/receipts/m23-corpus-ee4982119af7244086187c9b643cb9c1823c6b0a2b796f861d2d4077edb5256d.json`); G3 sealed-holdout commitment format PASS, format only, nothing sealed (`#122`), commitment owner signature with TEST keys (`#130` `7fb59d3`). Sealing and real owner signing are BLOCKED_OPERATOR.

### DIRAC-0 (ARCH-0023, PROPOSED; downstream consumer, not a milestone)
- Status 2026-10-01: no table row. DIRAC-0 (`docs/plans/dirac/DIRAC-0-SPEC.md`) consumes OSC-2, E1, M20 `OMEGA_TENSOR`, ESTIMATION and TURING and creates no new tensor, measurement, GPU-runtime or evidence system. `DIRAC_PREP_FROZEN` is proposed, not passed; no roadmap milestone changes.

### M40 — `AIEN_SUCCESSION`
Closed-loop self-improvement: AIEN-N designs AIEN-N+1 under Physics canary control.

---

**Addendum 2026-10-01 (language course correction):** [ADR 0024](../docs/adr/0024-rust-scaffolding-omega-destination.md) supersedes the Rust-to-C migration plan. Rust is scaffolding, Omega is the destination, and C stays only where hardware-justified. No milestone, gate or order in this roadmap changes.

<!-- HISTORICAL-PROVENANCE:BEGIN -->
## 4. Historical Provenance: Superseded Roadmap Generations

> **HISTORICAL — NOT NORMATIVE.** Retained only to explain older references. Do not cite these numbers.

- **28-milestone workstream roadmap (M0-M27, workstreams A-F):** ended at M27 `SOVEREIGN_MACHINE_CLOSURE`; its identifiers (e.g. `ATLAS_BAREMETAL_BOOT`, `PHYSICS_CORE_MEMBRANE`, `SENTINEL_GATE_1..3`) are retired.
- **31-milestone program-learning roadmap (M0-M30):** placed `AIEN_HUMAN_INTERFACE` at M27, `AIEN_RESIDENT` at M28, `OMEGA_CONTINUAL_LIBRARY_LEARNING` at M29, and `AIEN_SUCCESSION` at M30. Superseded by the 41-milestone roadmap above when Physics Zero (DOCTRINE-006) was ratified.
- **Alpha lineage:** `atlas.bin` was designated Alpha (`alpha.bin`) during early bootstrap drafting; the Alpha evidence bundle names (`alpha.manifest`, `alpha.sha256`, `alpha.decode`, `alpha.memory-map`, `alpha.control-flow`, `alpha.audit`) are retired in favour of `atlas.*`.
<!-- HISTORICAL-PROVENANCE:END -->
