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

//! Formal oracle bindings: tie a formal proof job to the exact evidence it ran on.
//!
//! A formal job (a theorem prover or model checker run) is one more
//! qualification procedure. Its result becomes an ordinary `EvidenceReceiptV1`
//! of kind [`KIND`]. No new receipt fields and no new canonical encoding exist:
//! each bound item is a stable assertion ID, the four input digests sit in
//! `input_artifacts`, the oracle toolchain sits in `toolchain`, and the output
//! digest and dependencies use the native fields. Names are tool neutral
//! (`oracle_toolchain`, `proof_source_digest`) so a later oracle can replace
//! the bootstrap one without a format change (ADR 0027 point 5).
//!
//! A PASS here is one evidence input bound to one source and model revision.
//! It is not authority (ADR 0027 point 6); `authority` stays empty.
//!
//! Invalidation rests on [`crate::fingerprint`]: a job whose inputs include
//! the invariant manifest, the formal model and proof sources, and the covered
//! implementation file gets a new job key when any of them changes, so an old
//! stamp is never replayed for new evidence.

use std::path::{Path, PathBuf};

use crate::evidence::{Assertion, Receipt, Verdict};
use crate::fingerprint::inputs_digest;
use crate::import::{build_receipt, ImportRequest};

/// Receipt kind for formal oracle bindings.
pub const KIND: &str = "formal-oracle-binding";

pub const A_INVARIANT_ID: &str = "oracle_binding.invariant_id";
pub const A_INVARIANT_VERSION: &str = "oracle_binding.invariant_version";
pub const A_MODEL_DIGEST: &str = "oracle_binding.model_digest";
pub const A_ORACLE_TOOLCHAIN: &str = "oracle_binding.oracle_toolchain";
pub const A_PROOF_SOURCE_DIGEST: &str = "oracle_binding.proof_source_digest";
pub const A_COVERED_IMPL_DIGEST: &str = "oracle_binding.covered_impl_digest";
pub const A_MANIFEST_DIGEST: &str = "oracle_binding.manifest_digest";
/// Prefix of one assertion per theorem checked: `oracle_binding.theorem:<name>`.
pub const A_THEOREM_PREFIX: &str = "oracle_binding.theorem:";

const SOURCE: &str = "aien-proof formal bind";

/// Everything a formal job binds, as plain data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalBinding {
    pub invariant_id: String,
    pub invariant_version: u32,
    /// Digest of the formal model files.
    pub model_digest: String,
    /// Oracle toolchain identity (version, commit, platform), free text.
    pub oracle_toolchain: String,
    /// Digest of all proof sources the oracle compiled (model, theorems, corpus, mutants).
    pub proof_source_digest: String,
    /// Digest of the implementation file(s) the model claims to cover.
    pub covered_impl_digest: String,
    /// Digest of the invariant manifest.
    pub manifest_digest: String,
    pub theorems_checked: Vec<String>,
}

/// Paths (relative to `base`) that make up each slice of a formal job.
#[derive(Debug, Clone)]
pub struct FormalInputs {
    pub base: PathBuf,
    pub manifest: PathBuf,
    pub model: Vec<PathBuf>,
    pub proof_source: Vec<PathBuf>,
    pub covered: Vec<PathBuf>,
}

impl FormalInputs {
    /// One digest over every input of the job. The job writes it into its
    /// captured output so a receipt can refuse output from a different input set.
    pub fn run_digest(&self) -> Result<String, String> {
        let mut all = vec![self.manifest.clone()];
        all.extend(self.model.iter().cloned());
        all.extend(self.proof_source.iter().cloned());
        all.extend(self.covered.iter().cloned());
        inputs_digest(&self.base, &all).map_err(|e| format!("cannot digest inputs: {e}"))
    }

    fn slice(&self, paths: &[PathBuf]) -> Result<String, String> {
        if paths.is_empty() {
            return Err("formal binding needs at least one path per input slice".to_string());
        }
        inputs_digest(&self.base, paths).map_err(|e| format!("cannot digest inputs: {e}"))
    }
}

/// Read `id` and `version` from the top level of an invariant manifest.
/// Only the leading key/value block (before the first table header) is read.
pub fn manifest_identity(path: &Path) -> Result<(String, u32), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read manifest: {e}"))?;
    let (mut id, mut version) = (None, None);
    for line in text.lines() {
        if line.starts_with('[') {
            break;
        }
        if let Some(rest) = line.strip_prefix("id = ") {
            id = Some(rest.trim().trim_matches('"').to_string());
        } else if let Some(rest) = line.strip_prefix("version = ") {
            version = rest.trim().parse::<u32>().ok();
        }
    }
    match (id, version) {
        (Some(i), Some(v)) if !i.is_empty() => Ok((i, v)),
        _ => Err("manifest has no top level id and numeric version".to_string()),
    }
}

/// Result of the job taken from the oracle's own final line, `PREFIX: PASS|FAIL|NOT_RUN`.
/// Anything else is refused; there is no default.
pub fn parse_result_line(output: &str, prefix: &str) -> Result<Verdict, String> {
    let last = output
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| "output is empty".to_string())?;
    match last.strip_prefix(prefix).and_then(|r| r.strip_prefix(": ")) {
        Some("PASS") => Ok(Verdict::Pass),
        Some("FAIL") => Ok(Verdict::Fail),
        Some("NOT_RUN") => Ok(Verdict::Skipped),
        _ => Err(format!(
            "last output line is not a {prefix} result line: {last:?}"
        )),
    }
}

/// The job output must carry the digest of the exact inputs it ran on.
pub fn check_output_binds_inputs(output: &str, run_digest: &str) -> Result<(), String> {
    let want = format!("inputs_digest: {run_digest}");
    if output.lines().any(|l| l == want) {
        Ok(())
    } else {
        Err("output does not carry this input set's digest (stale or foreign output)".to_string())
    }
}

/// Compute the binding from files on disk.
pub fn compute_binding(
    inputs: &FormalInputs,
    oracle_toolchain: &str,
    theorems: &[String],
) -> Result<FormalBinding, String> {
    let (invariant_id, invariant_version) = manifest_identity(&inputs.base.join(&inputs.manifest))?;
    Ok(FormalBinding {
        invariant_id,
        invariant_version,
        model_digest: inputs.slice(&inputs.model)?,
        oracle_toolchain: oracle_toolchain.to_string(),
        proof_source_digest: inputs.slice(&inputs.proof_source)?,
        covered_impl_digest: inputs.slice(&inputs.covered)?,
        manifest_digest: inputs.slice(std::slice::from_ref(&inputs.manifest))?,
        theorems_checked: theorems.to_vec(),
    })
}

fn fact(id: &str, value: &str) -> Assertion {
    Assertion {
        id: id.to_string(),
        expected: String::new(),
        observed: value.to_string(),
        pass: true,
        source: SOURCE.to_string(),
        note: String::new(),
    }
}

/// Assertions carrying the binding. Theorem assertions pass only when the
/// job's result is PASS.
pub fn binding_assertions(b: &FormalBinding, result: Verdict) -> Vec<Assertion> {
    let mut out = vec![
        fact(A_INVARIANT_ID, &b.invariant_id),
        fact(A_INVARIANT_VERSION, &b.invariant_version.to_string()),
        fact(A_MODEL_DIGEST, &b.model_digest),
        fact(A_ORACLE_TOOLCHAIN, &b.oracle_toolchain),
        fact(A_PROOF_SOURCE_DIGEST, &b.proof_source_digest),
        fact(A_COVERED_IMPL_DIGEST, &b.covered_impl_digest),
        fact(A_MANIFEST_DIGEST, &b.manifest_digest),
    ];
    for t in &b.theorems_checked {
        let mut a = fact(&format!("{A_THEOREM_PREFIX}{t}"), "checked");
        a.pass = result == Verdict::Pass;
        out.push(a);
    }
    out
}

fn get<'a>(r: &'a Receipt, id: &str) -> Result<&'a str, String> {
    r.assertions
        .iter()
        .find(|a| a.id == id)
        .map(|a| a.observed.as_str())
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| format!("formal receipt lacks assertion {id}"))
}

/// Read the binding back out of a receipt, checking it is self consistent:
/// the four digests in `input_artifacts` equal the digest assertions, the
/// toolchain field equals the oracle toolchain, and at least one theorem is listed.
pub fn parse_binding(r: &Receipt) -> Result<FormalBinding, String> {
    if r.kind != KIND {
        return Err(format!("not a formal oracle binding: kind {:?}", r.kind));
    }
    let b = FormalBinding {
        invariant_id: get(r, A_INVARIANT_ID)?.to_string(),
        invariant_version: get(r, A_INVARIANT_VERSION)?
            .parse()
            .map_err(|_| "invariant_version is not a number".to_string())?,
        model_digest: get(r, A_MODEL_DIGEST)?.to_string(),
        oracle_toolchain: get(r, A_ORACLE_TOOLCHAIN)?.to_string(),
        proof_source_digest: get(r, A_PROOF_SOURCE_DIGEST)?.to_string(),
        covered_impl_digest: get(r, A_COVERED_IMPL_DIGEST)?.to_string(),
        manifest_digest: get(r, A_MANIFEST_DIGEST)?.to_string(),
        theorems_checked: r
            .assertions
            .iter()
            .filter_map(|a| a.id.strip_prefix(A_THEOREM_PREFIX))
            .map(str::to_string)
            .collect(),
    };
    if b.theorems_checked.is_empty() {
        return Err("formal receipt lists no theorems".to_string());
    }
    let want = artifact_digests(&b);
    if r.input_artifacts != want {
        return Err("input_artifacts do not match the bound digests".to_string());
    }
    if r.toolchain != b.oracle_toolchain {
        return Err("toolchain field differs from oracle_toolchain".to_string());
    }
    Ok(b)
}

fn artifact_digests(b: &FormalBinding) -> Vec<String> {
    [
        &b.manifest_digest,
        &b.model_digest,
        &b.proof_source_digest,
        &b.covered_impl_digest,
    ]
    .iter()
    .map(|d| format!("blake3:{d}"))
    .collect()
}

/// Profile checks for kind [`KIND`], called from `build_receipt`.
pub fn validate_formal_profile(
    req: &ImportRequest,
    assertions: &[Assertion],
    result: Verdict,
) -> Result<(), String> {
    if !req.authority.trim().is_empty() {
        return Err("a formal binding is evidence, not authority: authority must be empty".into());
    }
    if req.toolchain.trim().is_empty() {
        return Err("formal binding requires an oracle toolchain identity".to_string());
    }
    if req.tier != "HOST_TEST" {
        return Err(format!("formal binding refuses tier {:?}", req.tier));
    }
    for id in [
        A_INVARIANT_ID,
        A_INVARIANT_VERSION,
        A_MODEL_DIGEST,
        A_ORACLE_TOOLCHAIN,
        A_PROOF_SOURCE_DIGEST,
        A_COVERED_IMPL_DIGEST,
        A_MANIFEST_DIGEST,
    ] {
        let ok = assertions
            .iter()
            .any(|a| a.id == id && !a.observed.trim().is_empty());
        if !ok {
            return Err(format!("formal binding missing assertion {id}"));
        }
    }
    if !assertions
        .iter()
        .any(|a| a.id.starts_with(A_THEOREM_PREFIX))
    {
        return Err("formal binding lists no theorems".to_string());
    }
    if result == Verdict::Pass && assertions.iter().any(|a| !a.pass) {
        return Err("formal binding claims PASS but an assertion failed".to_string());
    }
    if req.input_artifacts.len() != 4 {
        return Err("formal binding needs the four input digests".to_string());
    }
    Ok(())
}

/// Facts a caller supplies beyond the binding itself.
#[derive(Debug, Clone)]
pub struct FormalRun {
    /// Repository and commit of the covered implementation.
    pub repo: String,
    pub commit: String,
    pub dirty: bool,
    pub machine: String,
    pub procedure: String,
    pub result: Verdict,
    pub output_digest: String,
    pub dependencies: Vec<String>,
    pub external_refs: Vec<String>,
    pub timestamp: Option<u64>,
}

/// Mint the receipt for a formal job.
pub fn bind_receipt(b: &FormalBinding, run: &FormalRun) -> Result<Receipt, String> {
    let req = ImportRequest {
        repo: run.repo.clone(),
        commit: run.commit.clone(),
        dirty: run.dirty,
        kind: KIND.to_string(),
        tier: "HOST_TEST".to_string(),
        toolchain: b.oracle_toolchain.clone(),
        procedure: run.procedure.clone(),
        machine: run.machine.clone(),
        input_artifacts: artifact_digests(b),
        dependencies: run.dependencies.clone(),
        output_digest: Some(run.output_digest.clone()),
        external_refs: run.external_refs.clone(),
        timestamp: run.timestamp,
        ..ImportRequest::default()
    };
    build_receipt(&req, binding_assertions(b, run.result), run.result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::verify_bytes;
    use crate::fingerprint::job_key;
    use crate::testutil::temp_dir;
    use std::fs;

    const MANIFEST: &str =
        "schema_version = 0\nid = \"AIEN.INV.TEST.V1\"\nversion = 3\n\n[[x]]\nid = \"nope\"\n";

    fn tree(root: &Path) -> FormalInputs {
        fs::create_dir_all(root.join("inv")).unwrap();
        fs::create_dir_all(root.join("lean/Model")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("inv/a.toml"), MANIFEST).unwrap();
        fs::write(root.join("lean/Model/Model.lean"), "def m := 1").unwrap();
        fs::write(
            root.join("lean/Theorems.lean"),
            "theorem t : True := trivial",
        )
        .unwrap();
        fs::write(root.join("src/cap.rs"), "fn derive() {}").unwrap();
        FormalInputs {
            base: root.to_path_buf(),
            manifest: "inv/a.toml".into(),
            model: vec!["lean/Model".into()],
            proof_source: vec!["lean".into()],
            covered: vec!["src/cap.rs".into()],
        }
    }

    fn all_inputs(i: &FormalInputs) -> Vec<PathBuf> {
        let mut v = vec![i.manifest.clone()];
        v.extend(i.model.clone());
        v.extend(i.proof_source.clone());
        v.extend(i.covered.clone());
        v
    }

    fn key(i: &FormalInputs) -> String {
        job_key(
            &i.base,
            "formal-test-v1",
            &["sh".into(), "run.sh".into()],
            "lean 4",
            &all_inputs(i),
        )
        .unwrap()
    }

    fn run() -> FormalRun {
        FormalRun {
            repo: "https://github.com/aien-dev/aienos".into(),
            commit: "a".repeat(40),
            dirty: false,
            machine: "spark".into(),
            procedure: "formal-test-v1".into(),
            result: Verdict::Pass,
            output_digest: "c".repeat(64),
            dependencies: vec![],
            external_refs: vec![],
            timestamp: Some(1),
        }
    }

    fn binding(i: &FormalInputs) -> FormalBinding {
        compute_binding(i, "lean 4.34.1", &["t1".into(), "t2".into()]).unwrap()
    }

    #[test]
    fn manifest_identity_reads_top_level_only() {
        let root = temp_dir("formal-id");
        let i = tree(&root);
        assert_eq!(
            manifest_identity(&root.join(&i.manifest)).unwrap(),
            ("AIEN.INV.TEST.V1".to_string(), 3)
        );
    }

    #[test]
    fn receipt_round_trips_and_verifies() {
        let root = temp_dir("formal-rt");
        let i = tree(&root);
        let b = binding(&i);
        let r = bind_receipt(&b, &run()).unwrap();
        let text = serde_json::to_vec(&r).unwrap();
        assert!(verify_bytes(&text).unwrap().1.ok());
        assert_eq!(parse_binding(&r).unwrap(), b);
        assert_eq!(r.output_digest, "c".repeat(64));
        assert_eq!(r.authority, "");
    }

    #[test]
    fn changed_implementation_model_proof_or_manifest_changes_key_digest_and_receipt() {
        for (rel, what) in [
            ("src/cap.rs", "implementation"),
            ("lean/Model/Model.lean", "model"),
            ("lean/Theorems.lean", "proof source"),
            ("inv/a.toml", "manifest"),
        ] {
            let root = temp_dir("formal-inv");
            let i = tree(&root);
            let (k0, d0, b0) = (key(&i), i.run_digest().unwrap(), binding(&i));
            let id0 = bind_receipt(&b0, &run()).unwrap().id;
            let old = fs::read_to_string(root.join(rel)).unwrap();
            fs::write(root.join(rel), format!("{old}\n-- changed")).unwrap();
            assert_ne!(k0, key(&i), "{what}: job key must change");
            assert_ne!(
                d0,
                i.run_digest().unwrap(),
                "{what}: run digest must change"
            );
            let b1 = binding(&i);
            assert_ne!(b0, b1, "{what}: binding must change");
            assert_ne!(
                id0,
                bind_receipt(&b1, &run()).unwrap().id,
                "{what}: receipt id"
            );
        }
    }

    #[test]
    fn each_slice_digest_moves_only_with_its_own_files() {
        let root = temp_dir("formal-slice");
        let i = tree(&root);
        let b0 = binding(&i);
        fs::write(root.join("src/cap.rs"), "fn derive() { /* x */ }").unwrap();
        let b1 = binding(&i);
        assert_ne!(b0.covered_impl_digest, b1.covered_impl_digest);
        assert_eq!(b0.model_digest, b1.model_digest);
        assert_eq!(b0.proof_source_digest, b1.proof_source_digest);
        assert_eq!(b0.manifest_digest, b1.manifest_digest);
    }

    #[test]
    fn lake_build_output_does_not_change_the_key() {
        let root = temp_dir("formal-lake");
        let i = tree(&root);
        let k0 = key(&i);
        fs::create_dir_all(root.join("lean/.lake/build")).unwrap();
        fs::write(root.join("lean/.lake/build/x.olean"), "bin").unwrap();
        assert_eq!(k0, key(&i));
    }

    #[test]
    fn result_line_must_match_the_oracle_prefix() {
        assert_eq!(
            parse_result_line("a\nCAP_FORMAL_BOOTSTRAP: PASS\n", "CAP_FORMAL_BOOTSTRAP"),
            Ok(Verdict::Pass)
        );
        assert_eq!(
            parse_result_line("CAP_FORMAL_BOOTSTRAP: FAIL", "CAP_FORMAL_BOOTSTRAP"),
            Ok(Verdict::Fail)
        );
        assert_eq!(
            parse_result_line("CAP_FORMAL_BOOTSTRAP: NOT_RUN", "CAP_FORMAL_BOOTSTRAP"),
            Ok(Verdict::Skipped)
        );
        assert!(
            parse_result_line("RECEIPT_FORMAL_BOOTSTRAP: PASS", "CAP_FORMAL_BOOTSTRAP").is_err()
        );
        assert!(
            parse_result_line("CAP_FORMAL_BOOTSTRAP: PASS\nextra", "CAP_FORMAL_BOOTSTRAP").is_err()
        );
        assert!(parse_result_line("", "CAP_FORMAL_BOOTSTRAP").is_err());
    }

    #[test]
    fn output_from_another_input_set_is_refused() {
        let root = temp_dir("formal-stale");
        let i = tree(&root);
        let d = i.run_digest().unwrap();
        let out = format!("inputs_digest: {d}\nCAP_FORMAL_BOOTSTRAP: PASS\n");
        assert!(check_output_binds_inputs(&out, &d).is_ok());
        fs::write(root.join("src/cap.rs"), "fn derive() { evil() }").unwrap();
        assert!(check_output_binds_inputs(&out, &i.run_digest().unwrap()).is_err());
    }

    #[test]
    fn failing_result_is_never_sealed_and_theorems_are_not_passing() {
        let root = temp_dir("formal-fail");
        let b = binding(&tree(&root));
        let mut r = run();
        r.result = Verdict::Fail;
        assert!(
            bind_receipt(&b, &r).is_err(),
            "the receipt layer refuses to seal FAIL"
        );
        assert!(binding_assertions(&b, Verdict::Fail)
            .iter()
            .filter(|a| a.id.starts_with(A_THEOREM_PREFIX))
            .all(|a| !a.pass));
    }

    #[test]
    fn profile_refuses_incomplete_or_authoritative_bindings() {
        let root = temp_dir("formal-refuse");
        let b = binding(&tree(&root));
        let mut no_theorems = b.clone();
        no_theorems.theorems_checked.clear();
        assert!(bind_receipt(&no_theorems, &run()).is_err());
        let mut no_tool = b.clone();
        no_tool.oracle_toolchain = " ".into();
        assert!(bind_receipt(&no_tool, &run()).is_err());
        let mut r = run();
        r.output_digest = "zz".into();
        assert!(bind_receipt(&b, &r).is_err());
    }

    #[test]
    fn tampered_artifacts_fail_binding_parse() {
        let root = temp_dir("formal-tamper");
        let b = binding(&tree(&root));
        let mut r = bind_receipt(&b, &run()).unwrap();
        r.input_artifacts[3] = format!("blake3:{}", "e".repeat(64));
        assert!(parse_binding(&r).is_err());
    }
}
