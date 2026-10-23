#!/usr/bin/env python3
"""Exercise real local CLI output; never post comments or contact GitHub."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

LEGACY = re.compile(r"SparkOS|\boath\b|CONSTITUTION\.md", re.IGNORECASE)
SOURCE = "3ccf32ed42c645d33d4c44fe08eca5d21a1858b9"

def check_text(text):
    hits = LEGACY.findall(text)
    if hits:
        print("REFUSE retired project terms:", hits)
        return 1
    print("PASS current project terms")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", default=SOURCE)
    args = parser.parse_args()
    source = args.source
    if re.fullmatch(r"[0-9a-f]{40}", source) is None:
        parser.error("--source must be an exact full lowercase commit SHA")
    if args.check_output:
        return check_text(args.check_output.read_text())
    if args.output is None:
        parser.error("--output is required; it must not already exist")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    # No source rewriting, old binary substitution or metadata-only success.
    inputs = [*sorted((root / "crates/spark-inquisitor").rglob("*.rs")),
              root / "crates/spark-inquisitor/Cargo.toml", root / "Cargo.toml",
              root / "Cargo.lock", root / "rust-toolchain.toml"]
    for f in inputs:
        relative = f.relative_to(root).as_posix()
        pinned = subprocess.check_output(["git", "show", source + ":" + relative], cwd=root)
        if f.read_bytes() != pinned:
            raise RuntimeError("source differs from qualified PR head: " + relative)
    binary = root / "target/debug/spark-inquisitor"
    manifest = {"source": source, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "synthetic_fixtures": True, "real_native_cli": True, "complete": False, "cases": []}
    env = dict(os.environ, NO_COLOR="1", CLICOLOR="0")
    with tempfile.TemporaryDirectory(prefix="inquisitor477-", dir=out) as temp:
        p = Path(temp)
        fixtures = {
            "clean.diff": "diff --git a/lib.rs b/lib.rs\n--- a/lib.rs\n+++ b/lib.rs\n@@ -0,0 +1 @@\n+pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
            "telemetry.diff": "diff --git a/web.js b/web.js\n--- a/web.js\n+++ b/web.js\n@@ -0,0 +1 @@\n+fetch(\"https://www.google-analytics.com/collect\");\n",
            "issue.txt": "Reproducible parser panic on an empty response. Steps: pass an empty response to the parser. Expected: a refusal, not a panic.\n",
            "valid.txt": "I solemnly certify and affirm the Sovereign Contributor Oath. This PR introduces zero telemetry and zero tracking. It is built to advance open software and human freedom.\n",
            "empty.txt": "",
        }
        for name, data in fixtures.items():
            (p / name).write_text(data)
        cases = [
            ("help", ["--help"], 0),
            ("doctor", ["doctor"], 0),
            ("interview", ["interview", "--author", "synthetic-fixture", "--pr", "1", "--title", "Synthetic fixture"], 0),
            ("review-clean", ["review", "--author", "synthetic-fixture", "--pr", "1", "--diff", str(p/"clean.diff")], 0),
            ("review-telemetry", ["review", "--author", "synthetic-fixture", "--pr", "1", "--diff", str(p/"telemetry.diff")], 1),
            ("audit-clean", ["audit", "--diff", str(p/"clean.diff")], 0),
            ("audit-telemetry", ["audit", "--diff", str(p/"telemetry.diff")], 1),
            ("triage", ["triage-issue", "--author", "synthetic-fixture", "--issue", "1", "--body", str(p/"issue.txt")], 0),
            ("evaluate-legacy-compatible", ["evaluate", "--author", "synthetic-fixture", "--testimony", str(p/"valid.txt")], 0),
            ("evaluate-empty-refused", ["evaluate", "--author", "synthetic-fixture", "--testimony", str(p/"empty.txt")], 1),
        ]
        for name, argv, expected in cases:
            r = subprocess.run([str(binary), *argv], env=env, capture_output=True, timeout=30)
            (out/(name+".stdout")).write_bytes(r.stdout)
            (out/(name+".stderr")).write_bytes(r.stderr)
            text = (r.stdout+r.stderr).decode()
            hits = LEGACY.findall(text)
            row = {"name": name, "argv": argv, "exit": r.returncode, "expected_exit": expected,
                   "legacy_hits": hits, "stdout_sha256": hashlib.sha256(r.stdout).hexdigest(),
                   "stderr_sha256": hashlib.sha256(r.stderr).hexdigest()}
            manifest["cases"].append(row)
            (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            if r.returncode != expected or hits:
                raise RuntimeError("CLI case refused: " + json.dumps(row))
        combined = "\n".join((out/(name+".stdout")).read_text() for name, _, _ in cases)
        for required in ["AIENOS", "Stapleton Doctrine", "COVENANT.md", "AGPL-3.0-or-later"]:
            if required not in combined:
                raise RuntimeError("required current standard absent: " + required)
    manifest["complete"] = True
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"cases": len(manifest["cases"]), "result": "PASS", "source": source}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
