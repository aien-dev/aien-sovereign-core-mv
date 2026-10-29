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

//! C side of the check (used from `build.rs`): headers, a compiled layout probe, and `nm`.

use crate::header;
use crate::manifest::Manifest;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Inputs for [`check_c`].
pub struct CCheck<'a> {
    pub manifest: &'a Manifest,
    /// Directory the manifest's `header` paths are relative to (the omega checkout).
    pub root: &'a Path,
    /// Extra `-I` directories the headers need (omega `src`, physics `m16`, ...).
    pub includes: Vec<PathBuf>,
    /// The archive that will be linked; every manifest `fn` must be a defined global symbol.
    /// `None` skips the symbol check (header-only use in tests).
    pub archive: Option<&'a Path>,
    /// Scratch directory for the probe source and binary.
    pub work: &'a Path,
    /// C compiler (`$CC` or `cc`).
    pub cc: String,
}

/// Run every C-side check. `Ok(report)` lists what was checked; `Err` lists every mismatch.
pub fn check_c(c: &CCheck) -> Result<String, Vec<String>> {
    let m = c.manifest;
    let mut errs = Vec::new();
    let mut report = String::new();

    // 1. Headers: field names and order, prototypes, callback typedefs.
    let mut parsed = header::Header::default();
    for h in &m.headers {
        let p = c.root.join(h);
        match std::fs::read_to_string(&p) {
            Ok(src) => {
                let one = header::parse(&src);
                parsed.structs.extend(one.structs);
                parsed.fns.extend(one.fns);
                parsed.fnptrs.extend(one.fnptrs);
            }
            Err(e) => errs.push(format!("cannot read header {}: {e}", p.display())),
        }
    }
    for (s, ..) in &m.structs {
        let want = m.field_names(s);
        match parsed.structs.get(s) {
            None => errs.push(format!("struct {s}: not declared in the headers")),
            Some(got) if *got != want => errs.push(format!(
                "struct {s}: C fields {got:?} != manifest fields {want:?} (added, removed, renamed or reordered)"
            )),
            Some(_) => {}
        }
    }
    for (n, sig) in &m.fns {
        match parsed.fns.get(n) {
            None => errs.push(format!("fn {n}: no prototype in the headers")),
            Some(got) if got != sig => {
                errs.push(format!("fn {n}: C signature `{got}` != manifest `{sig}`"))
            }
            Some(_) => {}
        }
    }
    for (n, sig) in &m.fnptrs {
        match parsed.fnptrs.get(n) {
            None => errs.push(format!("fnptr {n}: no typedef in the headers")),
            Some(got) if got != sig => errs.push(format!(
                "fnptr {n}: C signature `{got}` != manifest `{sig}`"
            )),
            Some(_) => {}
        }
    }
    let _ = writeln!(
        report,
        "headers: {} structs, {} functions, {} callback types",
        m.structs.len(),
        m.fns.len(),
        m.fnptrs.len()
    );

    // 2. Probe: compiled against the same headers, prints the layout facts.
    match run_probe(c) {
        Ok(lines) => {
            let want: Vec<String> = m
                .fact_lines()
                .into_iter()
                .filter(|l| !l.starts_with("fn") && !l.starts_with("fnptr"))
                .collect();
            for w in &want {
                if !lines.contains(w) {
                    let n = if w.starts_with("field ") { 3 } else { 2 };
                    let key: Vec<&str> = w.split_whitespace().take(n).collect();
                    let key = key.join(" ");
                    let got = lines
                        .iter()
                        .find(|l| l.starts_with(&format!("{key} ")))
                        .cloned()
                        .unwrap_or_else(|| "(absent)".into());
                    errs.push(format!("layout: C `{got}` != manifest `{w}`"));
                }
            }
            let _ = writeln!(report, "probe: {} layout/constant facts match", want.len());
        }
        Err(e) => errs.push(e),
    }

    // 3. Symbols: every declared function is defined by the archive.
    if let Some(a) = c.archive {
        match defined_symbols(a) {
            Ok(syms) => {
                for (n, _) in &m.fns {
                    if !syms.iter().any(|s| s == n) {
                        errs.push(format!("symbol {n}: not defined in {}", a.display()));
                    }
                }
                let _ = writeln!(report, "nm: {} functions defined", m.fns.len());
            }
            Err(e) => errs.push(e),
        }
    }
    if errs.is_empty() {
        Ok(report)
    } else {
        Err(errs)
    }
}

/// The probe's C source for a manifest (public for tests and for the evidence record).
pub fn probe_source(m: &Manifest) -> String {
    let mut s = String::from(
        "#include <stdio.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include <stdint.h>\n",
    );
    for h in &m.headers {
        let _ = writeln!(s, "#include \"{h}\"");
    }
    s.push_str(
        "_Static_assert(sizeof(long) == 8 && sizeof(void *) == 8, \"LP64 only\");\n\
         #define K(x) _Generic((x), _Bool: \"bool\", char: \"c_char\", signed char: \"i8\", \
         unsigned char: \"u8\", short: \"i16\", unsigned short: \"u16\", int: \"i32\", \
         unsigned int: \"u32\", long: \"i64\", unsigned long: \"u64\", long long: \"i64\", \
         unsigned long long: \"u64\", float: \"f32\", double: \"f64\", default: \"other\")\n\
         int main(void) {\n",
    );
    for (st, size, align) in &m.structs {
        let _ = (size, align);
        let _ = writeln!(
            s,
            "    printf(\"struct {st} %zu %zu\\n\", sizeof({st}), _Alignof({st}));"
        );
        for (fs, f, _, _, _, n) in &m.fields {
            if fs != st {
                continue;
            }
            let el = if *n > 1 {
                format!("((({st} *)0)->{f}[0])")
            } else {
                format!("((({st} *)0)->{f})")
            };
            let _ = writeln!(
                s,
                "    printf(\"field {st} {f} %zu %zu %s %zu\\n\", offsetof({st}, {f}), \
                 sizeof((({st} *)0)->{f}), K({el}), sizeof((({st} *)0)->{f}) / sizeof({el}));"
            );
        }
    }
    for (n, _) in &m.consts {
        let _ = writeln!(s, "    printf(\"const {n} %lld\\n\", (long long)({n}));");
    }
    s.push_str("    return 0;\n}\n");
    s
}

fn run_probe(c: &CCheck) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(c.work).map_err(|e| format!("probe dir: {e}"))?;
    let src = c.work.join("abi_probe.c");
    let bin = c.work.join("abi_probe");
    std::fs::write(&src, probe_source(c.manifest)).map_err(|e| format!("probe source: {e}"))?;
    let mut cmd = Command::new(&c.cc);
    cmd.arg("-std=gnu11")
        .arg("-D_GNU_SOURCE")
        .arg("-w")
        .arg("-I")
        .arg(c.root);
    for i in &c.includes {
        cmd.arg("-I").arg(i);
    }
    let out = cmd
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", c.cc))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let first: Vec<&str> = err
            .lines()
            .filter(|l| l.contains("error"))
            .take(6)
            .collect();
        return Err(format!(
            "layout probe did not compile against the headers (a field or constant the manifest names is missing or changed shape): {}",
            first.join(" | ")
        ));
    }
    let run = Command::new(&bin)
        .output()
        .map_err(|e| format!("cannot run the layout probe: {e}"))?;
    if !run.status.success() {
        return Err("layout probe exited nonzero".into());
    }
    Ok(String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(str::to_string)
        .collect())
}

/// Global symbols the archive defines (`nm -g --defined-only`).
pub fn defined_symbols(archive: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("nm")
        .arg("-g")
        .arg("--defined-only")
        .arg(archive)
        .output()
        .map_err(|e| format!("cannot run nm on {}: {e}", archive.display()))?;
    if !out.status.success() {
        return Err(format!(
            "nm failed on {} (corrupt or truncated archive?): {}",
            archive.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            (w.len() == 3 && matches!(w[1], "T" | "D" | "R" | "B")).then(|| w[2].to_string())
        })
        .collect())
}
