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

//! The C-side check against small synthetic headers and archives: one compatible case and one
//! negative case per kind of drift the mission names (equal-size reorder, width, signedness,
//! parameter type, ABI version constant, missing symbol, corrupted archive).
//! Needs `cc`, `ar` and `nm` (the same tools the native build needs).

use aien_abi_conformance::{check_c, CCheck, Manifest};
use std::path::PathBuf;
use std::process::Command;

const MANIFEST: &str = "\
abi demo 2
header demo.h
struct DemoInfo 24 8
field DemoInfo version 0 4 u32 1
field DemoInfo rc 4 4 i32 1
field DemoInfo name 8 4 c_char 4
field DemoInfo ok 12 1 bool 1
field DemoInfo total 16 8 u64 1
const DEMO_ABI_VERSION 2
const DEMO_E_ARG -1
fnptr DemoCb fn(*mut c_void, u64) -> i32
fn demo_open fn(*const c_char, u32, DemoCb, *mut DemoInfo) -> i32
fn demo_close fn(*mut c_void) -> ()
";

const HEADER: &str = "\
#include <stdint.h>
#include <stdbool.h>
#define DEMO_ABI_VERSION 2u
enum { DEMO_OK = 0, DEMO_E_ARG = -1 };
typedef int (*DemoCb)(void *ctx, uint64_t x);
typedef struct {
    uint32_t version;
    int32_t rc;
    char name[4];
    bool ok;
    uint64_t total;
} DemoInfo;
int demo_open(const char *dir, uint32_t n, DemoCb cb, DemoInfo *out);
void demo_close(void *h);
";

const SOURCE: &str = "\
#include \"demo.h\"
int demo_open(const char *dir, uint32_t n, DemoCb cb, DemoInfo *out) { (void)dir; (void)n; (void)cb; (void)out; return 0; }
void demo_close(void *h) { (void)h; }
";

struct Case {
    dir: PathBuf,
}

impl Case {
    fn new(tag: &str, header: &str, source: &str) -> Case {
        let dir = std::env::temp_dir().join(format!("abi-c-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("demo.h"), header).unwrap();
        std::fs::write(dir.join("demo.c"), source).unwrap();
        let ok = Command::new("cc")
            .current_dir(&dir)
            .args(["-c", "-o", "demo.o", "demo.c"])
            .status()
            .unwrap()
            .success();
        assert!(ok, "cc demo.c");
        let ok = Command::new("ar")
            .current_dir(&dir)
            .args(["rcs", "libdemo.a", "demo.o"])
            .status()
            .unwrap()
            .success();
        assert!(ok, "ar");
        Case { dir }
    }

    fn check(&self, manifest: &str) -> Result<String, Vec<String>> {
        let m = Manifest::parse(manifest).unwrap();
        let archive = self.dir.join("libdemo.a");
        let work = self.dir.join("probe");
        check_c(&CCheck {
            manifest: &m,
            root: &self.dir,
            includes: vec![],
            archive: Some(archive.as_path()),
            work: work.as_path(),
            cc: "cc".into(),
        })
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn expect_err(r: Result<String, Vec<String>>, needle: &str) {
    match r {
        Ok(rep) => panic!("expected a failure mentioning `{needle}`, got PASS:\n{rep}"),
        Err(e) => assert!(
            e.iter().any(|l| l.contains(needle)),
            "no error mentions `{needle}`: {e:#?}"
        ),
    }
}

#[test]
fn compatible_header_and_archive_pass() {
    let c = Case::new("ok", HEADER, SOURCE);
    let rep = c.check(MANIFEST).expect("compatible case must pass");
    assert!(
        rep.contains("probe: 8 layout/constant facts match"),
        "{rep}"
    );
}

#[test]
fn equal_size_reorder_is_caught() {
    // version and rc swapped: same size, same alignment, same total
    let h = HEADER.replace(
        "    uint32_t version;\n    int32_t rc;",
        "    int32_t rc;\n    uint32_t version;",
    );
    let c = Case::new("reorder", &h, SOURCE);
    expect_err(c.check(MANIFEST), "reordered");
    expect_err(c.check(MANIFEST), "field DemoInfo version 4 4 u32 1");
}

#[test]
fn width_change_is_caught() {
    let h = HEADER.replace("uint32_t version;", "uint64_t version;");
    let c = Case::new("width", &h, SOURCE);
    expect_err(c.check(MANIFEST), "struct DemoInfo");
}

#[test]
fn signedness_change_is_caught() {
    let h = HEADER.replace("int32_t rc;", "uint32_t rc;");
    let c = Case::new("sign", &h, SOURCE);
    expect_err(c.check(MANIFEST), "field DemoInfo rc 4 4 u32 1");
}

#[test]
fn bool_to_byte_change_is_caught() {
    let h = HEADER.replace("bool ok;", "uint8_t ok;");
    let c = Case::new("bool", &h, SOURCE);
    expect_err(c.check(MANIFEST), "field DemoInfo ok 12 1 u8 1");
}

#[test]
fn parameter_type_change_is_caught() {
    let h = HEADER.replace(
        "int demo_open(const char *dir, uint32_t n,",
        "int demo_open(const char *dir, uint64_t n,",
    );
    let s = SOURCE.replace("uint32_t n", "uint64_t n");
    let c = Case::new("param", &h, &s);
    expect_err(c.check(MANIFEST), "fn demo_open: C signature");
}

#[test]
fn callback_signature_change_is_caught() {
    let h = HEADER.replace("(void *ctx, uint64_t x)", "(void *ctx, uint32_t x)");
    let c = Case::new("cb", &h, SOURCE);
    expect_err(c.check(MANIFEST), "fnptr DemoCb");
}

#[test]
fn abi_version_mismatch_is_caught() {
    let h = HEADER.replace("#define DEMO_ABI_VERSION 2u", "#define DEMO_ABI_VERSION 3u");
    let c = Case::new("ver", &h, SOURCE);
    expect_err(c.check(MANIFEST), "const DEMO_ABI_VERSION 3");
}

#[test]
fn removed_field_is_caught_even_when_size_is_kept() {
    // `ok` dropped: the struct keeps 24 bytes because of padding
    let h = HEADER.replace("    bool ok;\n", "");
    let c = Case::new("drop", &h, SOURCE);
    let r = c.check(MANIFEST);
    expect_err(r, "reordered");
}

#[test]
fn missing_symbol_is_caught() {
    let s = SOURCE.replace("void demo_close(void *h) { (void)h; }\n", "");
    let c = Case::new("sym", HEADER, &s);
    expect_err(c.check(MANIFEST), "symbol demo_close: not defined");
}

#[test]
fn corrupted_archive_is_caught() {
    let c = Case::new("corrupt", HEADER, SOURCE);
    let a = c.dir.join("libdemo.a");
    let bytes = std::fs::read(&a).unwrap();
    std::fs::write(&a, &bytes[..bytes.len() / 3]).unwrap();
    let r = c.check(MANIFEST);
    match r {
        Ok(rep) => panic!("truncated archive passed: {rep}"),
        Err(e) => assert!(
            e.iter()
                .any(|l| l.contains("nm failed") || l.contains("not defined")),
            "{e:#?}"
        ),
    }
}

#[test]
fn missing_header_struct_is_caught() {
    let c = Case::new("nostruct", HEADER, SOURCE);
    let m = format!("{MANIFEST}struct Ghost 8 8\nfield Ghost a 0 8 u64 1\n");
    expect_err(c.check(&m), "struct Ghost: not declared");
}
