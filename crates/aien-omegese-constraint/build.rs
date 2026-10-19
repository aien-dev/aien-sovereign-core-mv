//! Links omega's Omegese reader as a library (`tools/omegese/omegese_lib.c`: `omegese.c`, unchanged,
//! with one entry point) for [`LinkedOracle`](crate::linked::LinkedOracle). Offline, no make: one
//! C compile of the pinned source. Order:
//!   1. `AIEN_OMEGESE_LIB=<libomegese.a>`: link that archive (a development override).
//!   2. `AIEN_OMEGESE_DIR=<omega checkout>`: its HEAD must equal `omegese.lock`; `omegese_lib.c` is
//!      compiled into OUT_DIR with `$CC` (default `cc`), the flags of omega `mk/omegese.mk`.
//!   3. otherwise: nothing is linked, and `LinkedOracle::new` says how to link it. Nothing in the
//!      workspace needs the library unless it asks for the linked oracle, so the build never fails
//!      for want of it.
//! `cfg(has_omegese_lib)` is set only when the library is linked.
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omegese_lib)");
    for v in ["AIEN_OMEGESE_LIB", "AIEN_OMEGESE_DIR", "CC"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lock = root.join("omegese.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let pinned = std::fs::read_to_string(&lock)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", lock.display()))
        .trim()
        .to_string();
    assert!(
        pinned.len() == 40 && pinned.bytes().all(|b| b.is_ascii_hexdigit()),
        "omegese.lock must hold one full 40-hex sha"
    );
    println!("cargo:rustc-env=AIEN_OMEGESE_PINNED_SHA={pinned}");
    if let Ok(lib) = std::env::var("AIEN_OMEGESE_LIB") {
        let lib = PathBuf::from(lib);
        assert!(lib.is_file(), "AIEN_OMEGESE_LIB {} is not a file", lib.display());
        println!("cargo:rerun-if-changed={}", lib.display());
        println!("cargo:rustc-env=AIEN_OMEGESE_LIB_SOURCE=prebuilt {}", lib.display());
        link(&lib);
        return;
    }
    let Ok(dir) = std::env::var("AIEN_OMEGESE_DIR") else {
        println!("cargo:rustc-env=AIEN_OMEGESE_LIB_SOURCE=none");
        return;
    };
    let dir = PathBuf::from(dir);
    let head = Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    assert!(
        head == pinned,
        "omega checkout {} is at '{head}', omegese.lock pins {pinned}",
        dir.display()
    );
    let src = dir.join("tools/omegese");
    for f in ["omegese_lib.c", "omegese_lib.h", "omegese.c"] {
        println!("cargo:rerun-if-changed={}", src.join(f).display());
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("omegese_lib.o");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let st = Command::new(&cc)
        .args(["-std=gnu11", "-Wall", "-Wextra", "-Werror", "-D_GNU_SOURCE", "-O2", "-fPIC", "-c", "-o"])
        .arg(&obj)
        .arg(src.join("omegese_lib.c"))
        .status()
        .unwrap_or_else(|e| panic!("run {cc}: {e}"));
    assert!(st.success(), "{cc} omegese_lib.c failed");
    let lib = out.join("libomegese.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("run ar");
    assert!(st.success(), "ar libomegese.a failed");
    println!("cargo:rustc-env=AIEN_OMEGESE_LIB_SOURCE=omega {pinned}");
    link(&lib);
}

fn link(lib: &Path) {
    let dir = lib.parent().expect("lib has a parent dir");
    let name = lib
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_prefix("lib"))
        .expect("the archive is named lib<name>.a");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static={name}");
    println!("cargo:rustc-cfg=has_omegese_lib");
}
