//! Links omega `librx_compose.a` (COMPOSITION-2 + the rxc_host ABI facade).
//!
//! Offline: nothing is downloaded. Order:
//!   1. `AIEN_FORCE_CPU_STUB=1`          -> stub (no library; every call returns
//!      `ComposeError::Unavailable`).
//!   2. `AIEN_OMEGA_COMPOSE_LIB=<file>`  -> link that prebuilt archive (not sha-checked).
//!   3. `AIEN_OMEGA_COMPOSE_DIR=<checkout>`, else `AIEN_OMEGA_DIR` (the checkout
//!      aien-omega-gpu builds from) -> its HEAD must equal omega.lock, then
//!      `make <OUT_DIR>/librx_compose.a` runs in it. `AIEN_OMEGA_COMPOSE_SHA=<40-hex>`
//!      replaces the expected sha: a documented developer override for building
//!      against an unmerged omega branch, never used for a release or a receipt. `AIEN_PHYSICS_DIR` (default `<omega>/../physics`) is passed
//!      as PHYSICS_DIR and `AIEN_AIENOS_LOCK_REPO` (if set) as AIENOS_LOCK_REPO;
//!      omega's own lock checks still run.
//!   4. otherwise                        -> stub, with a build warning.
//!
//! `AIEN_OMEGA_COMPOSE_DIR` exists so a developer override never touches the
//! checkout aien-omega-gpu checks against omega.lock.
//! `cfg(has_omega_compose)` is set only when the real library is linked.
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_omega_compose)");
    for v in [
        "AIEN_FORCE_CPU_STUB",
        "AIEN_OMEGA_COMPOSE_LIB",
        "AIEN_OMEGA_COMPOSE_DIR",
        "AIEN_OMEGA_DIR",
        "AIEN_OMEGA_COMPOSE_SHA",
        "AIEN_PHYSICS_DIR",
        "AIEN_AIENOS_LOCK_REPO",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let lock_path = Path::new(&env("CARGO_MANIFEST_DIR")).join("../../omega.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let pinned = std::fs::read_to_string(&lock_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", lock_path.display()))
        .trim()
        .to_string();
    assert!(is_sha(&pinned), "omega.lock must hold one full 40-hex sha");
    let expected = match std::env::var("AIEN_OMEGA_COMPOSE_SHA") {
        Ok(s) => {
            let s = s.trim().to_string();
            assert!(
                is_sha(&s),
                "AIEN_OMEGA_COMPOSE_SHA must be a full 40-hex sha"
            );
            println!("cargo:warning=aien-omega-compose: AIEN_OMEGA_COMPOSE_SHA={s} overrides omega.lock {pinned} (developer override, not omega.lock)");
            s
        }
        Err(_) => pinned.clone(),
    };
    println!("cargo:rustc-env=AIEN_OMEGA_COMPOSE_EXPECTED_SHA={expected}");

    if std::env::var("AIEN_FORCE_CPU_STUB").as_deref() == Ok("1") {
        return;
    }
    if let Ok(lib) = std::env::var("AIEN_OMEGA_COMPOSE_LIB") {
        let lib = PathBuf::from(lib);
        assert!(
            lib.is_file(),
            "AIEN_OMEGA_COMPOSE_LIB {} is not a file",
            lib.display()
        );
        link(&lib);
        return;
    }
    let Ok(dir) =
        std::env::var("AIEN_OMEGA_COMPOSE_DIR").or_else(|_| std::env::var("AIEN_OMEGA_DIR"))
    else {
        println!("cargo:warning=aien-omega-compose: no AIEN_OMEGA_COMPOSE_DIR, AIEN_OMEGA_DIR or AIEN_OMEGA_COMPOSE_LIB, building the stub");
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
        head == expected,
        "omega checkout {} is at '{head}', expected {expected}",
        dir.display()
    );
    let physics = std::env::var("AIEN_PHYSICS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("../physics"));
    let out = PathBuf::from(env("OUT_DIR")).join("omega-compose-build");
    std::fs::create_dir_all(&out).expect("create omega build dir");
    let mut make = Command::new("make");
    make.current_dir(&dir)
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .arg(format!("OUT_DIR={}", out.display()))
        .arg(format!("PHYSICS_DIR={}", physics.display()));
    if let Ok(repo) = std::env::var("AIEN_AIENOS_LOCK_REPO") {
        make.arg(format!("AIENOS_LOCK_REPO={repo}"));
    }
    let status = make
        .arg(out.join("librx_compose.a"))
        .status()
        .expect("run make");
    assert!(
        status.success(),
        "make librx_compose.a failed in {}",
        dir.display()
    );
    link(&out.join("librx_compose.a"));
}

fn link(lib: &Path) {
    let dir = lib.parent().expect("lib has a parent dir");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=rx_compose");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rustc-cfg=has_omega_compose");
    println!("cargo:linked=1");
}

fn is_sha(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"))
}
