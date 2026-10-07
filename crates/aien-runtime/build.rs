//! `cfg(compose_linked)` when aien-omega-compose linked the real
//! librx_compose.a (its build script's `links` metadata), so tests that need
//! the Cortex journal are reported IGNORED in a stub build instead of passing
//! vacuously.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(compose_linked)");
    println!("cargo:rerun-if-env-changed=DEP_RX_COMPOSE_LINKED");
    if std::env::var("DEP_RX_COMPOSE_LINKED").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=compose_linked");
    }
}
